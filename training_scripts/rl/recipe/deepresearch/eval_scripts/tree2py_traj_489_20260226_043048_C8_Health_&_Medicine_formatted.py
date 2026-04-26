import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "norovirus_cruise_2025"
TASK_DESCRIPTION = (
    "During the 2024-2025 norovirus season, multiple cruise ship outbreaks were reported to the CDC's Vessel "
    "Sanitation Program. Identify one cruise ship norovirus outbreak in 2025 that occurred during the peak norovirus "
    "season (November through April) and had a passenger illness rate exceeding 10%. For the identified outbreak, "
    "provide the following details: (1) cruise line name, (2) ship name, (3) voyage start date, (4) voyage end date, "
    "(5) total number of passengers onboard, (6) number of passengers who were ill, (7) passenger illness percentage, "
    "(8) total number of crew onboard, (9) number of crew who were ill, (10) crew illness percentage, "
    "(11) total onboard population, (12) total number of people ill, "
    "(13) overall illness percentage of the total onboard population, (14) predominant symptoms reported, "
    "and (15) the CDC Vessel Sanitation Program reference URL for this specific outbreak."
)

OUTBREAK_YEAR = 2025
PEAK_SEASON_MONTHS = ["November", "December", "January", "February", "March", "April"]

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class OutbreakInfo(BaseModel):
    cruise_line: Optional[str] = None
    ship_name: Optional[str] = None
    voyage_start_date: Optional[str] = None
    voyage_end_date: Optional[str] = None

    total_passengers: Optional[str] = None
    passengers_ill: Optional[str] = None
    passenger_illness_percentage: Optional[str] = None

    total_crew: Optional[str] = None
    crew_ill: Optional[str] = None
    crew_illness_percentage: Optional[str] = None

    total_onboard_population: Optional[str] = None
    total_people_ill: Optional[str] = None
    overall_illness_percentage: Optional[str] = None

    predominant_symptoms: List[str] = Field(default_factory=list)

    cdc_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outbreak() -> str:
    return (
        "From the answer, extract exactly one cruise ship norovirus outbreak that meets ALL of the following:\n"
        "- Occurred in the calendar year 2025;\n"
        "- Took place during the peak norovirus season (November through April);\n"
        "- Passenger illness rate exceeded 10%;\n"
        "- Includes a CDC Vessel Sanitation Program (VSP) outbreak reference URL specific to that outbreak.\n\n"
        "If multiple outbreaks are mentioned, choose the first one that satisfies the above constraints. "
        "If none satisfy, return null for all fields and an empty URL list.\n\n"
        "For the chosen outbreak, extract the following fields exactly as written in the answer:\n"
        "1) cruise_line: cruise line name;\n"
        "2) ship_name: ship name;\n"
        "3) voyage_start_date: voyage start date (string, keep formatting as-is);\n"
        "4) voyage_end_date: voyage end date (string, keep formatting as-is);\n"
        "5) total_passengers: total number of passengers onboard (string; allow commas/words);\n"
        "6) passengers_ill: number of passengers who were ill (string);\n"
        "7) passenger_illness_percentage: passenger illness percentage (string, e.g., '12.3%');\n"
        "8) total_crew: total number of crew onboard (string);\n"
        "9) crew_ill: number of crew who were ill (string);\n"
        "10) crew_illness_percentage: crew illness percentage (string);\n"
        "11) total_onboard_population: passengers + crew (string); if not given, return null;\n"
        "12) total_people_ill: passengers ill + crew ill (string); if not given, return null;\n"
        "13) overall_illness_percentage: total ill / total onboard population (string); if not given, return null;\n"
        "14) predominant_symptoms: list of symptoms mentioned (e.g., ['vomiting','diarrhea']);\n"
        "15) cdc_reference_urls: list of the CDC VSP outbreak URL(s) cited for this specific outbreak; "
        "extract only valid URLs explicitly present in the answer.\n\n"
        "Do not invent or infer any values. Use null for missing fields. "
        "For URLs, extract actual hyperlinks or plain URLs exactly as shown."
    )


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_outbreak_identification(
    evaluator: Evaluator,
    parent_node,
    outbreak: OutbreakInfo,
) -> None:
    """
    Build and verify the 'outbreak_identification' subtree.
    This node is critical: if it fails, subsequent details are skipped due to the root being sequential.
    """
    ident_node = evaluator.add_parallel(
        id="outbreak_identification",
        desc="Correctly identify the cruise ship outbreak that meets all specified criteria",
        parent=parent_node,
        critical=True,
    )

    primary_url: Optional[str] = outbreak.cdc_reference_urls[0] if outbreak.cdc_reference_urls else None

    # Critical gate: CDC VSP source must be provided
    cdc_source_available = evaluator.add_custom_node(
        result=bool(primary_url and primary_url.strip()),
        id="cdc_source_available",
        desc="CDC Vessel Sanitation Program outbreak reference URL is provided in the answer",
        parent=ident_node,
        critical=True,
    )

    # Critical gates: ship name and cruise line must be provided to form concrete claims
    cruise_line_provided = evaluator.add_custom_node(
        result=bool(outbreak.cruise_line and outbreak.cruise_line.strip()),
        id="cruise_line_provided",
        desc="Cruise line name is provided in the answer",
        parent=ident_node,
        critical=True,
    )
    ship_name_provided = evaluator.add_custom_node(
        result=bool(outbreak.ship_name and outbreak.ship_name.strip()),
        id="ship_name_provided",
        desc="Ship name is provided in the answer",
        parent=ident_node,
        critical=True,
    )

    # Leaf: Cruise line
    cruise_line_leaf = evaluator.add_leaf(
        id="cruise_line",
        desc="Provide the correct cruise line name",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The cruise line for this outbreak was '{outbreak.cruise_line}'.",
        node=cruise_line_leaf,
        sources=primary_url,
        additional_instruction=(
            "Verify on the CDC VSP outbreak page. Allow minor naming variants (e.g., 'Royal Caribbean' vs "
            "'Royal Caribbean International'). Case-insensitive comparison."
        ),
    )

    # Leaf: Ship name
    ship_name_leaf = evaluator.add_leaf(
        id="ship_name",
        desc="Provide the correct ship name",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The ship involved in this outbreak was '{outbreak.ship_name}'.",
        node=ship_name_leaf,
        sources=primary_url,
        additional_instruction="Verify the ship name stated on the CDC VSP outbreak page. Case-insensitive comparison.",
    )

    # Leaf: Outbreak year is 2025
    outbreak_year_leaf = evaluator.add_leaf(
        id="outbreak_year",
        desc="Outbreak occurred in 2025",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"This outbreak occurred in {OUTBREAK_YEAR}.",
        node=outbreak_year_leaf,
        sources=primary_url,
        additional_instruction=(
            "Check the voyage start/end dates and confirm the year is 2025. Either start or end date year equal to 2025 "
            "is acceptable."
        ),
    )

    # Leaf: Peak season timing (Nov–Apr)
    peak_season_leaf = evaluator.add_leaf(
        id="peak_season_timing",
        desc="Outbreak occurred during norovirus peak season (November through April)",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This outbreak occurred during the norovirus peak season (November through April).",
        node=peak_season_leaf,
        sources=primary_url,
        additional_instruction=(
            "Use the voyage dates on the CDC page. Peak season months are November, December, January, February, March, "
            "and April. If the voyage dates fall entirely within these months, or start/end within these months, consider it within peak season."
        ),
    )

    # Leaf: Passenger illness rate exceeded 10%
    pax_threshold_leaf = evaluator.add_leaf(
        id="passenger_illness_threshold",
        desc="Passenger illness rate exceeded 10%",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The passenger illness percentage exceeded 10%.",
        node=pax_threshold_leaf,
        sources=primary_url,
        additional_instruction=(
            "Check the 'passenger illness percentage' reported on the CDC page and confirm it is strictly greater than 10%."
        ),
    )

    # Leaf: Causative agent was norovirus
    agent_leaf = evaluator.add_leaf(
        id="causative_agent",
        desc="Causative agent confirmed as norovirus",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The causative agent for this outbreak was norovirus.",
        node=agent_leaf,
        sources=primary_url,
        additional_instruction="Confirm that the CDC page explicitly states norovirus as the causative agent.",
    )


async def verify_outbreak_details(
    evaluator: Evaluator,
    parent_node,
    outbreak: OutbreakInfo,
) -> None:
    """
    Build and verify the 'outbreak_details' subtree.
    This node is non-critical to allow partial credit across many factual fields.
    """
    details_node = evaluator.add_parallel(
        id="outbreak_details",
        desc="Provide accurate and comprehensive details about the identified outbreak",
        parent=parent_node,
        critical=False,
    )

    primary_url: Optional[str] = outbreak.cdc_reference_urls[0] if outbreak.cdc_reference_urls else None

    # Helper to add detail leaf with optional existence gate (non-critical)
    async def _add_detail_leaf(
        node_id: str,
        desc: str,
        value: Optional[str],
        claim_prefix: str,
        additional_instruction: str,
    ):
        provided_gate = evaluator.add_custom_node(
            result=bool(value and str(value).strip()),
            id=f"{node_id}_provided",
            desc=f"{desc} is provided in the answer",
            parent=details_node,
            critical=False,  # Non-critical: missing values should not fail entire details node
        )

        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=details_node,
            critical=False,
        )

        # If the value is missing, skip verification gracefully via extra prerequisite
        await evaluator.verify(
            claim=f"{claim_prefix} {value}.",
            node=leaf,
            sources=primary_url,
            additional_instruction=additional_instruction,
            extra_prerequisites=[provided_gate],
        )

    # Voyage dates
    await _add_detail_leaf(
        node_id="voyage_start_date",
        desc="Provide the correct voyage start date",
        value=outbreak.voyage_start_date,
        claim_prefix="The voyage start date was",
        additional_instruction="Verify the start date on the CDC VSP outbreak page; allow minor formatting differences.",
    )
    await _add_detail_leaf(
        node_id="voyage_end_date",
        desc="Provide the correct voyage end date",
        value=outbreak.voyage_end_date,
        claim_prefix="The voyage end date was",
        additional_instruction="Verify the end date on the CDC VSP outbreak page; allow minor formatting differences.",
    )

    # Passenger counts and percentage
    await _add_detail_leaf(
        node_id="total_passengers",
        desc="Provide the correct total number of passengers onboard",
        value=outbreak.total_passengers,
        claim_prefix="The total number of passengers onboard was",
        additional_instruction="Verify the passenger total on the CDC page; allow commas and minor formatting.",
    )
    await _add_detail_leaf(
        node_id="passengers_ill",
        desc="Provide the correct number of passengers who were ill",
        value=outbreak.passengers_ill,
        claim_prefix="The number of passengers who were ill was",
        additional_instruction="Verify the passenger illness count on the CDC page; allow commas and minor formatting.",
    )
    await _add_detail_leaf(
        node_id="passenger_illness_percentage",
        desc="Provide the correct passenger illness percentage",
        value=outbreak.passenger_illness_percentage,
        claim_prefix="The passenger illness percentage was",
        additional_instruction=(
            "Verify the passenger illness percentage on the CDC page; allow small rounding differences "
            "and formatting such as a trailing '%'."
        ),
    )

    # Crew counts and percentage
    await _add_detail_leaf(
        node_id="total_crew",
        desc="Provide the correct total number of crew onboard",
        value=outbreak.total_crew,
        claim_prefix="The total number of crew onboard was",
        additional_instruction="Verify the crew total on the CDC page; allow commas and minor formatting.",
    )
    await _add_detail_leaf(
        node_id="crew_ill",
        desc="Provide the correct number of crew who were ill",
        value=outbreak.crew_ill,
        claim_prefix="The number of crew who were ill was",
        additional_instruction="Verify the crew illness count on the CDC page; allow commas and minor formatting.",
    )
    await _add_detail_leaf(
        node_id="crew_illness_percentage",
        desc="Provide the correct crew illness percentage",
        value=outbreak.crew_illness_percentage,
        claim_prefix="The crew illness percentage was",
        additional_instruction=(
            "Verify the crew illness percentage on the CDC page; allow small rounding differences "
            "and formatting such as a trailing '%'."
        ),
    )

    # Totals and overall percentage (may be computed)
    await _add_detail_leaf(
        node_id="total_onboard_population",
        desc="Provide the correct total onboard population (passengers + crew)",
        value=outbreak.total_onboard_population,
        claim_prefix="The total onboard population (passengers + crew) was",
        additional_instruction=(
            "If the page shows passengers and crew totals but not the combined number, confirm that the value equals "
            "passengers + crew. Allow minor formatting differences (e.g., commas)."
        ),
    )
    await _add_detail_leaf(
        node_id="total_people_ill",
        desc="Provide the correct total number of people ill (passengers + crew)",
        value=outbreak.total_people_ill,
        claim_prefix="The total number of people ill (passengers + crew) was",
        additional_instruction=(
            "If the page shows passengers ill and crew ill but not the combined number, confirm that the value equals "
            "passengers ill + crew ill. Allow minor formatting differences (e.g., commas)."
        ),
    )
    await _add_detail_leaf(
        node_id="overall_illness_percentage",
        desc="Provide the correct overall percentage of people ill from total onboard population",
        value=outbreak.overall_illness_percentage,
        claim_prefix="The overall illness percentage (total ill / total onboard population) was",
        additional_instruction=(
            "If the page provides counts for total ill and total onboard, you may verify this percentage via computation. "
            "Allow reasonable rounding differences (e.g., 12.6% ≈ 12.6 percent)."
        ),
    )

    # Predominant symptoms
    symptoms_text = ", ".join(outbreak.predominant_symptoms) if outbreak.predominant_symptoms else None
    await _add_detail_leaf(
        node_id="predominant_symptoms",
        desc="Provide the predominant symptoms reported during the outbreak",
        value=symptoms_text,
        claim_prefix="The predominant symptoms reported were",
        additional_instruction=(
            "Verify the symptom list on the CDC page. Allow synonyms (e.g., vomiting/emesis, stomach cramps/abdominal cramps) "
            "and minor ordering differences."
        ),
    )

    # CDC reference URL verification (non-critical leaf)
    cdc_url_leaf = evaluator.add_leaf(
        id="cdc_reference_url",
        desc="Provide the CDC Vessel Sanitation Program URL for this specific outbreak",
        parent=details_node,
        critical=False,
    )
    await evaluator.verify(
        claim="This webpage is the CDC Vessel Sanitation Program outbreak report for the specific cruise ship outbreak.",
        node=cdc_url_leaf,
        sources=primary_url,
        additional_instruction=(
            "Confirm the page belongs to CDC and corresponds to a VSP outbreak report (ship- and voyage-specific). "
            "Look for indicators like 'Vessel Sanitation Program', 'Outbreak', ship name, voyage dates, and norovirus."
        ),
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an agent's answer for the 2025 cruise ship norovirus outbreak task.
    """
    evaluator = Evaluator()
    # IMPORTANT: Root is non-critical to allow non-critical detail leaves under its children per framework constraints.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Identification first; if it fails, details are skipped
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

    # Extract outbreak info from the answer
    outbreak = await evaluator.extract(
        prompt=prompt_extract_outbreak(),
        template_class=OutbreakInfo,
        extraction_name="selected_outbreak",
    )

    # Record helpful custom info (e.g., selected CDC URL and constraints)
    primary_url: Optional[str] = outbreak.cdc_reference_urls[0] if outbreak.cdc_reference_urls else None
    evaluator.add_custom_info(
        info={
            "selected_cdc_url": primary_url or "None",
            "constraints": {
                "year": OUTBREAK_YEAR,
                "peak_season_months": PEAK_SEASON_MONTHS,
                "passenger_illness_threshold": "> 10%",
            },
        },
        info_type="selection_context",
    )

    # Build identification and details verification
    await verify_outbreak_identification(evaluator, root, outbreak)
    await verify_outbreak_details(evaluator, root, outbreak)

    # Return structured summary
    return evaluator.get_summary()