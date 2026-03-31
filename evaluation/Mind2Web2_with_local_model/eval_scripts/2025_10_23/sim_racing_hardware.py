import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sim_racing_hardware"
TASK_DESCRIPTION = """
Research and document the exact personal sim racing setups used by the following professional drivers: Max Verstappen, Lando Norris, and Charles Leclerc. For each driver, specify the exact brand and model of their personal wheelbase, steering wheel, pedals, rig frame (cockpit), and monitor, along with purchase links. Also for each driver, identify one online racing competition they have participated in, and provide the simulator platform or game used for the event.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class DriversList(BaseModel):
    """Model for the list of drivers mentioned in the answer"""
    drivers: List[str] = Field(default_factory=list)


class Equipment(BaseModel):
    """Model for a specific piece of sim racing equipment"""
    brand: Optional[str] = None
    model: Optional[str] = None


class OnlineCompetition(BaseModel):
    """Model for online racing competition information"""
    competition_name: Optional[str] = None
    simulator_platform: Optional[str] = None


class DriverSetup(BaseModel):
    """Model for a driver's complete sim racing setup"""
    wheelbase: Optional[Equipment] = None
    steering_wheel: Optional[Equipment] = None
    pedals: Optional[Equipment] = None
    rig_frame: Optional[Equipment] = None
    monitor: Optional[Equipment] = None
    competition: Optional[OnlineCompetition] = None


class ProvLinks(BaseModel):
    """Model for extracted URLs"""
    links: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_drivers_list() -> str:
    """Prompt to extract the list of drivers mentioned in the answer"""
    return """
    Extract a list of the drivers mentioned in the answer about sim racing setups. 
    Only include Max Verstappen, Lando Norris, and Charles Leclerc if they are mentioned.
    Return the list as an array of strings containing only their names.
    """


def prompt_extract_driver_equipment(driver_name: str) -> str:
    """Prompt to extract equipment details for a specific driver"""
    return f"""
    Extract the detailed sim racing equipment information for {driver_name} from the answer.
    For each equipment item (wheelbase, steering wheel, pedals, rig frame/cockpit, and monitor):
    1. Extract the brand name exactly as mentioned.
    2. Extract the model name exactly as mentioned.

    If any information is missing from the answer, set the corresponding field to null.
    """


def prompt_extract_driver_competition(driver_name: str) -> str:
    """Prompt to extract competition details for a specific driver"""
    return f"""
    Extract information about {driver_name}'s participation in online racing competitions from the answer.

    Specifically extract:
    1. The name of one online racing competition {driver_name} has participated in.
    2. The simulator platform or game used for that competition.

    If any information is missing from the answer, set the corresponding field to null.
    """


def prompt_extract_equipment_urls(driver_name: str, equipment_type: str, brand_model: str) -> str:
    """Prompt to extract URLs for a specific piece of equipment"""
    return f"""
    Extract any purchase links or URLs provided in the answer for {driver_name}'s {equipment_type} ({brand_model}).
    Only extract URLs that are specifically for purchasing this exact equipment.

    If no specific URL is provided for this equipment, return an empty list.
    """


def prompt_extract_competition_urls(driver_name: str, competition_name: str) -> str:
    """Prompt to extract URLs for a specific competition"""
    return f"""
    Extract any URLs provided in the answer that are related to {driver_name}'s participation in {competition_name}.
    Only extract URLs that are specifically about this competition or provide evidence of participation.

    If no specific URL is provided for this competition, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Equipment verification functions                                            #
# --------------------------------------------------------------------------- #
async def verify_equipment_item(
        evaluator: Evaluator,
        parent_node,
        equipment: Optional[Equipment],
        equipment_type: str,
        driver_name: str,
        idx: int
) -> None:
    """
    Verify a specific piece of equipment for a driver, including both correctness and provenance.
    """
    # Create node for this specific equipment item - using SEQUENTIAL strategy
    equipment_node = evaluator.add_sequential(
        id=f"{driver_name.lower().replace(' ', '_')}_{equipment_type}_{idx}",
        desc=f"Verification of {driver_name}'s {equipment_type} (brand, model, and purchase link)",
        parent=parent_node,
        critical=False
    )

    # Format the brand/model for display and verification
    brand_model = f"{equipment.brand or ''} {equipment.model or ''}".strip() if equipment else ""

    # Extract all relevant URLs for this equipment
    url_extraction_prompt = f"""
    Extract all URLs mentioned in the answer that are related to {driver_name}'s {equipment_type}.
    Include both informational URLs (that describe their setup) and purchase links.
    Return an empty list if no URLs are provided for this specific equipment item.
    """

    urls = await evaluator.extract(
        prompt=url_extraction_prompt,
        template_class=ProvLinks,
        extraction_name=f"{driver_name}_{equipment_type}_urls"
    )

    # 1. Check if equipment information exists
    equipment_exists = evaluator.add_custom_node(
        result=bool(equipment and (equipment.brand or equipment.model) and urls.links),
        id=f"{driver_name.lower().replace(' ', '_')}_{equipment_type}_exists_{idx}",
        desc=f"Check if {equipment_type} information and URLs exist for {driver_name}",
        parent=equipment_node,
        critical=True
    )

    # 2. Verify equipment specifications correctness with provenance
    specs_node = evaluator.add_leaf(
        id=f"{driver_name.lower().replace(' ', '_')}_{equipment_type}_specs_{idx}",
        desc=f"Verification that {driver_name} uses {brand_model} as their {equipment_type} (with source verification)",
        parent=equipment_node,
        critical=True
    )

    specs_claim = f"{driver_name} uses {brand_model} as their personal {equipment_type} in their sim racing setup"
    await evaluator.verify(
        claim=specs_claim,
        node=specs_node,
        sources=urls.links,
        additional_instruction="Allow for minor variations in the driver's name spelling or common nicknames when verifying this claim."
    )

    # 3. Verify purchase link specifically
    link_node = evaluator.add_leaf(
        id=f"{driver_name.lower().replace(' ', '_')}_{equipment_type}_purchase_link_{idx}",
        desc=f"Verification that a valid purchase link is provided for {driver_name}'s {equipment_type} ({brand_model})",
        parent=equipment_node,
        critical=True
    )

    # Extract URLs specifically for purchasing this equipment
    purchase_url_prompt = f"""
    Extract only the purchase links provided in the answer for {driver_name}'s {equipment_type} ({brand_model}).
    These should be URLs that allow someone to buy this exact equipment.
    Return an empty list if no purchase URLs are provided for this specific equipment item.
    """

    purchase_urls = await evaluator.extract(
        prompt=purchase_url_prompt,
        template_class=ProvLinks,
        extraction_name=f"{driver_name}_{equipment_type}_purchase_urls"
    )

    link_claim = f"This is a valid purchase link for the {brand_model} {equipment_type}"
    await evaluator.verify(
        claim=link_claim,
        node=link_node,
        sources=purchase_urls.links
    )


# --------------------------------------------------------------------------- #
# Competition verification function                                           #
# --------------------------------------------------------------------------- #
async def verify_competition(
        evaluator: Evaluator,
        parent_node,
        competition: Optional[OnlineCompetition],
        driver_name: str,
        idx: int
) -> None:
    """
    Verify an online competition and simulator platform for a driver.
    """
    competition_node = evaluator.add_sequential(
        id=f"{driver_name.lower().replace(' ', '_')}_competition_{idx}",
        desc=f"Verification of {driver_name}'s online racing competition participation and simulator platform",
        parent=parent_node,
        critical=False
    )

    competition_name = competition.competition_name if competition else ""

    # Extract URLs for the competition
    urls = await evaluator.extract(
        prompt=prompt_extract_competition_urls(driver_name, competition_name),
        template_class=ProvLinks,
        extraction_name=f"{driver_name}_competition_urls"
    )

    # 1. Check if competition information exists
    competition_exists = evaluator.add_custom_node(
        result=bool(competition and competition.competition_name and urls.links),
        id=f"{driver_name.lower().replace(' ', '_')}_competition_exists_{idx}",
        desc=f"Check if competition information and URLs exist for {driver_name}",
        parent=competition_node,
        critical=True
    )

    # 2. Verify competition participation
    participation_node = evaluator.add_leaf(
        id=f"{driver_name.lower().replace(' ', '_')}_competition_participation_{idx}",
        desc=f"Verification that {driver_name} has participated in the online racing competition: {competition_name}",
        parent=competition_node,
        critical=False
    )

    participation_claim = f"{driver_name} has participated in the online racing competition called {competition_name}"
    await evaluator.verify(
        claim=participation_claim,
        node=participation_node,
        sources=urls.links,
        additional_instruction="Allow for minor variations in the driver's name spelling or common nicknames when verifying participation."
    )

    # 3. Verify simulator platform
    platform_node = evaluator.add_leaf(
        id=f"{driver_name.lower().replace(' ', '_')}_simulator_platform_{idx}",
        desc=f"Verification that {competition_name} uses {competition.simulator_platform or 'N/A'} as the simulator platform or game",
        parent=competition_node,
        critical=False
    )

    simulator_claim = f"The online racing competition {competition_name} uses {competition.simulator_platform or 'N/A'} as the simulator platform or game"
    await evaluator.verify(
        claim=simulator_claim,
        node=platform_node,
        sources=urls.links
    )


# --------------------------------------------------------------------------- #
# Driver verification function                                                #
# --------------------------------------------------------------------------- #
async def verify_driver_setup(
        evaluator: Evaluator,
        parent_node,
        driver_name: str,
        idx: int
) -> None:
    """
    Verify the complete sim racing setup for a single driver.
    """
    driver_node = evaluator.add_parallel(
        id=f"{driver_name.lower().replace(' ', '_')}_setup_{idx}",
        desc=f"Verification of {driver_name}'s complete sim racing setup including equipment and competition participation",
        parent=parent_node,
        critical=False
    )

    # Check if driver is included in the answer
    inclusion_claim = f"The answer includes information about {driver_name}'s sim racing setup"
    driver_included = await evaluator.verify(
        claim=inclusion_claim,
        node=None,
        additional_instruction="Check if the answer contains any information about this driver's sim racing equipment or competition participation. Allow for minor variations in name spelling, capitalization. The key is to identify if this specific person is being discussed."
    )
    
    # Add critical existence check
    driver_exists = evaluator.add_custom_node(
        result=driver_included,
        id=f"{driver_name.lower().replace(' ', '_')}_exists_{idx}",
        desc=f"Check if {driver_name} is included in the answer",
        parent=driver_node,
        critical=True
    )

    # 1. Extract and verify equipment details
    driver_equipment = await evaluator.extract(
        prompt=prompt_extract_driver_equipment(driver_name),
        template_class=DriverSetup,
        extraction_name=f"{driver_name}_equipment"
    )

    equipment_node = evaluator.add_parallel(
        id=f"{driver_name.lower().replace(' ', '_')}_equipment_{idx}",
        desc=f"Verification of {driver_name}'s sim racing equipment (wheelbase, steering wheel, pedals, rig frame, monitor)",
        parent=driver_node,
        critical=False
    )

    # Verify each equipment item individually
    equipment_types = [
        ("wheelbase", driver_equipment.wheelbase),
        ("steering_wheel", driver_equipment.steering_wheel),
        ("pedals", driver_equipment.pedals),
        ("rig_frame", driver_equipment.rig_frame),
        ("monitor", driver_equipment.monitor)
    ]

    for eq_idx, (equipment_type, equipment) in enumerate(equipment_types):
        await verify_equipment_item(
            evaluator=evaluator,
            parent_node=equipment_node,
            equipment=equipment,
            equipment_type=equipment_type,
            driver_name=driver_name,
            idx=eq_idx
        )

    # 2. Extract and verify competition participation
    driver_competition = await evaluator.extract(
        prompt=prompt_extract_driver_competition(driver_name),
        template_class=OnlineCompetition,
        extraction_name=f"{driver_name}_competition"
    )

    await verify_competition(
        evaluator=evaluator,
        parent_node=driver_node,
        competition=driver_competition,
        driver_name=driver_name,
        idx=idx
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer to the sim racing hardware task.
    """
    # Set up evaluator
    evaluator = Evaluator()
    
    # Initialize evaluator with parallel strategy for root
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
        default_model=model if model else JUDGE_MODEL
    )

    # Expected driver names for this task
    expected_drivers = ["Max Verstappen", "Lando Norris", "Charles Leclerc"]

    # Process each expected driver uniformly - add directly to root
    for idx, expected_driver in enumerate(expected_drivers):
        await verify_driver_setup(
            evaluator=evaluator,
            parent_node=root,  # Add directly to root
            driver_name=expected_driver,
            idx=idx
        )

    # Collect extracted information for debugging and reporting
    evaluator.add_custom_info({
        "expected_drivers": expected_drivers
    }, "driver_coverage")

    # Return structured result
    return evaluator.get_summary()