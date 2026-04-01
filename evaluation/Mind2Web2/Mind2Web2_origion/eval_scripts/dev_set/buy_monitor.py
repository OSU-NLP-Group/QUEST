import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import VerificationNode, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "buy_monitor"
TASK_DESCRIPTION = """
I'm interested in buying a 34'' curved monitor. Find at least 5 models and a minimum of 3 websites to purchase online for each model. For each website, please provide the link to the specific purchasing page along with its price.
"""
REQUIRED_MODELS = 5
REQUIRED_WEBSITES_PER_MODEL = 3

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Website(BaseModel):
    """Represents a website where a monitor can be purchased."""
    url: Optional[str] = None
    price: Optional[str] = None

class Monitor(BaseModel):
    """Represents a monitor model with its purchasing websites."""
    model_name: Optional[str] = None
    websites: List[Website] = Field(default_factory=list)

class MonitorName(BaseModel):
    """Simple model to extract just the monitor name."""
    model_name: Optional[str] = None

class ExtractedMonitorNames(BaseModel):
    """Container for all extracted monitor model names."""
    monitors: List[MonitorName] = Field(default_factory=list)

class ExtractedWebsites(BaseModel):
    """Container for websites extracted for a specific monitor."""
    websites: List[Website] = Field(default_factory=list)

class ExtractedMonitors(BaseModel):
    """Container for all extracted monitor models with their websites."""
    monitors: List[Monitor] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_monitor_names() -> str:
    return """
    Extract all 34" curved monitor models mentioned in the answer. Focus only on extracting the model names at this stage.
    
    Format requirements:
    - Extract each monitor model name as precisely as stated in the answer
    - Extract only the name/identifier of the monitor, without additional details
    - Do not include website or price information at this stage
    - Do not invent or add any information that is not explicitly mentioned in the answer
    - If the model name includes a model number, brand, or specific identifier, include this information
    """

def prompt_extract_websites_for_monitor(model_name: str) -> str:
    return f"""
    Extract all websites where the monitor model '{model_name}' can be purchased, according to the answer.
    
    For each website:
    1. Extract the URL of the purchasing page
    2. Extract the price mentioned for this specific monitor model
    
    Format requirements:
    - Extract all websites mentioned for this specific monitor model
    - Extract both the URL and price for each website if available
    - If any information is missing (e.g., a URL or price is not provided), set the corresponding field to null
    - Do not invent or add any information that is not explicitly mentioned in the answer
    - Only include websites that are specifically associated with the '{model_name}' model
    """


# --------------------------------------------------------------------------- #
# Verification functions for monitors and websites                            #
# --------------------------------------------------------------------------- #
async def verify_is_34inch_curved_monitor(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    monitor: Monitor,
    model_index: int
) -> None:
    """
    Verify that the monitor is a 34-inch curved monitor.
    This is a critical check for each monitor.
    """
    # Create parallel node for verification
    verification_parent = evaluator.add_parallel(
        id=f"model_{model_index}_34inch_verification",
        desc=f"Model {model_index}: Verification that '{monitor.model_name}' is a 34-inch curved monitor",
        parent=parent_node,
        critical=True
    )
    
    # Add existence check
    existence_check = evaluator.add_custom_node(
        result=bool(monitor.model_name),
        id=f"model_{model_index}_name_exists",
        desc=f"Model {model_index}: Monitor name exists",
        parent=verification_parent,
        critical=True
    )
    
    # Add verification node
    is_34inch_node = evaluator.add_leaf(
        id=f"model_{model_index}_is_34inch_curved",
        desc=f"Model {model_index}: The monitor '{monitor.model_name}' is a 34-inch curved monitor",
        parent=verification_parent,
        critical=True
    )
    
    # Verify that this is a 34-inch curved monitor
    claim = f"The '{monitor.model_name}' is a 34-inch curved monitor."
    
    # Collect all URLs for the model to use for verification
    urls = [website.url for website in monitor.websites if website.url]
    
    await evaluator.verify(
        claim=claim,
        node=is_34inch_node,
        sources=urls if urls else None,
        additional_instruction="Verify that the monitor is specifically described as a 34-inch curved monitor."
    )


async def verify_single_website(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    website: Website,
    model_name: str,
    model_index: int,
    website_index: int
) -> None:
    """
    Verify all aspects of a single website listing.
    """
    website_node = evaluator.add_parallel(
        id=f"model_{model_index}_website_{website_index}",
        desc=f"Model {model_index}, Website {website_index}: Complete verification of purchase website for '{model_name}'",
        parent=parent_node,
        critical=True
    )
    
    # Add existence checks
    data_exists_node = evaluator.add_custom_node(
        result=bool(website.url) and bool(website.price),
        id=f"model_{model_index}_website_{website_index}_data_exists",
        desc=f"Model {model_index}, Website {website_index}: Website has both URL and price",
        parent=website_node,
        critical=True
    )
    
    # 1. Verify the URL sells the specified monitor
    sells_monitor_node = evaluator.add_leaf(
        id=f"model_{model_index}_website_{website_index}_sells_monitor",
        desc=f"Model {model_index}, Website {website_index}: The URL '{website.url}' sells the '{model_name}' monitor",
        parent=website_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The URL '{website.url}' is for a page that sells the '{model_name}' monitor.",
        node=sells_monitor_node,
        sources=website.url,
        additional_instruction="Verify that the page is specifically for selling the mentioned monitor model."
    )
    
    # 2. Verify the price matches what's on the URL
    price_matches_node = evaluator.add_leaf(
        id=f"model_{model_index}_website_{website_index}_price_matches",
        desc=f"Model {model_index}, Website {website_index}: The stated price '{website.price}' matches the page",
        parent=website_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The stated price '{website.price}' for the '{model_name}' monitor appears on the purchase page at '{website.url}'.",
        node=price_matches_node,
        sources=website.url,
        additional_instruction="Accept the price as correct if it matches either the original price or any deal/special price shown on the page. Account for minor formatting differences in how the price is displayed."
    )


async def verify_single_monitor(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    monitor: Monitor,
    model_index: int
) -> None:
    """
    Verify a single monitor model and all its websites.
    """
    monitor_node = evaluator.add_sequential(
        id=f"model_{model_index}",
        desc=f"Model {model_index}: Complete verification of monitor model and its websites",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit
    )
    
    # Verify this is a 34-inch curved monitor
    await verify_is_34inch_curved_monitor(
        evaluator=evaluator,
        parent_node=monitor_node,
        monitor=monitor,
        model_index=model_index
    )
    
    # Create websites parent node
    websites_node = evaluator.add_parallel(
        id=f"model_{model_index}_websites",
        desc=f"Model {model_index}: Verification of all websites for '{monitor.model_name}'",
        parent=monitor_node,
        critical=True
    )
    
    # Verify each of the required websites (3)
    while len(monitor.websites) < REQUIRED_WEBSITES_PER_MODEL:
        monitor.websites.append(Website())
    
    for website_index in len(monitor.websites[:REQUIRED_WEBSITES_PER_MODEL]):
        await verify_single_website(
            evaluator=evaluator,
            parent_node=websites_node,
            website=monitor.websites[website_index],
            model_name=monitor.model_name,
            model_index=model_index,
            website_index=website_index
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------------- #
    evaluator = Evaluator()
    evaluator.initialize(
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

    # -------- 2. Extract structured info from the answer ---------------- #
    # First, extract all monitor model names
    parsed_monitor_names = await evaluator.extract(
        prompt=prompt_extract_monitor_names(),
        template_class=ExtractedMonitorNames,
        extraction_name="monitor_names"
    )
    
    # Create a container for the fully extracted monitors with websites
    fully_extracted_monitors = ExtractedMonitors(monitors=[])
    
    # For each monitor, extract its websites
    for monitor_name_obj in parsed_monitor_names.monitors:
        if monitor_name_obj.model_name:
            # Extract websites for this specific monitor
            websites = await evaluator.extract(
                prompt=prompt_extract_websites_for_monitor(monitor_name_obj.model_name),
                template_class=ExtractedWebsites,
                extraction_name=f"websites_for_{monitor_name_obj.model_name}"
            )
            
            # Create a complete monitor object with its websites
            monitor = Monitor(
                model_name=monitor_name_obj.model_name,
                websites=websites.websites
            )
            
            # Add to our full container
            fully_extracted_monitors.monitors.append(monitor)
    
    logger.info(f"Extracted {len(fully_extracted_monitors.monitors)} monitors with their websites")

    # Pad to ensure we have REQUIRED_MODELS monitors (even if empty)
    while len(fully_extracted_monitors.monitors) < REQUIRED_MODELS:
        fully_extracted_monitors.monitors.append(Monitor())

    # -------- 3. Build verification tree -------------------------------- #
    # Verify each of the required models (5)
    for model_index in range(REQUIRED_MODELS):
        await verify_single_monitor(
            evaluator=evaluator,
            parent_node=evaluator.root,
            monitor=fully_extracted_monitors.monitors[model_index],
            model_index=model_index
        )

    # -------- 4. Get final score and summary ---------------------------- #
    final_score = evaluator.score()
    
    # Add custom information
    evaluator.add_custom_info(
        {
            "monitors_found": len([m for m in fully_extracted_monitors.monitors if m.model_name]),
            "required_monitors": REQUIRED_MODELS,
            "required_websites_per_model": REQUIRED_WEBSITES_PER_MODEL
        },
        info_type="task_requirements"
    )

    # -------- 5. Return structured result ------------------------------- #
    return evaluator.get_summary()