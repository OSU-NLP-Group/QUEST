import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import create_evaluator, Extractor, Verifier
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator
from mind2web2.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "kids_stem"
TASK_DESCRIPTION = """
My 10-year-old is really into science and tech, and I'd love to plan a fun and educational day for them in Chicago. Can you help me design a one-day itinerary with **three different stops**, all offering **hands-on or interactive STEM-related experiences** (science, technology, engineering, or math)? 

For each place, please include:

- The **name**, **address**, and a link to the **official website**  
- The **ticket prices** for one child and one adult  
- One **unique interactive exhibit or activity** the location offers

Make sure each stop has a **different kind of hands-on experience**. Thanks!
"""

# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class TicketPrice(BaseModel):
    """Represents ticket pricing for a location."""
    child_price: Optional[str] = None
    adult_price: Optional[str] = None

class LocationDetails(BaseModel):
    """Represents the basic details of a STEM location without activity info."""
    address: Optional[str] = None
    website: Optional[str] = None
    ticket_prices: Optional[TicketPrice] = None

class LocationNames(BaseModel):
    """Model to extract just the names of the STEM locations."""
    names: List[str] = Field(default_factory=list)

class ActivityDescription(BaseModel):
    """Model to extract just the activity description."""
    description: Optional[str] = None

class LocationURLs(BaseModel):
    """Model to extract all URLs for a specific location."""
    urls: List[str] = Field(default_factory=list)

class ActivityAnalysis(BaseModel):
    """Model for analyzing the nature of an activity."""
    is_interactive: Optional[bool] = None
    is_hands_on: Optional[bool] = None
    is_stem: Optional[bool] = None   # Science, Technology, Engineering, Math

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_location_names() -> str:
    return """
    Extract the names of all STEM locations/destinations mentioned in the answer.
    
    The task asked for three different stops in Chicago offering hands-on or interactive STEM-related experiences.
    Return just the names of these locations in a list.
    
    Include all locations mentioned in the answer (up to 5) even if there are more than three.
    If no locations are mentioned, return an empty list.
    """

def prompt_extract_location_details(location_name: str) -> str:
    return f"""
    Extract the details for the STEM location named "{location_name}" from the answer.
    
    Extract the following information:
    1. The complete address (if provided)
    2. The website URL (if provided)
    3. The ticket prices for one child and one adult (if provided)
    
    If any information is missing, set that field to null.
    If the ticket price is given as a range or approximate value, include that as stated.
    DO NOT include information about activities or exhibits - those will be extracted separately.
    """

def prompt_extract_location_activity(location_name: str) -> str:
    return f"""
    Extract information about the interactive exhibit or activity mentioned for the location named "{location_name}" in the answer.
    
    Extract a description of one unique interactive exhibit or activity that this location offers.
    Focus ONLY on the description of the activity itself, not other details about the location.
    
    If no interactive activity is mentioned for this location, return null for the description.
    """

def prompt_extract_location_urls(location_name: str) -> str:
    return f"""
    Extract all URLs mentioned in the answer that are associated with the location: "{location_name}".
    
    This includes:
    1. The official website URL
    2. Any other URLs that might contain information about this location
    
    Return the URLs as a list. If no URLs are found, return an empty list.
    """

def prompt_analyze_stem_activity(activity_description: str) -> str:
    return f"""
    Analyze the following activity description to determine if it's interactive, hands-on, and which STEM category it belongs to:
    
    "{activity_description}"
    
    Determine:
    1. Is the activity interactive? (Does it involve user participation?)
    2. Is it hands-on? (Does it involve physical manipulation or creation?)
    3. Is the activity STEM-related? (Science, Technology, Engineering, or Math)
    
    If you cannot determine any of these aspects, set the corresponding field to null.
    """

# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_location(
    evaluator: Evaluator,
    parent_node,
    location_data: Dict,
    idx: int,
    location_urls: List[str]
) -> Dict:
    """Verify all aspects of a single location."""
    
    # Create location node
    location_node = evaluator.add_parallel(
        id=f"location_{idx}",
        desc=f"Location {idx}: {location_data['name'] or 'Not provided'} meets the requirements",
        parent=parent_node,
        critical=False
    )
    
    # Add completeness check
    completeness = evaluator.add_custom_node(
        result=all(location_data[field] is not None for field in ['name', 'address', 'website', 'activity_description']),
        id=f"location_{idx}_completeness",
        desc=f"Location {idx} has all required information",
        parent=location_node,
        critical=True
    )

    # Verify location exists
    existence_node = evaluator.add_leaf(
        id=f"location_{idx}_exists",
        desc=f"Location {idx} named '{location_data['name']}' exists as a STEM-related destination in Chicago",
        parent=location_node,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The following is a real STEM-related destination in Chicago: {location_data['name']}",
        node=existence_node,
        sources=location_urls,
        additional_instruction="Check if this is a real place that exists in Chicago and offers STEM-related activities."
    )
    
    # Verify address
    address_node = evaluator.add_leaf(
        id=f"location_{idx}_address",
        desc=f"Location {idx} '{location_data['name']}' has an accurate address in Chicago: '{location_data['address']}'",
        parent=location_node,
        critical=False
    )
    
    await evaluator.verify(
        claim=f"The address '{location_data['address']}' is the correct address for {location_data['name']} in Chicago",
        node=address_node,
        sources=location_urls
    )
    
    # Verify website
    website_node = evaluator.add_leaf(
        id=f"location_{idx}_website",
        desc=f"Location {idx} '{location_data['name']}' has a valid official website URL: '{location_data['website']}'",
        parent=location_node,
        critical=False
    )
    
    await evaluator.verify(
        claim=f"This webpage (URL: {location_data['website']}) is the official website for {location_data['name']}",
        node=website_node,
        sources=[location_data['website']],
        additional_instruction="Check if this URL leads to the official website of the mentioned location. Look for matching name and confirm it's an official site, not a third-party booking or review site."
    )
    
    # Verify ticket prices
    prices_node = evaluator.add_leaf(
        id=f"location_{idx}_ticket_prices",
        desc=f"Location {idx} '{location_data['name']}' has accurate ticket prices",
        parent=location_node,
        critical=False
    )
    
    child_price_str = location_data['child_price'] or "not provided"
    adult_price_str = location_data['adult_price'] or "not provided"
    
    await evaluator.verify(
        claim=f"The ticket prices for {location_data['name']} are: Child: {child_price_str}, Adult: {adult_price_str}",
        node=prices_node,
        sources=location_urls,
        additional_instruction="Check if the mentioned ticket prices for children and adults match the information on the official website or another reliable source. If prices are given as ranges or approximate values, they should be considered correct if they fall within the actual price range."
    )
    
    # Verify interactive activity
    activity_analysis = {}

    # Create parent node for activity verification
    activity_parent = evaluator.add_parallel(
        id=f"location_{idx}_interactive_activity",
        desc=f"Location {idx} offers the mentioned interactive STEM activity",
        parent=location_node,
        critical=True
    )
    
    # First analyze the activity
    analyzed_activity = await evaluator.extract(
        prompt=prompt_analyze_stem_activity(location_data['activity_description']),
        template_class=ActivityAnalysis,
        extraction_name=f"activity_analysis_{idx}"
    )
    
    activity_analysis = {
        "is_stem": analyzed_activity.is_stem,
        "is_interactive": analyzed_activity.is_interactive,
        "is_hands_on": analyzed_activity.is_hands_on
    }
    
    # Check if activity is interactive/hands-on STEM
    interactive_check = evaluator.add_custom_node(
        result=(analyzed_activity.is_interactive or analyzed_activity.is_hands_on) and analyzed_activity.is_stem,
        id=f"location_{idx}_activity_is_interactive_stem",
        desc=f"Activity is interactive/hands-on and STEM-related",
        parent=activity_parent,
        critical=True
    )
    
    # Verify activity exists
    activity_exists_node = evaluator.add_leaf(
        id=f"location_{idx}_activity_exists",
        desc=f"The activity '{location_data['activity_description']}' exists at '{location_data['name']}'",
        parent=activity_parent,
        critical=True
    )
    
    await evaluator.verify(
        claim=f"The following activity or exhibit exists at {location_data['name']}: {location_data['activity_description']}",
        node=activity_exists_node,
        sources=location_urls
    )
    
    return activity_analysis


async def verify_different_experiences(
    evaluator: Evaluator,
    parent_node,
    location_activities: List[Dict]
) -> None:
    """Verify that each of the three locations offers a different kind of hands-on experience."""
    
    # Filter valid activities
    valid_activities = [act for act in location_activities if act.get("activity_description")]

    if len(valid_activities) >= 2:
        exp_descriptions = []
        for i, act in enumerate(valid_activities):
            location_name = act["location_name"]
            description = act["activity_description"]
            exp_descriptions.append(f"Location {i+1} ({location_name}): {description}")
        
        experiences_text = "\n".join(exp_descriptions)
        
        different_node = evaluator.add_leaf(
            id="experiences_are_different",
            desc="The experiences are sufficiently different from each other",
            parent=parent_node,
            critical=True
        )
        
        await evaluator.verify(
            claim=f"The following STEM experiences are different from each other in terms of either STEM category (Science, Technology, Engineering, Math) or nature of activity (interactive vs hands-on):\n{experiences_text}",
            node=different_node,
            additional_instruction="Check if the locations offer DIFFERENT kinds of hands-on or interactive experiences. Consider both the STEM category (Science, Technology, Engineering, Math) and the nature of the activity (e.g., demonstration, building, coding, experimenting). They should be sufficiently different from each other to provide varied experiences."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id="different_experiences",
            desc="Only one hands-on STEM experience is provided.",
            parent=parent_node,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Helper extraction functions                                                 #
# --------------------------------------------------------------------------- #
async def extract_location_urls(
    evaluator: Evaluator,
    location_name: str, 
    main_url: Optional[str] = None
) -> List[str]:
    """Extract all URLs mentioned for a specific location."""
    urls = []
    
    if main_url:
        urls.append(main_url)
    
    extracted_urls = await evaluator.extract(
        prompt=prompt_extract_location_urls(location_name),
        template_class=LocationURLs
    )
    
    # Combine all unique URLs
    unique_urls = set(urls)
    if extracted_urls and extracted_urls.urls:
        unique_urls.update(extracted_urls.urls)
    
    return list(unique_urls)

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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
        strategy=AggregationStrategy.PARALLEL
    )

    # -------- 2. First extract location names ----------------------------- #
    location_names = await evaluator.extract(
        prompt=prompt_extract_location_names(),
        template_class=LocationNames
    )
    
    # Ensure we have at least 3 locations to evaluate, limit to first 3
    locations_to_evaluate = location_names.names[:3] if location_names.names else []
    
    # Create a list to store extracted data and analysis results
    locations_data = []
    
    # -------- 3. For each location name, extract details separately from activity info ---- #
    for name in locations_to_evaluate:
        # Extract location details (address, website, ticket prices)
        location_details = await evaluator.extract(
            prompt=prompt_extract_location_details(name),
            template_class=LocationDetails
        )
        
        # Extract activity description separately
        activity_description = await evaluator.extract(
            prompt=prompt_extract_location_activity(name),
            template_class=ActivityDescription
        )
        
        # Store extracted data
        location_data = {
            "name": name,
            "address": location_details.address,
            "website": location_details.website,
            "child_price": location_details.ticket_prices.child_price if location_details.ticket_prices else None,
            "adult_price": location_details.ticket_prices.adult_price if location_details.ticket_prices else None,
            "activity_description": activity_description.description if activity_description else None
        }
        
        locations_data.append(location_data)

    # Pad with empty locations if needed
    while len(locations_data) < 3:
        locations_data.append({
            "name": None,
            "address": None,
            "website": None,
            "child_price": None,
            "adult_price": None,
            "activity_description": None
        })
    
    # -------- 4. Build verification tree -------------------------------- #
    
    # Store activity analysis results for the different experiences check
    activity_analysis_results = []
    
    # Create individual verification nodes for each location
    for idx, location_data in enumerate(locations_data, 1):
        # Extract URLs for this location
        location_urls = []
        if location_data["name"]:
            location_urls = await extract_location_urls(
                evaluator,
                location_data["name"], 
                location_data["website"]
            )
        location_data["urls"] = location_urls
        
        # Verify location
        activity_analysis = await verify_location(
            evaluator,
            root,
            location_data,
            idx,
            location_urls
        )
        
        # Store activity analysis for different experiences check
        if location_data["name"] and location_data["activity_description"]:
            activity_analysis_results.append({
                "location_name": location_data["name"],
                "activity_description": location_data["activity_description"],
                "is_stem": activity_analysis.get("is_stem"),
                "is_interactive": activity_analysis.get("is_interactive"),
                "is_hands_on": activity_analysis.get("is_hands_on")
            })
    
    # Check for different kinds of hands-on experiences across locations
    await verify_different_experiences(evaluator, root, activity_analysis_results)

    # -------- 5. Get summary ---------------------------------------- #
    summary = evaluator.get_summary()
    
    # Add custom info about the locations
    evaluator.add_custom_info({
        "itinerary": [
            {
                "name": loc["name"],
                "address": loc["address"],
                "website": loc["website"],
                "ticket_prices": {
                    "child": loc["child_price"],
                    "adult": loc["adult_price"]
                },
                "activity": loc["activity_description"],
                "is_stem": next((a["is_stem"] for a in activity_analysis_results 
                                 if a["location_name"] == loc["name"]), None),
                "is_interactive": next((a["is_interactive"] for a in activity_analysis_results 
                                      if a["location_name"] == loc["name"]), None),
                "is_hands_on": next((a["is_hands_on"] for a in activity_analysis_results 
                                   if a["location_name"] == loc["name"]), None),
                "urls": loc["urls"]
            } for loc in locations_data
        ],
        "locations_count": len([loc for loc in locations_data if loc["name"]])
    }, info_type="evaluation_details")

    # -------- 6. Return the summary ------------------------------- #
    return summary