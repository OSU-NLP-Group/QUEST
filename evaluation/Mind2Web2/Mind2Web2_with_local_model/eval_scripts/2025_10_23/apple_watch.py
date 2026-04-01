import asyncio
import logging
from typing import Optional, List, Dict

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "apple_watch"
TASK_DESCRIPTION = """
I am compiling a list of 6 news reports about incidents where an Apple Watch helped save someone's life. Please help me identify 6 distinct incidents covered by reputable news sources. The incidents should meet the following criteria:

1. All incidents must have occurred in the United States.
2. Each incident must have occurred in a different U.S. state.
3. The news report should explicitly include:
- The time (at least the month) of the incident
- The specific location (city)
- The nature of the incident or medical emergency
- The Apple Watch features involved, clearly stating if multiple features contributed.
4. The news reports should be from independent, reputable news sources and not from Apple's official communications or PR releases.

For each incident, please provide:

- The required information as specified above.
- A direct link to the news article reporting the incident.

Ensure all links provided clearly mention and verify the Apple Watch's role in the described incidents.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                        #
# --------------------------------------------------------------------------- #
class IncidentInfo(BaseModel):
    time: Optional[str]
    location_city: Optional[str]
    location_state: Optional[str]
    nature_of_incident: Optional[str]
    apple_watch_features: Optional[str]
    news_source_urls: List[str] = Field(default_factory=list)


class ExtractedIncidents(BaseModel):
    incidents: List[IncidentInfo] = Field(default_factory=list)


class UrlExtraction(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_incidents() -> str:
    return """
    Extract information about Apple Watch life-saving incidents from the answer. For each incident mentioned, extract:

    1. time: The time when the incident occurred (at least month and year)
    2. location_city: The specific city where the incident occurred
    3. location_state: The U.S. state where the incident occurred (if the answer only mentions a country or region, set this to null; if the answer only mentions a city without a state, infer the state from the city name if possible)
    4. nature_of_incident: Description of the medical emergency or incident
    5. apple_watch_features: The specific Apple Watch features that helped (e.g., fall detection, heart rate monitoring, ECG, emergency SOS)
    6. news_source_urls: All URLs mentioned as sources for this specific incident

    Extract exactly as mentioned in the answer. If any information is missing or unclear, set the field to null.
    """


def prompt_extract_all_urls() -> str:
    return """
    Extract all URLs mentioned in the answer that appear to be news article links or source links. 
    Include any URL that might be used to verify the Apple Watch incidents described.
    """


# --------------------------------------------------------------------------- #
# Individual incident verification functions                                   #
# --------------------------------------------------------------------------- #
async def verify_incident_completeness(
        evaluator: Evaluator,
        parent_node,
        incident: IncidentInfo,
        incident_index: int,
) -> None:
    """Verify that an incident has all required information fields."""
    
    # Create a parent node for all completeness checks
    completeness_node = evaluator.add_parallel(
        id=f"incident_{incident_index}_completeness",
        desc=f"Incident {incident_index + 1} has all required information fields",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(incident.location_city and incident.location_state and incident.nature_of_incident and incident.apple_watch_features and incident.time),
        id=f"incident_{incident_index}_info",
        desc=f"Incident {incident_index + 1} includes required info",
        parent=completeness_node,
        critical=True
    )

    # Time verification
    time_node = evaluator.add_leaf(
        id=f"incident_{incident_index}_time",
        desc=f"Incident {incident_index + 1} includes time information (at least month)",
        parent=completeness_node,
        critical=True,
    )

    has_time = await evaluator.verify(
        claim=f"The time information '{incident.time}' includes at least the month when the incident occurred",
        node=time_node,
        additional_instruction="Check if the time information includes at least a month (e.g., 'January 2023', 'March 2022', etc.)"
    )



async def verify_incident_provenance(
        evaluator: Evaluator,
        parent_node,
        incident: IncidentInfo,
        incident_index: int,
        all_urls: List[str],
) -> None:
    """Verify that incident information is supported by provided URLs and sources are reputable."""

    provenance_parent = evaluator.add_parallel(
        id=f"incident_{incident_index}_provenance_wrapper",
        desc=f"Incident {incident_index + 1} provenance verification",
        parent=parent_node,
        critical=True
    )

    # Combine incident-specific URLs with all extracted URLs for verification
    verification_urls = list(set(incident.news_source_urls + all_urls))

    # Add existence check for URLs
    urls_exist = evaluator.add_custom_node(
        result=bool(verification_urls),
        id=f"incident_{incident_index}_urls_exist",
        desc=f"Incident {incident_index + 1} has URLs for verification",
        parent=provenance_parent,
        critical=True
    )

    # Create comprehensive claim combining all incident details and source verification
    incident_details = []

    incident_details.append(f"occurred in {incident.time}")

    incident_details.append(f"took place in {incident.location_city}, {incident.location_state}")

    incident_details.append(f"involved {incident.nature_of_incident}")

    incident_details.append(f"Apple Watch helped through {incident.apple_watch_features}")

    provenance_node = evaluator.add_leaf(
        id=f"incident_{incident_index}_provenance",
        desc=f"Incident {incident_index + 1} information is fully supported by reputable news sources",
        parent=provenance_parent,
        critical=True
    )

    comprehensive_claim = (
        f"This news article reports an Apple Watch life-saving incident that "
        f"{', and '.join(incident_details) if incident_details else 'has no details'}. "
        f"The news source must NOT be from Apple's official communications, PR releases, or apple.com."
    )

    # Always call verify - let the existence checks handle missing data
    await evaluator.verify(
        claim=comprehensive_claim,
        node=provenance_node,
        sources=verification_urls,  # Pass the list directly
        additional_instruction=(
            "Verify that the news article contains ALL the specified details about the Apple Watch incident "
            "AND confirm the source is from an independent news organization (NOT Apple's official channels). "
            "Requirements:\n"
            "1. INCIDENT TIME (This is tricky!!! Pay attention to this): Must include at least month. You should carefully distinguish between the actual incident date "
            "and the press release or publication date, where the latter should not be treated as a correct incident date. If the exact month of the incident is unclear from the text, only having the date of the press release should NOT be treated as a success.\n"
            "2. LOCATION: City must exactly match the article. State may be inferred from context, news source, "
            "or city name if not explicitly stated (i.e., the state name is okay to be missing in the answer, or the webpage).\n"
            "3. APPLE WATCH FEATURES: Must accurately match those described in the article. Minor variations "
            "in wording are acceptable, but really make sure all critical features for the life saving is mentioned, and there are no wrong features that are not mentioned at all in the article (i.e., no fabricated features should be included.).\n"
            "4. SOURCE INDEPENDENCE: Must be from news outlets, not Apple PR or marketing materials.\n"
        )
    )


async def verify_single_incident(
        evaluator: Evaluator,
        parent_node,
        incident: IncidentInfo,
        incident_index: int,
        all_urls: List[str],
) -> None:
    """Verify a single incident meets all requirements."""

    incident_node = evaluator.add_parallel(
        id=f"incident_{incident_index}",
        desc=f"Incident {incident_index + 1} meets all requirements and is properly substantiated",
        parent=parent_node
    )

    # Verify completeness (all critical)
    await verify_incident_completeness(evaluator, incident_node, incident, incident_index)

    # Verify provenance (critical)
    await verify_incident_provenance(evaluator, incident_node, incident, incident_index, all_urls)


# --------------------------------------------------------------------------- #
# Global verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_state_uniqueness(
        evaluator: Evaluator,
        parent_node,
        incidents: List[IncidentInfo],
) -> None:
    """Verify that incidents occur in different U.S. states."""

    states = [incident.location_state for incident in incidents if incident.location_state]
    unique_states = set(states)

    if len(states) <= 1:
        is_unique = True
    else:
        is_unique = len(states) == len(unique_states)

    uniqueness_wrapper = evaluator.add_parallel(
            id="state_uniqueness_wrapper",
            desc="State uniqueness verification",
            parent=parent_node,
            critical=True
        )
    
    # Add uniqueness check
    evaluator.add_custom_node(
        result=is_unique,  # We already know they're unique
        id="states_are_unique",
        desc="States are unique (no duplicates)",
        parent=uniqueness_wrapper,
        critical=True
    )

    # Verify states are actually U.S. states
    valid_states_node = evaluator.add_leaf(
        id="valid_us_states",
        desc="All states are valid U.S. states",
        parent=uniqueness_wrapper,
        critical=True
    )

    states_claim = f"The following are all valid U.S. states: {', '.join(unique_states)}"
    await evaluator.verify(
        claim=states_claim,
        node=valid_states_node,
        additional_instruction="Verify that all listed locations are valid U.S. states (not countries, territories, or invalid locations)"
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
    Evaluate a single answer about Apple Watch life-saving incidents.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        agent_name=agent_name,
        answer_name=answer_name,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # -------- 2. Extract structured info from the answer ----------------- #
    extracted_incidents = await evaluator.extract(
        prompt=prompt_extract_incidents(),
        template_class=ExtractedIncidents,
        extraction_name="incidents"
    )

    extracted_urls = await evaluator.extract(
        prompt=prompt_extract_all_urls(),
        template_class=UrlExtraction,
        extraction_name="all_urls"
    )

    # -------- 3. Build verification tree -------------------------------- #
    # Individual incident verification
    incidents_node = evaluator.add_parallel(
        id="individual_incidents",
        desc="Individual incidents meet requirements and are properly substantiated",
    )

    # Take first 6 incidents for evaluation, pad if needed
    actual_incidents = extracted_incidents.incidents[:6]
    
    # Pad missing incidents with empty objects
    while len(actual_incidents) < 6:
        actual_incidents.append(IncidentInfo())

    # Verify all 6 incidents (real or empty)
    for i, incident in enumerate(actual_incidents):
        await verify_single_incident(
            evaluator,
            incidents_node,
            incident,
            i,
            extracted_urls.urls
        )

    # State uniqueness verification
    await verify_state_uniqueness(evaluator, evaluator.root, extracted_incidents.incidents[:6])

    # -------- 4. Get evaluation summary --------------------------------- #
    # Add custom information
    valid_incidents = [inc for inc in extracted_incidents.incidents[:6] 
                      if inc.location_state]
    evaluator.add_custom_info({
        "incident_count": len([inc for inc in extracted_incidents.incidents[:6] 
                              if inc.nature_of_incident]),  # Count real incidents
        "unique_states": list(set([inc.location_state for inc in valid_incidents])),
        "url_count": len(extracted_urls.urls)
    }, "evaluation_stats")

    return evaluator.get_summary()