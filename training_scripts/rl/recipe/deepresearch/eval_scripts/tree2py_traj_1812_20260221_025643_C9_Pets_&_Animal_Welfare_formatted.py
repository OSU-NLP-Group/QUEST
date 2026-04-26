import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pets_animal_welfare_2025"
TASK_DESCRIPTION = """I need to gather comprehensive information about four major topics in pets and animal welfare from 2025. Please provide detailed, well-sourced information for each of the following:

1. National Dog Show 2025 Best in Show Winner
2. AKC National Championship 2025 Sporting Group Winner
3. Whiting Ranch Wilderness Park Mountain Lion Incident (November 2025)
4. Patagonia Pumas Research Study (Published December 2025)
"""

# Ground truth / expected facts from the rubric for reference in verification prompts
NDS_EXPECTED = {
    "breed": "Belgian Sheepdog",
    "call_name": "Soleil",
    "registered_name": "GCHS Prairiewind's Songs of Summer at La Neige",
    "handler_name": "Daniel Martin",
    "handler_location": "Princeton, North Carolina",
    "broadcast_network": "NBC",
    "broadcast_date": "November 27, 2025",
    "broadcast_time": "12:00 p.m. to 2:00 p.m. local time",
    "group_advanced_from": "Herding Group",
}

AKC_EXPECTED = {
    "breed": "Gordon Setter",
    "breed_type": "setter",
    "call_name": "River",
    "registered_name": "GCHB CH Tamarack Valley View River Of Dreams",
    "owner_options": ["Dr. Ellen Shanahan", "Stacy Threlfall"],
    "owner_location": "Great Barrington, Massachusetts",
    "event_dates": "December 13–14, 2025",
    "event_venue": "Orange County Convention Center (Orlando Convention Center), Orlando, Florida",
    "total_dogs_competing": "5,500",
    "results_broadcast_network": "ABC",
    "winners_announced_date": "December 28, 2025",
    "also_won_2024": True,
}

WHITING_EXPECTED = {
    "park_closure_date": "November 4, 2025",
    "sighting_date": "November 3, 2025",
    "first_sighting_time": "approximately 4:00 p.m.",
    "second_sighting_time": "approximately 5:30 p.m.",
    "behavior_description": "following people but eventually running off",
    "evidence_type": "video",
    "research_partner": "UC Davis Wildlife Health Center (UC Davis)",
    # Location is not explicitly provided in rubric; common usage: Davis, California
    "research_partner_location_hint": "Davis, California",
    "park_reopening_date": "November 26, 2025",
}

PATAGONIA_EXPECTED = {
    "journal_name": "Proceedings of the Royal Society B",
    "publication_date": "December 17, 2025",
    "lead_author_name": "Mitchell Serota",
    "lead_author_affiliation": "University of California Berkeley",
    "coauthor_name": "Emiliano Donadio",
    "coauthor_organization": "Fundación Rewilding Argentina",
    "study_park_name": "Monte León National Park",
    "study_country": "Argentina",
    "penguin_species": "Magellanic penguin (Spheniscus magellanicus)",
    "breeding_pairs_count": "more than 40,000",
    "coastline_length": "approximately 2 kilometers",
    "breeding_season_start": "approximately September",
    "breeding_season_end": "approximately April",
    "camera_traps_count": "32",
    "gps_collared_pumas_count": "14",
    "tracking_period_start": "September 2019",
    "tracking_period_end": "January 2023",
    "penguin_hunting_collared_pumas": "9 out of 14",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NationalDogShowInfo(BaseModel):
    breed: Optional[str] = None
    call_name: Optional[str] = None
    registered_name: Optional[str] = None
    handler_name: Optional[str] = None
    handler_location: Optional[str] = None
    broadcast_network: Optional[str] = None
    broadcast_date: Optional[str] = None
    broadcast_time: Optional[str] = None
    group_advanced_from: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AKCSportingInfo(BaseModel):
    breed: Optional[str] = None
    breed_type: Optional[str] = None  # e.g., setter, retriever, spaniel, pointer
    call_name: Optional[str] = None
    registered_name: Optional[str] = None
    owner_names: List[str] = Field(default_factory=list)
    owner_location: Optional[str] = None
    event_dates: Optional[str] = None
    event_venue: Optional[str] = None
    total_dogs_competing: Optional[str] = None  # Keep as string to allow variations
    results_broadcast_network: Optional[str] = None
    winners_announced_date: Optional[str] = None
    also_won_2024: Optional[str] = None  # "yes"/"no"/None as string
    urls: List[str] = Field(default_factory=list)


class WhitingRanchIncidentInfo(BaseModel):
    park_closure_date: Optional[str] = None
    sighting_date: Optional[str] = None
    first_sighting_time: Optional[str] = None
    second_sighting_time: Optional[str] = None
    behavior_description: Optional[str] = None
    evidence_type: Optional[str] = None
    research_partner_name_and_location: Optional[str] = None
    park_reopening_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PatagoniaPumasStudyInfo(BaseModel):
    journal_name: Optional[str] = None
    publication_date: Optional[str] = None
    lead_author_name: Optional[str] = None
    lead_author_affiliation: Optional[str] = None
    coauthor_name: Optional[str] = None
    coauthor_organization: Optional[str] = None
    study_park_name: Optional[str] = None
    study_country: Optional[str] = None
    penguin_species: Optional[str] = None
    breeding_pairs_count: Optional[str] = None
    coastline_length: Optional[str] = None
    breeding_season_start: Optional[str] = None
    breeding_season_end: Optional[str] = None
    camera_traps_count: Optional[str] = None
    gps_collared_pumas_count: Optional[str] = None
    tracking_period_start: Optional[str] = None
    tracking_period_end: Optional[str] = None
    penguin_hunting_collared_pumas: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PetsAnimalWelfare2025Extraction(BaseModel):
    national_dog_show: Optional[NationalDogShowInfo] = None
    akc_sporting: Optional[AKCSportingInfo] = None
    whiting_ranch_incident: Optional[WhitingRanchIncidentInfo] = None
    patagonia_study: Optional[PatagoniaPumasStudyInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract the requested information for four topics from the provided answer text. For each topic, include exactly the fields listed. If any field is missing from the answer, return null for that field. For URLs, extract all valid URLs mentioned for that topic. Do not invent or infer information.

    1) National Dog Show 2025 Best in Show Winner (televised Thanksgiving Day):
       - breed
       - call_name
       - registered_name
       - handler_name
       - handler_location (city and state)
       - broadcast_network
       - broadcast_date
       - broadcast_time
       - group_advanced_from
       - urls (array of URLs provided for this topic)

    2) AKC National Championship 2025 Sporting Group Winner (Orlando, December):
       - breed
       - breed_type (setter / retriever / spaniel / pointer)
       - call_name
       - registered_name
       - owner_names (array; list all owners mentioned; at least one)
       - owner_location (city and state)
       - event_dates
       - event_venue (venue name and city/state)
       - total_dogs_competing
       - results_broadcast_network
       - winners_announced_date
       - also_won_2024 (string "yes" or "no" if mentioned; otherwise null)
       - urls (array of URLs provided for this topic)

    3) Whiting Ranch Wilderness Park Mountain Lion Incident (November 2025, Orange County CA):
       - park_closure_date
       - sighting_date
       - first_sighting_time
       - second_sighting_time
       - behavior_description
       - evidence_type (e.g., video)
       - research_partner_name_and_location (name and location string)
       - park_reopening_date
       - urls (array of URLs provided for this topic)

    4) Patagonia pumas hunting penguins research study (published December 2025):
       - journal_name
       - publication_date
       - lead_author_name
       - lead_author_affiliation
       - coauthor_name
       - coauthor_organization
       - study_park_name
       - study_country
       - penguin_species
       - breeding_pairs_count
       - coastline_length
       - breeding_season_start
       - breeding_season_end
       - camera_traps_count
       - gps_collared_pumas_count
       - tracking_period_start
       - tracking_period_end
       - penguin_hunting_collared_pumas
       - urls (array of URLs provided for this topic)

    Return a JSON object with keys:
    - national_dog_show
    - akc_sporting
    - whiting_ranch_incident
    - patagonia_study
    Each key maps to its corresponding object with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_national_dog_show(evaluator: Evaluator, parent_node, nds: Optional[NationalDogShowInfo]) -> None:
    topic_node = evaluator.add_parallel(
        id="national_dog_show_2025",
        desc="National Dog Show 2025 Best in Show winner details (all required fields + supporting URLs)",
        parent=parent_node,
        critical=True
    )

    # Leaf: Winner_Breed
    breed_node = evaluator.add_leaf(
        id="nds_winner_breed",
        desc="Best in Show winner breed is Belgian Sheepdog",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner's breed is {NDS_EXPECTED['breed']}.",
        node=breed_node,
        additional_instruction="Judge based on the answer text; do not rely on outside knowledge. Allow minor casing differences."
    )

    # Leaf: Winner_Call_Name
    call_node = evaluator.add_leaf(
        id="nds_winner_call_name",
        desc="Best in Show winner call name is Soleil",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner's call name (nickname) is {NDS_EXPECTED['call_name']}.",
        node=call_node,
        additional_instruction="Judge based on the answer text; allow minor spelling variants and casing."
    )

    # Leaf: Winner_Registered_Name
    reg_node = evaluator.add_leaf(
        id="nds_winner_registered_name",
        desc="Best in Show winner full registered name is provided as GCHS Prairiewind's Songs of Summer at La Neige (allow minor spelling variants per constraint)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner's full registered name matches '{NDS_EXPECTED['registered_name']}' (allow minor formatting variants, punctuation, or abbreviations).",
        node=reg_node,
        additional_instruction="Consider reasonable punctuation, spacing, apostrophes, and AKC title abbreviations as equivalent."
    )

    # Leaf: Handler_Name
    handler_node = evaluator.add_leaf(
        id="nds_handler_name",
        desc="Handler name is Daniel Martin",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The handler's name mentioned is {NDS_EXPECTED['handler_name']}.",
        node=handler_node,
        additional_instruction="Judge from answer; allow inclusion of middle initials or suffixes."
    )

    # Leaf: Handler_Location
    handler_loc_node = evaluator.add_leaf(
        id="nds_handler_location",
        desc="Handler location (city and state) is Princeton, North Carolina",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The handler's location (city and state) is {NDS_EXPECTED['handler_location']}.",
        node=handler_loc_node,
        additional_instruction="Allow minor formatting variants (e.g., 'Princeton, NC')."
    )

    # Leaf: Broadcast_Network
    bnet_node = evaluator.add_leaf(
        id="nds_broadcast_network",
        desc="Broadcast network is NBC",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The National Dog Show was broadcast on {NDS_EXPECTED['broadcast_network']}.",
        node=bnet_node,
        additional_instruction="Judge from answer; network abbreviations acceptable."
    )

    # Leaf: Broadcast_Date
    bdate_node = evaluator.add_leaf(
        id="nds_broadcast_date",
        desc="Broadcast date is November 27, 2025 (Thanksgiving Day)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The broadcast date of the National Dog Show was {NDS_EXPECTED['broadcast_date']}.",
        node=bdate_node,
        additional_instruction="Judge from answer; allow 'Thanksgiving Day 2025' to be equivalent."
    )

    # Leaf: Broadcast_Time
    btime_node = evaluator.add_leaf(
        id="nds_broadcast_time",
        desc="Broadcast time is 12:00 p.m. to 2:00 p.m. local time",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The broadcast time was {NDS_EXPECTED['broadcast_time']}.",
        node=btime_node,
        additional_instruction="Treat 'noon to 2 PM' and similar variants as equivalent; local time phrasing acceptable."
    )

    # Leaf: Group_Advanced_From
    group_node = evaluator.add_leaf(
        id="nds_group_advanced_from",
        desc="Winner advanced from the Herding Group before winning Best in Show",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Best in Show winner advanced from the {NDS_EXPECTED['group_advanced_from']}.",
        node=group_node,
        additional_instruction="Judge from answer; allow variants like 'Herding'."
    )

    # Leaf: Supporting_URLs (verify via URLs if present; fallback to simple check)
    urls = _safe_urls(nds.urls if nds else [])
    sup_node = evaluator.add_leaf(
        id="nds_supporting_urls",
        desc="Provides valid URL reference(s) from acceptable sources (official source, news outlet) that collectively support the provided National Dog Show required details",
        parent=topic_node,
        critical=True
    )
    if urls:
        claim = ("These provided sources collectively support the National Dog Show 2025 Best in Show details, "
                 "including breed (Belgian Sheepdog), call name (Soleil), registered name, handler and location, "
                 "broadcast network/date/time, and advancement from the Herding Group.")
        await evaluator.verify(
            claim=claim,
            node=sup_node,
            sources=urls,
            additional_instruction="Check each URL to confirm the details. If at least one official or reputable source clearly supports the majority of these details, judge accordingly."
        )
    else:
        await evaluator.verify(
            claim="The answer includes valid URL references that support the National Dog Show 2025 Best in Show details.",
            node=sup_node,
            additional_instruction="Judge from the answer text. If no URLs are provided for this topic, mark this as incorrect."
        )


async def verify_akc_sporting(evaluator: Evaluator, parent_node, akc: Optional[AKCSportingInfo]) -> None:
    topic_node = evaluator.add_parallel(
        id="akc_national_championship_2025_sporting",
        desc="AKC National Championship 2025 Sporting Group winner details (all required fields + supporting URLs)",
        parent=parent_node,
        critical=True
    )

    # Winner_Breed
    breed_node = evaluator.add_leaf(
        id="akc_winner_breed",
        desc="Sporting Group winner breed is Gordon Setter",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Sporting Group winner's breed is {AKC_EXPECTED['breed']}.",
        node=breed_node,
        additional_instruction="Judge based on answer; allow minor casing."
    )

    # Winner_Breed_Type
    btype_node = evaluator.add_leaf(
        id="akc_winner_breed_type",
        desc="Breed type within Sporting Group is correctly identified as setter",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The breed type within the Sporting Group is correctly identified as {AKC_EXPECTED['breed_type']}.",
        node=btype_node,
        additional_instruction="Judge from answer; allow variants ('Setter')."
    )

    # Winner_Call_Name
    call_node = evaluator.add_leaf(
        id="akc_winner_call_name",
        desc="Sporting Group winner call name is River",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The Sporting Group winner's call name is {AKC_EXPECTED['call_name']}.",
        node=call_node,
        additional_instruction="Judge from answer; allow minor spelling."
    )

    # Winner_Registered_Name
    reg_node = evaluator.add_leaf(
        id="akc_winner_registered_name",
        desc="Sporting Group winner full registered name is GCHB CH Tamarack Valley View River Of Dreams",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winner's full registered name matches '{AKC_EXPECTED['registered_name']}' (allow title/formatting variants).",
        node=reg_node,
        additional_instruction="Consider reasonable formatting, punctuation, title abbreviations as equivalent."
    )

    # Owner_Name (at least one of the specified)
    owner_node = evaluator.add_leaf(
        id="akc_owner_name",
        desc="At least one owner is named (Dr. Ellen Shanahan and/or Stacy Threlfall)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one owner listed is either Dr. Ellen Shanahan or Stacy Threlfall.",
        node=owner_node,
        additional_instruction="Judge from answer; allow variants like 'Ellen Shanahan' (without title) or middle initials."
    )

    # Owner_Location
    owner_loc_node = evaluator.add_leaf(
        id="akc_owner_location",
        desc="Owner location (city and state) is Great Barrington, Massachusetts",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The owner location (city/state) is {AKC_EXPECTED['owner_location']}.",
        node=owner_loc_node,
        additional_instruction="Allow variants like 'Great Barrington, MA'."
    )

    # Event_Dates
    edates_node = evaluator.add_leaf(
        id="akc_event_dates",
        desc="Event dates are December 13–14, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The event dates are {AKC_EXPECTED['event_dates']}.",
        node=edates_node,
        additional_instruction="Judge from answer; allow en dash vs hyphen variations."
    )

    # Event_Venue
    venue_node = evaluator.add_leaf(
        id="akc_event_venue",
        desc="Venue is Orlando County Convention Center / Orlando Convention Center in Orlando, Florida",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The venue is the Orange County Convention Center (also referred to as Orlando Convention Center) in Orlando, Florida.",
        node=venue_node,
        additional_instruction="Allow naming variants: 'Orange County Convention Center', 'Orlando Convention Center'."
    )

    # Total_Dogs_Competing
    total_node = evaluator.add_leaf(
        id="akc_total_dogs_competing",
        desc="Total number of dogs competing is 5,500",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total number of dogs competing is {AKC_EXPECTED['total_dogs_competing']}.",
        node=total_node,
        additional_instruction="Allow minor formatting (commas, rounding); 5500 ≈ 5,500."
    )

    # Results_Broadcast_Network
    rnet_node = evaluator.add_leaf(
        id="akc_results_broadcast_network",
        desc="Results broadcast network is ABC",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The results broadcast network was {AKC_EXPECTED['results_broadcast_network']}.",
        node=rnet_node,
        additional_instruction="Judge from answer; abbreviations acceptable."
    )

    # Winners_Announced_Date
    wdate_node = evaluator.add_leaf(
        id="akc_winners_announced_date",
        desc="Date winners were announced/broadcast is December 28, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The date the winners were announced/broadcast was {AKC_EXPECTED['winners_announced_date']}.",
        node=wdate_node,
        additional_instruction="Judge from answer; allow minor date formatting variants."
    )

    # Also_Won_2024_Sporting_Group
    also_node = evaluator.add_leaf(
        id="akc_also_won_2024_sporting_group",
        desc="Notes that River also won the Sporting Group at the 2024 AKC National Championship",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The dog 'River' also won the Sporting Group at the 2024 AKC National Championship.",
        node=also_node,
        additional_instruction="Judge from answer; allow phrasing variants indicating the same accomplishment."
    )

    # Supporting_URLs
    urls = _safe_urls(akc.urls if akc else [])
    sup_node = evaluator.add_leaf(
        id="akc_supporting_urls",
        desc="Provides valid URL reference(s) from acceptable sources (e.g., AKC, news outlet) that collectively support the provided AKC required details",
        parent=topic_node,
        critical=True
    )
    if urls:
        claim = ("These provided sources collectively support the AKC National Championship 2025 Sporting Group winner details, "
                 "including breed/type (Gordon Setter, setter), call/registered names (River; GCHB CH Tamarack Valley View River Of Dreams), "
                 "owner(s) and location, event dates/venue, total dogs competing, broadcast network/date, and note about 2024 win.")
        await evaluator.verify(
            claim=claim,
            node=sup_node,
            sources=urls,
            additional_instruction="Confirm as many of the listed details as possible via AKC or reputable sources."
        )
    else:
        await evaluator.verify(
            claim="The answer includes valid URL references that support the AKC Sporting Group winner details.",
            node=sup_node,
            additional_instruction="Judge from the answer text. If no URLs are provided for this topic, mark this as incorrect."
        )


async def verify_whiting_ranch(evaluator: Evaluator, parent_node, wr: Optional[WhitingRanchIncidentInfo]) -> None:
    topic_node = evaluator.add_parallel(
        id="whiting_ranch_incident_nov_2025",
        desc="Whiting Ranch Wilderness Park mountain lion incident and closure (Nov 2025) details (all required fields + supporting URLs)",
        parent=parent_node,
        critical=True
    )

    # Park_Closure_Date
    close_node = evaluator.add_leaf(
        id="whiting_park_closure_date",
        desc="Park closure date is November 4, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park closure date is {WHITING_EXPECTED['park_closure_date']}.",
        node=close_node,
        additional_instruction="Judge from answer; allow minor date formatting variants."
    )

    # Sighting_Date
    sight_node = evaluator.add_leaf(
        id="whiting_sighting_date",
        desc="Mountain lion sightings date is November 3, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The mountain lion sightings occurred on {WHITING_EXPECTED['sighting_date']}.",
        node=sight_node,
        additional_instruction="Judge from answer; allow minor formatting variants."
    )

    # First_Sighting_Time
    first_node = evaluator.add_leaf(
        id="whiting_first_sighting_time",
        desc="First sighting time is approximately 4:00 p.m.",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The first sighting time is {WHITING_EXPECTED['first_sighting_time']}.",
        node=first_node,
        additional_instruction="Times are approximate; allow variants like 'around 4 pm'."
    )

    # Second_Sighting_Time
    second_node = evaluator.add_leaf(
        id="whiting_second_sighting_time",
        desc="Second sighting time is approximately 5:30 p.m.",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The second sighting time is {WHITING_EXPECTED['second_sighting_time']}.",
        node=second_node,
        additional_instruction="Times are approximate; allow variants like 'about 5:30 pm'."
    )

    # Behavior_Description
    behavior_node = evaluator.add_leaf(
        id="whiting_behavior_description",
        desc="Reported behavior described as following people but eventually running off",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The reported mountain lion behavior is described as {WHITING_EXPECTED['behavior_description']}.",
        node=behavior_node,
        additional_instruction="Judge from answer; allow paraphrases indicating following people and then running off."
    )

    # Evidence_Type
    evidence_node = evaluator.add_leaf(
        id="whiting_evidence_type",
        desc="Evidence shared with OC Parks is video",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The evidence shared with OC Parks was {WHITING_EXPECTED['evidence_type']}.",
        node=evidence_node,
        additional_instruction="Judge from answer; allow variants indicating 'video evidence'."
    )

    # Research_Partner_Name_And_Location
    partner_node = evaluator.add_leaf(
        id="whiting_research_partner",
        desc="Research/academic partner is UC Davis Wildlife Health Center (UC Davis), and includes the partner’s location information as requested",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim="The research/academic partner worked with OC Parks was UC Davis Wildlife Health Center (UC Davis), and the answer includes the partner’s location (city/state).",
        node=partner_node,
        additional_instruction="Judge from answer; location can be stated as 'Davis, California' or equivalent."
    )

    # Park_Reopening_Date
    reopen_node = evaluator.add_leaf(
        id="whiting_park_reopening_date",
        desc="Park reopening date is November 26, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The park reopened on {WHITING_EXPECTED['park_reopening_date']}.",
        node=reopen_node,
        additional_instruction="Judge from answer; allow minor date formatting."
    )

    # Supporting_URLs
    urls = _safe_urls(wr.urls if wr else [])
    sup_node = evaluator.add_leaf(
        id="whiting_supporting_urls",
        desc="Provides valid URL reference(s) from acceptable sources (official OC Parks and/or news outlet) that collectively support the provided incident required details",
        parent=topic_node,
        critical=True
    )
    if urls:
        claim = ("These provided sources collectively support Whiting Ranch Wilderness Park November 2025 incident details, "
                 "including closure (Nov 4), sightings (Nov 3) with approximate times, behavior description, evidence (video), "
                 "research partner (UC Davis Wildlife Health Center), and reopening (Nov 26).")
        await evaluator.verify(
            claim=claim,
            node=sup_node,
            sources=urls,
            additional_instruction="Confirm details via OC Parks official communications or reputable news outlets."
        )
    else:
        await evaluator.verify(
            claim="The answer includes valid URL references that support the Whiting Ranch incident details.",
            node=sup_node,
            additional_instruction="Judge from the answer text. If no URLs are provided for this topic, mark this as incorrect."
        )


async def verify_patagonia_study(evaluator: Evaluator, parent_node, ps: Optional[PatagoniaPumasStudyInfo]) -> None:
    topic_node = evaluator.add_parallel(
        id="patagonia_pumas_penguins_study_dec_2025",
        desc="Patagonia pumas hunting penguins research study (published Dec 2025) details (all required fields + supporting URLs)",
        parent=parent_node,
        critical=True
    )

    # Journal_Name
    journal_node = evaluator.add_leaf(
        id="patagonia_journal_name",
        desc="Journal is Proceedings of the Royal Society B",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The study was published in {PATAGONIA_EXPECTED['journal_name']}.",
        node=journal_node,
        additional_instruction="Judge from answer; allow abbreviation 'Proc. R. Soc. B'."
    )

    # Publication_Date
    pdate_node = evaluator.add_leaf(
        id="patagonia_publication_date",
        desc="Publication date is December 17, 2025",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The publication date is {PATAGONIA_EXPECTED['publication_date']}.",
        node=pdate_node,
        additional_instruction="Judge from answer; allow minor formatting variants."
    )

    # Lead_Author_Name
    lead_node = evaluator.add_leaf(
        id="patagonia_lead_author_name",
        desc="Lead author is Mitchell Serota",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The lead author is {PATAGONIA_EXPECTED['lead_author_name']}.",
        node=lead_node,
        additional_instruction="Judge from answer; allow middle initials."
    )

    # Lead_Author_Affiliation
    aff_node = evaluator.add_leaf(
        id="patagonia_lead_author_affiliation",
        desc="Lead author affiliation is University of California Berkeley (UC Berkeley)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The lead author affiliation is {PATAGONIA_EXPECTED['lead_author_affiliation']}.",
        node=aff_node,
        additional_instruction="Allow variants like 'UC Berkeley' or department names attached."
    )

    # Coauthor_Name
    co_name_node = evaluator.add_leaf(
        id="patagonia_coauthor_name",
        desc="At least one co-author is Emiliano Donadio",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least one co-author listed is {PATAGONIA_EXPECTED['coauthor_name']}.",
        node=co_name_node,
        additional_instruction="Judge from answer; allow accent marks."
    )

    # Coauthor_Organization
    co_org_node = evaluator.add_leaf(
        id="patagonia_coauthor_organization",
        desc="Co-author organization is Fundación Rewilding Argentina",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The co-author's organization is {PATAGONIA_EXPECTED['coauthor_organization']}.",
        node=co_org_node,
        additional_instruction="Judge from answer; allow minor spelling variations and accents."
    )

    # Study_Park_Name
    park_node = evaluator.add_leaf(
        id="patagonia_study_park_name",
        desc="Study conducted in Monte León National Park (Monte Leon National Park)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The study was conducted in {PATAGONIA_EXPECTED['study_park_name']}.",
        node=park_node,
        additional_instruction="Allow minor accent differences (León vs Leon)."
    )

    # Study_Country
    country_node = evaluator.add_leaf(
        id="patagonia_study_country",
        desc="Country is Argentina",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The study country is {PATAGONIA_EXPECTED['study_country']}.",
        node=country_node,
        additional_instruction="Judge from answer."
    )

    # Penguin_Species
    species_node = evaluator.add_leaf(
        id="patagonia_penguin_species",
        desc="Penguin species is Magellanic penguin (Spheniscus magellanicus)",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The penguin species studied is {PATAGONIA_EXPECTED['penguin_species']}.",
        node=species_node,
        additional_instruction="Judge from answer; allow scientific name formatting variants."
    )

    # Breeding_Pairs_Count
    pairs_node = evaluator.add_leaf(
        id="patagonia_breeding_pairs_count",
        desc="Penguin colony has more than 40,000 breeding pairs",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The penguin colony has {PATAGONIA_EXPECTED['breeding_pairs_count']}.",
        node=pairs_node,
        additional_instruction="Judge from answer; 'over 40,000' equivalent."
    )

    # Coastline_Length
    coast_node = evaluator.add_leaf(
        id="patagonia_coastline_length",
        desc="Colony occupies approximately 2 kilometers of coastline",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The colony occupies {PATAGONIA_EXPECTED['coastline_length']}.",
        node=coast_node,
        additional_instruction="Judge from answer; allow 'about 2 km'."
    )

    # Breeding_Season_Start
    bstart_node = evaluator.add_leaf(
        id="patagonia_breeding_season_start",
        desc="Breeding season start is approximately September",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The breeding season starts in {PATAGONIA_EXPECTED['breeding_season_start']}.",
        node=bstart_node,
        additional_instruction="Judge from answer; months phrasing variants acceptable."
    )

    # Breeding_Season_End
    bend_node = evaluator.add_leaf(
        id="patagonia_breeding_season_end",
        desc="Breeding season end is approximately April",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The breeding season ends in {PATAGONIA_EXPECTED['breeding_season_end']}.",
        node=bend_node,
        additional_instruction="Judge from answer; months phrasing variants acceptable."
    )

    # Camera_Traps_Count
    cams_node = evaluator.add_leaf(
        id="patagonia_camera_traps_count",
        desc="Number of camera traps used is 32",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of camera traps used was {PATAGONIA_EXPECTED['camera_traps_count']}.",
        node=cams_node,
        additional_instruction="Judge from answer; allow numeric formatting variants."
    )

    # GPS_Collared_Pumas_Count
    gps_node = evaluator.add_leaf(
        id="patagonia_gps_collared_pumas_count",
        desc="Number of GPS-collared pumas is 14",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of GPS-collared pumas was {PATAGONIA_EXPECTED['gps_collared_pumas_count']}.",
        node=gps_node,
        additional_instruction="Judge from answer; allow numeric formatting variants."
    )

    # Tracking_Period_Start
    tstart_node = evaluator.add_leaf(
        id="patagonia_tracking_period_start",
        desc="Puma tracking period start is September 2019",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The puma tracking period started in {PATAGONIA_EXPECTED['tracking_period_start']}.",
        node=tstart_node,
        additional_instruction="Judge from answer; allow variants like 'Sep 2019'."
    )

    # Tracking_Period_End
    tend_node = evaluator.add_leaf(
        id="patagonia_tracking_period_end",
        desc="Puma tracking period end is January 2023",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The puma tracking period ended in {PATAGONIA_EXPECTED['tracking_period_end']}.",
        node=tend_node,
        additional_instruction="Judge from answer; allow variants like 'Jan 2023'."
    )

    # Penguin_Hunting_Collared_Pumas
    hunt_node = evaluator.add_leaf(
        id="patagonia_penguin_hunting_collared_pumas",
        desc="Number of collared pumas that hunted penguins is 9 out of 14",
        parent=topic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The number of collared pumas that hunted penguins is {PATAGONIA_EXPECTED['penguin_hunting_collared_pumas']}.",
        node=hunt_node,
        additional_instruction="Judge from answer; allow variants indicating 9 of 14 collared pumas."
    )

    # Supporting_URLs
    urls = _safe_urls(ps.urls if ps else [])
    sup_node = evaluator.add_leaf(
        id="patagonia_supporting_urls",
        desc="Provides valid URL reference(s) to the original research publication (journal/DOI landing page) and/or acceptable news outlet sources that collectively support the provided study required details",
        parent=topic_node,
        critical=True
    )
    if urls:
        claim = ("These sources collectively support the Patagonia pumas study details, including journal/date, authors/affiliations, "
                 "park/country, penguin species, colony size and coastline, breeding season, camera traps and GPS collars, tracking period, "
                 "and how many collared pumas hunted penguins.")
        await evaluator.verify(
            claim=claim,
            node=sup_node,
            sources=urls,
            additional_instruction="Prefer verifying via the journal or DOI landing page; reputable news outlets acceptable."
        )
    else:
        await evaluator.verify(
            claim="The answer includes valid URL references that support the Patagonia pumas study details.",
            node=sup_node,
            additional_instruction="Judge from the answer text. If no URLs are provided for this topic, mark this as incorrect."
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 'pets_animal_welfare_2025' task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator (framework's root is non-critical by design)
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

    # Create a critical task node under the framework root to mirror rubric's critical Root
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Provide complete information for all four specified 2025 pets/animal-welfare topics, with valid supporting URL references for the provided details",
        parent=root,
        critical=True
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=PetsAnimalWelfare2025Extraction,
        extraction_name="pets_animal_welfare_2025_extraction"
    )

    # Record ground truth info for transparency
    evaluator.add_ground_truth({
        "National_Dog_Show_Expected": NDS_EXPECTED,
        "AKC_Sporting_Expected": AKC_EXPECTED,
        "Whiting_Ranch_Expected": WHITING_EXPECTED,
        "Patagonia_Study_Expected": PATAGONIA_EXPECTED
    }, gt_type="expected_facts")

    # Add some custom info (URL stats) to summary
    evaluator.add_custom_info(
        {
            "nds_url_count": len(_safe_urls(extracted.national_dog_show.urls if extracted.national_dog_show else [])),
            "akc_url_count": len(_safe_urls(extracted.akc_sporting.urls if extracted.akc_sporting else [])),
            "whiting_url_count": len(_safe_urls(extracted.whiting_ranch_incident.urls if extracted.whiting_ranch_incident else [])),
            "patagonia_url_count": len(_safe_urls(extracted.patagonia_study.urls if extracted.patagonia_study else [])),
        },
        info_type="url_counts",
        info_name="url_statistics"
    )

    # Build verification tree per topic
    await verify_national_dog_show(evaluator, task_root, extracted.national_dog_show)
    await verify_akc_sporting(evaluator, task_root, extracted.akc_sporting)
    await verify_whiting_ranch(evaluator, task_root, extracted.whiting_ranch_incident)
    await verify_patagonia_study(evaluator, task_root, extracted.patagonia_study)

    # Return structured summary
    return evaluator.get_summary()