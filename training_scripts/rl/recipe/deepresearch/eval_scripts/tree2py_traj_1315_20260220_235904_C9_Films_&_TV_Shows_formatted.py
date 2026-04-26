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
TASK_ID = "cbs_2024_25_four_items"
TASK_DESCRIPTION = (
    "For the CBS 2024-25 broadcast season, identify the following four items with detailed information and reference URLs:\n\n"
    "1. A primetime series that premiered with a special 'sneak peek' episode before moving to its regular weekly timeslot. Provide: (a) the show title, (b) the date of the sneak peek premiere, (c) the date it moved to its regular timeslot, and (d) the name of the lead star.\n\n"
    "2. A primetime series that premiered with a 2-hour event episode in October 2024. Provide: (a) the show title, (b) the exact premiere date, (c) confirmation that it was a 2-hour premiere format, and (d) the name of the established franchise it belongs to.\n\n"
    "3. A primetime comedy series that is a direct sequel or spinoff of another CBS sitcom. Provide: (a) the show title, (b) the name of the parent show it spins off from, (c) the full names of the two lead actors who carry their roles from the parent show, and (d) the nature of the relationship (sequel/spinoff/continuation).\n\n"
    "4. The filming location for Hell's Kitchen Season 23. Provide: (a) the full name of the resort or casino venue, (b) the city or town, (c) the state, and (d) the complete street address.\n\n"
    "For each piece of information, include reference URLs that confirm your answers."
)

# Ground truth expectations derived from rubric
EXPECTED = {
    "item1": {
        "title": "Matlock",
        "sneak_peek_date": "September 22, 2024",
        "regular_timeslot_date": "October 17, 2024",
        "lead_star": "Kathy Bates",
    },
    "item2": {
        "title": "NCIS: Origins",
        "premiere_date": "October 14, 2024",
        "two_hour_premiere": "2-hour",  # descriptor; used in instruction
        "franchise": "NCIS",
    },
    "item3": {
        "title": "Georgie & Mandy's First Marriage",
        "parent_show": "Young Sheldon",
        "lead_actors": ["Montana Jordan", "Emily Osment"],
        "relationship_type_allowed": ["sequel", "spinoff", "continuation"],
    },
    "item4": {
        "venue_name": "Foxwoods Resort Casino",
        "city": "Mashantucket",
        "state": "Connecticut",
        "street_address": "350 Trolley Line Boulevard, Ledyard, CT 06338",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Item1Extraction(BaseModel):
    title: Optional[str] = None
    sneak_peek_date: Optional[str] = None
    regular_timeslot_date: Optional[str] = None
    lead_star: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Item2Extraction(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    two_hour_format: Optional[str] = None  # e.g., "2-hour", "two-hour", or a sentence confirming it
    franchise: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Item3Extraction(BaseModel):
    title: Optional[str] = None
    parent_show: Optional[str] = None
    lead_actors: List[str] = Field(default_factory=list)  # Expect two names
    reprise_statement: Optional[str] = None  # sentence indicating they reprise roles
    relationship_type: Optional[str] = None  # "sequel"/"spinoff"/"continuation" or similar
    sources: List[str] = Field(default_factory=list)


class Item4Extraction(BaseModel):
    venue_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    street_address: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AnswerExtraction(BaseModel):
    item1: Optional[Item1Extraction] = None
    item2: Optional[Item2Extraction] = None
    item3: Optional[Item3Extraction] = None
    item4: Optional[Item4Extraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_items() -> str:
    return """
    Extract the four requested items and their attributes exactly as presented in the answer. 
    If multiple candidates are discussed for an item, extract the first one that matches the item's constraint.
    If any attribute is missing, return null for that attribute. 
    For each item, also extract all explicit URLs from the answer that are intended to support that item's facts.

    Return a JSON object with the following structure:

    {
      "item1": {
        "title": string | null,
        "sneak_peek_date": string | null,   // e.g., "September 22, 2024" (allow alternative formats in the answer)
        "regular_timeslot_date": string | null, // e.g., "October 17, 2024"
        "lead_star": string | null,
        "sources": string[]                 // all URLs explicitly cited for this item
      },
      "item2": {
        "title": string | null,
        "premiere_date": string | null,     // exact date in October 2024
        "two_hour_format": string | null,   // text that indicates it was a 2-hour premiere if claimed (e.g., "2-hour premiere")
        "franchise": string | null,         // e.g., "NCIS"
        "sources": string[]
      },
      "item3": {
        "title": string | null,
        "parent_show": string | null,       // e.g., "Young Sheldon"
        "lead_actors": string[],            // list of lead actors' full names (expect two names)
        "reprise_statement": string | null, // sentence or phrase showing they reprise/carry over their roles
        "relationship_type": string | null, // one of sequel/spinoff/continuation (or similar wording used in the answer)
        "sources": string[]
      },
      "item4": {
        "venue_name": string | null,        // e.g., "Foxwoods Resort Casino"
        "city": string | null,              // e.g., "Mashantucket"
        "state": string | null,             // e.g., "Connecticut"
        "street_address": string | null,    // e.g., "350 Trolley Line Boulevard, Ledyard, CT 06338"
        "sources": string[]
      }
    }

    SPECIAL RULES FOR URL EXTRACTION:
    - Only extract URLs that are explicitly present in the answer.
    - Accept both plain URLs and markdown-formatted links; output the actual URL string.
    - If a URL is missing protocol, prepend "http://".
    - If no URLs are provided for an item, return an empty array for that item's "sources".
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm_list(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification functions: build subtrees per item                             #
# --------------------------------------------------------------------------- #
async def verify_item1(evaluator: Evaluator, parent_node, item: Optional[Item1Extraction]) -> None:
    node = evaluator.add_parallel(
        id="item_1_sneak_peek_series",
        desc="Sneak-peek-before-regular-timeslot primetime series (constraint-specified).",
        parent=parent_node,
        critical=False
    )

    title_val = (item.title if item else None) or ""
    sneak_val = (item.sneak_peek_date if item else None) or ""
    regular_val = (item.regular_timeslot_date if item else None) or ""
    star_val = (item.lead_star if item else None) or ""
    sources = _norm_list(item.sources if item else [])

    # Value checks (simple equality against expected)
    n1 = evaluator.add_leaf(
        id="item_1_title_is_matlock",
        desc="Show title is exactly 'Matlock'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 1 title '{title_val}' matches 'Matlock' (case-insensitive, ignore trivial punctuation/formatting).",
        node=n1,
        additional_instruction="Treat 'Matlock' equivalently even with minor punctuation or capitalization differences."
    )

    n2 = evaluator.add_leaf(
        id="item_1_sneak_peek_date_is_2024_09_22",
        desc="Sneak peek premiere date is exactly September 22, 2024.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 1 sneak peek premiere date '{sneak_val}' equals 'September 22, 2024' (accept variants like 'Sept. 22, 2024' or '9/22/2024').",
        node=n2,
        additional_instruction="Allow standard date formatting variants corresponding to September 22, 2024."
    )

    n3 = evaluator.add_leaf(
        id="item_1_regular_timeslot_is_2024_10_17",
        desc="Date it moved to its regular timeslot is exactly October 17, 2024.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 1 regular timeslot date '{regular_val}' equals 'October 17, 2024' (accept variants like 'Oct. 17, 2024' or '10/17/2024').",
        node=n3,
        additional_instruction="Allow standard date formatting variants corresponding to October 17, 2024."
    )

    n4 = evaluator.add_leaf(
        id="item_1_lead_star_is_kathy_bates",
        desc="Lead star is exactly 'Kathy Bates'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 1 lead star '{star_val}' matches 'Kathy Bates' (case-insensitive; allow middle initials if any).",
        node=n4,
        additional_instruction="Treat 'Kathy Bates' equivalently with minor name formatting variants."
    )

    # Reference support subgroup
    refs_group = evaluator.add_parallel(
        id="item_1_refs_support",
        desc="Reference URLs support Item 1 attributes",
        parent=node,
        critical=True
    )

    src_exist = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item_1_sources_present",
        desc="Item 1 has reference URL(s) provided.",
        parent=refs_group,
        critical=True
    )

    # Title support
    s1 = evaluator.add_leaf(
        id="item_1_support_title",
        desc="Item 1 sources support the show title 'Matlock'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show title is 'Matlock'.",
        node=s1,
        sources=sources,
        additional_instruction="Verify on any provided URL that the show is named 'Matlock'."
    )

    # Sneak peek support
    s2 = evaluator.add_leaf(
        id="item_1_support_sneak_peek_date",
        desc="Item 1 sources support the sneak peek premiere date September 22, 2024.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Matlock premiered with a special 'sneak peek' episode on September 22, 2024 (on CBS).",
        node=s2,
        sources=sources,
        additional_instruction="Look for wording like 'sneak peek', 'special preview', or similar confirming the date 9/22/2024."
    )

    # Regular timeslot support
    s3 = evaluator.add_leaf(
        id="item_1_support_regular_timeslot_date",
        desc="Item 1 sources support the regular weekly timeslot date October 17, 2024.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Matlock moved to its regular weekly timeslot on October 17, 2024.",
        node=s3,
        sources=sources,
        additional_instruction="Look for explicit mention of the regular weekly timeslot beginning on 10/17/2024."
    )

    # Lead star support
    s4 = evaluator.add_leaf(
        id="item_1_support_lead_star",
        desc="Item 1 sources support that the lead star is Kathy Bates.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Kathy Bates is the lead star of Matlock.",
        node=s4,
        sources=sources,
        additional_instruction="The page should explicitly connect Kathy Bates as lead star of Matlock."
    )


async def verify_item2(evaluator: Evaluator, parent_node, item: Optional[Item2Extraction]) -> None:
    node = evaluator.add_parallel(
        id="item_2_two_hour_oct2024",
        desc="Primetime series with a 2-hour event premiere episode in October 2024 (constraint-specified).",
        parent=parent_node,
        critical=False
    )

    title_val = (item.title if item else None) or ""
    pdate_val = (item.premiere_date if item else None) or ""
    two_hr_val = (item.two_hour_format if item else None) or ""
    franchise_val = (item.franchise if item else None) or ""
    sources = _norm_list(item.sources if item else [])

    # Value checks
    n1 = evaluator.add_leaf(
        id="item_2_title_is_ncis_origins",
        desc="Show title is exactly 'NCIS: Origins'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 2 title '{title_val}' matches 'NCIS: Origins' (case-insensitive; allow colon and spacing variants).",
        node=n1,
        additional_instruction="Treat 'NCIS: Origins' equivalently with minor punctuation/case variants."
    )

    n2 = evaluator.add_leaf(
        id="item_2_premiere_date_2024_10_14",
        desc="Exact premiere date is October 14, 2024.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 2 premiere date '{pdate_val}' equals 'October 14, 2024' (allow variants 'Oct. 14, 2024' or '10/14/2024').",
        node=n2,
        additional_instruction="Allow standard date formatting variants corresponding to October 14, 2024."
    )

    n3 = evaluator.add_leaf(
        id="item_2_two_hour_event_confirmed",
        desc="Answer explicitly confirms the premiere was a 2-hour event / 2-hour episode format.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 2 information ('{two_hr_val}') explicitly confirms the premiere was a 2-hour event.",
        node=n3,
        additional_instruction="If the extracted text indicates '2-hour', 'two-hour', or equivalent wording, consider it confirmed."
    )

    n4 = evaluator.add_leaf(
        id="item_2_franchise_is_ncis",
        desc="Established franchise named is exactly 'NCIS'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 2 franchise '{franchise_val}' matches 'NCIS' (case-insensitive).",
        node=n4,
        additional_instruction="Treat 'NCIS' equivalently with minor punctuation/case variants."
    )

    # Reference support subgroup
    refs_group = evaluator.add_parallel(
        id="item_2_refs_support",
        desc="Reference URLs support Item 2 attributes",
        parent=node,
        critical=True
    )

    src_exist = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item_2_sources_present",
        desc="Item 2 has reference URL(s) provided.",
        parent=refs_group,
        critical=True
    )

    s1 = evaluator.add_leaf(
        id="item_2_support_title",
        desc="Item 2 sources support the show title 'NCIS: Origins'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show title is 'NCIS: Origins'.",
        node=s1,
        sources=sources,
        additional_instruction="Verify the series is titled 'NCIS: Origins'."
    )

    s2 = evaluator.add_leaf(
        id="item_2_support_premiere_date",
        desc="Item 2 sources support the premiere date October 14, 2024.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="NCIS: Origins premiered on October 14, 2024.",
        node=s2,
        sources=sources,
        additional_instruction="Look for explicit mention of the exact premiere date 10/14/2024."
    )

    s3 = evaluator.add_leaf(
        id="item_2_support_two_hour_format",
        desc="Item 2 sources support that it was a 2-hour premiere format.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The series NCIS: Origins had a 2-hour premiere episode/event.",
        node=s3,
        sources=sources,
        additional_instruction="Wording like 'two-hour premiere', '2-hour event' qualifies."
    )

    s4 = evaluator.add_leaf(
        id="item_2_support_franchise",
        desc="Item 2 sources support that the franchise is 'NCIS'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="NCIS: Origins belongs to the 'NCIS' franchise.",
        node=s4,
        sources=sources,
        additional_instruction="Page should clearly connect the show as part of the NCIS franchise."
    )


async def verify_item3(evaluator: Evaluator, parent_node, item: Optional[Item3Extraction]) -> None:
    node = evaluator.add_parallel(
        id="item_3_cbs_sitcom_spinoff",
        desc="Primetime comedy that is a direct sequel/spinoff of another CBS sitcom (constraint-specified).",
        parent=parent_node,
        critical=False
    )

    title_val = (item.title if item else None) or ""
    parent_show_val = (item.parent_show if item else None) or ""
    lead_actors_list = (item.lead_actors if item else None) or []
    reprise_val = (item.reprise_statement if item else None) or ""
    rel_type_val = (item.relationship_type if item else None) or ""
    sources = _norm_list(item.sources if item else [])

    # Value checks
    n1 = evaluator.add_leaf(
        id="item_3_title_is_georgie_mandy_first_marriage",
        desc="Show title is exactly 'Georgie & Mandy's First Marriage'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 3 title '{title_val}' matches \"Georgie & Mandy's First Marriage\" (case-insensitive; ignore minor punctuation variants).",
        node=n1,
        additional_instruction="Treat minor punctuation and capitalization variants as equivalent."
    )

    n2 = evaluator.add_leaf(
        id="item_3_parent_is_young_sheldon",
        desc="Parent show is exactly 'Young Sheldon'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 3 parent show '{parent_show_val}' matches 'Young Sheldon' (case-insensitive).",
        node=n2,
        additional_instruction="Treat 'Young Sheldon' equivalently with minor formatting variants."
    )

    n3 = evaluator.add_leaf(
        id="item_3_lead_actors_are_montana_jordan_and_emily_osment",
        desc="Provides the two lead actors' full names, and they are exactly 'Montana Jordan' and 'Emily Osment'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 3 lead actors {lead_actors_list} contain exactly 'Montana Jordan' and 'Emily Osment' (order-insensitive; allow minor name formatting variants).",
        node=n3,
        additional_instruction="Confirm both names are present; allow case-insensitive and minor punctuation variants."
    )

    n4 = evaluator.add_leaf(
        id="item_3_actors_reprise_roles",
        desc="Answer explicitly indicates the two lead actors carry over/reprise their roles from the parent show.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 3 information ('{reprise_val}') explicitly indicates that Montana Jordan and Emily Osment reprise/carry over their roles from 'Young Sheldon'.",
        node=n4,
        additional_instruction="If the statement clearly asserts they reprise their roles, consider this confirmed."
    )

    n5 = evaluator.add_leaf(
        id="item_3_relationship_type_valid",
        desc="Answer characterizes the relationship as a direct sequel/spinoff/continuation consistent with the constraints.",
        parent=node,
        critical=True
    )
    allowed = ", ".join(EXPECTED["item3"]["relationship_type_allowed"])
    await evaluator.verify(
        claim=f"The extracted Item 3 relationship type '{rel_type_val}' is one of [{allowed}] and correctly describes the show's relationship to 'Young Sheldon'.",
        node=n5,
        additional_instruction="Accept synonyms for 'spinoff', 'sequel', or 'continuation' if clearly equivalent."
    )

    # Reference support subgroup
    refs_group = evaluator.add_parallel(
        id="item_3_refs_support",
        desc="Reference URLs support Item 3 attributes",
        parent=node,
        critical=True
    )

    src_exist = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item_3_sources_present",
        desc="Item 3 has reference URL(s) provided.",
        parent=refs_group,
        critical=True
    )

    s1 = evaluator.add_leaf(
        id="item_3_support_title",
        desc="Item 3 sources support the show title 'Georgie & Mandy's First Marriage'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The show title is \"Georgie & Mandy's First Marriage\".",
        node=s1,
        sources=sources,
        additional_instruction="Verify that the series uses this exact title (allowing minor punctuation)."
    )

    s2 = evaluator.add_leaf(
        id="item_3_support_parent_show",
        desc="Item 3 sources support that the parent show is 'Young Sheldon'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="\"Georgie & Mandy's First Marriage\" is a direct sequel/spinoff/continuation of 'Young Sheldon'.",
        node=s2,
        sources=sources,
        additional_instruction="The page should explicitly connect the new series to 'Young Sheldon'."
    )

    s3 = evaluator.add_leaf(
        id="item_3_support_lead_actors",
        desc="Item 3 sources support that the two lead actors are Montana Jordan and Emily Osment.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The lead actors for \"Georgie & Mandy's First Marriage\" are Montana Jordan and Emily Osment.",
        node=s3,
        sources=sources,
        additional_instruction="Look for explicit listing of both names as leads."
    )

    s4 = evaluator.add_leaf(
        id="item_3_support_reprise_roles",
        desc="Item 3 sources support that the two leads reprise their roles from 'Young Sheldon'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="Montana Jordan and Emily Osment reprise/carry over their roles from 'Young Sheldon' in the new series.",
        node=s4,
        sources=sources,
        additional_instruction="The page should clearly state that these actors reprise their roles."
    )

    s5 = evaluator.add_leaf(
        id="item_3_support_relationship_type",
        desc="Item 3 sources support the relationship characterization (sequel/spinoff/continuation).",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="\"Georgie & Mandy's First Marriage\" is characterized as a spinoff/sequel/continuation of 'Young Sheldon'.",
        node=s5,
        sources=sources,
        additional_instruction="Any equivalent phrasing that clearly conveys spinoff/sequel/continuation is acceptable."
    )


async def verify_item4(evaluator: Evaluator, parent_node, item: Optional[Item4Extraction]) -> None:
    node = evaluator.add_parallel(
        id="item_4_hk_s23_location",
        desc="Filming location details for Hell's Kitchen Season 23 (constraint-specified).",
        parent=parent_node,
        critical=False
    )

    venue_val = (item.venue_name if item else None) or ""
    city_val = (item.city if item else None) or ""
    state_val = (item.state if item else None) or ""
    addr_val = (item.street_address if item else None) or ""
    sources = _norm_list(item.sources if item else [])

    # Value checks
    n1 = evaluator.add_leaf(
        id="item_4_venue_is_foxwoods",
        desc="Venue name is exactly 'Foxwoods Resort Casino'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 4 venue '{venue_val}' matches 'Foxwoods Resort Casino' (case-insensitive; allow minor punctuation variants).",
        node=n1,
        additional_instruction="Treat minor punctuation and capitalization variants as equivalent."
    )

    n2 = evaluator.add_leaf(
        id="item_4_city_is_mashantucket",
        desc="City/town is exactly 'Mashantucket'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 4 city/town '{city_val}' matches 'Mashantucket' (case-insensitive).",
        node=n2,
        additional_instruction="Accept 'Mashantucket' exactly (allow case-insensitive)."
    )

    n3 = evaluator.add_leaf(
        id="item_4_state_is_connecticut",
        desc="State is exactly 'Connecticut'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 4 state '{state_val}' matches 'Connecticut' (case-insensitive).",
        node=n3,
        additional_instruction="Accept 'Connecticut' exactly (allow case-insensitive)."
    )

    n4 = evaluator.add_leaf(
        id="item_4_address_is_full",
        desc="Complete street address is exactly '350 Trolley Line Boulevard, Ledyard, CT 06338'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The extracted Item 4 street address '{addr_val}' matches '350 Trolley Line Boulevard, Ledyard, CT 06338' (allow minor punctuation/abbreviation variants).",
        node=n4,
        additional_instruction="Accept standard postal abbreviations if equivalent (e.g., 'Blvd' for 'Boulevard')."
    )

    # Reference support subgroup
    refs_group = evaluator.add_parallel(
        id="item_4_refs_support",
        desc="Reference URLs support Item 4 attributes",
        parent=node,
        critical=True
    )

    src_exist = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="item_4_sources_present",
        desc="Item 4 has reference URL(s) provided.",
        parent=refs_group,
        critical=True
    )

    s1 = evaluator.add_leaf(
        id="item_4_support_venue",
        desc="Item 4 sources support the venue name 'Foxwoods Resort Casino'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The filming location venue is Foxwoods Resort Casino.",
        node=s1,
        sources=sources,
        additional_instruction="The page should explicitly state the venue name."
    )

    s2 = evaluator.add_leaf(
        id="item_4_support_city",
        desc="Item 4 sources support the city/town 'Mashantucket'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The filming location city/town is Mashantucket.",
        node=s2,
        sources=sources,
        additional_instruction="The page should indicate the municipality as 'Mashantucket'."
    )

    s3 = evaluator.add_leaf(
        id="item_4_support_state",
        desc="Item 4 sources support the state 'Connecticut'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The filming location state is Connecticut.",
        node=s3,
        sources=sources,
        additional_instruction="Look for explicit 'Connecticut' as the state."
    )

    s4 = evaluator.add_leaf(
        id="item_4_support_address",
        desc="Item 4 sources support the full street address '350 Trolley Line Boulevard, Ledyard, CT 06338'.",
        parent=refs_group,
        critical=True
    )
    await evaluator.verify(
        claim="The full street address is 350 Trolley Line Boulevard, Ledyard, CT 06338.",
        node=s4,
        sources=sources,
        additional_instruction="The page should contain this complete address (minor formatting differences are acceptable)."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root should be non-critical to allow mixed children)
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

    # Extract all items in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_all_items(),
        template_class=AnswerExtraction,
        extraction_name="all_items_extraction"
    )

    # Add ground truth for transparency
    evaluator.add_ground_truth(
        {
            "expected_item_1": EXPECTED["item1"],
            "expected_item_2": EXPECTED["item2"],
            "expected_item_3": {
                "title": EXPECTED["item3"]["title"],
                "parent_show": EXPECTED["item3"]["parent_show"],
                "lead_actors": EXPECTED["item3"]["lead_actors"],
                "relationship_type_allowed": EXPECTED["item3"]["relationship_type_allowed"],
            },
            "expected_item_4": EXPECTED["item4"],
        },
        gt_type="expected_values"
    )

    # Build tree per item
    await verify_item1(evaluator, root, extracted.item1)
    await verify_item2(evaluator, root, extracted.item2)
    await verify_item3(evaluator, root, extracted.item3)
    await verify_item4(evaluator, root, extracted.item4)

    return evaluator.get_summary()