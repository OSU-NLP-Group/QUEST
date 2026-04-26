import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tahoe_westshore_csp_campground_rv_ada_pet"
TASK_DESCRIPTION = (
    "A family with a 28-foot RV trailer, a family member who uses a wheelchair and requires ADA-accessible facilities, "
    "and a pet dog is planning a camping trip in the Lake Tahoe area of Northern California for 5 days in July 2026. "
    "Identify ONE suitable California State Park campground on the west shore of Lake Tahoe that can accommodate their RV "
    "and has ADA-accessible campsites. Provide comprehensive details about this campground including: "
    "(1) the campground name and location, "
    "(2) RV/trailer accommodation capabilities, "
    "(3) the number and specific site numbers of ADA-accessible campsites, "
    "(4) detailed pet policies including where dogs are and are not allowed, "
    "(5) hookup availability (water/electric/sewer), "
    "(6) shower facility availability and accessibility, "
    "(7) dump station availability, "
    "(8) reservation booking window and opening time, "
    "(9) maximum occupancy per campsite, "
    "(10) lake or beach access availability, "
    "(11) fire ring provisions, "
    "(12) food storage requirements and facilities, and "
    "(13) total campsite count. Provide official California State Parks sources for all information."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
def _list_field() -> List[str]:
    return []


class RVInfo(BaseModel):
    accepted: Optional[str] = None
    length_info: Optional[str] = None
    rv_urls: List[str] = Field(default_factory=_list_field)


class AccessibilityInfo(BaseModel):
    accessible_sites_exist: Optional[str] = None
    number_of_accessible_sites: Optional[str] = None
    specific_site_numbers: List[str] = Field(default_factory=_list_field)
    accessible_restrooms: Optional[str] = None
    accessibility_urls: List[str] = Field(default_factory=_list_field)


class PetPolicyInfo(BaseModel):
    dogs_allowed_campground: Optional[str] = None
    leash_requirement: Optional[str] = None
    allowed_areas: List[str] = Field(default_factory=_list_field)
    restricted_areas: List[str] = Field(default_factory=_list_field)
    pet_urls: List[str] = Field(default_factory=_list_field)


class HookupsInfo(BaseModel):
    water: Optional[str] = None
    electric: Optional[str] = None
    sewer: Optional[str] = None
    status_summary: Optional[str] = None
    hookup_urls: List[str] = Field(default_factory=_list_field)


class ShowersInfo(BaseModel):
    showers_available: Optional[str] = None
    shower_accessibility: Optional[str] = None
    showers_urls: List[str] = Field(default_factory=_list_field)


class DumpStationInfo(BaseModel):
    dump_station_available: Optional[str] = None
    dump_urls: List[str] = Field(default_factory=_list_field)


class ReservationInfo(BaseModel):
    booking_window: Optional[str] = None
    booking_open_time: Optional[str] = None
    ada_reservation_requirement: Optional[str] = None
    reservation_urls: List[str] = Field(default_factory=_list_field)


class OccupancyInfo(BaseModel):
    max_people_per_site: Optional[str] = None
    occupancy_urls: List[str] = Field(default_factory=_list_field)


class LakeAccessInfo(BaseModel):
    lake_beach_access: Optional[str] = None
    lake_urls: List[str] = Field(default_factory=_list_field)


class FireInfo(BaseModel):
    fire_rings_present: Optional[str] = None
    fire_urls: List[str] = Field(default_factory=_list_field)


class FoodStorageInfo(BaseModel):
    bear_storage_required: Optional[str] = None
    storage_facilities: Optional[str] = None
    storage_urls: List[str] = Field(default_factory=_list_field)


class CampsiteCountInfo(BaseModel):
    total_campsites: Optional[str] = None
    count_urls: List[str] = Field(default_factory=_list_field)


class CampgroundExtraction(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    campground_url: Optional[str] = None

    rv_info: RVInfo = Field(default_factory=RVInfo)
    accessibility_info: AccessibilityInfo = Field(default_factory=AccessibilityInfo)
    pet_info: PetPolicyInfo = Field(default_factory=PetPolicyInfo)
    hookups_info: HookupsInfo = Field(default_factory=HookupsInfo)
    showers_info: ShowersInfo = Field(default_factory=ShowersInfo)
    dump_info: DumpStationInfo = Field(default_factory=DumpStationInfo)
    reservation_info: ReservationInfo = Field(default_factory=ReservationInfo)
    occupancy_info: OccupancyInfo = Field(default_factory=OccupancyInfo)
    lake_info: LakeAccessInfo = Field(default_factory=LakeAccessInfo)
    fire_info: FireInfo = Field(default_factory=FireInfo)
    storage_info: FoodStorageInfo = Field(default_factory=FoodStorageInfo)
    count_info: CampsiteCountInfo = Field(default_factory=CampsiteCountInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground_details() -> str:
    return (
        "Extract structured details for ONE California State Park campground on the west shore of Lake Tahoe that the answer identified. "
        "Return a JSON object matching the CampgroundExtraction schema below. Extract ONLY what is explicitly stated in the answer. "
        "For each category, also extract official California State Parks or ReserveCalifornia URLs that support the information.\n"
        "\nSchema and fields to extract:\n"
        "- name: Official campground name (string)\n"
        "- location: Brief location description (string), ideally indicating 'west shore of Lake Tahoe'\n"
        "- campground_url: The official California State Parks campground page URL (parks.ca.gov) or the park page that describes the campground\n"
        "\nRV_Accommodation (rv_info):\n"
        "- accepted: Statement whether RVs/trailers are accepted (string)\n"
        "- length_info: RV/trailer length limits or accommodation information; if the answer claims 28-foot compatibility, include that statement (string)\n"
        "- rv_urls: List of URLs that support RV/trailer accommodation and length info (list of strings)\n"
        "\nADA_Accessibility (accessibility_info):\n"
        "- accessible_sites_exist: Statement confirming accessible campsites exist (string)\n"
        "- number_of_accessible_sites: Total number of accessible campsites (string)\n"
        "- specific_site_numbers: Specific site numbers for accessible campsites (list of strings)\n"
        "- accessible_restrooms: Statement confirming accessible restroom/shower facilities (string)\n"
        "- accessibility_urls: URLs supporting accessibility claims (list of strings)\n"
        "\nPet_Policies (pet_info):\n"
        "- dogs_allowed_campground: Statement confirming dogs are allowed in the campground (string)\n"
        "- leash_requirement: Leash length requirement (should be 6 feet) (string)\n"
        "- allowed_areas: Where dogs are allowed (list of strings)\n"
        "- restricted_areas: Where dogs are not allowed (list of strings)\n"
        "- pet_urls: URLs supporting pet policies (list of strings)\n"
        "\nHookup_Availability (hookups_info):\n"
        "- water: Availability of water hookups (string)\n"
        "- electric: Availability of electric hookups (string)\n"
        "- sewer: Availability of sewer hookups (string)\n"
        "- status_summary: Overall summary such as 'no hookups' or details (string)\n"
        "- hookup_urls: URLs supporting hookup info (list of strings)\n"
        "\nShower_Facilities (showers_info):\n"
        "- showers_available: Statement whether showers are available (string)\n"
        "- shower_accessibility: Statement about wheelchair accessibility of showers (string)\n"
        "- showers_urls: URLs supporting shower info (list of strings)\n"
        "\nDump_Station (dump_info):\n"
        "- dump_station_available: Statement about dump station availability (string)\n"
        "- dump_urls: URLs supporting dump station info (list of strings)\n"
        "\nReservation_Requirements (reservation_info):\n"
        "- booking_window: How far in advance reservations can be made (string; California State Parks is 6 months)\n"
        "- booking_open_time: Time reservations open each day (string; should be 8 a.m. PST/PDT)\n"
        "- ada_reservation_requirement: Statement about DMV Disabled Placard or License Plate requirement for accessible sites (string)\n"
        "- reservation_urls: URLs supporting reservation info (list of strings)\n"
        "\nMaximum_Occupancy (occupancy_info):\n"
        "- max_people_per_site: Maximum occupancy per campsite (string; standard is 8) \n"
        "- occupancy_urls: URLs supporting occupancy info (list of strings)\n"
        "\nLake_Access (lake_info):\n"
        "- lake_beach_access: Statement about lake or beach access availability (string)\n"
        "- lake_urls: URLs supporting lake access info (list of strings)\n"
        "\nFire_Facilities (fire_info):\n"
        "- fire_rings_present: Statement about fire rings provided (string)\n"
        "- fire_urls: URLs supporting fire facilities info (list of strings)\n"
        "\nFood_Storage (storage_info):\n"
        "- bear_storage_required: Statement about bear-proof storage requirements (string)\n"
        "- storage_facilities: Statement confirming bear-proof lockers/boxes provided (string)\n"
        "- storage_urls: URLs supporting food storage info (list of strings)\n"
        "\nTotal_Campsite_Count (count_info):\n"
        "- total_campsites: Total number of campsites (string)\n"
        "- count_urls: URLs supporting campsite count (list of strings)\n"
        "\nRules:\n"
        "1) Do NOT invent information; if something is missing in the answer, return null or an empty list for that field.\n"
        "2) For URL fields, extract only actual URLs shown in the answer; prefer parks.ca.gov and reservecalifornia.com.\n"
        "3) If a URL is missing a protocol, prepend http://.\n"
        "4) Return exactly one campground's information."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def pick_sources(*url_lists: List[str], extra_url: Optional[str] = None) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str) and u.strip():
                combined.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    if not deduped and extra_url and isinstance(extra_url, str) and extra_url.strip():
        return [extra_url.strip()]
    return deduped


def has_official_source(urls: List[str]) -> bool:
    for u in urls:
        if not isinstance(u, str):
            continue
        lower = u.lower()
        if ("parks.ca.gov" in lower) or ("reservecalifornia.com" in lower):
            return True
    return False


def str_or_empty(x: Optional[str]) -> str:
    return x if (isinstance(x, str) and x.strip()) else ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_campground_identification(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_sequential(
        id="Campground_Identification",
        desc="Correctly identifies a California State Park campground on the west shore of Lake Tahoe",
        parent=root,
        critical=True,
    )

    # Campground_Name
    name_leaf = evaluator.add_leaf(
        id="Campground_Name",
        desc="Provides the official name of the campground",
        parent=node,
        critical=True,
    )
    name_claim = f"The official name of the campground is '{str_or_empty(data.name)}'."
    name_sources = pick_sources(
        [str_or_empty(data.campground_url)],
        data.rv_info.rv_urls,
        data.accessibility_info.accessibility_urls,
        data.pet_info.pet_urls,
        extra_url=data.campground_url,
    )
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Verify the campground's official name on California State Parks and/or ReserveCalifornia pages. Allow minor variations such as 'State Park Campground' appended to the park name.",
    )

    # Location_Verification
    location_leaf = evaluator.add_leaf(
        id="Location_Verification",
        desc="Confirms the campground is located on the west shore of Lake Tahoe within California State Parks system",
        parent=node,
        critical=True,
    )
    location_claim = (
        f"The campground is located on the west shore of Lake Tahoe and is part of the California State Parks system."
    )
    location_sources = pick_sources(
        [str_or_empty(data.campground_url)],
        data.pet_info.pet_urls,
        data.lake_info.lake_urls,
        extra_url=data.campground_url,
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=location_sources,
        additional_instruction="Confirm that the official page indicates the park/campground is on Lake Tahoe's west shore (phrases like 'on the west shore of Lake Tahoe' or equivalent) and that it is indeed a California State Park.",
    )

    # Reference_URL_Campground
    ref_urls = pick_sources([str_or_empty(data.campground_url)])
    evaluator.add_custom_node(
        result=(len(ref_urls) > 0 and has_official_source(ref_urls)),
        id="Reference_URL_Campground",
        desc="Provides official California State Parks URL for the identified campground",
        parent=node,
        critical=True,
    )


async def build_rv_accommodation(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="RV_Accommodation",
        desc="Provides information about RV/trailer accommodation capabilities",
        parent=root,
        critical=True,
    )
    rv_sources = pick_sources(data.rv_info.rv_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # RV_Accepted
    rv_accepted_leaf = evaluator.add_leaf(
        id="RV_Accepted",
        desc="Confirms that RVs or trailers are accepted at the campground",
        parent=node,
        critical=True,
    )
    rv_accepted_claim = f"RVs or trailers are accepted at {str_or_empty(data.name)}."
    await evaluator.verify(
        claim=rv_accepted_claim,
        node=rv_accepted_leaf,
        sources=rv_sources,
        additional_instruction="Check the official pages for indications that RVs and/or trailers are permitted in the campground, not just tents.",
    )

    # Length_Information
    length_leaf = evaluator.add_leaf(
        id="Length_Information",
        desc="Provides information about RV/trailer length limits or accommodation",
        parent=node,
        critical=True,
    )
    # Prefer exact extracted info; if missing, verify 28-foot compatibility (family requirement)
    if str_or_empty(data.rv_info.length_info):
        length_claim = f"RV/trailer length information: {data.rv_info.length_info}."
    else:
        length_claim = "The campground can accommodate a 28-foot RV or trailer."
    await evaluator.verify(
        claim=length_claim,
        node=length_leaf,
        sources=rv_sources,
        additional_instruction="Confirm the site maximum vehicle/RV length or an explicit statement that a 28-foot RV/trailer can be accommodated. Use ReserveCalifornia site details if needed.",
    )

    # Reference_URL_RV
    evaluator.add_custom_node(
        result=(len(rv_sources) > 0 and has_official_source(rv_sources)),
        id="Reference_URL_RV",
        desc="Provides reference URL supporting RV accommodation information",
        parent=node,
        critical=True,
    )


async def build_ada_accessibility(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="ADA_Accessibility",
        desc="Provides complete information about ADA-accessible campsites",
        parent=root,
        critical=True,
    )
    acc_sources = pick_sources(data.accessibility_info.accessibility_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Accessible_Sites_Exist
    exist_leaf = evaluator.add_leaf(
        id="Accessible_Sites_Exist",
        desc="Confirms that ADA-accessible campsites are available",
        parent=node,
        critical=True,
    )
    exist_claim = "ADA-accessible campsites are available at the campground."
    await evaluator.verify(
        claim=exist_claim,
        node=exist_leaf,
        sources=acc_sources,
        additional_instruction="Look for explicit mention of accessible/ADA campsites or accessible facilities on official pages.",
    )

    # Number_of_Accessible_Sites
    num_leaf = evaluator.add_leaf(
        id="Number_of_Accessible_Sites",
        desc="States the total number of ADA-accessible campsites",
        parent=node,
        critical=True,
    )
    num_claim = f"The total number of ADA-accessible campsites is {str_or_empty(data.accessibility_info.number_of_accessible_sites)}."
    await evaluator.verify(
        claim=num_claim,
        node=num_leaf,
        sources=acc_sources,
        additional_instruction="Verify the total count of accessible campsites from official CA State Parks or ReserveCalifornia pages.",
    )

    # Specific_Site_Numbers
    sites_leaf = evaluator.add_leaf(
        id="Specific_Site_Numbers",
        desc="Provides specific site numbers of accessible campsites",
        parent=node,
        critical=True,
    )
    site_numbers_text = ", ".join(data.accessibility_info.specific_site_numbers) if data.accessibility_info.specific_site_numbers else ""
    sites_claim = f"The ADA-accessible campsite numbers include: {site_numbers_text}."
    await evaluator.verify(
        claim=sites_claim,
        node=sites_leaf,
        sources=acc_sources,
        additional_instruction="Confirm the specific accessible campsite numbers as listed on official sources.",
    )

    # Accessible_Restrooms
    rest_leaf = evaluator.add_leaf(
        id="Accessible_Restrooms",
        desc="Confirms accessible restroom/shower facilities are available",
        parent=node,
        critical=True,
    )
    rest_claim = "Accessible restroom and/or shower facilities are available at the campground."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_leaf,
        sources=acc_sources,
        additional_instruction="Look for mentions of ADA accessible restrooms or shower facilities on the official park or campground pages.",
    )

    # Reference_URL_Accessibility
    evaluator.add_custom_node(
        result=(len(acc_sources) > 0 and has_official_source(acc_sources)),
        id="Reference_URL_Accessibility",
        desc="Provides reference URL for accessibility information",
        parent=node,
        critical=True,
    )


async def build_pet_policies(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Pet_Policies",
        desc="Provides comprehensive pet policies including where dogs are allowed and restricted",
        parent=root,
        critical=True,
    )
    pet_sources = pick_sources(data.pet_info.pet_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Dogs_Allowed_Campground
    dogs_leaf = evaluator.add_leaf(
        id="Dogs_Allowed_Campground",
        desc="Confirms dogs are allowed in the campground",
        parent=node,
        critical=True,
    )
    dogs_claim = "Dogs are allowed in the campground."
    await evaluator.verify(
        claim=dogs_claim,
        node=dogs_leaf,
        sources=pet_sources,
        additional_instruction="Verify that dogs are permitted in the developed campground area on official pages.",
    )

    # Leash_Requirement
    leash_leaf = evaluator.add_leaf(
        id="Leash_Requirement",
        desc="States the leash length requirement (must be 6 feet for California State Parks)",
        parent=node,
        critical=True,
    )
    leash_claim = f"The leash length requirement is {str_or_empty(data.pet_info.leash_requirement)}."
    await evaluator.verify(
        claim=leash_claim,
        node=leash_leaf,
        sources=pet_sources,
        additional_instruction="Confirm that the leash length requirement is 6 feet per California State Parks policy.",
    )

    # Allowed_Areas
    allowed_leaf = evaluator.add_leaf(
        id="Allowed_Areas",
        desc="Specifies where dogs are allowed (campground, paved areas, etc.)",
        parent=node,
        critical=True,
    )
    allowed_text = ", ".join(data.pet_info.allowed_areas) if data.pet_info.allowed_areas else ""
    allowed_claim = f"Dogs are allowed in these areas: {allowed_text}."
    await evaluator.verify(
        claim=allowed_claim,
        node=allowed_leaf,
        sources=pet_sources,
        additional_instruction="Confirm areas where dogs are permitted such as campgrounds, paved areas, parking lots, etc.",
    )

    # Restricted_Areas
    restricted_leaf = evaluator.add_leaf(
        id="Restricted_Areas",
        desc="Specifies where dogs are not allowed (trails, beaches, etc.)",
        parent=node,
        critical=True,
    )
    restricted_text = ", ".join(data.pet_info.restricted_areas) if data.pet_info.restricted_areas else ""
    restricted_claim = f"Dogs are not allowed in these areas: {restricted_text}."
    await evaluator.verify(
        claim=restricted_claim,
        node=restricted_leaf,
        sources=pet_sources,
        additional_instruction="Confirm areas where dogs are prohibited such as trails, beaches, or other natural areas, per official park rules.",
    )

    # Reference_URL_Pets
    evaluator.add_custom_node(
        result=(len(pet_sources) > 0 and has_official_source(pet_sources)),
        id="Reference_URL_Pets",
        desc="Provides reference URL for pet policy information",
        parent=node,
        critical=True,
    )


async def build_hookups(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Hookup_Availability",
        desc="Provides information about water, electric, and sewer hookup availability",
        parent=root,
        critical=True,
    )
    hook_sources = pick_sources(data.hookups_info.hookup_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Hookup_Status
    hookup_leaf = evaluator.add_leaf(
        id="Hookup_Status",
        desc="States whether water, electric, or sewer hookups are available",
        parent=node,
        critical=True,
    )
    status_text = (
        f"Water: {str_or_empty(data.hookups_info.water)}; "
        f"Electric: {str_or_empty(data.hookups_info.electric)}; "
        f"Sewer: {str_or_empty(data.hookups_info.sewer)}; "
        f"Summary: {str_or_empty(data.hookups_info.status_summary)}"
    )
    hookup_claim = f"Hookup availability details - {status_text}"
    await evaluator.verify(
        claim=hookup_claim,
        node=hookup_leaf,
        sources=hook_sources,
        additional_instruction="Verify hookup availability (water, electric, sewer) or confirm that there are no hookups, per official sources.",
    )

    # Reference_URL_Hookups
    evaluator.add_custom_node(
        result=(len(hook_sources) > 0 and has_official_source(hook_sources)),
        id="Reference_URL_Hookups",
        desc="Provides reference URL for hookup information",
        parent=node,
        critical=True,
    )


async def build_showers(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Shower_Facilities",
        desc="Provides information about shower availability and accessibility",
        parent=root,
        critical=True,
    )
    shower_sources = pick_sources(data.showers_info.showers_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Showers_Available
    showers_leaf = evaluator.add_leaf(
        id="Showers_Available",
        desc="Confirms whether showers are available",
        parent=node,
        critical=True,
    )
    showers_claim = f"Showers availability: {str_or_empty(data.showers_info.showers_available)}."
    await evaluator.verify(
        claim=showers_claim,
        node=showers_leaf,
        sources=shower_sources,
        additional_instruction="Verify whether showers are available in the campground on official sources.",
    )

    # Shower_Accessibility
    shower_access_leaf = evaluator.add_leaf(
        id="Shower_Accessibility",
        desc="States whether showers are accessible for wheelchair users",
        parent=node,
        critical=True,
    )
    shower_access_claim = f"Shower accessibility for wheelchair users: {str_or_empty(data.showers_info.shower_accessibility)}."
    await evaluator.verify(
        claim=shower_access_claim,
        node=shower_access_leaf,
        sources=shower_sources,
        additional_instruction="Look for ADA accessible shower/restroom mentions or accessibility information on official pages.",
    )

    # Reference_URL_Showers
    evaluator.add_custom_node(
        result=(len(shower_sources) > 0 and has_official_source(shower_sources)),
        id="Reference_URL_Showers",
        desc="Provides reference URL for shower facility information",
        parent=node,
        critical=True,
    )


async def build_dump_station(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Dump_Station",
        desc="Provides information about dump station availability",
        parent=root,
        critical=True,
    )
    dump_sources = pick_sources(data.dump_info.dump_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Dump_Station_Available
    dump_leaf = evaluator.add_leaf(
        id="Dump_Station_Available",
        desc="Confirms whether a dump station is available",
        parent=node,
        critical=True,
    )
    dump_claim = f"Dump station availability: {str_or_empty(data.dump_info.dump_station_available)}."
    await evaluator.verify(
        claim=dump_claim,
        node=dump_leaf,
        sources=dump_sources,
        additional_instruction="Verify whether a dump station is present/available for the campground on official sources.",
    )

    # Reference_URL_Dump
    evaluator.add_custom_node(
        result=(len(dump_sources) > 0 and has_official_source(dump_sources)),
        id="Reference_URL_Dump",
        desc="Provides reference URL for dump station information",
        parent=node,
        critical=True,
    )


async def build_reservations(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reservation_Requirements",
        desc="Provides complete reservation booking information",
        parent=root,
        critical=True,
    )
    res_sources = pick_sources(data.reservation_info.reservation_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Booking_Window
    window_leaf = evaluator.add_leaf(
        id="Booking_Window",
        desc="States how far in advance reservations can be made (must be 6 months for California State Parks)",
        parent=node,
        critical=True,
    )
    window_claim = f"Reservations can be made {str_or_empty(data.reservation_info.booking_window)} in advance."
    await evaluator.verify(
        claim=window_claim,
        node=window_leaf,
        sources=res_sources,
        additional_instruction="Confirm that California State Parks reservations open a rolling 6 months in advance via ReserveCalifornia or official policy pages.",
    )

    # Booking_Time
    time_leaf = evaluator.add_leaf(
        id="Booking_Time",
        desc="States what time reservations open each day (must be 8 a.m. PST/PDT for California State Parks)",
        parent=node,
        critical=True,
    )
    time_claim = f"Reservations open at {str_or_empty(data.reservation_info.booking_open_time)} each day."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=res_sources,
        additional_instruction="Confirm that reservations open at 8 a.m. PST/PDT (local time) for California State Parks via official sources.",
    )

    # ADA_Reservation_Requirement
    ada_res_leaf = evaluator.add_leaf(
        id="ADA_Reservation_Requirement",
        desc="States that accessible campsite reservations require DMV Disabled Placard or License Plate",
        parent=node,
        critical=True,
    )
    ada_res_claim = f"Accessible campsite reservations require: {str_or_empty(data.reservation_info.ada_reservation_requirement)}."
    await evaluator.verify(
        claim=ada_res_claim,
        node=ada_res_leaf,
        sources=res_sources,
        additional_instruction="Verify the requirement for a DMV Disabled Placard or Disabled Person License Plate to reserve accessible sites, per official policy.",
    )

    # Reference_URL_Reservations
    evaluator.add_custom_node(
        result=(len(res_sources) > 0 and has_official_source(res_sources)),
        id="Reference_URL_Reservations",
        desc="Provides reference URL for reservation information",
        parent=node,
        critical=True,
    )


async def build_occupancy(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Maximum_Occupancy",
        desc="Provides maximum occupancy per campsite",
        parent=root,
        critical=True,
    )
    occ_sources = pick_sources(data.occupancy_info.occupancy_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # People_Limit
    people_leaf = evaluator.add_leaf(
        id="People_Limit",
        desc="States maximum number of people per campsite (standard is 8 for California State Parks)",
        parent=node,
        critical=True,
    )
    people_claim = f"The maximum number of people per campsite is {str_or_empty(data.occupancy_info.max_people_per_site)}."
    await evaluator.verify(
        claim=people_claim,
        node=people_leaf,
        sources=occ_sources,
        additional_instruction="Confirm the per-site occupancy limit (commonly 8) for this campground on official sources.",
    )

    # Reference_URL_Occupancy
    evaluator.add_custom_node(
        result=(len(occ_sources) > 0 and has_official_source(occ_sources)),
        id="Reference_URL_Occupancy",
        desc="Provides reference URL for occupancy information",
        parent=node,
        critical=True,
    )


async def build_lake_access(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Lake_Access",
        desc="Provides information about lake or beach access",
        parent=root,
        critical=True,
    )
    lake_sources = pick_sources(data.lake_info.lake_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Lake_Access_Available
    lake_leaf = evaluator.add_leaf(
        id="Lake_Access_Available",
        desc="Confirms whether lake or beach access is available from the campground",
        parent=node,
        critical=True,
    )
    lake_claim = f"Lake or beach access availability: {str_or_empty(data.lake_info.lake_beach_access)}."
    await evaluator.verify(
        claim=lake_claim,
        node=lake_leaf,
        sources=lake_sources,
        additional_instruction="Verify access to Lake Tahoe beaches or shoreline from the campground (even if via trails) on official sources.",
    )

    # Reference_URL_Lake
    evaluator.add_custom_node(
        result=(len(lake_sources) > 0 and has_official_source(lake_sources)),
        id="Reference_URL_Lake",
        desc="Provides reference URL for lake access information",
        parent=node,
        critical=True,
    )


async def build_fire_facilities(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Fire_Facilities",
        desc="Provides information about fire rings and fire regulations",
        parent=root,
        critical=True,
    )
    fire_sources = pick_sources(data.fire_info.fire_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Fire_Rings_Present
    fire_leaf = evaluator.add_leaf(
        id="Fire_Rings_Present",
        desc="Confirms whether fire rings are provided at campsites",
        parent=node,
        critical=True,
    )
    fire_claim = f"Fire ring provision: {str_or_empty(data.fire_info.fire_rings_present)}."
    await evaluator.verify(
        claim=fire_claim,
        node=fire_leaf,
        sources=fire_sources,
        additional_instruction="Verify that individual campsites provide a fire ring or whether fires are subject to seasonal restrictions.",
    )

    # Reference_URL_Fire
    evaluator.add_custom_node(
        result=(len(fire_sources) > 0 and has_official_source(fire_sources)),
        id="Reference_URL_Fire",
        desc="Provides reference URL for fire facility information",
        parent=node,
        critical=True,
    )


async def build_food_storage(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Food_Storage",
        desc="Provides information about food storage requirements and bear-proof facilities",
        parent=root,
        critical=True,
    )
    storage_sources = pick_sources(data.storage_info.storage_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Bear_Storage_Required
    bear_leaf = evaluator.add_leaf(
        id="Bear_Storage_Required",
        desc="States that bear-proof food storage is required or provided",
        parent=node,
        critical=True,
    )
    bear_claim = f"Bear-proof food storage requirement: {str_or_empty(data.storage_info.bear_storage_required)}."
    await evaluator.verify(
        claim=bear_claim,
        node=bear_leaf,
        sources=storage_sources,
        additional_instruction="Confirm that bear-resistant food storage is required/provided per official park rules.",
    )

    # Storage_Facilities
    storage_leaf = evaluator.add_leaf(
        id="Storage_Facilities",
        desc="Confirms that bear-proof storage facilities (lockers/boxes) are provided at sites",
        parent=node,
        critical=True,
    )
    storage_claim = f"Bear-proof storage facilities provided: {str_or_empty(data.storage_info.storage_facilities)}."
    await evaluator.verify(
        claim=storage_claim,
        node=storage_leaf,
        sources=storage_sources,
        additional_instruction="Verify presence of bear boxes/lockers at individual campsites per official sources.",
    )

    # Reference_URL_Storage
    evaluator.add_custom_node(
        result=(len(storage_sources) > 0 and has_official_source(storage_sources)),
        id="Reference_URL_Storage",
        desc="Provides reference URL for food storage information",
        parent=node,
        critical=True,
    )


async def build_total_campsite_count(evaluator: Evaluator, root: Any, data: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Total_Campsite_Count",
        desc="Provides total number of campsites at the campground",
        parent=root,
        critical=True,
    )
    count_sources = pick_sources(data.count_info.count_urls, [str_or_empty(data.campground_url)], extra_url=data.campground_url)

    # Campsite_Number
    count_leaf = evaluator.add_leaf(
        id="Campsite_Number",
        desc="States the total number of campsites",
        parent=node,
        critical=True,
    )
    count_claim = f"The total number of campsites at the campground is {str_or_empty(data.count_info.total_campsites)}."
    await evaluator.verify(
        claim=count_claim,
        node=count_leaf,
        sources=count_sources,
        additional_instruction="Verify the total count of campsites from official CA State Parks or ReserveCalifornia sources.",
    )

    # Reference_URL_Count
    evaluator.add_custom_node(
        result=(len(count_sources) > 0 and has_official_source(count_sources)),
        id="Reference_URL_Count",
        desc="Provides reference URL for campsite count information",
        parent=node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Complete and accurate information about a suitable Lake Tahoe west shore California State Park campground meeting all family requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Root must be critical; ensure children critical consistency
    root.critical = True

    # Extract all details from the answer
    extracted: CampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_campground_details(),
        template_class=CampgroundExtraction,
        extraction_name="campground_details",
    )

    # Build verification tree according to rubric
    await build_campground_identification(evaluator, root, extracted)
    await build_rv_accommodation(evaluator, root, extracted)
    await build_ada_accessibility(evaluator, root, extracted)
    await build_pet_policies(evaluator, root, extracted)
    await build_hookups(evaluator, root, extracted)
    await build_showers(evaluator, root, extracted)
    await build_dump_station(evaluator, root, extracted)
    await build_reservations(evaluator, root, extracted)
    await build_occupancy(evaluator, root, extracted)
    await build_lake_access(evaluator, root, extracted)
    await build_fire_facilities(evaluator, root, extracted)
    await build_food_storage(evaluator, root, extracted)
    await build_total_campsite_count(evaluator, root, extracted)

    return evaluator.get_summary()