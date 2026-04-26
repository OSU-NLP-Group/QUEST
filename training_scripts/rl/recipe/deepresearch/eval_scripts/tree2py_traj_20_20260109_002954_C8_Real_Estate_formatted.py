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
TASK_ID = "duke_energy_center_requirements"
TASK_DESCRIPTION = """A technology consulting firm with 35 employees is evaluating the Duke Energy Center (550 South Tryon Street, Charlotte, NC 28202) as a potential office location. Verify whether this building meets ALL of the following requirements:

1. Building Classification: Confirmed Class A office building designation
2. Location: Located within Charlotte city limits (not suburbs)
3. Building Size: Multi-story building with at least 3 floors
4. Elevator Access: Has elevator systems for vertical transportation
5. Parking: Provides on-site parking facilities
6. Public Transportation: Within 0.5 miles (approximately 2,600 feet) of a public transportation stop (bus or light rail station)
7. Conference Facilities: Has conference room or meeting space facilities available
8. Internet Infrastructure: Has modern telecommunications/high-speed internet infrastructure
9. ADA Compliance: Building is ADA compliant with accessible entrances and facilities
10. Fire Protection: Has active fire protection system (sprinklers)
11. Security: Has building security systems or personnel
12. Nearby Dining: Has restaurants or food options within 2 blocks (approximately 500-600 feet)
13. Green Certification: Has ENERGY STAR or LEED certification (any level)

For each requirement, provide verification with supporting evidence and reference URL(s) from reliable sources (building websites, official listings, government records, or reputable news sources). Your answer should clearly state whether the Duke Energy Center meets each requirement and provide the supporting documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BuildingIdentity(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)


class GenericRequirement(BaseModel):
    status: Optional[str] = None  # Expect "meets" or "does not meet" (synonyms OK)
    details: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)


class MultiStoryInfo(BaseModel):
    status: Optional[str] = None
    floors_count_or_note: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)


class TransitInfo(BaseModel):
    status: Optional[str] = None
    stop_name: Optional[str] = None
    stop_type: Optional[str] = None  # "bus" or "light rail" (or synonym)
    distance: Optional[str] = None   # e.g., "0.2 miles", "800 ft", "2 blocks"
    evidence_urls: List[str] = Field(default_factory=list)


class DiningInfo(BaseModel):
    status: Optional[str] = None
    place_name: Optional[str] = None
    distance: Optional[str] = None   # e.g., "450 ft", "1 block"
    evidence_urls: List[str] = Field(default_factory=list)


class GreenInfo(BaseModel):
    status: Optional[str] = None
    certification_type: Optional[str] = None  # "LEED" or "ENERGY STAR"
    certification_level_or_score: Optional[str] = None  # e.g., "Platinum", "Gold", ENERGY STAR Score
    evidence_urls: List[str] = Field(default_factory=list)


class DukeEvalExtraction(BaseModel):
    identity: Optional[BuildingIdentity] = None

    class_a: Optional[GenericRequirement] = None
    city_limits: Optional[GenericRequirement] = None
    multi_story_building: Optional[MultiStoryInfo] = None
    elevator_access: Optional[GenericRequirement] = None
    parking_availability: Optional[GenericRequirement] = None
    public_transportation_proximity: Optional[TransitInfo] = None
    conference_facilities: Optional[GenericRequirement] = None
    internet_infrastructure: Optional[GenericRequirement] = None
    ada_compliance: Optional[GenericRequirement] = None
    fire_protection_system: Optional[GenericRequirement] = None
    security_features: Optional[GenericRequirement] = None
    nearby_dining_options: Optional[DiningInfo] = None
    green_certification: Optional[GreenInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_duke_requirements() -> str:
    return """
    Extract the requested structured information from the answer about Duke Energy Center (550 South Tryon Street, Charlotte, NC 28202).
    Follow these rules:
    - For each requirement, extract `status` exactly as either "meets" or "does not meet" if the answer clearly states it. If the answer uses synonyms (e.g., "yes", "no", "compliant", "not compliant", "available", "not available"), normalize to "meets" or "does not meet".
    - Extract all evidence URLs that the answer explicitly cites for each requirement; return an empty array if none were provided.
    - Do not invent URLs. Only include URLs present in the answer.

    Fields to extract:

    identity:
      - name: building name as stated in answer
      - address: full address as stated in answer
      - identity_urls: list of URL(s) that support the identity/address

    class_a:
      - status: "meets" or "does not meet"
      - details: optional short note (e.g., "Class A office tower")
      - evidence_urls: URL(s) supporting Class A designation

    city_limits:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "Charlotte, NC 28202")
      - evidence_urls: URL(s) supporting in-city location

    multi_story_building:
      - status: "meets" or "does not meet"
      - floors_count_or_note: e.g., "48 stories", "skyscraper", ">=3 floors"
      - evidence_urls: URL(s) stating floors/stories

    elevator_access:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "elevators present")
      - evidence_urls: URL(s) supporting elevator availability

    parking_availability:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "on-site garage")
      - evidence_urls: URL(s) supporting on-site parking

    public_transportation_proximity:
      - status: "meets" or "does not meet"
      - stop_name: name of a specific stop/station if provided
      - stop_type: "bus" or "light rail" (or synonym)
      - distance: distance or proximity statement as provided (e.g., "0.2 mi", "800 ft", "2 blocks")
      - evidence_urls: URL(s) supporting the stop and distance

    conference_facilities:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "conference rooms")
      - evidence_urls: URL(s) supporting conference/meeting space availability

    internet_infrastructure:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "fiber/high-speed")
      - evidence_urls: URL(s) supporting modern telecom/high-speed Internet

    ada_compliance:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "ADA-compliant entrances")
      - evidence_urls: URL(s) supporting ADA accessibility

    fire_protection_system:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "sprinklered")
      - evidence_urls: URL(s) supporting sprinklers/fire protection

    security_features:
      - status: "meets" or "does not meet"
      - details: optional note (e.g., "24/7 security")
      - evidence_urls: URL(s) supporting security presence

    nearby_dining_options:
      - status: "meets" or "does not meet"
      - place_name: a specific nearby restaurant/food option if provided
      - distance: proximity (e.g., "450 ft", "1 block", "500-600 ft")
      - evidence_urls: URL(s) supporting the dining option and distance

    green_certification:
      - status: "meets" or "does not meet"
      - certification_type: "LEED" or "ENERGY STAR" (if specified)
      - certification_level_or_score: e.g., "Platinum", "Gold", or ENERGY STAR score if provided
      - evidence_urls: URL(s) confirming the certification

    If any field is missing in the answer, set it to null or an empty list appropriately.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic cleanup: strip whitespace, dedupe while keeping order
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _status_to_claim(status: Optional[str], positive: str, negative: str) -> str:
    """
    Create a claim string corresponding to the extracted status.
    If status is None or unrecognized, default to the positive claim text.
    """
    if not status:
        return positive
    s = status.strip().lower()
    if s in {"meets", "meet", "yes", "available", "compliant", "true"}:
        return positive
    if s in {"does not meet", "doesn't meet", "no", "not available", "non-compliant", "false", "doesnt meet"}:
        return negative
    # Fallback to positive
    return positive


async def _add_urls_present_check(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    urls: List[str],
    desc: str
):
    """
    Add a critical custom node that asserts at least one evidence URL was provided.
    """
    evaluator.add_custom_node(
        result=len(_safe_urls(urls)) > 0,
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification builders per rubric section                                    #
# --------------------------------------------------------------------------- #
async def verify_building_identification(evaluator: Evaluator, root, data: DukeEvalExtraction):
    # Parent node: sequential, critical
    parent = evaluator.add_sequential(
        id="building_identification",
        desc="Correctly identify the target building (name and address match the prompt)",
        parent=root,
        critical=True
    )

    # 1) identity_statement (leaf)
    leaf_stmt = evaluator.add_leaf(
        id="identity_statement",
        desc="Explicitly state the building name and full address as Duke Energy Center, 550 South Tryon Street, Charlotte, NC 28202",
        parent=parent,
        critical=True
    )
    claim_stmt = "The answer explicitly states the building as 'Duke Energy Center' and provides the full address '550 South Tryon Street, Charlotte, NC 28202'."
    await evaluator.verify(
        claim=claim_stmt,
        node=leaf_stmt,
        additional_instruction="Check the answer text for the exact building name and full postal address as specified. Allow minor formatting differences (e.g., abbreviations like 'St.' for 'Street')."
    )

    # 2) identity evidence URL presence (custom check to enforce provenance)
    identity_urls = _safe_urls(data.identity.identity_urls if data.identity else [])
    await _add_urls_present_check(
        evaluator,
        parent_node=parent,
        node_id="identity_urls_provided",
        urls=identity_urls,
        desc="At least one identity/address evidence URL is provided"
    )

    # 3) identity_evidence_url (leaf) – verify page supports the identity/address
    leaf_evid = evaluator.add_leaf(
        id="identity_evidence_url",
        desc="Provide at least one evidence URL that supports the building identity/address (reliable source)",
        parent=parent,
        critical=True
    )
    claim_evid = "The referenced webpage(s) show the building named 'Duke Energy Center' at '550 South Tryon Street, Charlotte, NC 28202'."
    await evaluator.verify(
        claim=claim_evid,
        node=leaf_evid,
        sources=identity_urls,
        additional_instruction="Prefer authoritative sources (owner/property website, official listings, government records). The page should explicitly show the building name and full address."
    )


async def verify_generic_requirement(
    evaluator: Evaluator,
    root,
    node_id: str,
    node_desc: str,
    meet_status_leaf_id: str,
    meet_status_leaf_desc: str,
    evidence_leaf_id: str,
    evidence_leaf_desc: str,
    status: Optional[str],
    urls: List[str],
    positive_claim: str,
    negative_claim: str,
    url_presence_check_id: str,
    url_presence_desc: str,
    additional_instruction: str
):
    # Parent sequential node
    parent = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=root,
        critical=True
    )

    # Meet status leaf: Check the answer clearly states meets/does not meet
    leaf_status = evaluator.add_leaf(
        id=meet_status_leaf_id,
        desc=meet_status_leaf_desc,
        parent=parent,
        critical=True
    )
    claim_status = f"The answer clearly states whether the building meets or does not meet this requirement: {node_desc}."
    await evaluator.verify(
        claim=claim_status,
        node=leaf_status,
        additional_instruction="Accept clear synonyms (e.g., 'yes/no', 'compliant/not compliant', 'available/not available'). The answer must contain an explicit determination."
    )

    # Evidence URL presence (custom)
    await _add_urls_present_check(
        evaluator,
        parent_node=parent,
        node_id=url_presence_check_id,
        urls=urls,
        desc=url_presence_desc
    )

    # Evidence URL verification
    leaf_evid = evaluator.add_leaf(
        id=evidence_leaf_id,
        desc=evidence_leaf_desc,
        parent=parent,
        critical=True
    )
    claim = _status_to_claim(status, positive_claim, negative_claim)
    await evaluator.verify(
        claim=claim,
        node=leaf_evid,
        sources=urls,
        additional_instruction=additional_instruction
    )


async def verify_multi_story(evaluator: Evaluator, root, info: Optional[MultiStoryInfo]):
    urls = _safe_urls(info.evidence_urls if info else [])
    floors_note = (info.floors_count_or_note if info and info.floors_count_or_note else "").strip()

    positive = "The building is a multi-story office building with at least 3 floors."
    if floors_note:
        positive = f"The building is a multi-story office building and has at least 3 floors (e.g., '{floors_note}')."
    negative = "The building does not have at least 3 floors."

    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="multi_story_building",
        node_desc="Requirement 3: Multi-story building with at least 3 floors",
        meet_status_leaf_id="floors_meet_status",
        meet_status_leaf_desc="Clearly state whether the building has at least 3 floors (meets/does not meet)",
        evidence_leaf_id="floors_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL stating the number of floors/stories (reliable source)",
        status=info.status if info else None,
        urls=urls,
        positive_claim=positive,
        negative_claim=negative,
        url_presence_check_id="floors_urls_provided",
        url_presence_desc="At least one URL is provided that states the number of floors/stories",
        additional_instruction="Verify the page mentions total floors/stories or clearly indicates a multi-story tower (≥3 floors). Prefer authoritative sources."
    )


async def verify_transit(evaluator: Evaluator, root, info: Optional[TransitInfo]):
    # Parent sequential
    parent = evaluator.add_sequential(
        id="public_transportation_proximity",
        desc="Requirement 6: Within 0.5 miles (~2,600 feet) of a public transportation stop (bus or light rail)",
        parent=root,
        critical=True
    )

    # 6.1 Meet status
    leaf_status = evaluator.add_leaf(
        id="transit_meet_status",
        desc="Clearly state whether the public transportation proximity requirement is met (meets/does not meet)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly states whether the building meets or does not meet the public transportation proximity requirement (within 0.5 miles).",
        node=leaf_status,
        additional_instruction="Accept synonyms and clear phrasing that indicates compliance or non-compliance."
    )

    # 6.2 Details (parallel)
    details_node = evaluator.add_parallel(
        id="transit_details",
        desc="Provide a specific nearby transit stop and verify it is within 0.5 miles",
        parent=parent,
        critical=True
    )

    # 6.2.a Transit stop identification (leaf)
    stop_name = (info.stop_name if info and info.stop_name else "").strip()
    stop_type = (info.stop_type if info and info.stop_type else "").strip()
    leaf_stop = evaluator.add_leaf(
        id="transit_stop_identification",
        desc="Name at least one specific qualifying transit stop/station and its type (bus or light rail)",
        parent=details_node,
        critical=True
    )
    # If no stop_name present, this check should fail. We'll craft claim that uses extracted fields.
    if stop_name:
        claim_stop = f"The answer identifies a nearby {stop_type or 'transit'} stop named '{stop_name}'."
    else:
        claim_stop = "The answer identifies a specific nearby transit stop/station by name."
    await evaluator.verify(
        claim=claim_stop,
        node=leaf_stop,
        additional_instruction="Look for an explicit stop or station name and its type (bus/light rail)."
    )

    # URLs presence
    urls = _safe_urls(info.evidence_urls if info else [])
    await _add_urls_present_check(
        evaluator,
        parent_node=details_node,
        node_id="transit_urls_provided",
        urls=urls,
        desc="At least one transit evidence URL is provided"
    )

    # 6.2.b Distance verification (leaf) – with sources
    distance_text = (info.distance if info and info.distance else "").strip()
    leaf_dist = evaluator.add_leaf(
        id="transit_distance_verification",
        desc="Provide a distance value showing the stop is <= 0.5 miles from the building (with method/source implied by citation)",
        parent=details_node,
        critical=True
    )
    if distance_text:
        claim_dist = f"The distance from Duke Energy Center to {stop_name or 'the named transit stop'} is '{distance_text}' and is less than or equal to 0.5 miles (~2,600 feet)."
    else:
        claim_dist = "The cited evidence indicates the named transit stop is within 0.5 miles (~2,600 feet) of Duke Energy Center."
    await evaluator.verify(
        claim=claim_dist,
        node=leaf_dist,
        sources=urls,
        additional_instruction="Confirm the measured or stated distance on the provided page(s). Accept maps/walking distance outputs. ≤ 0.5 miles qualifies."
    )

    # 6.2.c Evidence URL verification (leaf) – stop/location support
    leaf_evid = evaluator.add_leaf(
        id="transit_evidence_url",
        desc="Provide at least one evidence URL supporting the stop location and/or distance (reliable source)",
        parent=details_node,
        critical=True
    )
    claim_evid = f"The referenced page(s) confirm the existence and location of the {stop_type or 'transit'} stop '{stop_name or 'the named stop'}' near Duke Energy Center."
    await evaluator.verify(
        claim=claim_evid,
        node=leaf_evid,
        sources=urls,
        additional_instruction="Prefer official transit operator pages or reliable mapping outputs. The page should indicate the stop location near the building."
    )


async def verify_dining(evaluator: Evaluator, root, info: Optional[DiningInfo]):
    # Parent sequential
    parent = evaluator.add_sequential(
        id="nearby_dining_options",
        desc="Requirement 12: Restaurants/food options within 2 blocks (~500–600 feet)",
        parent=root,
        critical=True
    )

    # 12.1 Meet status
    leaf_status = evaluator.add_leaf(
        id="dining_meet_status",
        desc="Clearly state whether the nearby dining requirement is met (meets/does not meet)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly states whether the building meets or does not meet the nearby dining requirement (within ~500–600 feet or within 2 blocks).",
        node=leaf_status,
        additional_instruction="Accept synonyms indicating proximity compliance."
    )

    # 12.2 Details (parallel)
    details_node = evaluator.add_parallel(
        id="dining_details",
        desc="Identify at least one nearby dining option and verify proximity within ~500–600 feet / 2 blocks",
        parent=parent,
        critical=True
    )

    # Identification
    place_name = (info.place_name if info and info.place_name else "").strip()
    leaf_ident = evaluator.add_leaf(
        id="dining_identification",
        desc="Name at least one specific restaurant/food option",
        parent=details_node,
        critical=True
    )
    if place_name:
        claim_ident = f"The answer identifies a nearby dining option named '{place_name}'."
    else:
        claim_ident = "The answer identifies at least one specific nearby restaurant/food option by name."
    await evaluator.verify(
        claim=claim_ident,
        node=leaf_ident,
        additional_instruction="Look for a named restaurant/food option explicitly listed."
    )

    # URLs presence
    urls = _safe_urls(info.evidence_urls if info else [])
    await _add_urls_present_check(
        evaluator,
        parent_node=details_node,
        node_id="dining_urls_provided",
        urls=urls,
        desc="At least one dining evidence URL is provided"
    )

    # Distance verification
    distance_text = (info.distance if info and info.distance else "").strip()
    leaf_dist = evaluator.add_leaf(
        id="dining_distance_verification",
        desc="Provide a distance/proximity value indicating the option is within ~500–600 feet (or explicitly within 2 blocks) of the building",
        parent=details_node,
        critical=True
    )
    if distance_text:
        claim_dist = f"The cited proximity for '{place_name or 'the named dining option'}' is '{distance_text}' and is within ~500–600 feet (or explicitly within 2 blocks) of Duke Energy Center."
    else:
        claim_dist = "The cited evidence shows the named dining option is within ~500–600 feet or explicitly within 2 blocks of Duke Energy Center."
    await evaluator.verify(
        claim=claim_dist,
        node=leaf_dist,
        sources=urls,
        additional_instruction="Confirm the distance or proximity on the provided page(s). Accept reliable maps or site listings that show an explicit distance or block count."
    )

    # Evidence URL verification
    leaf_evid = evaluator.add_leaf(
        id="dining_evidence_url",
        desc="Provide at least one evidence URL supporting the dining option and/or proximity (reliable source)",
        parent=details_node,
        critical=True
    )
    claim_evid = f"The referenced page(s) confirm the location of '{place_name or 'the named dining option'}' and its proximity to Duke Energy Center."
    await evaluator.verify(
        claim=claim_evid,
        node=leaf_evid,
        sources=urls,
        additional_instruction="Pages may include business listings, official websites, or reliable maps that show the restaurant's location relative to the building."
    )


async def verify_green(evaluator: Evaluator, root, info: Optional[GreenInfo]):
    # Parent sequential
    parent = evaluator.add_sequential(
        id="green_certification",
        desc="Requirement 13: Has ENERGY STAR or LEED certification (any level)",
        parent=root,
        critical=True
    )

    # Meet status
    leaf_status = evaluator.add_leaf(
        id="green_meet_status",
        desc="Clearly state whether green certification requirement is met (meets/does not meet)",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly states whether the building meets or does not meet the green certification requirement (ENERGY STAR or LEED).",
        node=leaf_status,
        additional_instruction="Accept explicit determination using clear synonyms."
    )

    # Details (parallel)
    details_node = evaluator.add_parallel(
        id="green_details",
        desc="Specify a qualifying certification and support it with evidence",
        parent=parent,
        critical=True
    )

    # Certification identification
    cert_type = (info.certification_type if info and info.certification_type else "").strip()
    cert_level = (info.certification_level_or_score if info and info.certification_level_or_score else "").strip()

    leaf_cert = evaluator.add_leaf(
        id="certification_type_and_level",
        desc="Identify certification type (ENERGY STAR or LEED) and level/score if applicable",
        parent=details_node,
        critical=True
    )
    if cert_type and cert_level:
        claim_cert = f"The answer identifies a qualifying certification for the building: {cert_type} ({cert_level})."
    elif cert_type:
        claim_cert = f"The answer identifies a qualifying certification for the building: {cert_type}."
    else:
        claim_cert = "The answer identifies at least one qualifying certification (ENERGY STAR or LEED) for the building."
    await evaluator.verify(
        claim=claim_cert,
        node=leaf_cert,
        additional_instruction="Check the answer for explicit mention of a LEED level (e.g., Platinum/Gold) or ENERGY STAR certification/score when available."
    )

    # URLs presence
    urls = _safe_urls(info.evidence_urls if info else [])
    await _add_urls_present_check(
        evaluator,
        parent_node=details_node,
        node_id="green_urls_provided",
        urls=urls,
        desc="At least one certification evidence URL is provided"
    )

    # Evidence URL verification
    leaf_evid = evaluator.add_leaf(
        id="green_evidence_url",
        desc="Provide at least one evidence URL confirming the certification (reliable source)",
        parent=details_node,
        critical=True
    )
    if cert_type and cert_level:
        pos_claim = f"The referenced page(s) confirm that the building has {cert_type} certification ({cert_level})."
    elif cert_type:
        pos_claim = f"The referenced page(s) confirm that the building has {cert_type} certification."
    else:
        pos_claim = "The referenced page(s) confirm that the building has an ENERGY STAR or LEED certification."
    neg_claim = "The referenced page(s) indicate the building does not have ENERGY STAR or LEED certification."

    status = info.status if info else None
    claim = _status_to_claim(status, pos_claim, neg_claim)

    await evaluator.verify(
        claim=claim,
        node=leaf_evid,
        sources=urls,
        additional_instruction="Prefer authoritative sources: USGBC/LEED directory, ENERGY STAR registry, property owner site, or reputable media referencing the certification."
    )


# --------------------------------------------------------------------------- #
# Requirement-specific wrappers using the generic verifier                    #
# --------------------------------------------------------------------------- #
async def verify_class_a(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="class_a_designation",
        node_desc="Requirement 1: Confirmed Class A office building designation",
        meet_status_leaf_id="class_a_meet_status",
        meet_status_leaf_desc="Clearly state whether the building meets the Class A designation requirement (meets/does not meet)",
        evidence_leaf_id="class_a_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL that supports the Class A designation claim (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building is designated as a Class A (or higher, e.g., Trophy/Class AA) office building.",
        negative_claim="The building is not designated as a Class A office building.",
        url_presence_check_id="class_a_urls_provided",
        url_presence_desc="At least one Class A designation evidence URL is provided",
        additional_instruction="Verify that the page explicitly describes the building as 'Class A' (or equivalent top-tier classification such as Trophy/Class AA). Prefer authoritative/owner sources."
    )


async def verify_city_limits(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="charlotte_city_limits",
        node_desc="Requirement 2: Located within Charlotte city limits (not suburbs)",
        meet_status_leaf_id="city_limits_meet_status",
        meet_status_leaf_desc="Clearly state whether the building is within Charlotte city limits (meets/does not meet)",
        evidence_leaf_id="city_limits_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting that the address is within Charlotte city limits (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The address '550 South Tryon Street, Charlotte, NC 28202' lies within the municipal boundaries of the City of Charlotte.",
        negative_claim="The address '550 South Tryon Street, Charlotte, NC 28202' is not within the City of Charlotte municipal limits.",
        url_presence_check_id="city_limits_urls_provided",
        url_presence_desc="At least one city-limits evidence URL is provided",
        additional_instruction="Prefer government or official boundary sources, or authoritative address/property records indicating the address belongs to the City of Charlotte."
    )


async def verify_elevator(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="elevator_access",
        node_desc="Requirement 4: Has elevator systems for vertical transportation",
        meet_status_leaf_id="elevators_meet_status",
        meet_status_leaf_desc="Clearly state whether the building has elevators (meets/does not meet)",
        evidence_leaf_id="elevators_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting elevator availability (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building has elevator systems (lifts) for vertical transportation.",
        negative_claim="The building does not have elevator systems for vertical transportation.",
        url_presence_check_id="elevators_urls_provided",
        url_presence_desc="At least one elevator evidence URL is provided",
        additional_instruction="Look for amenities, building fact sheets, or specifications mentioning elevators."
    )


async def verify_parking(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="parking_availability",
        node_desc="Requirement 5: Provides on-site parking facilities",
        meet_status_leaf_id="parking_meet_status",
        meet_status_leaf_desc="Clearly state whether on-site parking is provided (meets/does not meet)",
        evidence_leaf_id="parking_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting on-site parking availability (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building provides on-site parking facilities (e.g., an on-site garage/lot).",
        negative_claim="The building does not provide on-site parking facilities.",
        url_presence_check_id="parking_urls_provided",
        url_presence_desc="At least one parking evidence URL is provided",
        additional_instruction="Prefer owner/manager/property brochure or authoritative listings that explicitly mention on-site parking/garage."
    )


async def verify_conference(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="conference_facilities",
        node_desc="Requirement 7: Has conference room or meeting space facilities available",
        meet_status_leaf_id="conference_meet_status",
        meet_status_leaf_desc="Clearly state whether conference/meeting facilities are available (meets/does not meet)",
        evidence_leaf_id="conference_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting conference/meeting facilities availability (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building has conference rooms or meeting space facilities available.",
        negative_claim="The building does not have conference rooms or meeting space facilities available.",
        url_presence_check_id="conference_urls_provided",
        url_presence_desc="At least one conference/meeting evidence URL is provided",
        additional_instruction="Look for amenities, floor plans, or services listing conference/meeting rooms."
    )


async def verify_internet(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="internet_infrastructure",
        node_desc="Requirement 8: Has modern telecommunications/high-speed internet infrastructure",
        meet_status_leaf_id="internet_meet_status",
        meet_status_leaf_desc="Clearly state whether modern telecom/high-speed internet infrastructure is available (meets/does not meet)",
        evidence_leaf_id="internet_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting telecom/internet infrastructure (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building has modern telecommunications/high-speed Internet infrastructure (e.g., fiber or equivalent high-speed service).",
        negative_claim="The building does not have modern telecommunications/high-speed Internet infrastructure.",
        url_presence_check_id="internet_urls_provided",
        url_presence_desc="At least one telecom/internet evidence URL is provided",
        additional_instruction="Look for telecom/IT amenities in official/authoritative property descriptions."
    )


async def verify_ada(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="ada_compliance",
        node_desc="Requirement 9: ADA compliant with accessible entrances and facilities",
        meet_status_leaf_id="ada_meet_status",
        meet_status_leaf_desc="Clearly state whether ADA compliance/accessibility requirement is met (meets/does not meet)",
        evidence_leaf_id="ada_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting ADA compliance or specific accessibility features (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building is ADA compliant, with accessible entrances and facilities.",
        negative_claim="The building is not ADA compliant with accessible entrances and facilities.",
        url_presence_check_id="ada_urls_provided",
        url_presence_desc="At least one ADA/accessibility evidence URL is provided",
        additional_instruction="Prefer authoritative building documentation or accessibility statements indicating ADA-compliant features."
    )


async def verify_fire(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="fire_protection_system",
        node_desc="Requirement 10: Has active fire protection system (sprinklers)",
        meet_status_leaf_id="sprinklers_meet_status",
        meet_status_leaf_desc="Clearly state whether sprinklers/fire protection system requirement is met (meets/does not meet)",
        evidence_leaf_id="sprinklers_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting sprinkler/fire protection presence (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building has an active fire protection system, including sprinklers.",
        negative_claim="The building does not have an active fire protection system with sprinklers.",
        url_presence_check_id="sprinklers_urls_provided",
        url_presence_desc="At least one fire protection/sprinklers evidence URL is provided",
        additional_instruction="Look for building specs, life-safety summaries, or authoritative listings mentioning sprinklers/fire protection."
    )


async def verify_security(evaluator: Evaluator, root, req: Optional[GenericRequirement]):
    urls = _safe_urls(req.evidence_urls if req else [])
    await verify_generic_requirement(
        evaluator=evaluator,
        root=root,
        node_id="security_features",
        node_desc="Requirement 11: Has building security systems or personnel",
        meet_status_leaf_id="security_meet_status",
        meet_status_leaf_desc="Clearly state whether security systems/personnel are present (meets/does not meet)",
        evidence_leaf_id="security_evidence_url",
        evidence_leaf_desc="Provide at least one evidence URL supporting the security claim (reliable source)",
        status=req.status if req else None,
        urls=urls,
        positive_claim="The building has security systems and/or on-site security personnel.",
        negative_claim="The building does not have building security systems or personnel.",
        url_presence_check_id="security_urls_provided",
        url_presence_desc="At least one security evidence URL is provided",
        additional_instruction="Look for mentions of security staff, 24/7 security, access control, or surveillance in authoritative property documentation."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for Duke Energy Center suitability requirements.
    """
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

    # Extract structured info from the answer (single pass)
    extracted: DukeEvalExtraction = await evaluator.extract(
        prompt=prompt_extract_duke_requirements(),
        template_class=DukeEvalExtraction,
        extraction_name="duke_energy_center_extraction"
    )

    # Add ground truth context (expected building ID elements)
    evaluator.add_ground_truth({
        "expected_building_name": "Duke Energy Center",
        "expected_address": "550 South Tryon Street, Charlotte, NC 28202"
    }, gt_type="expected_identity")

    # Root is a parallel aggregator; all children are critical as per rubric.

    # Building identification
    await verify_building_identification(evaluator, root, extracted)

    # Requirement 1: Class A
    await verify_class_a(evaluator, root, extracted.class_a)

    # Requirement 2: City limits
    await verify_city_limits(evaluator, root, extracted.city_limits)

    # Requirement 3: Multi-story ≥3 floors
    await verify_multi_story(evaluator, root, extracted.multi_story_building)

    # Requirement 4: Elevators
    await verify_elevator(evaluator, root, extracted.elevator_access)

    # Requirement 5: Parking
    await verify_parking(evaluator, root, extracted.parking_availability)

    # Requirement 6: Public transportation proximity
    await verify_transit(evaluator, root, extracted.public_transportation_proximity)

    # Requirement 7: Conference facilities
    await verify_conference(evaluator, root, extracted.conference_facilities)

    # Requirement 8: Internet infrastructure
    await verify_internet(evaluator, root, extracted.internet_infrastructure)

    # Requirement 9: ADA compliance
    await verify_ada(evaluator, root, extracted.ada_compliance)

    # Requirement 10: Fire protection
    await verify_fire(evaluator, root, extracted.fire_protection_system)

    # Requirement 11: Security
    await verify_security(evaluator, root, extracted.security_features)

    # Requirement 12: Nearby dining
    await verify_dining(evaluator, root, extracted.nearby_dining_options)

    # Requirement 13: Green certification
    await verify_green(evaluator, root, extracted.green_certification)

    # Return structured evaluation summary
    return evaluator.get_summary()