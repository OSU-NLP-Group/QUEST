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
TASK_ID = "wildlife_referral_fc"
TASK_DESCRIPTION = """
A veterinary clinic in Fort Collins, Colorado needs to create a wildlife rehabilitation referral guide for their staff. The guide must cover five common scenarios: (1) injured songbirds or waterfowl, (2) injured raptors (birds of prey), (3) injured small mammals, (4) injured reptiles or amphibians, and (5) mountain lion sightings or incidents. For each scenario, identify the correct wildlife rehabilitation facility or reporting protocol, provide the facility name, contact phone number, physical address (when applicable), and briefly note their primary species specialization or service.
"""

# Ground-truth anchors used for verification claims
GREENWOOD = {
    "name": "Greenwood Wildlife Rehabilitation Center",
    "aliases": ["Greenwood Wildlife Center", "Greenwood Wildlife Rehab Center", "Greenwood Wildlife"],
    "phone": "(303) 823-8455",
    "domain": "greenwoodwildlife.org",
    "location_note": "Lyons, Colorado (Boulder County)",
    "species_note": "treats wild birds (including songbirds and waterfowl) and mammals",
}

RMRP = {
    "name": "Rocky Mountain Raptor Program",
    "aliases": ["RMRP", "Rocky Mountain Raptor Programme"],  # include a reasonable variant
    "after_hours": "(970) 222-0322",
    "domain": "rmrp.org",
    "species_note": "specializes in rehabilitation of birds of prey (raptors)",
}

NCWC = {
    "name": "Northern Colorado Wildlife Center",
    "aliases": ["NCWC", "Northern CO Wildlife Center"],
    "phone": "(970) 283-7822",
    "domain": "nocowildlife.org",
    "address": "2637 Midpoint Dr, Suite E, Fort Collins, CO 80525",
    "species_note": "provides rehabilitation for reptiles, amphibians, and small mammals",
}

CPW = {
    "name": "Colorado Parks and Wildlife",
    "aliases": ["CPW"],
    "domain": "cpw.state.co.us",
    "fort_collins_phone": "(970) 472-4300",
    "reporting_note": "mountain lion incidents must be reported immediately to Colorado Parks and Wildlife",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ScenarioEntry(BaseModel):
    facility_name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    specialization: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ReferralGuideExtraction(BaseModel):
    songbird_waterfowl: Optional[ScenarioEntry] = None
    raptor: Optional[ScenarioEntry] = None
    small_mammal: Optional[ScenarioEntry] = None
    reptile_amphibian: Optional[ScenarioEntry] = None
    mountain_lion: Optional[ScenarioEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_referral_guide() -> str:
    return """
    Extract the referral guide information provided in the answer for the five scenarios below.
    For each scenario, extract the following fields from the answer exactly as written:
    - facility_name: The facility/agency name (use the exact wording provided)
    - phone: The contact phone number (if provided; keep the original formatting)
    - address: The physical address or location note (if provided; keep the original formatting)
    - specialization: A brief note about species specialization or service (if provided; short phrase from the answer)
    - source_urls: A list of all URLs the answer cites for this scenario (only URLs explicitly present in the answer)

    Scenarios (use these keys):
    - songbird_waterfowl: injured songbirds or waterfowl
    - raptor: injured raptors (birds of prey)
    - small_mammal: injured small mammals
    - reptile_amphibian: injured reptiles or amphibians
    - mountain_lion: mountain lion sightings or incidents (reporting protocol/agency)

    Rules:
    - If any field is missing in the answer for a scenario, set it to null (for strings) or [] for source_urls.
    - Only include URLs that appear in the answer text (plain or markdown links). Do not invent any URLs.
    - Do not normalize or reformat values; return what appears in the answer (except ensure URLs include a protocol; if missing, prepend http://).

    Return a JSON object with keys: songbird_waterfowl, raptor, small_mammal, reptile_amphibian, mountain_lion.
    Each key should contain an object with: facility_name, phone, address, specialization, source_urls.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(xs: Optional[List[str]]) -> List[str]:
    return xs or []


def _domain_in_sources(sources: List[str], allowed_domains: List[str]) -> bool:
    import re
    domains = []
    for u in sources:
        try:
            # Simple domain extraction
            from urllib.parse import urlparse
            netloc = urlparse(u).netloc.lower()
            if netloc:
                domains.append(netloc)
        except Exception:
            # fallback regex
            m = re.match(r"^(?:https?://)?([^/]+)/?", u.strip(), re.I)
            if m:
                domains.append(m.group(1).lower())
    for d in domains:
        for allowed in allowed_domains:
            if allowed.lower() in d:
                return True
    return False


def _guess_selected_small_mammal_facility(entry: ScenarioEntry) -> Optional[str]:
    """
    Heuristic guess which facility the answer chose for the small mammal scenario.
    Returns "NCWC", "GREENWOOD", or None.
    """
    name = (entry.facility_name or "").lower()
    sources = _safe_list(entry.source_urls)

    if _domain_in_sources(sources, [NCWC["domain"]]):
        return "NCWC"
    if _domain_in_sources(sources, [GREENWOOD["domain"]]):
        return "GREENWOOD"

    if "northern colorado wildlife" in name or "ncwc" in name:
        return "NCWC"
    if "greenwood" in name:
        return "GREENWOOD"
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_songbird_waterfowl_nodes(evaluator: Evaluator, root, data: Optional[ScenarioEntry]) -> None:
    scen = evaluator.add_sequential(
        id="Songbird_Waterfowl_Scenario",
        desc="Complete and accurate referral information for injured songbirds or waterfowl",
        parent=root,
        critical=False,
    )

    greenwood = evaluator.add_parallel(
        id="Greenwood_Referral",
        desc="Identifies Greenwood Wildlife Rehabilitation Center with complete information",
        parent=scen,
        critical=True,  # Critical parent => all children must be critical in this framework
    )

    # Facility Name check (simple equivalence)
    name_node = evaluator.add_leaf(
        id="Greenwood_Facility_Name",
        desc="Correctly names the facility as Greenwood Wildlife Rehabilitation Center or Greenwood Wildlife Center",
        parent=greenwood,
        critical=True,
    )
    claimed_name = data.facility_name if data else ""
    await evaluator.verify(
        claim=f'The identified facility name "{claimed_name}" refers to Greenwood Wildlife Rehabilitation Center (also known as "Greenwood Wildlife Center"). Consider minor variants equivalent.',
        node=name_node,
        additional_instruction="Allow reasonable name variants and abbreviations; focus on whether the name clearly refers to Greenwood Wildlife Rehabilitation Center."
    )

    # URL reference from greenwoodwildlife.org (custom domain presence)
    name_url_node = evaluator.add_custom_node(
        result=data is not None and _domain_in_sources(_safe_list(data.source_urls), [GREENWOOD["domain"]]),
        id="Greenwood_Name_URL",
        desc="Provides URL reference from greenwoodwildlife.org domain",
        parent=greenwood,
        critical=True,
    )

    # Phone number check (verify by URLs)
    phone_node = evaluator.add_leaf(
        id="Greenwood_Phone_Number",
        desc="Provides the correct phone number (303) 823-8455",
        parent=greenwood,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The phone number for Greenwood Wildlife Rehabilitation Center is {GREENWOOD["phone"]}.',
        node=phone_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept common formatting variants like 303-823-8455 or (303) 823-8455 or 303 823 8455."
    )

    # Explicit phone URL presence (any URL is acceptable for verification path)
    phone_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="Greenwood_Phone_URL",
        desc="Provides URL reference for phone number verification",
        parent=greenwood,
        critical=True,
    )

    # Location check
    location_node = evaluator.add_leaf(
        id="Greenwood_Location",
        desc="Provides physical location information (Lyons, Colorado or Boulder County area)",
        parent=greenwood,
        critical=True,
    )
    await evaluator.verify(
        claim=f'{GREENWOOD["name"]} is located in Lyons, Colorado (Boulder County).',
        node=location_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept variants like 'Lyons, CO' or text clearly indicating the facility is in Lyons in Boulder County."
    )

    # Location URL presence
    location_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="Greenwood_Location_URL",
        desc="Provides URL reference for location verification",
        parent=greenwood,
        critical=True,
    )

    # Specialization check
    spec_node = evaluator.add_leaf(
        id="Greenwood_Specialization",
        desc="Notes that Greenwood treats wild birds, waterfowl, and mammals",
        parent=greenwood,
        critical=True,
    )
    await evaluator.verify(
        claim=f'{GREENWOOD["name"]} {GREENWOOD["species_note"]}.',
        node=spec_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Look for language indicating Greenwood accepts/treats birds (including songbirds and waterfowl) and mammals."
    )


async def build_raptor_nodes(evaluator: Evaluator, root, data: Optional[ScenarioEntry]) -> None:
    scen = evaluator.add_sequential(
        id="Raptor_Scenario",
        desc="Complete and accurate referral information for injured raptors or birds of prey",
        parent=root,
        critical=False,
    )

    rmrp = evaluator.add_parallel(
        id="RMRP_Referral",
        desc="Identifies Rocky Mountain Raptor Program with complete information",
        parent=scen,
        critical=True,
    )

    # Facility Name
    name_node = evaluator.add_leaf(
        id="RMRP_Facility_Name",
        desc="Correctly names the facility as Rocky Mountain Raptor Program or RMRP",
        parent=rmrp,
        critical=True,
    )
    claimed_name = data.facility_name if data else ""
    await evaluator.verify(
        claim=f'The identified facility name "{claimed_name}" refers to the Rocky Mountain Raptor Program (RMRP).',
        node=name_node,
        additional_instruction="Allow reasonable variants like 'RMRP'."
    )

    # Name URL presence (any is acceptable for this node)
    name_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="RMRP_Name_URL",
        desc="Provides URL reference for RMRP verification",
        parent=rmrp,
        critical=True,
    )

    # Phone (after-hours hotline)
    phone_node = evaluator.add_leaf(
        id="RMRP_Phone_Number",
        desc="Provides the after-hours hotline number (970) 222-0322",
        parent=rmrp,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The after-hours hotline for the Rocky Mountain Raptor Program is {RMRP["after_hours"]}.',
        node=phone_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Focus on after-hours/emergency hotline; accept formatting variants like 970-222-0322 or (970) 222-0322."
    )

    # Phone URL presence
    phone_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="RMRP_Phone_URL",
        desc="Provides URL reference for phone number verification",
        parent=rmrp,
        critical=True,
    )

    # Specialization
    spec_node = evaluator.add_leaf(
        id="RMRP_Specialization",
        desc="Notes that RMRP specializes in birds of prey or raptors",
        parent=rmrp,
        critical=True,
    )
    await evaluator.verify(
        claim=f'{RMRP["name"]} {RMRP["species_note"]}.',
        node=spec_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Look for text indicating focus on raptors/birds of prey."
    )


async def build_small_mammal_nodes(evaluator: Evaluator, root, data: Optional[ScenarioEntry]) -> None:
    scen = evaluator.add_sequential(
        id="Small_Mammal_Scenario",
        desc="Complete and accurate referral information for injured small mammals",
        parent=root,
        critical=False,
    )

    ref = evaluator.add_parallel(
        id="Mammal_Facility_Referral",
        desc="Identifies an appropriate facility that accepts small mammals with complete information",
        parent=scen,
        critical=True,
    )

    # Facility Name must be NCWC or Greenwood
    name_node = evaluator.add_leaf(
        id="Mammal_Facility_Name",
        desc="Names either Northern Colorado Wildlife Center or Greenwood Wildlife Center as both accept small mammals",
        parent=ref,
        critical=True,
    )
    claimed_name = data.facility_name if data else ""
    allowed_names = f'"{NCWC["name"]}" (aka "NCWC") or "{GREENWOOD["name"]}" (aka "Greenwood Wildlife Center")'
    await evaluator.verify(
        claim=f'The facility name "{claimed_name}" refers to either {allowed_names}.',
        node=name_node,
        additional_instruction="Accept reasonable variants and abbreviations for the two facilities."
    )

    # Facility URL presence with appropriate domain for the selected facility
    selected = _guess_selected_small_mammal_facility(data or ScenarioEntry())
    if selected == "NCWC":
        url_ok = data is not None and _domain_in_sources(_safe_list(data.source_urls), [NCWC["domain"]])
    elif selected == "GREENWOOD":
        url_ok = data is not None and _domain_in_sources(_safe_list(data.source_urls), [GREENWOOD["domain"]])
    else:
        # If unknown selection, require at least some URL to be present
        url_ok = data is not None and len(_safe_list(data.source_urls)) > 0

    url_node = evaluator.add_custom_node(
        result=url_ok,
        id="Mammal_Facility_URL",
        desc="Provides URL reference confirming the facility accepts small mammals",
        parent=ref,
        critical=True,
    )

    # Phone (depends on chosen facility)
    phone_node = evaluator.add_leaf(
        id="Mammal_Contact_Phone",
        desc="Provides correct phone number for the selected facility (970-283-7822 for NCWC or 303-823-8455 for Greenwood)",
        parent=ref,
        critical=True,
    )
    if selected == "NCWC":
        expected_phone = NCWC["phone"]
        facility_for_claim = NCWC["name"]
    else:
        # Default to Greenwood if ambiguous
        expected_phone = GREENWOOD["phone"]
        facility_for_claim = GREENWOOD["name"]

    await evaluator.verify(
        claim=f'The phone number for {facility_for_claim} is {expected_phone}.',
        node=phone_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept common formatting variants such as parentheses, dashes, or spaces."
    )

    # Phone URL presence
    phone_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="Mammal_Phone_URL",
        desc="Provides URL reference for phone number verification",
        parent=ref,
        critical=True,
    )

    # Location (depends on chosen facility)
    location_node = evaluator.add_leaf(
        id="Mammal_Facility_Location",
        desc="Provides appropriate location information for the selected facility",
        parent=ref,
        critical=True,
    )
    if selected == "NCWC":
        loc_claim = f'{NCWC["name"]} is located in Fort Collins, Colorado (address may be listed as {NCWC["address"]}).'
    else:
        loc_claim = f'{GREENWOOD["name"]} is located in Lyons, Colorado (Boulder County).'
    await evaluator.verify(
        claim=loc_claim,
        node=location_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Minor address formatting differences (Dr vs Drive, Suite vs Ste, punctuation) are acceptable."
    )

    # Specialization
    spec_node = evaluator.add_leaf(
        id="Mammal_Specialization",
        desc="Notes the species specialization of the selected facility",
        parent=ref,
        critical=True,
    )
    if selected == "NCWC":
        spec_claim = f'{NCWC["name"]} {NCWC["species_note"]}.'
    else:
        spec_claim = f'{GREENWOOD["name"]} {GREENWOOD["species_note"]}.'
    await evaluator.verify(
        claim=spec_claim,
        node=spec_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Confirm the facility accepts small mammals; for NCWC, also reptiles/amphibians; for Greenwood, birds and mammals."
    )


async def build_reptile_amphibian_nodes(evaluator: Evaluator, root, data: Optional[ScenarioEntry]) -> None:
    scen = evaluator.add_sequential(
        id="Reptile_Amphibian_Scenario",
        desc="Complete and accurate referral information for injured reptiles or amphibians",
        parent=root,
        critical=False,
    )

    ncwc = evaluator.add_parallel(
        id="NCWC_Reptile_Referral",
        desc="Identifies Northern Colorado Wildlife Center with complete information",
        parent=scen,
        critical=True,
    )

    # Facility Name
    name_node = evaluator.add_leaf(
        id="NCWC_Facility_Name",
        desc="Correctly names the facility as Northern Colorado Wildlife Center or NCWC",
        parent=ncwc,
        critical=True,
    )
    claimed_name = data.facility_name if data else ""
    await evaluator.verify(
        claim=f'The identified facility name "{claimed_name}" refers to Northern Colorado Wildlife Center (also known as "NCWC").',
        node=name_node,
        additional_instruction="Allow reasonable variants and abbreviations."
    )

    # Domain URL presence from nocowildlife.org
    name_url_node = evaluator.add_custom_node(
        result=data is not None and _domain_in_sources(_safe_list(data.source_urls), [NCWC["domain"]]),
        id="NCWC_Name_URL",
        desc="Provides URL reference from nocowildlife.org domain",
        parent=ncwc,
        critical=True,
    )

    # Phone (970) 283-7822
    phone_node = evaluator.add_leaf(
        id="NCWC_Phone_Number",
        desc="Provides the correct phone number (970) 283-7822",
        parent=ncwc,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The phone number for {NCWC["name"]} is {NCWC["phone"]}.',
        node=phone_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept common formatting variants such as 970-283-7822 or (970) 283-7822."
    )

    # Phone URL presence
    phone_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="NCWC_Phone_URL",
        desc="Provides URL reference for phone number verification",
        parent=ncwc,
        critical=True,
    )

    # Physical Address
    addr_node = evaluator.add_leaf(
        id="NCWC_Physical_Address",
        desc="Provides the physical address: 2637 Midpoint Dr, Suite E, Fort Collins, CO 80525",
        parent=ncwc,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The physical address for {NCWC["name"]} is {NCWC["address"]}.',
        node=addr_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Allow minor formatting variations (e.g., 'Dr' vs 'Drive', 'Suite' vs 'Ste', punctuation)."
    )

    # Address URL presence
    addr_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="NCWC_Address_URL",
        desc="Provides URL reference for address verification",
        parent=ncwc,
        critical=True,
    )

    # Specialization
    spec_node = evaluator.add_leaf(
        id="NCWC_Specialization",
        desc="Notes that NCWC provides rehabilitation for reptiles, amphibians, and small mammals",
        parent=ncwc,
        critical=True,
    )
    await evaluator.verify(
        claim=f'{NCWC["name"]} {NCWC["species_note"]}.',
        node=spec_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Look for explicit mention of reptiles, amphibians, and small mammals."
    )


async def build_mountain_lion_nodes(evaluator: Evaluator, root, data: Optional[ScenarioEntry]) -> None:
    scen = evaluator.add_sequential(
        id="Mountain_Lion_Scenario",
        desc="Complete and accurate protocol for mountain lion sightings or incidents",
        parent=root,
        critical=False,
    )

    cpw = evaluator.add_parallel(
        id="CPW_Protocol",
        desc="Identifies Colorado Parks and Wildlife as the reporting agency with complete contact information",
        parent=scen,
        critical=True,
    )

    # Agency Name
    name_node = evaluator.add_leaf(
        id="CPW_Agency_Name",
        desc="Correctly identifies Colorado Parks and Wildlife or CPW as the agency for mountain lion incidents",
        parent=cpw,
        critical=True,
    )
    claimed_name = data.facility_name if data else ""
    await evaluator.verify(
        claim=f'The identified agency name "{claimed_name}" refers to Colorado Parks and Wildlife (CPW).',
        node=name_node,
        additional_instruction="Accept the abbreviation 'CPW'."
    )

    # CPW domain presence
    cpw_url_node = evaluator.add_custom_node(
        result=data is not None and _domain_in_sources(_safe_list(data.source_urls), [CPW["domain"]]),
        id="CPW_Agency_URL",
        desc="Provides URL reference from cpw.state.co.us domain",
        parent=cpw,
        critical=True,
    )

    # CPW Fort Collins office phone
    phone_node = evaluator.add_leaf(
        id="CPW_Contact_Number",
        desc="Provides the CPW Fort Collins office phone number (970) 472-4300",
        parent=cpw,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The Colorado Parks and Wildlife Fort Collins office phone number is {CPW["fort_collins_phone"]}.',
        node=phone_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept formatting variants like 970-472-4300 or (970) 472-4300."
    )

    # Contact URL presence
    contact_url_node = evaluator.add_custom_node(
        result=data is not None and len(_safe_list(data.source_urls)) > 0,
        id="CPW_Contact_URL",
        desc="Provides URL reference for contact information verification",
        parent=cpw,
        critical=True,
    )

    # Reporting note
    note_node = evaluator.add_leaf(
        id="CPW_Reporting_Note",
        desc="Notes that mountain lion incidents must be reported immediately to CPW",
        parent=cpw,
        critical=True,
    )
    await evaluator.verify(
        claim=f'In Colorado (Fort Collins area), {CPW["reporting_note"]}.',
        node=note_node,
        sources=_safe_list(data.source_urls) if data else [],
        additional_instruction="Accept synonymous language like 'contact immediately', 'report at once', or 'call CPW right away'."
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
    Evaluate an answer for the Fort Collins wildlife referral guide task.
    """
    # Initialize evaluator (root parallel as rubric)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_referral_guide(),
        template_class=ReferralGuideExtraction,
        extraction_name="wildlife_referral_extraction",
    )

    # Optionally record ground-truth anchors to help interpret results
    evaluator.add_ground_truth({
        "expected_anchors": {
            "songbird_waterfowl": {"facility": GREENWOOD["name"], "domain": GREENWOOD["domain"], "phone": GREENWOOD["phone"]},
            "raptor": {"facility": RMRP["name"], "phone_after_hours": RMRP["after_hours"]},
            "small_mammal": {"facility_options": [NCWC["name"], GREENWOOD["name"]]},
            "reptile_amphibian": {"facility": NCWC["name"], "phone": NCWC["phone"], "address": NCWC["address"]},
            "mountain_lion": {"agency": CPW["name"], "fort_collins_phone": CPW["fort_collins_phone"], "domain": CPW["domain"]},
        }
    })

    # Build each scenario subtree
    await build_songbird_waterfowl_nodes(evaluator, root, extracted.songbird_waterfowl)
    await build_raptor_nodes(evaluator, root, extracted.raptor)
    await build_small_mammal_nodes(evaluator, root, extracted.small_mammal)
    await build_reptile_amphibian_nodes(evaluator, root, extracted.reptile_amphibian)
    await build_mountain_lion_nodes(evaluator, root, extracted.mountain_lion)

    # Return evaluation summary
    return evaluator.get_summary()