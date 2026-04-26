import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cal_coast_campgrounds"
TASK_DESCRIPTION = """
I'm planning a summer camping trip for three families traveling together along the California coast. We need to find three different campgrounds that meet our group's specific requirements. Each campground must be: (1) Located within 50 miles of the California coast; (2) Managed by either California State Parks or a National Forest; (3) Able to accommodate RVs up to at least 35 feet in length; (4) Have sites with full hookups (water, electric, sewer) OR at minimum water and electric hookups; (5) Able to accommodate our group of approximately 20 people through either group camping sites OR at least 4 individual sites that can be reserved together; (6) Have designated ADA accessible campsites and accessible restroom facilities; (7) Provide flush toilet facilities (not just vault toilets) and potable water on-site; (8) Allow dogs in the campground. For each of the three campgrounds, please provide: the campground name, full address or specific location, the managing agency, a direct link to the official reservation page (ReserveCalifornia.com or Recreation.gov), confirmation that it meets the RV length and hookup requirements, details about group capacity or number of sites available, confirmation of ADA accessibility features, information about amenities (restrooms, water, and any shower facilities), the pet policy, and the advance reservation window and typical camping fees.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None  # address or specific location
    managing_agency: Optional[str] = None

    reservation_url: Optional[str] = None  # must be ReserveCalifornia.com or Recreation.gov
    additional_urls: List[str] = Field(default_factory=list)

    rv_length: Optional[str] = None  # e.g., "up to 40 ft", "35+ ft", etc.
    hookups: Optional[str] = None  # e.g., "full hookups", "water & electric", etc.

    group_accommodation: Optional[str] = None  # free-form summary as stated in the answer
    group_site_capacity: Optional[str] = None  # e.g., "25 people", "20-40", etc.
    num_individual_sites_together: Optional[str] = None  # e.g., "4+ sites", "at least four sites"

    ada_campsites: Optional[str] = None  # statement confirming designated ADA sites
    ada_restrooms: Optional[str] = None  # statement confirming ADA-accessible restrooms

    toilets: Optional[str] = None  # e.g., "flush toilets", "vault", etc.
    potable_water: Optional[str] = None  # e.g., "potable water available"
    showers: Optional[str] = None  # e.g., "showers available", "no showers"

    dogs_allowed: Optional[str] = None  # e.g., "dogs allowed"
    pet_policy_details: Optional[str] = None  # leash rules, areas, limits, etc.

    reservation_window: Optional[str] = None  # e.g., "6 months in advance"
    typical_fees: Optional[str] = None  # e.g., "$35–$50/night", "varies by site"


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract every distinct campground the answer proposes (or meaningfully discusses as candidates) for the family's coastal California camping trip.

    For each campground, extract the following fields exactly as they appear in the answer text:
    - name: The campground name.
    - location: Full address or specific location/park name as provided.
    - managing_agency: The named managing agency if provided (e.g., "California State Parks", "Los Padres National Forest", "USFS", etc.).
    - reservation_url: A single direct URL to the official reservation page if one is given. It must be a URL on ReserveCalifornia.com or Recreation.gov (extract exactly the URL shown in the answer; do not invent).
    - additional_urls: Any other URLs provided in the answer for this campground (e.g., park pages, campground info pages). Return an array of URLs. If none, return [].
    - rv_length: The statement about RV length supported (e.g., "RVs up to 35 ft", "max 40 ft").
    - hookups: The statement about utility hookups (e.g., "full hookups", "water & electric", "no hookups").
    - group_accommodation: The summary phrase or sentence about group capacity or reserving multiple sites together.
    - group_site_capacity: Any numeric or textual capacity (e.g., "20 people", "25–40").
    - num_individual_sites_together: Any explicit statement about reserving 4 or more individual sites together.
    - ada_campsites: The explicit confirmation of designated ADA/accessible campsites, if present in the answer.
    - ada_restrooms: The explicit confirmation of ADA-accessible restrooms, if present in the answer.
    - toilets: The restroom type mentioned (e.g., "flush toilets", "vault toilets"). If multiple types, capture as text.
    - potable_water: Whether potable water is present on-site, as text.
    - showers: Whether showers are available. Record "showers available", "no showers", or the exact wording used.
    - dogs_allowed: Whether dogs are allowed, as text (e.g., "dogs allowed").
    - pet_policy_details: Any detailed rules (e.g., leash requirements, restricted areas, limits).
    - reservation_window: The advance reservation window if stated (e.g., "6 months in advance").
    - typical_fees: The typical camping fees or fee range as stated.

    IMPORTANT:
    - Extract only information explicitly present in the answer text. Do not infer or add any new info.
    - Include all distinct campgrounds mentioned, even if more than three are listed.
    - For URL fields, extract only valid, complete URLs explicitly present in the answer. If none, return null (for reservation_url) or [] (for additional_urls).
    - If a field is not present in the answer for a campground, return null for that field.

    Return a JSON object with a single field "campgrounds", which is an array of campground objects with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def get_sources_for_campground(cg: CampgroundItem) -> List[str]:
    """Combine reservation_url and additional_urls into a deduplicated list."""
    urls: List[str] = []
    if cg.reservation_url:
        urls.append(cg.reservation_url)
    urls.extend(cg.additional_urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def is_official_reservation_url(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        netloc = urlparse(url).netloc.lower()
        return ("reservecalifornia.com" in netloc) or ("recreation.gov" in netloc)
    except Exception:
        return False


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


# --------------------------------------------------------------------------- #
# Verification for a single campground                                        #
# --------------------------------------------------------------------------- #
async def verify_one_campground(
    evaluator: Evaluator,
    parent: VerificationNode,
    cg: CampgroundItem,
    idx: int,
    prior_names: List[str],
    total_mentioned_count: int,
) -> None:
    """
    Build verification nodes for one campground under `parent`.
    """
    cg_idx = idx + 1
    cg_node = evaluator.add_parallel(
        id=f"Campground_{cg_idx}",
        desc=f"Campground {cg_idx} (item {cg_idx}): meets all constraints and includes all required information",
        parent=parent,
        critical=False,
    )

    # Name provided (critical)
    name_exists = evaluator.add_custom_node(
        result=_nonempty(cg.name),
        id=f"CG{cg_idx}_Name",
        desc="Campground name is provided",
        parent=cg_node,
        critical=True,
    )

    # Location provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(cg.location),
        id=f"CG{cg_idx}_Location",
        desc="Full address or specific location is provided",
        parent=cg_node,
        critical=True,
    )

    # Reservation link existence and official domain (critical)
    reservation_ok = evaluator.add_custom_node(
        result=is_official_reservation_url(cg.reservation_url),
        id=f"CG{cg_idx}_ReservationLink",
        desc="A direct link to the official reservation page on ReserveCalifornia.com or Recreation.gov is provided",
        parent=cg_node,
        critical=True,
    )

    # Uniqueness checks for CG2 and CG3
    if cg_idx == 2:
        uniq2 = evaluator.add_leaf(
            id=f"CG2_UniqueFromCG1",
            desc="Campground 2 is different from Campground 1 (no duplicate campground)",
            parent=cg_node,
            critical=True,
        )
        n1 = prior_names[0] if prior_names else ""
        claim = f"Campground 2 '{cg.name or ''}' is a different campground than Campground 1 '{n1}'. Consider park/area synonyms; different loops within the same named campground should be treated as the same campground."
        await evaluator.verify(
            claim=claim,
            node=uniq2,
            additional_instruction="Judge distinctness by official campground identity, not by site loop names.",
        )

    if cg_idx == 3:
        # Unique from 1 and 2
        uniq3 = evaluator.add_leaf(
            id=f"CG3_UniqueFromCG1andCG2",
            desc="Campground 3 is different from Campground 1 and Campground 2 (no duplicate campground)",
            parent=cg_node,
            critical=True,
        )
        n1 = prior_names[0] if len(prior_names) > 0 else ""
        n2 = prior_names[1] if len(prior_names) > 1 else ""
        claim = f"Campground 3 '{cg.name or ''}' is a different campground than Campground 1 '{n1}' and Campground 2 '{n2}'. Consider park/area synonyms; different loops within the same named campground should be treated as the same campground."
        await evaluator.verify(
            claim=claim,
            node=uniq3,
            additional_instruction="Judge distinctness by official campground identity, not by site loop names.",
        )

        # Exactly three campgrounds only
        evaluator.add_custom_node(
            result=(total_mentioned_count == 3),
            id=f"CG3_OnlyThreeCampgrounds",
            desc="Response identifies exactly three campgrounds (no extra campgrounds beyond the three items)",
            parent=cg_node,
            critical=True,
        )

    # Build common sources list
    sources = get_sources_for_campground(cg)

    # Managing agency identified and valid (critical)
    managing_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_ManagingAgency",
        desc="Managing agency is identified and is either California State Parks or a National Forest",
        parent=cg_node,
        critical=True,
    )
    m_agency = cg.managing_agency or "unknown managing agency"
    claim = (
        f"The managing agency for {cg.name or 'this campground'} is {m_agency}, "
        f"and this agency is either California State Parks or a National Forest (managed by the U.S. Forest Service)."
    )
    await evaluator.verify(
        claim=claim,
        node=managing_leaf,
        sources=sources,
        additional_instruction=(
            "Accept if the page indicates California State Parks or explicitly references a National Forest (USFS). "
            "Pages may say 'Los Padres National Forest', 'Angeles National Forest', etc., which count as National Forest."
        ),
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Coastal proximity within 50 miles (critical)
    coastal_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_CoastalProximity",
        desc="Campground is located within 50 miles of the California coast",
        parent=cg_node,
        critical=True,
    )
    claim = (
        f"The campground {cg.name or ''} is located within 50 miles (≈80 km) of the California coastline."
    )
    await evaluator.verify(
        claim=claim,
        node=coastal_leaf,
        sources=sources,
        additional_instruction=(
            "Use the location information on the page (address, map, or description). "
            "If clearly in a coastal city/park (e.g., within typical coastal counties) or the page indicates proximity to the ocean, consider it within 50 miles."
        ),
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # RV length >= 35 ft (critical)
    rv_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_RV_Length",
        desc="Confirms campground can accommodate RVs up to at least 35 feet in length",
        parent=cg_node,
        critical=True,
    )
    claim = (
        f"The campground can accommodate RVs up to at least 35 feet in length. "
        f"Extracted statement: '{cg.rv_length or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim,
        node=rv_leaf,
        sources=sources,
        additional_instruction=(
            "If any site or the campground maximum allows 35 ft or more, consider this satisfied. "
            "Per-site limits may vary; it's sufficient that some sites meet 35+ ft."
        ),
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Hookups (full OR at least water+electric) (critical)
    hookups_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Hookups",
        desc="Confirms sites have full hookups (water/electric/sewer) OR at minimum water and electric hookups",
        parent=cg_node,
        critical=True,
    )
    claim = (
        f"The campground offers either full hookups (water, electric, and sewer) or at least water and electric hookups. "
        f"Extracted statement: '{cg.hookups or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim,
        node=hookups_leaf,
        sources=sources,
        additional_instruction="Accept 'full hookups', or 'water & electric'. If only 'electric' or only 'water', it does not satisfy.",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Group accommodation for ~20 people (critical)
    group_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_GroupAccommodation",
        desc="Confirms ability to accommodate ~20 people via group site(s) OR at least 4 individual sites reservable together, and provides the supporting capacity/site-count details",
        parent=cg_node,
        critical=True,
    )
    group_details = (
        f"Summary: {cg.group_accommodation or 'N/A'}; "
        f"Group capacity: {cg.group_site_capacity or 'N/A'}; "
        f"Sites together: {cg.num_individual_sites_together or 'N/A'}."
    )
    claim = (
        "This campground can accommodate approximately 20 people either via one or more group sites with capacity of at least 20, "
        "or by reserving at least four individual campsites together. "
        f"{group_details}"
    )
    await evaluator.verify(
        claim=claim,
        node=group_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the presence of a group site with capacity >= 20 OR confirm that at least 4 individual sites could be reserved together for the same dates. "
            "If the source explicitly states a group site capacity reaching ~20 or more, accept."
        ),
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # ADA accessible campsites (critical)
    ada_site_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_ADA_Campsites",
        desc="Designated ADA accessible campsites are confirmed",
        parent=cg_node,
        critical=True,
    )
    claim = (
        "There are designated ADA-accessible campsites (or equivalent accessible sites) at this campground."
    )
    await evaluator.verify(
        claim=claim,
        node=ada_site_leaf,
        sources=sources,
        additional_instruction="Look for indications like 'ADA', 'accessible campsite', 'accessible site', or wheelchair icon designations.",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # ADA accessible restrooms (critical)
    ada_rr_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_ADA_Restrooms",
        desc="Accessible restroom facilities are confirmed",
        parent=cg_node,
        critical=True,
    )
    claim = "The campground has ADA-accessible restroom facilities."
    await evaluator.verify(
        claim=claim,
        node=ada_rr_leaf,
        sources=sources,
        additional_instruction="Look for 'accessible restrooms', 'ADA restrooms', or equivalent phrasing.",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Amenities: flush toilets + potable water + showers info stated (critical)
    amenities_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_Amenities",
        desc="Amenity info is provided and satisfies constraints: flush toilets (not only vault toilets), potable water on-site, and shower availability is stated (present or not)",
        parent=cg_node,
        critical=True,
    )
    claim = (
        "This campground provides flush toilet facilities (not only vault toilets) and has potable water on-site; "
        "and the presence or absence of showers is explicitly stated on the cited page(s)."
    )
    await evaluator.verify(
        claim=claim,
        node=amenities_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm: (1) Flush toilets present (or restrooms with flush). If only 'vault toilets', fail. "
            "(2) Potable water available on-site. "
            "(3) The page explicitly states whether showers are available or not (either is acceptable as long as stated)."
        ),
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Dogs allowed (critical)
    dogs_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_DogsAllowed",
        desc="Confirms dogs are allowed in the campground",
        parent=cg_node,
        critical=True,
    )
    claim = "Dogs are allowed in this campground."
    await evaluator.verify(
        claim=claim,
        node=dogs_leaf,
        sources=sources,
        additional_instruction="Accept typical pet policy language that confirms dogs are permitted (often on leash, with restrictions).",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Pet policy details (critical)
    pet_policy_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_PetPolicyDetails",
        desc="Pet policy details are provided (e.g., leash rules, restricted areas, limits)",
        parent=cg_node,
        critical=True,
    )
    claim = (
        "The pet policy includes specific details such as leash requirements, restricted areas, or other limits, "
        f"beyond merely saying 'dogs allowed'. Extracted: '{cg.pet_policy_details or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim,
        node=pet_policy_leaf,
        sources=sources,
        additional_instruction="Look for at least one concrete rule (e.g., must be on 6-foot leash, not allowed on trails/beaches, always attended, etc.).",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Advance reservation window (critical)
    window_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_AdvanceReservationWindow",
        desc="Advance reservation window information is provided",
        parent=cg_node,
        critical=True,
    )
    claim = (
        f"The page states an advance reservation window or booking policy for this campground. "
        f"Extracted: '{cg.reservation_window or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim,
        node=window_leaf,
        sources=sources,
        additional_instruction="Accept any explicit mention of booking lead time/window (e.g., 6 months in advance) or policy language conveying the same.",
        extra_prerequisites=[reservation_ok, name_exists],
    )

    # Typical fees (critical)
    fees_leaf = evaluator.add_leaf(
        id=f"CG{cg_idx}_TypicalFees",
        desc="Typical camping fees (or fee range) are provided",
        parent=cg_node,
        critical=True,
    )
    claim = (
        f"The page provides typical camping fees or a fee range for this campground. Extracted: '{cg.typical_fees or 'N/A'}'."
    )
    await evaluator.verify(
        claim=claim,
        node=fees_leaf,
        sources=sources,
        additional_instruction="Look for nightly site fees (range acceptable). If multiple site types, any representative typical fee information suffices.",
        extra_prerequisites=[reservation_ok, name_exists],
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
    Evaluate an answer for the California coastal campgrounds task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation at the root
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

    # Extract all campgrounds mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction",
    )

    # Record simple custom info for debugging
    total_mentioned = len(extracted.campgrounds or [])
    unique_names_lower = []
    for item in extracted.campgrounds:
        n = (item.name or "").strip()
        if n and n.lower() not in unique_names_lower:
            unique_names_lower.append(n.lower())
    evaluator.add_custom_info(
        info={"total_mentioned": total_mentioned, "unique_names_count": len(unique_names_lower)},
        info_type="counts",
        info_name="campground_counts",
    )

    # Select exactly three items for evaluation (pad with empty if fewer)
    selected: List[CampgroundItem] = list((extracted.campgrounds or [])[:3])
    while len(selected) < 3:
        selected.append(CampgroundItem())

    # Build verification tree per item
    prior_names: List[str] = []
    for idx, cg in enumerate(selected):
        await verify_one_campground(
            evaluator=evaluator,
            parent=root,
            cg=cg,
            idx=idx,
            prior_names=prior_names,
            total_mentioned_count=len(unique_names_lower) if total_mentioned else 0,
        )
        # Track names for uniqueness checks
        if _nonempty(cg.name):
            prior_names.append(cg.name.strip())

    # Return the evaluation summary
    return evaluator.get_summary()