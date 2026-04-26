import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tn_camping_four_campgrounds"
TASK_DESCRIPTION = """You are planning a camping road trip and need to identify four specific campgrounds in Tennessee that meet distinct criteria based on their location, capacity, and available amenities. Using official sources such as the National Park Service (nps.gov), Recreation.gov, and Tennessee State Parks (tnstateparks.com), provide the name, exact location, total site capacity, electric hookup availability details, and a reference URL for each of the following four campgrounds:

1. Campground A: A campground located within Great Smoky Mountains National Park along the Foothills Parkway. This campground must offer sites with electric and water hookups and must have a total capacity of more than 60 campsites.

2. Campground B: A campground located within Big South Fork National River and Recreation Area. This campground must have the largest number of sites with electric and water hookups among all campgrounds in Big South Fork.

3. Campground C: A Tennessee State Park campground located near Norris Lake. This campground must have exactly 50 campsites, and all 50 sites must offer 50-amp electric hookups along with water connections.

4. Campground D: A campground located within Great Smoky Mountains National Park that operates year-round. This campground must have a total capacity of more than 140 campsites but must not offer any electric hookup sites.

For each campground, provide:
- The campground name
- The specific park or recreation area where it is located
- The total number of campsites
- The number of sites with electric hookups (or confirmation of zero electric hookups)
- A reference URL from an official government source
"""

ALLOWED_A_DOMAINS = ["nps.gov", "recreation.gov"]
ALLOWED_B_DOMAINS = ["nps.gov", "recreation.gov"]
ALLOWED_C_DOMAINS = ["tnstateparks.com"]
ALLOWED_D_DOMAINS = ["nps.gov", "recreation.gov"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundFields(BaseModel):
    name: Optional[str] = None
    park_or_area: Optional[str] = None
    total_campsites: Optional[str] = None
    electric_sites: Optional[str] = None  # e.g., "0", "20", "none"
    electric_details: Optional[str] = None  # e.g., "50-amp", "30/50-amp", etc.
    water_hookups_info: Optional[str] = None  # e.g., "water hookups at all electric sites"
    is_open_year_round: Optional[str] = None  # "yes"/"no"/text
    location_context: Optional[str] = None  # e.g., "along Foothills Parkway", "near Norris Lake"
    reference_urls: List[str] = Field(default_factory=list)


class FourCampgroundsExtraction(BaseModel):
    campground_a: Optional[CampgroundFields] = None
    campground_b: Optional[CampgroundFields] = None
    campground_c: Optional[CampgroundFields] = None
    campground_d: Optional[CampgroundFields] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract structured information for four specifically labeled campgrounds (Campground A, B, C, D) from the provided answer text.

    For EACH campground, return the following fields:
    - name: the campground name exactly as stated
    - park_or_area: the specific park, national park unit, or state park the campground belongs to (e.g., "Great Smoky Mountains National Park", "Big South Fork National River and Recreation Area", "Norris Dam State Park")
    - total_campsites: the total number of campsites as presented (string; do not parse to a number)
    - electric_sites: the number of sites with electric hookups as presented (use string "0" or "none" if there are no electric hookups; otherwise the number as a string)
    - electric_details: any details about electric service (e.g., "50-amp", "30/50-amp") if stated
    - water_hookups_info: text describing water hookup availability if stated (e.g., "water hookups at each electric site", "no water hookups")
    - is_open_year_round: "yes" if the campground is described as open year-round, "no" otherwise, or null if unspecified
    - location_context: any location context text explicitly mentioned (e.g., "along the Foothills Parkway", "near Norris Lake")
    - reference_urls: an array of all URLs cited for this campground in the answer text. IMPORTANT: include only official sources:
        * Campground A, B, D: URLs must be on nps.gov or recreation.gov
        * Campground C: URLs must be on tnstateparks.com
      If multiple official URLs are provided, include them all. If none are present, return an empty array.

    Output a JSON object with the keys:
    - campground_a
    - campground_b
    - campground_c
    - campground_d
    Each key maps to an object with the fields described above.

    Do NOT invent values. If any field is not stated in the answer, set it to null (or an empty list for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _filter_official_urls(urls: List[str], allowed_domains: List[str]) -> List[str]:
    official = []
    for u in urls:
        if not _nonempty(u):
            continue
        try:
            uri = u if "://" in u else ("http://" + u)
            host = urlparse(uri).netloc.lower()
            if any(host.endswith(dom) for dom in allowed_domains):
                official.append(u)
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for u in official:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _mk_sources(item: Optional[CampgroundFields], allowed_domains: List[str]) -> List[str]:
    if not item or not item.reference_urls:
        return []
    return _filter_official_urls(item.reference_urls, allowed_domains)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_campground_a(evaluator: Evaluator, parent_node, item: Optional[CampgroundFields]) -> None:
    node = evaluator.add_parallel(
        id="Campground_A",
        desc="A campground in Great Smoky Mountains National Park along the Foothills Parkway with electric hookup sites and total capacity exceeding 60 sites",
        parent=parent_node,
        critical=False
    )

    sources = _mk_sources(item, ALLOWED_A_DOMAINS)

    # Reference URL (critical)
    ref_node = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Campground_A_Reference_URL",
        desc="A valid reference URL from an official source (nps.gov or recreation.gov) is provided",
        parent=node,
        critical=True
    )

    # Name (critical)
    if item and _nonempty(item.name):
        name_leaf = evaluator.add_leaf(
            id="Campground_A_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This page is about a campground named '{item.name}'.",
            node=name_leaf,
            sources=sources,
            additional_instruction="Confirm that the page clearly identifies the campground name (allow minor formatting or case variations).",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_A_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )

    # Park Location (critical)
    park_leaf = evaluator.add_leaf(
        id="Campground_A_Park_Location",
        desc="The campground is correctly identified as being in Great Smoky Mountains National Park along the Foothills Parkway",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground is located within Great Smoky Mountains National Park and is along or adjacent to the Foothills Parkway.",
        node=park_leaf,
        sources=sources,
        additional_instruction="Look for explicit mention of Great Smoky Mountains National Park and reference to the Foothills Parkway on the page (title, body, or map). Allow reasonable wording like 'on the Foothills Parkway' or 'near the Foothills Parkway'.",
        extra_prerequisites=[ref_node],
    )

    # Total Sites (critical) - must be provided and exceed 60
    if item and _nonempty(item.total_campsites):
        total_leaf = evaluator.add_leaf(
            id="Campground_A_Total_Sites",
            desc="The total number of campsites is correctly provided and exceeds 60",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This campground has a total of {item.total_campsites} campsites, and the total capacity exceeds 60 campsites.",
            node=total_leaf,
            sources=sources,
            additional_instruction="Confirm the total campsite count on the official page and that it is > 60.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_A_Total_Sites",
            desc="The total number of campsites is correctly provided and exceeds 60",
            parent=node,
            critical=True
        )

    # Electric hookups (critical) - must offer electric and water hookups; number specified
    if item and _nonempty(item.electric_sites):
        elec_leaf = evaluator.add_leaf(
            id="Campground_A_Electric_Hookups",
            desc="The campground offers electric and water hookup sites, and the number of such sites is correctly specified",
            parent=node,
            critical=True
        )
        details = f" ({item.electric_details})" if _nonempty(item.electric_details) else ""
        await evaluator.verify(
            claim=f"This campground offers electric and water hookup sites. Specifically, it has {item.electric_sites} sites with both electric{details} and water hookups.",
            node=elec_leaf,
            sources=sources,
            additional_instruction="Verify that electric and water hookups are offered and that the specified number of such sites matches the official information.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_A_Electric_Hookups",
            desc="The campground offers electric and water hookup sites, and the number of such sites is correctly specified",
            parent=node,
            critical=True
        )


async def verify_campground_b(evaluator: Evaluator, parent_node, item: Optional[CampgroundFields]) -> None:
    node = evaluator.add_parallel(
        id="Campground_B",
        desc="The campground in Big South Fork National River and Recreation Area with the largest number of electric and water hookup sites",
        parent=parent_node,
        critical=False
    )

    sources = _mk_sources(item, ALLOWED_B_DOMAINS)

    # Reference URL (critical)
    ref_node = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Campground_B_Reference_URL",
        desc="A valid reference URL from an official source (nps.gov or recreation.gov) is provided",
        parent=node,
        critical=True
    )

    # Name (critical)
    if item and _nonempty(item.name):
        name_leaf = evaluator.add_leaf(
            id="Campground_B_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This page is about a campground named '{item.name}'.",
            node=name_leaf,
            sources=sources,
            additional_instruction="Confirm the page clearly identifies the campground name.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_B_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )

    # Park Location (critical)
    park_leaf = evaluator.add_leaf(
        id="Campground_B_Park_Location",
        desc="The campground is correctly identified as being in Big South Fork National River and Recreation Area",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground is located within Big South Fork National River and Recreation Area (an NPS unit).",
        node=park_leaf,
        sources=sources,
        additional_instruction="Confirm the NPS unit 'Big South Fork National River and Recreation Area' on the page.",
        extra_prerequisites=[ref_node],
    )

    # Total Sites (critical) - must be provided
    if item and _nonempty(item.total_campsites):
        total_leaf = evaluator.add_leaf(
            id="Campground_B_Total_Sites",
            desc="The total number of campsites is correctly provided",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"The total number of campsites for this campground is {item.total_campsites}.",
            node=total_leaf,
            sources=sources,
            additional_instruction="Confirm the total campsite count on the official page.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_B_Total_Sites",
            desc="The total number of campsites is correctly provided",
            parent=node,
            critical=True
        )

    # Electric hookups (critical) - highest number of E+W sites among BSF campgrounds; number specified
    if item and _nonempty(item.electric_sites):
        elec_leaf = evaluator.add_leaf(
            id="Campground_B_Electric_Hookups",
            desc="The number of sites with electric and water hookups is correctly specified and is the highest among Big South Fork campgrounds",
            parent=node,
            critical=True
        )
        details = f" ({item.electric_details})" if _nonempty(item.electric_details) else ""
        await evaluator.verify(
            claim=f"This campground has {item.electric_sites} sites with both electric{details} and water hookups, which is the largest number among campgrounds within Big South Fork National River and Recreation Area.",
            node=elec_leaf,
            sources=sources,
            additional_instruction="Prefer explicit statements indicating it's the largest. If not explicitly stated, compare counts across provided official pages if available.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_B_Electric_Hookups",
            desc="The number of sites with electric and water hookups is correctly specified and is the highest among Big South Fork campgrounds",
            parent=node,
            critical=True
        )


async def verify_campground_c(evaluator: Evaluator, parent_node, item: Optional[CampgroundFields]) -> None:
    node = evaluator.add_parallel(
        id="Campground_C",
        desc="A Tennessee State Park campground near Norris Lake with exactly 50 campsites, all with 50-amp electric hookups",
        parent=parent_node,
        critical=False
    )

    sources = _mk_sources(item, ALLOWED_C_DOMAINS)

    # Reference URL (critical)
    ref_node = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Campground_C_Reference_URL",
        desc="A valid reference URL from an official source (tnstateparks.com) is provided",
        parent=node,
        critical=True
    )

    # Name (critical)
    if item and _nonempty(item.name):
        name_leaf = evaluator.add_leaf(
            id="Campground_C_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This page is about a campground named '{item.name}'.",
            node=name_leaf,
            sources=sources,
            additional_instruction="Confirm the page clearly identifies the campground name.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_C_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )

    # Park/Location (critical) - Tennessee State Park near Norris Lake
    park_leaf = evaluator.add_leaf(
        id="Campground_C_Park_Location",
        desc="The campground is correctly identified as being in a Tennessee State Park near Norris Lake",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground is located in a Tennessee State Park and is near or on Norris Lake.",
        node=park_leaf,
        sources=sources,
        additional_instruction="Look for 'Tennessee State Parks' branding and explicit mention of Norris Lake proximity (e.g., 'near Norris Lake', 'on the shores of Norris Lake', or similar).",
        extra_prerequisites=[ref_node],
    )

    # Total Sites (critical) - exactly 50 (must be provided)
    if item and _nonempty(item.total_campsites):
        total_leaf = evaluator.add_leaf(
            id="Campground_C_Total_Sites",
            desc="The total number of campsites is correctly provided as exactly 50",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim="This campground has exactly 50 campsites.",
            node=total_leaf,
            sources=sources,
            additional_instruction="Verify the total campsite count is exactly 50.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_C_Total_Sites",
            desc="The total number of campsites is correctly provided as exactly 50",
            parent=node,
            critical=True
        )

    # Electric hookups (critical) - all 50 sites have 50-amp electric and water
    elec_leaf = evaluator.add_leaf(
        id="Campground_C_Electric_Hookups",
        desc="All 50 sites have 50-amp electric and water hookups",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="All 50 campsites at this campground have 50-amp electric hookups and water connections.",
        node=elec_leaf,
        sources=sources,
        additional_instruction="Confirm both 50-amp electric service and water connections are available at all 50 sites.",
        extra_prerequisites=[ref_node],
    )


async def verify_campground_d(evaluator: Evaluator, parent_node, item: Optional[CampgroundFields]) -> None:
    node = evaluator.add_parallel(
        id="Campground_D",
        desc="A campground in Great Smoky Mountains National Park that is open year-round, has more than 140 total sites, but offers no electric hookups",
        parent=parent_node,
        critical=False
    )

    sources = _mk_sources(item, ALLOWED_D_DOMAINS)

    # Reference URL (critical)
    ref_node = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="Campground_D_Reference_URL",
        desc="A valid reference URL from an official source (nps.gov or recreation.gov) is provided",
        parent=node,
        critical=True
    )

    # Name (critical)
    if item and _nonempty(item.name):
        name_leaf = evaluator.add_leaf(
            id="Campground_D_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This page is about a campground named '{item.name}'.",
            node=name_leaf,
            sources=sources,
            additional_instruction="Confirm the page clearly identifies the campground name.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_D_Name",
            desc="The campground name is correctly identified",
            parent=node,
            critical=True
        )

    # Park Location (critical)
    park_leaf = evaluator.add_leaf(
        id="Campground_D_Park_Location",
        desc="The campground is correctly identified as being in Great Smoky Mountains National Park",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground is located within Great Smoky Mountains National Park.",
        node=park_leaf,
        sources=sources,
        additional_instruction="Confirm explicit mention of Great Smoky Mountains National Park.",
        extra_prerequisites=[ref_node],
    )

    # Operational Status (critical) - open year-round
    op_leaf = evaluator.add_leaf(
        id="Campground_D_Operational_Status",
        desc="The campground is confirmed to be open year-round",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground operates year-round (open all year).",
        node=op_leaf,
        sources=sources,
        additional_instruction="Confirm 'open year-round' or equivalent wording on the official page.",
        extra_prerequisites=[ref_node],
    )

    # Total Sites (critical) - must be provided and exceed 140
    if item and _nonempty(item.total_campsites):
        total_leaf = evaluator.add_leaf(
            id="Campground_D_Total_Sites",
            desc="The total number of campsites is correctly provided and exceeds 140",
            parent=node,
            critical=True
        )
        await evaluator.verify(
            claim=f"This campground has a total of {item.total_campsites} campsites, and the total capacity exceeds 140 campsites.",
            node=total_leaf,
            sources=sources,
            additional_instruction="Confirm the total campsite count on the official page and that it is > 140.",
            extra_prerequisites=[ref_node],
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="Campground_D_Total_Sites",
            desc="The total number of campsites is correctly provided and exceeds 140",
            parent=node,
            critical=True
        )

    # Electric hookups (critical) - confirmed zero
    elec_leaf = evaluator.add_leaf(
        id="Campground_D_Electric_Hookups",
        desc="The campground is confirmed to have zero electric hookup sites",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground offers no electric hookups (0 sites with electric service).",
        node=elec_leaf,
        sources=sources,
        additional_instruction="Confirm that electric hookups are not available at this campground.",
        extra_prerequisites=[ref_node],
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Tennessee campgrounds task and return a structured summary.
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=FourCampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Add custom info: domains policy used
    evaluator.add_custom_info(
        info={
            "allowed_domains": {
                "Campground_A": ALLOWED_A_DOMAINS,
                "Campground_B": ALLOWED_B_DOMAINS,
                "Campground_C": ALLOWED_C_DOMAINS,
                "Campground_D": ALLOWED_D_DOMAINS
            }
        },
        info_type="policy",
        info_name="domain_policy"
    )

    # Build verification subtrees for A, B, C, D
    await verify_campground_a(evaluator, root, extracted.campground_a)
    await verify_campground_b(evaluator, root, extracted.campground_b)
    await verify_campground_c(evaluator, root, extracted.campground_c)
    await verify_campground_d(evaluator, root, extracted.campground_d)

    return evaluator.get_summary()