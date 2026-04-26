import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_indie_bookstores_2024"
TASK_DESCRIPTION = (
    "Identify a minimum of four independent bookstores in California that were operational in 2024 and meet all of the following criteria:\n\n"
    "1. Each bookstore must be independently owned (not part of major chains such as Barnes & Noble).\n"
    "2. Each bookstore must have a physical location in California with a complete address including street address, city, and ZIP code.\n"
    "3. Each bookstore must have publicly available contact information (phone number or email address).\n"
    "4. Each bookstore must have publicly available operating hours information.\n"
    "5. Each bookstore must demonstrate capability to host author events or community programming (evidenced by event listings, event calendar, or documented history of hosting such events).\n"
    "6. Each bookstore must have an official website or be listed on a recognized independent bookstore directory.\n"
    "7. The selected bookstores must collectively represent at least three different California regions (such as Northern California/Bay Area, Southern California/Los Angeles area, Central California, or San Diego area).\n\n"
    "For each bookstore, provide:\n"
    "- The official bookstore name\n"
    "- Complete physical address (street address, city, ZIP code)\n"
    "- Contact information (phone number or email)\n"
    "- Operating hours\n"
    "- Evidence of events capability (description of their event programming or specific examples)\n"
    "- Official website URL\n"
    "- Reference URLs that verify each piece of information"
)

RECOGNIZED_DIRECTORY_DOMAINS = [
    "indiebound.org",
    "bookshop.org",
    "findabookstore.americanbooksellers.org",  # ABA directory
    "find.indiebound.org",  # legacy indiebound store finder
]


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Bookstore(BaseModel):
    name: Optional[str] = None

    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    address_full: Optional[str] = None

    phone: Optional[str] = None
    email: Optional[str] = None

    hours: Optional[str] = None
    events_evidence: Optional[str] = None

    official_website: Optional[str] = None
    directory_url: Optional[str] = None

    region: Optional[str] = None

    # Reference URLs for verification of each attribute
    address_refs: List[str] = Field(default_factory=list)
    contact_refs: List[str] = Field(default_factory=list)
    hours_refs: List[str] = Field(default_factory=list)
    events_refs: List[str] = Field(default_factory=list)
    operational_2024_refs: List[str] = Field(default_factory=list)
    independent_refs: List[str] = Field(default_factory=list)
    web_presence_refs: List[str] = Field(default_factory=list)


class BookstoresExtraction(BaseModel):
    bookstores: List[Bookstore] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bookstores() -> str:
    return """
Extract up to 5 bookstores presented in the answer that claim to be independent bookstores in California.

For each bookstore, extract these fields exactly as present in the answer:

- name: Official bookstore name
- street_address: Street address (line without city/state/zip if provided)
- city: City
- state: State abbreviation or name (e.g., CA or California)
- zip_code: ZIP code (5-digit or ZIP+4 is acceptable; extract as a string)
- address_full: The complete address string as written (if present)
- phone: Phone number (if present)
- email: Email address (if present)
- hours: Operating hours text (freeform, as summarized or listed in the answer)
- events_evidence: Short description of events capability (e.g., “hosts author readings; event calendar at …”) if provided
- official_website: The official bookstore website URL (if any)
- directory_url: A recognized independent bookstore directory listing URL (if any)
- region: Region label if explicitly stated in the answer (e.g., “Bay Area”, “Central California”, “San Diego”)

Also extract the reference URLs (if any) that the answer claims support each attribute:
- address_refs: list of URLs that verify the address
- contact_refs: list of URLs that verify phone/email
- hours_refs: list of URLs that verify operating hours
- events_refs: list of URLs that verify events capability
- operational_2024_refs: list of URLs that demonstrate the store was operational in 2024 (e.g., 2024 events/hours posts)
- independent_refs: list of URLs that support independent ownership (e.g., About page, ABA/IndieBound/Bookshop listing)
- web_presence_refs: list of URLs for official website or recognized directory listings

Rules:
- Return an array field 'bookstores' with one object per bookstore found (up to 5).
- Do not invent any information. If an item is missing in the answer, set it to null (for string fields) or [] (for URL lists).
- Only include URLs that are explicitly present in the answer (plain URL or inside markdown links). If a URL is missing a protocol, prepend http://
- Prefer keeping values as strings; do not normalize into numbers.
    """.strip()


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _unique_urls(*items: Optional[List[str] | str | None]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, str):
            u = it.strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
        else:
            for u in it:
                u = (u or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


def _full_address(store: Bookstore) -> Optional[str]:
    if store.address_full and store.address_full.strip():
        return store.address_full.strip()
    parts = []
    if store.street_address:
        parts.append(store.street_address.strip())
    if store.city:
        parts.append(store.city.strip())
    state = None
    if store.state and store.state.strip():
        # normalize spacing, but keep original string
        state = store.state.strip()
    zip_code = store.zip_code.strip() if store.zip_code else None
    tail = ", ".join([p for p in [store.city, state] if p and p.strip()])
    # Compose "street, city, state zip"
    if parts and (tail or zip_code):
        left = parts[0]
        right = " ".join([p for p in [tail, zip_code] if p and p.strip()]).strip()
        if tail and zip_code:
            return f"{left}, {tail} {zip_code}"
        elif tail:
            return f"{left}, {tail}"
        else:
            return f"{left}, {zip_code}"
    return None


def _pick_web_presence_urls(store: Bookstore) -> List[str]:
    return _unique_urls(store.official_website, store.directory_url, store.web_presence_refs)


def _categorize_region(store: Bookstore) -> str:
    # Attempt to categorize into one of:
    # "Northern California/Bay Area", "Southern California/Los Angeles area", "Central California", "San Diego area", "Unknown"
    # Heuristics based on ZIP prefix and/or city keywords.
    z = (store.zip_code or "").strip()
    zip3 = None
    if len(z) >= 5 and z[:5].isdigit():
        try:
            zip3 = int(z[:3])
        except Exception:
            zip3 = None

    city = (store.city or "").lower()

    # San Diego Area
    if zip3 in range(919, 922) or zip3 == 919 or zip3 == 920 or zip3 == 921:
        return "San Diego area"
    if any(k in city for k in ["san diego", "la jolla", "chula vista", "carlsbad", "escondido", "oceanside", "encinitas", "del mar", "poway", "vista"]):
        return "San Diego area"

    # Southern California / Los Angeles area
    if zip3 is not None and (
        900 <= zip3 <= 918 or    # LA core + Pasadena region
        922 <= zip3 <= 928 or    # Inland Empire/Orange County ranges
        930 <= zip3 <= 935       # Ventura, Santa Barbara, parts of LA County North
    ):
        return "Southern California/Los Angeles area"
    if any(k in city for k in [
        "los angeles", "santa monica", "pasadena", "burbank", "glendale", "long beach",
        "anaheim", "irvine", "santa ana", "newport beach", "torrance", "inglewood", "fullerton"
    ]):
        return "Southern California/Los Angeles area"

    # Central California
    if zip3 is not None and (
        932 <= zip3 <= 939 or    # Kern/Tulare/Fresno/Monterey/SLO regions
        952 <= zip3 <= 953       # Stockton/Modesto corridor
    ):
        return "Central California"
    if any(k in city for k in [
        "fresno", "visalia", "bakersfield", "salinas", "monterey", "san luis obispo", "modesto", "stockton", "merced"
    ]):
        return "Central California"

    # Northern California / Bay Area (catch-all for remaining north)
    if zip3 is not None and (940 <= zip3 <= 961):
        return "Northern California/Bay Area"
    if any(k in city for k in [
        "san francisco", "oakland", "berkeley", "san jose", "palo alto", "mountain view", "redwood city", "sunnyvale",
        "santa rosa", "napa", "sacramento", "davis", "san mateo", "fremont", "walnut creek"
    ]):
        return "Northern California/Bay Area"

    return "Unknown"


# --------------------------------------------------------------------------- #
# Verification for a single bookstore                                         #
# --------------------------------------------------------------------------- #
async def verify_bookstore(
    evaluator: Evaluator,
    parent_node,
    bs: Bookstore,
    index_one_based: int
) -> None:
    # Wrapper node for this bookstore
    store_node = evaluator.add_parallel(
        id=f"bookstore_{index_one_based}",
        desc=f"Bookstore {index_one_based} (if provided) satisfies all per-bookstore criteria and required fields.",
        parent=parent_node,
        critical=False
    )

    # Critical existence: name provided
    name_node = evaluator.add_custom_node(
        result=bool(bs.name and bs.name.strip()),
        id=f"bookstore_{index_one_based}_name_provided",
        desc="Official bookstore name is provided.",
        parent=store_node,
        critical=True
    )

    # Critical: complete address fields present
    addr_complete = bool((bs.street_address and bs.street_address.strip())
                         and (bs.city and bs.city.strip())
                         and (bs.zip_code and bs.zip_code.strip()))
    complete_addr_node = evaluator.add_custom_node(
        result=addr_complete,
        id=f"bookstore_{index_one_based}_complete_address",
        desc="Complete address is provided (street address, city, ZIP code).",
        parent=store_node,
        critical=True
    )

    # Critical: web presence URL exists (official site or recognized directory)
    web_presence_exists = bool((bs.official_website and bs.official_website.strip())
                               or (bs.directory_url and bs.directory_url.strip())
                               or (len(bs.web_presence_refs) > 0))
    web_presence_node = evaluator.add_custom_node(
        result=web_presence_exists,
        id=f"bookstore_{index_one_based}_web_presence_url",
        desc="A URL is provided for either (a) the official website or (b) a recognized independent bookstore directory listing.",
        parent=store_node,
        critical=True
    )

    # Prepare common sources
    web_presence_sources = _pick_web_presence_urls(bs)

    # Critical leaf: Located in California
    located_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_located_in_california",
        desc="Bookstore has a physical location in California.",
        parent=store_node,
        critical=True
    )
    addr_str = _full_address(bs) or ""
    located_claim = f"The bookstore '{bs.name or ''}' has a physical location in California at address '{addr_str}'."
    located_sources = _unique_urls(bs.address_refs, web_presence_sources)
    await evaluator.verify(
        claim=located_claim,
        node=located_leaf,
        sources=located_sources,
        additional_instruction=(
            "Verify the store is in California (CA). Accept if the page shows 'CA' or 'California' in the address "
            "or a California ZIP code (ranges roughly 90000–96199). The location must correspond to a physical address."
        ),
        extra_prerequisites=[name_node, complete_addr_node, web_presence_node]
    )

    # Critical leaf: Independently owned (not a major chain)
    indep_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_independently_owned",
        desc="Bookstore is independently owned (not part of a major chain).",
        parent=store_node,
        critical=True
    )
    indep_claim = (
        f"The bookstore '{bs.name or ''}' is independently owned (not part of a major chain like Barnes & Noble)."
    )
    indep_sources = _unique_urls(bs.independent_refs, web_presence_sources)
    await evaluator.verify(
        claim=indep_claim,
        node=indep_leaf,
        sources=indep_sources,
        additional_instruction=(
            "Support can include statements such as 'independent', 'locally owned', 'family-owned', "
            "or a listing on recognized independent bookstore directories (e.g., IndieBound/ABA, Bookshop.org). "
            "If the evidence suggests a corporate chain (e.g., Barnes & Noble, Books-A-Million), mark incorrect."
        ),
        extra_prerequisites=[name_node, web_presence_node]
    )

    # Critical leaf: Operational in 2024
    op2024_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_operational_in_2024",
        desc="Evidence supports that the bookstore was operational in 2024.",
        parent=store_node,
        critical=True
    )
    op2024_claim = (
        f"The bookstore '{bs.name or ''}' was operational in 2024, evidenced by posts/pages in 2024 such as hours or events."
    )
    op2024_sources = _unique_urls(bs.operational_2024_refs, bs.events_refs, bs.hours_refs, web_presence_sources)
    await evaluator.verify(
        claim=op2024_claim,
        node=op2024_leaf,
        sources=op2024_sources,
        additional_instruction=(
            "Look for clear 2024 evidence (e.g., an events calendar showing 2024 dates, a 2024 blog/news post about store "
            "operations, or a 2024-dated hours/holiday schedule). If nothing indicates 2024, mark incorrect."
        ),
        extra_prerequisites=[name_node, web_presence_node]
    )

    # Critical leaf: Contact info (phone or email) publicly available
    contact_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_contact_info",
        desc="Public contact information is provided (phone number or email address).",
        parent=store_node,
        critical=True
    )
    contact_desc = []
    if bs.phone:
        contact_desc.append(f"phone '{bs.phone}'")
    if bs.email:
        contact_desc.append(f"email '{bs.email}'")
    contact_text = " and ".join(contact_desc) if contact_desc else "contact information"
    contact_claim = f"The bookstore '{bs.name or ''}' provides public {contact_text}."
    contact_sources = _unique_urls(bs.contact_refs, web_presence_sources)
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=contact_sources,
        additional_instruction=(
            "Accept if at least one contact method (phone or email) is clearly visible on the provided page(s). "
            "Minor formatting differences (e.g., (xxx) xxx-xxxx vs xxx-xxx-xxxx) are acceptable."
        ),
        extra_prerequisites=[name_node, web_presence_node]
    )

    # Critical leaf: Operating hours publicly available
    hours_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_operating_hours",
        desc="Public operating hours information is provided.",
        parent=store_node,
        critical=True
    )
    hours_claim = (
        f"The bookstore '{bs.name or ''}' publishes public operating hours on the referenced page(s)."
    )
    hours_sources = _unique_urls(bs.hours_refs, web_presence_sources)
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=hours_sources,
        additional_instruction=(
            "You only need to verify that hours are published (e.g., a weekly schedule or 'Hours' section). "
            "An exact textual match to the extracted hours string is not required."
        ),
        extra_prerequisites=[name_node, web_presence_node]
    )

    # Critical leaf: Events capability evidence
    events_leaf = evaluator.add_leaf(
        id=f"bookstore_{index_one_based}_events_capability_evidence",
        desc="Evidence is provided of capability to host author events or community programming (e.g., event listings/calendar/history).",
        parent=store_node,
        critical=True
    )
    events_claim = (
        f"The bookstore '{bs.name or ''}' hosts or can host author events/community programming, "
        f"as evidenced by event listings, calendars, or past event pages."
    )
    events_sources = _unique_urls(bs.events_refs, web_presence_sources)
    await evaluator.verify(
        claim=events_claim,
        node=events_leaf,
        sources=events_sources,
        additional_instruction=(
            "Accept if the referenced page shows upcoming/past events, an events calendar, author readings, book clubs, or similar community programming. "
            "External event listings (e.g., Eventbrite/Facebook) are acceptable if clearly tied to the bookstore."
        ),
        extra_prerequisites=[name_node, web_presence_node]
    )

    # References subtree (critical as a group)
    refs_node = evaluator.add_parallel(
        id=f"bookstore_{index_one_based}_references",
        desc="Reference URLs are provided that verify each required attribute for this bookstore.",
        parent=store_node,
        critical=True
    )

    # Reference existence checks (each critical)
    evaluator.add_custom_node(
        result=len(bs.address_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_address",
        desc="At least one reference URL verifies the bookstore's address.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(bs.contact_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_contact",
        desc="At least one reference URL verifies the bookstore's contact information.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(bs.hours_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_hours",
        desc="At least one reference URL verifies the bookstore's operating hours.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(bs.events_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_events",
        desc="At least one reference URL verifies the bookstore's events/community programming capability.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(bs.operational_2024_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_operational_2024",
        desc="At least one reference URL supports the claim the bookstore was operational in 2024.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(bs.independent_refs) > 0,
        id=f"bookstore_{index_one_based}_reference_for_independent_ownership",
        desc="At least one reference URL supports the claim the bookstore is independently owned / not a major chain.",
        parent=refs_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(web_presence_sources),
        id=f"bookstore_{index_one_based}_reference_for_web_presence",
        desc="At least one reference URL supports the official website or recognized directory listing used.",
        parent=refs_node,
        critical=True
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel; keep non-critical to allow partial credit
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
    # Enforce root non-critical to satisfy framework's critical-children constraint
    root.critical = False

    # Extract structured bookstores
    extracted = await evaluator.extract(
        prompt=prompt_extract_bookstores(),
        template_class=BookstoresExtraction,
        extraction_name="bookstores_extraction"
    )

    # Keep at most 5 bookstores from the answer (the task asks for min 4; 5th optional)
    bookstores: List[Bookstore] = extracted.bookstores[:5] if extracted.bookstores else []

    # Critical: At least four bookstores provided (by name)
    num_with_names = sum(1 for b in bookstores if b.name and b.name.strip())
    evaluator.add_custom_node(
        result=(num_with_names >= 4),
        id="minimum_four_bookstores",
        desc="At least four bookstores are provided in the answer.",
        parent=root,
        critical=True
    )

    # Build per-bookstore verification nodes for up to 5 bookstores
    for i, bs in enumerate(bookstores, start=1):
        await verify_bookstore(evaluator, root, bs, i)

    # Regional diversity (critical): at least 3 distinct regions among selected bookstores
    # Use extracted 'region' if provided; else derive via ZIP/city heuristics.
    region_assignments: List[Tuple[str, str]] = []
    distinct_regions: set = set()
    for b in bookstores:
        if not (b and b.name and b.name.strip()):
            continue
        region = b.region.strip() if (b.region and b.region.strip()) else _categorize_region(b)
        if region and region.lower() != "unknown":
            distinct_regions.add(region)
        region_assignments.append((b.name or "Unknown name", region or "Unknown"))

    evaluator.add_custom_node(
        result=(len(distinct_regions) >= 3),
        id="regional_diversity",
        desc="The selected bookstores collectively represent at least three different California regions (regions may be evidenced by addresses and/or explicitly stated).",
        parent=root,
        critical=True
    )

    # Record auxiliary info for transparency
    evaluator.add_custom_info(
        info={
            "recognized_directory_domains": RECOGNIZED_DIRECTORY_DOMAINS,
            "num_bookstores_extracted": len(bookstores),
            "num_with_names": num_with_names,
            "region_assignments": region_assignments,
            "distinct_regions_count": len(distinct_regions),
            "distinct_regions": sorted(list(distinct_regions))
        },
        info_type="auxiliary",
        info_name="regional_and_directory_info"
    )

    return evaluator.get_summary()