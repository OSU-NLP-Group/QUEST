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
TASK_ID = "ca_indie_bookstores_events_2026"
TASK_DESCRIPTION = """
I'm planning a book tour across California for an upcoming fiction release in Spring 2026. I need to find three independent bookstores in California that can host author events, with each bookstore located in a different city. For each bookstore, provide the following information:

1. The bookstore's official name
2. Complete physical address (street address, city, and state)
3. Confirmation that it is an independent bookstore (not part of a national chain like Barnes & Noble)
4. Verification that it is listed in at least one recognized independent bookstore directory (such as IndieBound.org, CALIBA, or similar)
5. Confirmation that the bookstore hosts author events or has dedicated event space
6. Event space capacity, confirming it can accommodate at least 30 attendees
7. Contact information for event booking (email and/or phone number)
8. A reference URL to the bookstore's website or a verified directory listing

All bookstores must be currently operational as of March 2026, and no two bookstores should be located in the same city.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BookstoreItem(BaseModel):
    """Information about one bookstore as extracted from the answer."""
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    operational_status: Optional[str] = None  # e.g., "open", "operational", or a statement indicating current operation

    # Independence evidence
    independent_note: Optional[str] = None  # any textual confirmation in the answer
    directory_urls: List[str] = Field(default_factory=list)  # recognized directories (e.g., indiebound.org, caliba.org, bookweb.org)

    # Event capability
    hosts_author_events: Optional[str] = None  # textual confirmation or note
    event_capacity: Optional[str] = None  # number or range text like "40", "30+", "approx. 35"

    # Contacts
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None

    # References
    website_url: Optional[str] = None
    other_reference_urls: List[str] = Field(default_factory=list)


class BookstoresExtraction(BaseModel):
    """Top-level list of up to three bookstores extracted from the answer."""
    bookstores: List[BookstoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_bookstores() -> str:
    return """
Extract information for up to three independent bookstores in California described in the answer. Return at most three entries in the order they appear. For each bookstore, extract these exact fields:

- name: Official bookstore name as written in the answer.
- street: Street address (just the street line, no city/state).
- city: City name.
- state: State text exactly as shown (e.g., "CA", "California").
- operational_status: A short phrase from the answer indicating the store is currently open/operational as of March 2026; if not clearly stated, return null.
- independent_note: Any text in the answer explicitly confirming it is independent or not a chain; if absent, return null.
- directory_urls: Array of URLs to recognized independent bookstore directories mentioned in the answer (e.g., indiebound.org, caliba.org, bookweb.org/aba membership, or similar). Only include valid URLs explicitly present in the answer. If none provided, return an empty array.
- hosts_author_events: A short phrase from the answer indicating the store hosts author events or has an events space; if not clearly stated, return null.
- event_capacity: The stated or implied capacity from the answer (e.g., "30", "30+", "seats 40"); if not provided, return null.
- contact_email: A booking or general contact email from the answer, if provided; else null.
- contact_phone: A booking or store phone number from the answer, if provided; else null.
- website_url: The store’s official website URL if present in the answer; else null.
- other_reference_urls: Any additional relevant reference URLs from the answer (e.g., an events page URL, contact page URL, or other verified listing pages). Only include valid URLs present in the answer. If none, return an empty array.

Rules:
- Extract only what is explicitly present in the answer. Do not invent or infer.
- Ensure each URL is a valid absolute URL; if protocol is missing, prepend http://.
- If more than three bookstores are mentioned, include only the first three.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _collect_sources(store: BookstoreItem) -> List[str]:
    """Collect and de-duplicate all possible verification sources for a bookstore."""
    urls: List[str] = []
    if store.website_url:
        urls.append(store.website_url)
    urls.extend(store.directory_urls or [])
    urls.extend(store.other_reference_urls or [])

    # Simple de-duplication while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _norm_city(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    return city.strip().lower()


def _has_valid_directory(store: BookstoreItem) -> bool:
    """Check if at least one directory URL is provided."""
    return bool(store.directory_urls and len(store.directory_urls) > 0)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_bookstore(
    evaluator: Evaluator,
    parent_node,
    store: BookstoreItem,
    index: int,
    all_cities: List[Optional[str]],
) -> None:
    """
    Build verification tree and run checks for a single bookstore.
    Follows the rubric structure with critical/non-critical and parallel aggregations.
    """
    bs_idx = index + 1
    bs_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}",
        desc=f"{['First','Second','Third'][index]} independent bookstore meeting all requirements",
        parent=parent_node,
        critical=False
    )

    sources_all = _collect_sources(store)
    dir_sources = list(store.directory_urls or [])
    primary_ref = store.website_url or (dir_sources[0] if dir_sources else None)

    # ---------------- Identity (critical, parallel) ---------------- #
    identity_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Identity",
        desc="Bookstore identity and operational status",
        parent=bs_node,
        critical=True
    )

    # Bookstore_Name (critical, existence check)
    evaluator.add_custom_node(
        result=bool(store.name and store.name.strip()),
        id=f"Bookstore_{bs_idx}_Bookstore_Name",
        desc="Official name of the bookstore is provided",
        parent=identity_node,
        critical=True
    )

    # Operating_Status (critical, evidence-based)
    if sources_all:
        op_node = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Operating_Status",
            desc="Bookstore is currently operational as of March 2026",
            parent=identity_node,
            critical=True
        )
        claim = "As of March 2026, the bookstore appears to be currently operating (open to the public)."
        await evaluator.verify(
            claim=claim,
            node=op_node,
            sources=sources_all,
            additional_instruction="Use signals like an active website, current events calendar, recent posts or announcements on the site, hours pages, or updated directory listings to confirm current operation."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Operating_Status",
            desc="Bookstore is currently operational as of March 2026",
            parent=identity_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # ---------------- Location (critical, parallel) ---------------- #
    location_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Location",
        desc="Physical location information and verification",
        parent=bs_node,
        critical=True
    )

    # Address_Details (critical, parallel)
    addr_details_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Address_Details",
        desc="Complete address components",
        parent=location_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(store.street and store.street.strip()),
        id=f"Bookstore_{bs_idx}_Street_Address",
        desc="Complete street address is provided",
        parent=addr_details_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(store.city and store.city.strip()),
        id=f"Bookstore_{bs_idx}_City_Name",
        desc="City name is provided",
        parent=addr_details_node,
        critical=True
    )

    # State_California (critical - simple check on provided state text)
    state_text = (store.state or "").strip()
    if state_text:
        state_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_State_California",
            desc="State is confirmed as California",
            parent=addr_details_node,
            critical=True
        )
        claim = f"The provided state value '{state_text}' denotes California (accept 'CA', 'California', or reasonable abbreviations)."
        await evaluator.verify(
            claim=claim,
            node=state_leaf,
            additional_instruction="Treat case-insensitive 'CA', 'Calif.', and 'California' as equivalent."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_State_California",
            desc="State is confirmed as California",
            parent=addr_details_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Location_Validity (critical, parallel)
    loc_valid_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Location_Validity",
        desc="Location meets geographic constraints",
        parent=location_node,
        critical=True
    )

    # City_In_California (critical, evidence-based)
    if store.city and sources_all:
        city_ca_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_City_In_California",
            desc="City is located within California",
            parent=loc_valid_node,
            critical=True
        )
        claim = f"The bookstore is located in {store.city}, California."
        await evaluator.verify(
            claim=claim,
            node=city_ca_leaf,
            sources=sources_all,
            additional_instruction="Check the address line(s) on the provided URL(s) to confirm city and state within California."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_City_In_California",
            desc="City is located within California",
            parent=loc_valid_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # City_Uniqueness (critical, cross-item check)
    this_city_norm = _norm_city(store.city)
    others = [_norm_city(c) for i, c in enumerate(all_cities) if i != index and c]
    is_unique_city = bool(this_city_norm) and all(this_city_norm != oc for oc in others if oc)
    evaluator.add_custom_node(
        result=is_unique_city,
        id=f"Bookstore_{bs_idx}_City_Uniqueness",
        desc="City is different from the other two bookstores",
        parent=loc_valid_node,
        critical=True
    )

    # ---------------- Independence (critical, parallel) --------------- #
    indep_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Independence",
        desc="Verification of independent bookstore status",
        parent=bs_node,
        critical=True
    )

    # Independent_Status (critical, parallel)
    indep_status_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Independent_Status",
        desc="Bookstore is independently owned",
        parent=indep_node,
        critical=True
    )

    # Not_Chain_Store (critical, evidence-based)
    if sources_all:
        not_chain_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Not_Chain_Store",
            desc="Bookstore is not part of a national chain (e.g., not Barnes & Noble)",
            parent=indep_status_node,
            critical=True
        )
        claim = "This bookstore is independently owned and is not part of a national chain (e.g., Barnes & Noble, Books-A-Million)."
        await evaluator.verify(
            claim=claim,
            node=not_chain_leaf,
            sources=sources_all,
            additional_instruction="Accept strong evidence like membership/listing in recognized independent bookstore organizations (IndieBound/ABA, CALIBA) or 'independent' language on the official site."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Not_Chain_Store",
            desc="Bookstore is not part of a national chain (e.g., not Barnes & Noble)",
            parent=indep_status_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Directory_Verification (critical, parallel)
    dir_verif_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Directory_Verification",
        desc="Independent status verified through directory listing",
        parent=indep_node,
        critical=True
    )

    # Listed_In_Directory (critical, evidence-based, require directory url)
    if _has_valid_directory(store):
        listed_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Listed_In_Directory",
            desc="Bookstore is listed in at least one recognized independent bookstore directory (IndieBound, CALIBA, or similar)",
            parent=dir_verif_node,
            critical=True
        )
        claim = "The bookstore is listed on a recognized independent bookstore directory (e.g., IndieBound/ABA, CALIBA, or similar)."
        await evaluator.verify(
            claim=claim,
            node=listed_leaf,
            sources=dir_sources,
            additional_instruction="Verify that at least one provided URL is a listing page on a recognized independent bookstore directory, such as indiebound.org, bookweb.org (ABA membership), caliba.org, or a regional indie alliance."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Listed_In_Directory",
            desc="Bookstore is listed in at least one recognized independent bookstore directory (IndieBound, CALIBA, or similar)",
            parent=dir_verif_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # ---------------- Event_Capability (critical, parallel) ------------ #
    event_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Event_Capability",
        desc="Capacity to host author events",
        parent=bs_node,
        critical=True
    )

    # Event_Hosting (critical, parallel)
    hosting_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Event_Hosting",
        desc="Bookstore hosts or can host author events",
        parent=event_node,
        critical=True
    )

    # Has_Event_Space (critical, evidence-based)
    if sources_all:
        event_space_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Has_Event_Space",
            desc="Bookstore hosts author events or has dedicated event space",
            parent=hosting_node,
            critical=True
        )
        claim = "The bookstore hosts author events (e.g., readings, signings) or has a dedicated event space."
        await evaluator.verify(
            claim=claim,
            node=event_space_leaf,
            sources=sources_all,
            additional_instruction="Look for an events page, calendar, past event announcements, or mentions of an events area. Wording like 'author events', 'readings', 'signings' suffices."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Has_Event_Space",
            desc="Bookstore hosts author events or has dedicated event space",
            parent=hosting_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Event_Capacity (critical, parallel)
    capacity_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Event_Capacity",
        desc="Event space meets minimum capacity requirement",
        parent=event_node,
        critical=True
    )

    # Minimum_30_Attendees (critical, evidence-based)
    if sources_all:
        cap_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Minimum_30_Attendees",
            desc="Event space can accommodate at least 30 attendees",
            parent=capacity_node,
            critical=True
        )
        claim = "The bookstore's event space can accommodate at least 30 attendees."
        await evaluator.verify(
            claim=claim,
            node=cap_leaf,
            sources=sources_all,
            additional_instruction="Verify explicit capacity numbers or phrases like '30+' or 'seats about 40'. If multiple rooms, any room with capacity >= 30 counts."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Minimum_30_Attendees",
            desc="Event space can accommodate at least 30 attendees",
            parent=capacity_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Contact_Information (critical, evidence-based)
    if sources_all:
        contact_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Contact_Information",
            desc="Contact information (phone and/or email) for event booking is provided",
            parent=event_node,
            critical=True
        )
        claim = "There is a contact email address and/or phone number available for event booking or general inquiries."
        await evaluator.verify(
            claim=claim,
            node=contact_leaf,
            sources=sources_all,
            additional_instruction="Accept event-specific booking contacts or general store contacts (email/phone) visible on Contact or Events pages. A contact form alone is insufficient unless it’s clearly for booking."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Contact_Information",
            desc="Contact information (phone and/or email) for event booking is provided",
            parent=event_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # ---------------- Reference_Documentation (critical, parallel) ----- #
    ref_node = evaluator.add_parallel(
        id=f"Bookstore_{bs_idx}_Reference_Documentation",
        desc="Valid reference URL provided for verification",
        parent=bs_node,
        critical=True
    )

    # Reference_URL (critical, evidence-based) — must reference at least one concrete page
    if primary_ref:
        ref_leaf = evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Reference_URL",
            desc="Valid reference URL to bookstore's website or verified directory listing is provided",
            parent=ref_node,
            critical=True
        )
        # Try to include name and city for stronger match
        name_part = store.name or "the bookstore"
        where_part = f" in {store.city}, California" if store.city else ""
        claim = f"This URL is a valid page about {name_part}{where_part} (official site or a verified independent bookstore directory listing)."
        await evaluator.verify(
            claim=claim,
            node=ref_leaf,
            sources=primary_ref,
            additional_instruction="Confirm the page content references the bookstore’s official name (allowing minor variants) and, when available, the correct city/state."
        )
    else:
        evaluator.add_leaf(
            id=f"Bookstore_{bs_idx}_Reference_URL",
            desc="Valid reference URL to bookstore's website or verified directory listing is provided",
            parent=ref_node,
            critical=True,
            score=0.0,
            status="failed"
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the California independent bookstores event-hosting task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # As per rubric: top-level parallel aggregation across bookstores
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

    # Extract bookstores structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_bookstores(),
        template_class=BookstoresExtraction,
        extraction_name="bookstores_extraction"
    )

    # Normalize to exactly 3 bookstores (pad with empty items if needed)
    bookstores: List[BookstoreItem] = list(extracted.bookstores[:3])
    while len(bookstores) < 3:
        bookstores.append(BookstoreItem())

    # Build top-level rubric node (optional grouping as per input JSON)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Find three independent bookstores in California (each in a different city) that meet all specified criteria for hosting author events",
        parent=root,
        critical=False
    )

    # Pre-compute cities for uniqueness checks
    all_cities = [b.city for b in bookstores]

    # Verify each bookstore subtree
    for idx, store in enumerate(bookstores):
        await verify_bookstore(
            evaluator=evaluator,
            parent_node=task_node,
            store=store,
            index=idx,
            all_cities=all_cities
        )

    # Optional: record a small GT/policy note to the summary
    evaluator.add_ground_truth({
        "requirements": {
            "stores_required": 3,
            "state": "California",
            "unique_cities": True,
            "independent_directory_example": ["indiebound.org", "bookweb.org (ABA)", "caliba.org"],
            "min_capacity": 30,
            "operational_as_of": "March 2026"
        }
    }, gt_type="rubric_requirements")

    return evaluator.get_summary()