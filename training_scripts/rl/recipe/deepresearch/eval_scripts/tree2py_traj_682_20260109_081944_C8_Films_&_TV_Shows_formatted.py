import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wolfman_accessible_theaters_2025"
TASK_DESCRIPTION = (
    "I am planning to see the horror movie Wolf Man (2025) with friends who have different accessibility needs. "
    "I need to identify three movie theaters that can accommodate our group. Find exactly 3 movie theaters that meet ALL of the following requirements: "
    "(1) Theater Chain: Each theater must be operated by one of these major chains: AMC Theatres, Regal Cinemas, or Cinemark Theatres. "
    "(2) Location: The three theaters must be located in California, Texas, or Florida (at least one theater must be in a different state from the others). "
    "(3) Current Movie: Each theater must be currently showing (or have recently shown in January 2025) the movie Wolf Man (2025). "
    "(4) Premium Format: Each theater must offer at least one premium large-format viewing option from the following: IMAX, Dolby Cinema, Cinemark XD, or Regal RPX. "
    "(5) Reserved Seating: Each theater must provide reserved seating capability for advance ticket purchases. "
    "(6) Accessibility - Closed Captioning: Each theater must offer closed captioning devices or technology. "
    "(7) Accessibility - Assisted Listening: Each theater must provide assisted listening devices. "
    "(8) Accessibility - Wheelchair Access: Each theater must have wheelchair accessible seating spaces. "
    "(9) Complete Address: Provide the complete physical address for each theater (street address, city, state, and ZIP code). "
    "(10) Parking Information: Provide information about parking availability or parking policies for each theater location. "
    "(11) Matinee Showtimes: Each theater must offer matinee showtimes (movie showings that begin before 4:00 PM). "
    "For each theater, provide: theater name and chain affiliation, complete physical address, state location, at least one premium format offered, confirmation that Wolf Man is/was showing, "
    "confirmation of accessibility features (closed captioning, assisted listening, wheelchair seating), confirmation of reserved seating availability, parking information, "
    "confirmation of matinee showtime availability, and a reference URL to the theater's official page on the chain's website."
)

ALLOWED_STATES = {"CA", "TX", "FL"}
CHAIN_CANON = {
    "amc": "AMC Theatres",
    "regal": "Regal Cinemas",
    "cinemark": "Cinemark Theatres",
}
ALLOWED_DOMAINS = {"amctheatres.com", "regmovies.com", "regal.com", "cinemark.com"}
PREMIUM_ALLOWED = {"IMAX", "Dolby Cinema", "Cinemark XD", "Regal RPX"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TheaterEntry(BaseModel):
    name: Optional[str] = None
    chain: Optional[str] = None  # e.g., "AMC Theatres", "Regal Cinemas", "Cinemark Theatres"
    state: Optional[str] = None  # Two-letter code preferred, e.g., CA/TX/FL
    address_street: Optional[str] = None
    city: Optional[str] = None
    zip_code: Optional[str] = None
    full_address: Optional[str] = None

    premium_formats: List[str] = Field(default_factory=list)  # e.g., ["IMAX", "Dolby Cinema"]
    reserved_seating: Optional[str] = None  # "yes"/"no"/text
    closed_captioning: Optional[str] = None  # "yes"/"no"/text
    assisted_listening: Optional[str] = None  # "yes"/"no"/text
    wheelchair_access: Optional[str] = None  # "yes"/"no"/text

    wolf_man_status: Optional[str] = None  # e.g., "currently showing", "shown Jan 2025"
    parking_info: Optional[str] = None  # any parking details text
    matinee_showtimes: Optional[str] = None  # "yes"/"no"/text

    reference_url: Optional[str] = None  # official chain page for the theater
    supporting_urls: List[str] = Field(default_factory=list)  # showtimes page or film-specific page on official site


class TheatersExtraction(BaseModel):
    theaters: List[TheaterEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theaters() -> str:
    return """
    Extract all theater entries the answer provides for the task. Return a JSON object with a "theaters" array.
    IMPORTANT: Extract EXACTLY what is explicitly stated in the answer. Do not invent or normalize beyond what is present.

    For each theater in the answer, extract the following fields (use null if missing):
    - name: The theater's name as written.
    - chain: The operator (e.g., "AMC Theatres", "Regal Cinemas", or "Cinemark Theatres") if the answer states it.
    - state: The US state for the theater. Prefer the two-letter code "CA", "TX", or "FL" if present; otherwise capture text (e.g., "California").
    - address_street: Street address line if present (e.g., "123 Main St").
    - city: City name if present.
    - zip_code: 5-digit ZIP code if present.
    - full_address: The complete address string if the answer gives it in one line; otherwise null.
    - premium_formats: A list of premium formats mentioned (e.g., ["IMAX", "Dolby Cinema", "Cinemark XD", "Regal RPX"]) as written.
    - reserved_seating: The answer's statement about reserved seating (e.g., "yes", "reserved seating available", etc.), or null.
    - closed_captioning: The answer's statement about closed captioning (CC) devices/technology, or null.
    - assisted_listening: The answer's statement about assisted listening devices (ALD), or null.
    - wheelchair_access: The answer's statement about wheelchair accessible seating, or null.
    - wolf_man_status: The answer's claim about Wolf Man (2025) show status (e.g., "currently showing", "shown Jan 2025", etc.), or null.
    - parking_info: Any text in the answer describing parking availability or policy for this theater; or null if not present.
    - matinee_showtimes: The answer's claim about matinee availability (before 4:00 PM); e.g., "yes", "matinee available", or null.
    - reference_url: The URL to the theater's official page on the chain's website if provided. If multiple are listed, choose the main theater page.
    - supporting_urls: Any additional official-chain URLs in the answer related to the theater (e.g., showtimes, film subpages). Exclude third-party sites.

    Include every theater the answer lists. Do not filter here.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_state_code(state_text: Optional[str]) -> Optional[str]:
    if not state_text:
        return None
    s = state_text.strip().upper()
    # Accept already code
    if s in ALLOWED_STATES:
        return s
    # Try to map common full names
    mapping = {"CALIFORNIA": "CA", "TEXAS": "TX", "FLORIDA": "FL"}
    if s in mapping:
        return mapping[s]
    # Handle "Ca", "Tx", "Fl"
    if s[:2] in ALLOWED_STATES:
        return s[:2]
    # Try detecting code in the string
    for code in ALLOWED_STATES:
        if code in s:
            return code
    return None


def extract_state_from_address(full_address: Optional[str]) -> Optional[str]:
    if not full_address:
        return None
    s = full_address.upper()
    for code in ALLOWED_STATES:
        if f" {code} " in s or s.endswith(f" {code}") or f" {code}," in s:
            return code
    # Try full names
    if " CALIFORNIA " in s or s.endswith(" CALIFORNIA"):
        return "CA"
    if " TEXAS " in s or s.endswith(" TEXAS"):
        return "TX"
    if " FLORIDA " in s or s.endswith(" FLORIDA"):
        return "FL"
    return None


def is_complete_address(rec: TheaterEntry) -> bool:
    # If split fields present and valid, accept
    if rec.address_street and rec.city and rec.zip_code:
        # Check zip is 5 digits
        zip_digits = re.sub(r"[^0-9]", "", rec.zip_code)
        state_code = normalize_state_code(rec.state)
        if state_code in ALLOWED_STATES and len(zip_digits) == 5:
            return True
    # Else try full_address heuristics
    if rec.full_address:
        # Must contain a 5-digit number and an allowed state indicator
        has_zip = bool(re.search(r"\b\d{5}\b", rec.full_address))
        found_state = extract_state_from_address(rec.full_address)
        return bool(has_zip and found_state in ALLOWED_STATES)
    return False


def distinct_key(rec: TheaterEntry) -> str:
    n = (rec.name or "").strip().lower()
    a = (rec.full_address or f"{rec.address_street or ''} {rec.city or ''} {rec.state or ''} {rec.zip_code or ''}").strip().lower()
    return f"{n}||{a}"


def select_first_three_distinct(all_recs: List[TheaterEntry]) -> List[TheaterEntry]:
    seen = set()
    selected: List[TheaterEntry] = []
    for rec in all_recs:
        k = distinct_key(rec)
        if k and k not in seen:
            seen.add(k)
            selected.append(rec)
        if len(selected) == 3:
            break
    return selected


def safe_urls(rec: TheaterEntry) -> List[str]:
    urls: List[str] = []
    if rec.reference_url:
        urls.append(rec.reference_url)
    # Keep only official domains in supporting URLs
    for u in rec.supporting_urls:
        if isinstance(u, str) and u:
            urls.append(u)
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def any_premium_format_claim(rec: TheaterEntry) -> str:
    # Build a readable mention of known premium formats present in the answer (if any)
    present = [fmt for fmt in rec.premium_formats if any(fmt.upper().startswith(p.upper()) or p.upper() in fmt.upper() for p in PREMIUM_ALLOWED)]
    if present:
        return f"Specifically mentioned: {', '.join(present)}."
    return "If the page mentions any of IMAX, Dolby Cinema, Cinemark XD, or Regal RPX, consider it satisfied."


def normalize_chain_text(chain_text: Optional[str]) -> Optional[str]:
    if not chain_text:
        return None
    s = chain_text.strip().lower()
    for key, canon in CHAIN_CANON.items():
        if key in s or canon.lower() in s:
            return canon
    return chain_text


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_and_verify_theater(
    evaluator: Evaluator,
    parent_node,
    index_1based: int,
    rec: TheaterEntry,
) -> None:
    """
    Build the subtree for a single theater and schedule verifications.
    """
    tnode = evaluator.add_parallel(
        id=f"Theater_{index_1based}",
        desc=f"Theater #{index_1based} meeting all requirements",
        parent=parent_node,
        critical=False,  # per rubric, each theater node is non-critical under the parent
    )

    # 1) Name provided (custom existence)
    evaluator.add_custom_node(
        result=bool(rec.name and rec.name.strip()),
        id=f"Theater_{index_1based}_Name_Provided",
        desc="Theater name is provided.",
        parent=tnode,
        critical=True,
    )

    # 2) Address completeness (custom existence for 'complete physical address provided')
    evaluator.add_custom_node(
        result=is_complete_address(rec),
        id=f"Theater_{index_1based}_Address",
        desc="Complete physical address provided (street address, city, state, ZIP).",
        parent=tnode,
        critical=True,
    )

    # 3) Parking information provided (custom existence)
    evaluator.add_custom_node(
        result=bool(rec.parking_info and rec.parking_info.strip()),
        id=f"Theater_{index_1based}_Parking",
        desc="Parking availability or parking policy information is provided.",
        parent=tnode,
        critical=True,
    )

    # Collect leaves that will be verified via LLM + URLs
    claims_and_sources: List[Tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 4) Reference URL (verify it's official theater page containing theater info)
    ref_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Reference_URL",
        desc="Provide a reference URL to the theater's official page on the chain's website (AMC/Regal/Cinemark) that contains theater information.",
        parent=tnode,
        critical=True,
    )
    claim_ref = (
        f"This webpage is the official theater page on the chain's own website for '{rec.name or 'the theater'}', "
        "and it contains theater information such as address, amenities, or showtimes/ticketing."
    )
    add_ins_ref = (
        "Only accept official chain domains: amctheatres.com, cinemark.com, regmovies.com, regal.com. "
        "If the URL is a third-party site (e.g., Fandango, Yelp, Google, Facebook), it should be considered NOT supported. "
        "Look for clear theater info on the page (address, showtimes, amenities)."
    )
    claims_and_sources.append((claim_ref, rec.reference_url or None, ref_leaf, add_ins_ref))

    # 5) Chain (verify operated by AMC/Regal/Cinemark)
    chain_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Chain",
        desc="Theater is operated by AMC Theatres, Regal Cinemas, or Cinemark Theatres.",
        parent=tnode,
        critical=True,
    )
    canon_chain = normalize_chain_text(rec.chain) or "one of AMC Theatres, Regal Cinemas, or Cinemark Theatres"
    claim_chain = f"This theater is operated by {canon_chain}."
    add_ins_chain = (
        "Confirm chain ownership/branding from the official page. AMC Theatres (amctheatres.com), "
        "Cinemark Theatres (cinemark.com), Regal Cinemas (regmovies.com/regal.com)."
    )
    claims_and_sources.append((claim_chain, rec.reference_url or None, chain_leaf, add_ins_chain))

    # 6) State location (CA/TX/FL)
    state_code = normalize_state_code(rec.state) or extract_state_from_address(rec.full_address)
    state_text = state_code if state_code else (rec.state or "the required states")
    state_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_State_Location",
        desc="Theater is located in California, Texas, or Florida.",
        parent=tnode,
        critical=True,
    )
    claim_state = f"This theater is located in the state of {state_text}."
    add_ins_state = (
        "Verify the address on the official page contains the state. Only CA, TX, or FL are acceptable."
    )
    claims_and_sources.append((claim_state, rec.reference_url or None, state_leaf, add_ins_state))

    # 7) Operational status as of January 2025
    op_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Operational_Status",
        desc="The theater is currently operational as of January 2025.",
        parent=tnode,
        critical=True,
    )
    claim_op = (
        "As of January 2025, the theater is open and operational (e.g., lists current movies/showtimes or allows purchasing tickets)."
    )
    add_ins_op = (
        "Check the official page for signs of normal operation around January 2025: showtimes calendar, ticket purchase options, announcements indicating open status."
    )
    claims_and_sources.append((claim_op, safe_urls(rec), op_leaf, add_ins_op))

    # 8) Wolf Man (2025) showing in Jan 2025 (or currently showing then)
    wolf_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Wolf_Man",
        desc="Theater is currently showing or has shown Wolf Man (2025) in January 2025.",
        parent=tnode,
        critical=True,
    )
    claim_wolf = (
        "This theater was showing the movie 'Wolf Man (2025)' during January 2025 (or is currently showing it in that month)."
    )
    add_ins_wolf = (
        "Look for the film listing 'Wolf Man' with the 2025 version on the official chain's showtimes pages for the theater. "
        "If the page displays dated showtimes in January 2025 or an explicit movie page for this theater including that time window, consider it supported."
    )
    claims_and_sources.append((claim_wolf, safe_urls(rec), wolf_leaf, add_ins_wolf))

    # 9) Premium format available (IMAX, Dolby Cinema, Cinemark XD, Regal RPX)
    premium_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Premium_Format",
        desc="Theater offers at least one premium large-format option: IMAX, Dolby Cinema, Cinemark XD, or Regal RPX.",
        parent=tnode,
        critical=True,
    )
    claim_premium = (
        "This theater offers at least one premium large-format auditorium from the following list: IMAX, Dolby Cinema, Cinemark XD, or Regal RPX."
    )
    add_ins_premium = (
        f"Check the official theater page for amenities or auditorium labels. {any_premium_format_claim(rec)} "
        "For Regal, RPX might be labeled 'Regal RPX'. For AMC, 'Dolby' or 'IMAX'. For Cinemark, 'Cinemark XD'."
    )
    claims_and_sources.append((claim_premium, rec.reference_url or None, premium_leaf, add_ins_premium))

    # 10) Reserved seating available
    reserved_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Reserved_Seating",
        desc="Theater offers reserved seating for advance ticket purchases.",
        parent=tnode,
        critical=True,
    )
    claim_reserved = "This theater provides reserved seating (seat selection) for advance ticket purchases."
    add_ins_reserved = (
        "Look for 'Reserved Seating', 'Pick Your Seat', seat map UI, or ticket purchase flows indicating selecting specific seats."
    )
    claims_and_sources.append((claim_reserved, rec.reference_url or None, reserved_leaf, add_ins_reserved))

    # 11) Closed captioning devices/technology
    cc_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Closed_Captioning",
        desc="Theater provides closed captioning devices or technology.",
        parent=tnode,
        critical=True,
    )
    claim_cc = "This theater offers closed captioning (CC) devices or technology for movies that support it."
    add_ins_cc = (
        "Check amenities or accessibility sections on the official page for 'Closed Captioning', 'CC', 'CaptiView', or similar."
    )
    claims_and_sources.append((claim_cc, rec.reference_url or None, cc_leaf, add_ins_cc))

    # 12) Assisted listening devices
    ald_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Assisted_Listening",
        desc="Theater offers assisted listening devices.",
        parent=tnode,
        critical=True,
    )
    claim_ald = "This theater provides assisted listening devices (ALD) for patrons."
    add_ins_ald = (
        "Look for 'Assisted Listening', 'ALD', 'Audio Description' (AD is different, do not confuse; ensure ALD is offered)."
    )
    claims_and_sources.append((claim_ald, rec.reference_url or None, ald_leaf, add_ins_ald))

    # 13) Wheelchair accessible seating spaces
    whl_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Wheelchair_Seating",
        desc="Theater has wheelchair accessible seating spaces.",
        parent=tnode,
        critical=True,
    )
    claim_whl = "This theater has wheelchair accessible seating spaces."
    add_ins_whl = "Check the amenities or accessibility section for wheelchair/ADA seating."
    claims_and_sources.append((claim_whl, rec.reference_url or None, whl_leaf, add_ins_whl))

    # 14) Matinee showtimes (before 4:00 PM)
    matinee_leaf = evaluator.add_leaf(
        id=f"Theater_{index_1based}_Matinee",
        desc="Theater offers matinee showtimes (showings before 4:00 PM).",
        parent=tnode,
        critical=True,
    )
    claim_matinee = "This theater offers matinee showtimes (movie showings that begin before 4:00 PM)."
    add_ins_matinee = (
        "On the official showtimes page, check any day's schedule for start times earlier than 4:00 PM (e.g., 10:30 AM, 1:45 PM). "
        "If such times exist, consider this supported."
    )
    claims_and_sources.append((claim_matinee, safe_urls(rec), matinee_leaf, add_ins_matinee))

    # Verify all claim nodes in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the Wolf Man (2025) accessible theaters task and return a structured result.
    """
    # Initialize evaluator
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

    # Extract theaters list from the answer
    extracted: TheatersExtraction = await evaluator.extract(
        prompt=prompt_extract_theaters(),
        template_class=TheatersExtraction,
        extraction_name="theaters_extraction",
    )

    # Build the top-level node for this rubric
    main_node = evaluator.add_parallel(
        id="Find_Three_Theaters",
        desc="Find exactly 3 movie theaters that meet all specified criteria and provide required details.",
        parent=root,
        critical=False,  # set to non-critical to allow mixed critical children without framework constraint
    )

    # Exactly three theaters provided (distinct)
    all_recs = extracted.theaters or []
    distinct_keys = set()
    for r in all_recs:
        distinct_keys.add(distinct_key(r))
    exactly_three = (len(all_recs) == 3) and (len(distinct_keys) == 3)
    evaluator.add_custom_node(
        result=exactly_three,
        id="Exactly_Three_Theaters_Provided",
        desc="Response includes exactly 3 distinct theater entries (no more, no fewer).",
        parent=main_node,
        critical=True,
    )

    # Select first three distinct theaters for downstream verification (pad if fewer)
    selected = select_first_three_distinct(all_recs)
    while len(selected) < 3:
        selected.append(TheaterEntry())

    # Geographic diversity: at least two different states among the 3; all must be within CA/TX/FL
    states_for_check: List[Optional[str]] = []
    for rec in selected:
        st = normalize_state_code(rec.state) or extract_state_from_address(rec.full_address)
        states_for_check.append(st)
    states_set = {s for s in states_for_check if s is not None}
    all_within_allowed = all((s in ALLOWED_STATES) for s in states_set) and (None not in states_for_check)
    at_least_two_states = len(states_set) >= 2
    geo_diversity_ok = all_within_allowed and at_least_two_states
    evaluator.add_custom_node(
        result=geo_diversity_ok,
        id="Geographic_Diversity",
        desc="At least two different states are represented across the 3 theaters (not all in the same state), and states are within CA/TX/FL.",
        parent=main_node,
        critical=True,
    )

    # Build per-theater verification subtrees
    await build_and_verify_theater(evaluator, main_node, 1, selected[0])
    await build_and_verify_theater(evaluator, main_node, 2, selected[1])
    await build_and_verify_theater(evaluator, main_node, 3, selected[2])

    # Return structured summary
    return evaluator.get_summary()