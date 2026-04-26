import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_universities_football_research"
TASK_DESCRIPTION = (
    "A high school student-athlete is researching Big Ten Conference universities to compare athletic and academic opportunities. "
    "The student wants to understand the scale of football programs, campus facilities, and available academic support. Identify four Big Ten Conference universities with football programs and provide the following information for each:\n\n"
    "1. The university name and a URL to its official athletics website or football program page\n"
    "2. Confirmation that the university is a member of the Big Ten Conference\n"
    "3. Confirmation that the football program competes at the NCAA Division I FBS level\n"
    "4. The official seating capacity of the football stadium\n"
    "5. The complete physical address of the football stadium (including street address, city, state, and ZIP code)\n"
    "6. Contact information or an official URL for the university's academic support services for student-athletes (such as tutoring centers, academic counseling, or student-athlete academic services offices)\n"
    "7. Information about campus visit opportunities or tour scheduling for prospective students, including a URL or contact method\n\n"
    "All information must be current and verifiable through official university sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    """Represents one university entry extracted from the answer."""
    name: Optional[str] = None

    # Official athletics or football program URL
    athletics_url: Optional[str] = None

    # Sources for membership & FBS confirmation (prefer official university/athletics pages)
    membership_source_urls: List[str] = Field(default_factory=list)
    fbs_source_urls: List[str] = Field(default_factory=list)

    # Stadium information
    stadium_capacity: Optional[str] = None  # Keep as string to support formats like "102,780"
    stadium_address: Optional[str] = None   # Full address text
    stadium_info_urls: List[str] = Field(default_factory=list)  # Official stadium/athletics pages

    # Academic support services (URL or contact)
    academic_support_urls: List[str] = Field(default_factory=list)
    academic_support_contact: Optional[str] = None

    # Campus visit / tour info (URL or contact)
    visit_info_urls: List[str] = Field(default_factory=list)
    visit_info_contact: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    """Top-level extraction for up to four universities."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to FOUR distinct Big Ten universities with football programs as they appear in the ANSWER. "
        "Return a JSON object with a 'universities' array. Each array element must be an object with EXACTLY the following fields:\n\n"
        "- name: The university's name (string)\n"
        "- athletics_url: A URL to the university's official athletics website OR official football program page (string URL)\n"
        "- membership_source_urls: A list of official university/athletics URLs that explicitly confirm Big Ten membership (list of URLs). "
        "  If not explicitly provided, include any official athletics/university page from the answer that mentions Big Ten.\n"
        "- fbs_source_urls: A list of official university/athletics URLs that explicitly confirm NCAA Division I FBS competition (list of URLs). "
        "  If not explicitly provided, include any official athletics/university page from the answer that implies/mentions FBS.\n"
        "- stadium_capacity: The official seating capacity number for the football stadium (string as shown in the answer; keep numeric formatting such as commas if present)\n"
        "- stadium_address: The COMPLETE physical address of the football stadium (street address, city, state, ZIP) as a single string\n"
        "- stadium_info_urls: A list of official university/athletics URLs that provide/confirm the stadium capacity AND address (list of URLs)\n"
        "- academic_support_urls: A list of official university URLs for student-athlete academic support services (e.g., SAAS, tutoring, academic counseling). "
        "  If only contact info is provided in the answer and no URL is mentioned, return an empty list.\n"
        "- academic_support_contact: Any contact information (email/phone) for student-athlete academic support services as presented in the answer (string; null if not provided)\n"
        "- visit_info_urls: A list of official university URLs for campus visits/tours/visit scheduling (e.g., admissions visit page, visitor center). "
        "  If only contact info is provided in the answer and no URL is mentioned, return an empty list.\n"
        "- visit_info_contact: Any contact information (email/phone) for campus visits/tours as presented in the answer (string; null if not provided)\n\n"
        "IMPORTANT RULES:\n"
        "1) Extract ONLY from the given ANSWER. Do not invent or add missing info.\n"
        "2) All URLs must be exactly as shown in the ANSWER. If a URL is missing protocol, prepend 'http://'.\n"
        "3) Prefer official university/athletics sources (.edu domains or official athletics program domains) as presented.\n"
        "4) If more than four universities are present, include only the FIRST FOUR distinct universities in the order they appear.\n"
        "5) If fewer than four are present, include all available.\n"
        "6) Do not include external non-university sources (like Wikipedia) unless they are explicitly part of the ANSWER.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {0: "First", 1: "Second", 2: "Third", 3: "Fourth"}
    return mapping.get(n, f"Item_{n+1}")


def to_int_or_zero(capacity_text: Optional[str]) -> int:
    """Extract integer digits from capacity text; return 0 if invalid/missing."""
    if not capacity_text:
        return 0
    digits = re.sub(r"[^\d]", "", capacity_text)
    try:
        return int(digits) if digits else 0
    except Exception:
        return 0


def address_looks_complete(addr: Optional[str]) -> bool:
    """Heuristic check: must contain a ZIP code and at least one comma (to separate city/state)."""
    if not addr or not isinstance(addr, str):
        return False
    has_zip = re.search(r"\b\d{5}(?:-\d{4})?\b", addr) is not None
    has_comma = "," in addr
    return has_zip and has_comma


def ensure_url_list(urls: List[str], fallback: Optional[str]) -> List[str]:
    """Return urls if non-empty; otherwise fallback as single-item list if provided; otherwise empty list."""
    if urls and len(urls) > 0:
        return urls
    return [fallback] if fallback else []


# --------------------------------------------------------------------------- #
# Verification for one university                                             #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build verification subtree for a single university item and run checks.

    All children are marked critical to respect parent critical constraint in the verification tree.
    """
    # Create parent node for this university (critical to satisfy parent critical constraint)
    uni_node = evaluator.add_parallel(
        id=f"{ordinal(index).lower().replace(' ', '_')}_university",
        desc=f"Meets all requirements for the {ordinal(index)} university item.",
        parent=parent_node,
        critical=True,
    )

    # 1) University Name Provided (existence check)
    evaluator.add_custom_node(
        result=bool(uni.name and uni.name.strip()),
        id=f"university_{index}_name",
        desc="University name is provided.",
        parent=uni_node,
        critical=True,
    )

    # 2) Official athletics/football URL exists (existence check)
    evaluator.add_custom_node(
        result=bool(uni.athletics_url and uni.athletics_url.strip()),
        id=f"university_{index}_athletics_url_exists",
        desc="Official athletics/football URL is provided.",
        parent=uni_node,
        critical=True,
    )

    # 3) Official athletics/football URL is an official page (verification by URL)
    ath_official_leaf = evaluator.add_leaf(
        id=f"university_{index}_official_athletics_or_football_url",
        desc="Provides a URL to the university’s official athletics website or official football program page (official university/athletics source).",
        parent=uni_node,
        critical=True,
    )
    ath_claim = (
        f"This page is the official athletics website or official football program page for {uni.name}."
    )
    await evaluator.verify(
        claim=ath_claim,
        node=ath_official_leaf,
        sources=uni.athletics_url,
        additional_instruction=(
            "Verify official status by domain and on-page branding. "
            "Accept official athletics program domains (e.g., mgoblue.com, purduesports.com) or university athletics subdomains. "
            "The page should clearly represent the athletics department or football program of the named university."
        ),
    )

    # 4) Big Ten membership confirmed (verification by URLs; fallback to athletics_url)
    bigten_leaf = evaluator.add_leaf(
        id=f"university_{index}_big_ten_membership_confirmed",
        desc="Confirms the university is a member of the Big Ten Conference.",
        parent=uni_node,
        critical=True,
    )
    bigten_sources = ensure_url_list(uni.membership_source_urls, uni.athletics_url)
    bigten_claim = (
        f"{uni.name} is a current member of the Big Ten Conference."
    )
    await evaluator.verify(
        claim=bigten_claim,
        node=bigten_leaf,
        sources=bigten_sources,
        additional_instruction=(
            "Find explicit mention of 'Big Ten' membership on the provided official university/athletics sources. "
            "The page should clearly indicate participation or membership in the Big Ten Conference."
        ),
    )

    # 5) FBS level confirmed (verification by URLs; fallback to athletics_url)
    fbs_leaf = evaluator.add_leaf(
        id=f"university_{index}_fbs_level_confirmed",
        desc="Confirms the football program competes at the NCAA Division I FBS level.",
        parent=uni_node,
        critical=True,
    )
    fbs_sources = ensure_url_list(uni.fbs_source_urls, uni.athletics_url)
    fbs_claim = (
        f"The football program of {uni.name} competes at the NCAA Division I Football Bowl Subdivision (FBS) level."
    )
    await evaluator.verify(
        claim=fbs_claim,
        node=fbs_leaf,
        sources=fbs_sources,
        additional_instruction=(
            "Look for explicit mention of 'FBS' or 'Football Bowl Subdivision' on official university/athletics pages."
        ),
    )

    # 6) Stadium capacity number provided (existence/format check)
    evaluator.add_custom_node(
        result=(to_int_or_zero(uni.stadium_capacity) > 0),
        id=f"university_{index}_stadium_capacity_number",
        desc="Provides the official football stadium seating capacity as a specific number.",
        parent=uni_node,
        critical=True,
    )

    # 7) Stadium full physical address provided (existence/format check)
    evaluator.add_custom_node(
        result=address_looks_complete(uni.stadium_address),
        id=f"university_{index}_stadium_full_physical_address",
        desc="Provides the complete stadium physical address (street address, city, state, ZIP).",
        parent=uni_node,
        critical=True,
    )

    # 8) Official source URL(s) for stadium info supports capacity & address (verification by URLs)
    stadium_support_leaf = evaluator.add_leaf(
        id=f"university_{index}_official_source_url_for_stadium_info",
        desc="Provides an official university/athletics source URL that supports/verifies the stated stadium capacity and stadium address.",
        parent=uni_node,
        critical=True,
    )
    stadium_sources = ensure_url_list(uni.stadium_info_urls, uni.athletics_url)
    stadium_claim = (
        f"The official stadium/athletics page(s) for {uni.name} list the football stadium seating capacity as '{uni.stadium_capacity}' "
        f"and provide the physical address as '{uni.stadium_address}'."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_support_leaf,
        sources=stadium_sources,
        additional_instruction=(
            "Verify BOTH the capacity number and full stadium address are present on the official source(s). "
            "If multiple URLs are provided, any one of them clearly confirming these details is sufficient."
        ),
    )

    # 9) Academic support services presence (existence check: URL or contact)
    evaluator.add_custom_node(
        result=(bool(uni.academic_support_urls) or bool(uni.academic_support_contact)),
        id=f"university_{index}_academic_support_provided",
        desc="Academic support services info (URL or contact) is provided.",
        parent=uni_node,
        critical=True,
    )

    # 10) Academic support services verification (prefer URL verification)
    academic_support_leaf = evaluator.add_leaf(
        id=f"university_{index}_academic_support_services_url_or_contact",
        desc="Provides contact information or an official university URL for academic support services for student-athletes.",
        parent=uni_node,
        critical=True,
    )
    academic_sources = uni.academic_support_urls
    academic_claim = (
        f"The provided academic support page(s) for {uni.name} are official and relate specifically to student-athlete academic services "
        f"(such as tutoring, academic counseling, or Student-Athlete Academic Services)."
    )
    # If we have URLs, verify by URLs; otherwise, still perform a simple verification on the contact text.
    if academic_sources and len(academic_sources) > 0:
        await evaluator.verify(
            claim=academic_claim,
            node=academic_support_leaf,
            sources=academic_sources,
            additional_instruction=(
                "Confirm the page is an official university/athletics resource for student-athlete academic support. "
                "Look for terms like 'Student-Athlete Academic Services', 'Athletics Academic Support', 'tutoring', or 'academic counseling'."
            ),
        )
    else:
        await evaluator.verify(
            claim=f"Academic support contact is provided for {uni.name}: {uni.academic_support_contact or 'N/A'}.",
            node=academic_support_leaf,
            sources=None,
            additional_instruction=(
                "Since no URL was provided, judge only the presence of contact info from the answer context. "
                "If the contact appears official (e.g., uses a .edu email domain), consider it acceptable."
            ),
        )

    # 11) Campus visit/tour info presence (existence check: URL or contact)
    evaluator.add_custom_node(
        result=(bool(uni.visit_info_urls) or bool(uni.visit_info_contact)),
        id=f"university_{index}_campus_visit_provided",
        desc="Campus visit/tour info (URL or contact) is provided.",
        parent=uni_node,
        critical=True,
    )

    # 12) Campus visit/tour info verification (prefer URL verification)
    campus_visit_leaf = evaluator.add_leaf(
        id=f"university_{index}_campus_visit_or_tour_info_url_or_contact",
        desc="Provides campus visit/tour scheduling information including a URL or contact method from an official university source.",
        parent=uni_node,
        critical=True,
    )
    campus_sources = uni.visit_info_urls
    campus_claim = (
        f"The provided page(s) for {uni.name} offer official campus visit opportunities or tour scheduling information for prospective students."
    )
    if campus_sources and len(campus_sources) > 0:
        await evaluator.verify(
            claim=campus_claim,
            node=campus_visit_leaf,
            sources=campus_sources,
            additional_instruction=(
                "Confirm that the page is an official university site for campus tours/visits (e.g., admissions 'Visit' page or visitor center page) "
                "and provides scheduling or contact details."
            ),
        )
    else:
        await evaluator.verify(
            claim=f"Campus visit/tour contact is provided for {uni.name}: {uni.visit_info_contact or 'N/A'}.",
            node=campus_visit_leaf,
            sources=None,
            additional_instruction=(
                "Since no URL was provided, judge only the presence of contact info from the answer context."
            ),
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
    Evaluate an answer for the Big Ten universities football research task.
    """
    # Initialize evaluator with sequential root to enforce ordering:
    # First: check four universities provided; Second: verify per-university details.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Deduplicate and select first 4 distinct universities by name (case-insensitive)
    unique_items: List[UniversityItem] = []
    seen_names = set()
    for item in extraction.universities:
        nm = (item.name or "").strip().lower()
        if not nm:
            # still append if we need placeholders? We'll skip here to keep distinct requirement strict.
            continue
        if nm in seen_names:
            continue
        seen_names.add(nm)
        unique_items.append(item)
        if len(unique_items) >= 4:
            break

    # Step 1: Four distinct universities provided (critical leaf)
    evaluator.add_custom_node(
        result=(len(unique_items) >= 4),
        id="Four_Universities_Provided",
        desc="Provides information for four distinct universities (i.e., four separate items are present).",
        parent=root,
        critical=True,
    )

    # Step 2: Per-university requirements (critical, parallel parent)
    per_uni_node = evaluator.add_parallel(
        id="Per_University_Requirements",
        desc="For each of the four universities, all required fields are present and supported by official university sources/URLs as required.",
        parent=root,
        critical=True,
    )

    # Create university subtrees; use placeholders if fewer than 4 were extracted to keep tree structure consistent
    while len(unique_items) < 4:
        unique_items.append(UniversityItem())

    for idx in range(4):
        await verify_university(evaluator, per_uni_node, unique_items[idx], idx)

    # Return evaluation summary (includes verification tree)
    return evaluator.get_summary()