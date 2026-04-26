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
TASK_ID = "multi_awards_org_2024"
TASK_DESCRIPTION = """
Identify a U.S.-based news organization that won awards in multiple major 2024 journalism award ceremonies. The organization must have won at least one award in the 2024 Pulitzer Prizes in journalism categories and at least one award in another major 2024 journalism award ceremony (either the Edward R. Murrow Awards, the White House Correspondents' Association Awards, or the Reuters Journalists of the Year Awards). The awards won must span at least two different award categories. Provide the organization's name and headquarters city, along with reference URLs documenting the awards won.
"""

# Known Pulitzer journalism categories (non-exhaustive, covering common 2024 journalism categories)
PULITZER_JOURNALISM_CATEGORIES = {
    "public service",
    "investigative reporting",
    "breaking news reporting",
    "local reporting",
    "national reporting",
    "international reporting",
    "feature writing",
    "commentary",
    "criticism",
    "editorial writing",
    "editorial cartooning",
    "breaking news photography",
    "feature photography",
    "audio reporting",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AwardItem(BaseModel):
    """Information about a single award mention."""
    ceremony: Optional[str] = None              # e.g., "Pulitzer Prizes", "Edward R. Murrow Awards"
    year: Optional[str] = None                  # e.g., "2024"
    category: Optional[str] = None              # e.g., "Investigative Reporting"
    result: Optional[str] = None                # e.g., "Winner", "Won", "Awarded"
    urls: List[str] = Field(default_factory=list)


class OrganizationExtraction(BaseModel):
    """Information extracted from the answer."""
    org_name: Optional[str] = None
    headquarters_city: Optional[str] = None
    headquarters_urls: List[str] = Field(default_factory=list)
    awards: List[AwardItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_org_awards() -> str:
    return """
    Extract the following structured information about the organization described in the answer:

    1) org_name: The full name of the U.S.-based news organization.
    2) headquarters_city: The organization's headquarters city (do not include country/state).
    3) headquarters_urls: A list of reference URLs provided in the answer that support the headquarters city/location.
    4) awards: Extract every award mention provided for year 2024. For each award mention, include:
       - ceremony: The name of the award ceremony (e.g., "Pulitzer Prizes", "Edward R. Murrow Awards", "White House Correspondents' Association Awards", "Reuters Journalists of the Year Awards").
       - year: The year of the award (as stated in the answer, e.g., "2024").
       - category: The category of the award (e.g., "Investigative Reporting", "Public Service").
       - result: The outcome (e.g., "Winner", "Won", "Awarded"). If the answer says "Finalist", set result to "Finalist".
       - urls: All reference URLs in the answer that document this specific award win/recognition.

    Rules:
    - Only extract information explicitly present in the answer text. Do not invent or infer new details.
    - For any missing field, return null (or empty list for URLs).
    - Include all 2024 awards mentioned, even if they are not in the specified ceremonies. The evaluator will filter later.
    - URLs can be plain links or markdown links; extract the actual URL strings.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_text(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _is_pulitzer(ceremony: Optional[str]) -> bool:
    c = _normalize_text(ceremony)
    return "pulitzer" in c


def _is_murrow(ceremony: Optional[str]) -> bool:
    c = _normalize_text(ceremony)
    return "murrow" in c  # Edward R. Murrow Awards (RTDNA), includes national/regional


def _is_whca(ceremony: Optional[str]) -> bool:
    c = _normalize_text(ceremony)
    return "white house correspondents" in c or "whca" in c


def _is_reuters_joty(ceremony: Optional[str]) -> bool:
    c = _normalize_text(ceremony)
    return "reuters" in c and "journalists of the year" in c


def _is_allowed_second_ceremony(ceremony: Optional[str]) -> bool:
    return _is_murrow(ceremony) or _is_whca(ceremony) or _is_reuters_joty(ceremony)


def _is_winner(result: Optional[str]) -> bool:
    r = _normalize_text(result)
    if not r:
        return False
    # Consider various winner phrasings; exclude "finalist"
    return any(k in r for k in ["winner", "won", "award", "awarded", "recipient"]) and "finalist" not in r


def _is_year_2024(year: Optional[str]) -> bool:
    y = _normalize_text(year)
    return "2024" in y if y else False


def _category_is_journalism(category: Optional[str]) -> bool:
    cat = _normalize_text(category)
    if not cat:
        return False
    # Allow fuzzy matching: check if normalized category contains a known journalism category keyword
    return any(jc in cat for jc in PULITZER_JOURNALISM_CATEGORIES)


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _collect_award_urls(awards: List[AwardItem], predicate) -> List[str]:
    urls: List[str] = []
    for a in awards:
        if predicate(a):
            urls.extend(a.urls or [])
    return _dedup_urls(urls)


def _first_award(awards: List[AwardItem], predicate) -> Optional[AwardItem]:
    for a in awards:
        if predicate(a):
            return a
    return None


def _normalized_category(cat: Optional[str]) -> Optional[str]:
    if cat is None:
        return None
    return _normalize_text(cat)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: OrganizationExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and perform verifications.
    """

    # Top-level critical sequential node (since Evaluator.initialize creates a non-critical root)
    task_main = evaluator.add_sequential(
        id="task_main",
        desc="Identify a U.S.-based news organization that won awards in multiple major 2024 journalism award ceremonies and provide required details and sources",
        parent=evaluator.root,
        critical=True
    )

    # 1) Organization identified (critical leaf)
    org_identified_node = evaluator.add_custom_node(
        result=bool(extracted.org_name and extracted.org_name.strip()),
        id="organization_identified",
        desc="The organization's name is provided",
        parent=task_main,
        critical=True
    )

    # 2) Requirements check (parallel critical group)
    req_check_node = evaluator.add_parallel(
        id="requirements_check",
        desc="Verify the organization satisfies all award/location/category constraints and required sourcing",
        parent=task_main,
        critical=True
    )

    org_name = extracted.org_name or ""

    # 2.a) Headquarters city with source (split into two critical leaf checks under a critical sub-node)
    hq_group = evaluator.add_parallel(
        id="headquarters_city_with_source",
        desc="The organization's headquarters city is provided and supported by at least one reference URL",
        parent=req_check_node,
        critical=True
    )

    # Existence of headquarters city
    hq_city_provided = evaluator.add_custom_node(
        result=bool(extracted.headquarters_city and extracted.headquarters_city.strip()),
        id="hq_city_provided",
        desc="Headquarters city is provided",
        parent=hq_group,
        critical=True
    )

    # At least one source URL provided for HQ info
    hq_sources_exist = evaluator.add_custom_node(
        result=bool(extracted.headquarters_urls and len(extracted.headquarters_urls) > 0),
        id="hq_city_source_exists",
        desc="At least one HQ reference URL is provided",
        parent=hq_group,
        critical=True
    )

    # Verify HQ city against provided sources
    hq_city_verify = evaluator.add_leaf(
        id="hq_city_source_supported",
        desc="Headquarters city is supported by the provided reference URLs",
        parent=hq_group,
        critical=True
    )
    hq_city_claim = f"The headquarters city of '{org_name}' is '{extracted.headquarters_city or ''}'."
    await evaluator.verify(
        claim=hq_city_claim,
        node=hq_city_verify,
        sources=extracted.headquarters_urls,
        additional_instruction="Check the referenced page(s) to confirm the organization's headquarters city. Allow minor variants (e.g., 'New York' vs 'New York City')."
    )

    # 2.b) US-based verification (critical leaf)
    us_based_node = evaluator.add_leaf(
        id="us_based",
        desc="The organization is headquartered in the United States",
        parent=req_check_node,
        critical=True
    )
    us_based_claim = f"'{org_name}' is headquartered in the United States."
    await evaluator.verify(
        claim=us_based_claim,
        node=us_based_node,
        sources=extracted.headquarters_urls,
        additional_instruction="Use the referenced HQ page(s) to determine that the organization is US-based. City+state in the U.S. counts as US-based. Accept 'U.S.', 'USA', or 'United States of America' as equivalent."
    )

    # Prepare award filtering
    awards = extracted.awards or []
    pulitzer_wins = [a for a in awards if _is_pulitzer(a.ceremony) and _is_year_2024(a.year) and _is_winner(a.result)]
    pulitzer_urls = _collect_award_urls(awards, lambda a: _is_pulitzer(a.ceremony) and _is_year_2024(a.year))

    # Choose a specific Pulitzer award (if any) to make a concrete claim with category if available and journalism category
    pulitzer_award_for_claim = _first_award(
        pulitzer_wins,
        lambda a: True  # any winning Pulitzer in 2024
    )

    # 2.c) Pulitzer win source existence (critical leaf via custom)
    pulitzer_source_exist_node = evaluator.add_custom_node(
        result=bool(pulitzer_urls),
        id="pulitzer_win_source",
        desc="At least one reference URL is provided that documents the organization's 2024 Pulitzer win",
        parent=req_check_node,
        critical=True
    )

    # 2.d) Pulitzer win verification (critical leaf)
    pulitzer_win_node = evaluator.add_leaf(
        id="pulitzer_win",
        desc="The organization won at least one award in the 2024 Pulitzer Prizes in journalism categories",
        parent=req_check_node,
        critical=True
    )

    # Build claim for Pulitzer win
    if pulitzer_award_for_claim:
        pul_cat = pulitzer_award_for_claim.category or ""
        # If the extracted category is not clearly journalism, still state claim but add instruction to check journalism
        pulitzer_claim = (
            f"'{org_name}' won a 2024 Pulitzer Prize"
            + (f" in the '{pul_cat}' category." if pul_cat else ".")
        )
    else:
        # Fallback generic claim if no specific item extracted
        pulitzer_claim = f"'{org_name}' won at least one 2024 Pulitzer Prize in a journalism category."

    await evaluator.verify(
        claim=pulitzer_claim,
        node=pulitzer_win_node,
        sources=pulitzer_urls,
        additional_instruction=(
            "Confirm that the referenced page(s) show this organization as a winner in the 2024 Pulitzer Prizes. "
            "Only count journalism categories (e.g., Public Service, Investigative Reporting, Local/National/International Reporting, "
            "Feature Writing, Commentary, Criticism, Editorial Writing/Cartooning, Breaking News/Feature Photography, Audio Reporting). "
            "Do not count Letters, Drama & Music categories. Exclude 'Finalist'."
        )
    )

    # Prepare second ceremony wins
    second_wins = [
        a for a in awards
        if _is_allowed_second_ceremony(a.ceremony) and _is_year_2024(a.year) and _is_winner(a.result)
    ]
    second_award_urls = _collect_award_urls(awards, lambda a: _is_allowed_second_ceremony(a.ceremony) and _is_year_2024(a.year))
    second_award_for_claim = _first_award(second_wins, lambda a: True)

    # 2.e) Second award win source existence (critical leaf via custom)
    second_source_exist_node = evaluator.add_custom_node(
        result=bool(second_award_urls),
        id="second_award_win_source",
        desc="At least one reference URL is provided that documents the organization's win in the other major 2024 journalism award ceremony",
        parent=req_check_node,
        critical=True
    )

    # 2.f) Second award win verification (critical leaf)
    second_award_win_node = evaluator.add_leaf(
        id="second_award_win",
        desc="The organization won at least one award in another major 2024 journalism award ceremony (Edward R. Murrow Awards, White House Correspondents' Association Awards, or Reuters Journalists of the Year Awards)",
        parent=req_check_node,
        critical=True
    )

    if second_award_for_claim:
        ceremony_name = second_award_for_claim.ceremony or "another major 2024 award"
        sec_cat = second_award_for_claim.category or ""
        second_claim = (
            f"'{org_name}' won a 2024 {ceremony_name}"
            + (f" award in the '{sec_cat}' category." if sec_cat else " award.")
        )
    else:
        second_claim = (
            f"'{org_name}' won at least one award in 2024 at either the Edward R. Murrow Awards, "
            f"the White House Correspondents' Association Awards, or the Reuters Journalists of the Year Awards."
        )

    await evaluator.verify(
        claim=second_claim,
        node=second_award_win_node,
        sources=second_award_urls,
        additional_instruction=(
            "Confirm that the referenced page(s) show this organization as a winner in 2024 for one of the specified ceremonies. "
            "For Murrow Awards, both National and Regional are acceptable as long as it clearly indicates a win (not finalist). "
            "For WHCA, verify the awards (not scholarships) and ensure the organization is a winner. "
            "For Reuters Journalists of the Year, verify the organization is the winner or recipient where applicable."
        )
    )

    # 2.g) Multiple categories check (critical leaf via custom)
    # Collect distinct categories across all 2024 wins in targets (Pulitzer journalism + allowed second ceremonies)
    categories_set: set = set()

    for a in pulitzer_wins:
        cat = _normalized_category(a.category)
        if cat and _category_is_journalism(cat):
            categories_set.add(cat)

    for a in second_wins:
        cat = _normalized_category(a.category)
        if cat:
            categories_set.add(cat)

    multiple_categories_node = evaluator.add_custom_node(
        result=(len(categories_set) >= 2),
        id="multiple_categories",
        desc="The awards won span at least two different award categories (i.e., at least two distinct categories are identified)",
        parent=req_check_node,
        critical=True
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
    Evaluate an answer for the multi-awards organization 2024 task.
    """

    # Initialize evaluator with sequential root to match rubric flow
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_org_awards(),
        template_class=OrganizationExtraction,
        extraction_name="organization_awards_2024",
    )

    # Build verification tree and perform verification
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()