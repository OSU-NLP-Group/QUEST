import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nps_campgrounds_multi_constraints"
TASK_DESCRIPTION = """
Find 4 campgrounds, each in a different U.S. National Park, that ALL meet the following requirements:

1. The campground must accept advance reservations (through Recreation.gov or similar system) - not first-come-first-served only
2. The campground must be open for at least 6 consecutive months per year
3. The campground must have flush toilets (not vault toilets)
4. The campground must have potable drinking water available
5. The campground must allow pets (dogs)
6. The campground must have at least one ADA-accessible campsite
7. The campground must be able to accommodate RVs of at least 30 feet in length at some sites
8. The campground must be located within 10 miles (by road) of a park visitor center
9. The campground must have an RV dump station available either on-site at the campground or elsewhere within the same national park

For each of the 4 campgrounds, provide:
- Campground name
- National Park name
- Operating season (specific months/dates or year-round)
- A reference URL from an official National Park Service or Recreation.gov source

All four campgrounds must be in different national parks (no two campgrounds from the same park).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    campground_name: Optional[str] = None
    park_name: Optional[str] = None  # e.g., "Yosemite National Park"
    operating_season: Optional[str] = None  # e.g., "April–October", "Year-round", or specific dates
    source_urls: List[str] = Field(default_factory=list)  # Prefer official NPS (nps.gov) / Recreation.gov URLs


class CampgroundsExtraction(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to 6 campground entries mentioned in the answer (we will later evaluate the first 4).
    For each campground, extract the following fields:
    - campground_name: The campground's name as stated
    - park_name: The NATIONAL PARK unit name (e.g., "Yosemite National Park"). Do not use state parks, national forests, or other non–National Park units.
    - operating_season: The operating season text as stated (e.g., "April–October", "Year-round", or specific open/close dates)
    - source_urls: A list of all URLs provided in the answer that support facts about this campground. Include only official National Park Service (nps.gov) or Recreation.gov links when possible. If multiple official links are provided (e.g., campground page, facilities/amenities page, visitor center page, etc.), include them all for this campground. If the answer provides no URL for a campground, return an empty list for that campground.

    Return JSON with shape:
    {
      "campgrounds": [
        {
          "campground_name": "...",
          "park_name": "... National Park",
          "operating_season": "...",
          "source_urls": ["https://www.nps.gov/...", "https://www.recreation.gov/..."]
        },
        ...
      ]
    }

    Notes:
    - Only extract what is explicitly present in the provided answer.
    - Do not invent any URLs or fields.
    - For park_name, ensure it is a U.S. "National Park" unit (not "National Monument", "State Park", etc.).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_park_name(park: Optional[str]) -> str:
    """Normalize park unit name for uniqueness comparison."""
    if not park:
        return ""
    s = park.lower()
    # Remove common suffixes and conjunctions
    patterns = [
        r"\bnational park\b",
        r"\bnational parks\b",
        r"\bnational park and preserve\b",
        r"\bnational park & preserve\b",
        r"\band preserve\b",
        r"& preserve",
        r"\bnp\b",
        r"\bn\.p\.\b",
        r"\bthe\b",
    ]
    for p in patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    # Remove non-alphanumerics
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s.strip()


def compute_uniqueness_flags(items: List[CampgroundItem]) -> List[bool]:
    """Return list of booleans: each True iff that item's park_name is unique among the 4."""
    norms = [normalize_park_name(i.park_name) for i in items]
    counts: Dict[str, int] = {}
    for n in norms:
        counts[n] = counts.get(n, 0) + 1 if n else counts.get(n, 0) + 0  # empty string counted but will fail anyway
    flags = []
    for n in norms:
        if not n:
            flags.append(False)
        else:
            flags.append(counts.get(n, 0) == 1)
    return flags


def official_urls(urls: List[str]) -> List[str]:
    """Filter official NPS or Recreation.gov URLs."""
    res = []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        ul = u.lower().strip()
        if ul.startswith("http://") or ul.startswith("https://"):
            if "nps.gov" in ul or "recreation.gov" in ul:
                res.append(u)
    return res


def recgov_urls(urls: List[str]) -> List[str]:
    return [u for u in urls or [] if isinstance(u, str) and "recreation.gov" in u.lower()]


def nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


async def verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node,
    urls: List[str],
    additional_instruction: str
) -> None:
    """Invoke evaluator.verify with URLs if present; otherwise mark node failed."""
    if not urls:
        node.score = 0.0
        node.status = "failed"
        return
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification per campground                                                 #
# --------------------------------------------------------------------------- #
async def verify_campground(
    evaluator: Evaluator,
    parent_node,
    item: CampgroundItem,
    index_zero_based: int,
    all_items: List[CampgroundItem],
    unique_flags: List[bool],
) -> None:
    """
    Build the subtree for one campground and run verifications for each rubric leaf.
    """
    idx = index_zero_based + 1
    cg_parent = evaluator.add_parallel(
        id=f"campground_{idx}",
        desc=(
            "First qualifying campground with all requirements met" if idx == 1 else
            "Second qualifying campground with all requirements met" if idx == 2 else
            "Third qualifying campground with all requirements met" if idx == 3 else
            "Fourth qualifying campground with all requirements met"
        ),
        parent=parent_node,
        critical=False,  # Each campground contributes partial credit; leaves below are critical.
    )

    # Prepare URL sets
    all_official = official_urls(item.source_urls or [])
    rec_urls = recgov_urls(all_official)

    # 1) Park identity (critical)
    node_park_identity = evaluator.add_leaf(
        id=f"cg{idx}_park_identity",
        desc="Campground is located in a U.S. National Park (not state park) with name explicitly stated and NPS/Recreation.gov URL reference provided",
        parent=cg_parent,
        critical=True
    )
    # If key fields missing or no official source -> immediate fail
    if not (nonempty_str(item.campground_name) and nonempty_str(item.park_name) and all_official):
        node_park_identity.score = 0.0
        node_park_identity.status = "failed"
    else:
        claim = (
            f"The provided official page(s) indicate that '{item.campground_name}' is a campground located "
            f"within '{item.park_name}', which is a U.S. National Park (not a state park or other designation)."
        )
        add_ins = (
            "Confirm that the page is from an official NPS (nps.gov) or Recreation.gov domain and that the "
            "campground is explicitly within the named 'National Park' unit (not National Monument, State Park, "
            "National Forest, etc.). Allow minor naming variations such as 'National Park & Preserve' if it is still "
            "a National Park unit. If the provided pages do not clearly support this, mark as not supported."
        )
        await verify_with_urls_or_fail(evaluator, claim, node_park_identity, all_official, add_ins)

    # 2) Uniqueness of park across all four (critical, logical check)
    unique_ok = unique_flags[index_zero_based]
    evaluator.add_custom_node(
        result=unique_ok,
        id=f"cg{idx}_uniqueness",
        desc="This campground is in a different national park than the other three campgrounds",
        parent=cg_parent,
        critical=True
    )

    # 3) Operating season ≥ 6 consecutive months (critical)
    node_season = evaluator.add_leaf(
        id=f"cg{idx}_operating_season",
        desc="Campground is open for at least 6 consecutive months per year, with specific dates or year-round status stated and URL reference provided",
        parent=cg_parent,
        critical=True
    )
    claim = (
        f"The campground is open for at least 6 consecutive months per year. "
        f"The operating season is described as: '{item.operating_season or 'N/A'}'. "
        f"Accept 'year-round' as satisfying this requirement."
    )
    add_ins = (
        "Use the official page(s) to determine season dates. If the page lists opening/closing months or 'Season Dates' "
        "showing at least six consecutive months open, consider this satisfied. If multiple loops/sites differ, it's "
        "acceptable as long as the campground overall has at least one loop/site open ≥ 6 consecutive months. "
        "If season cannot be clearly established from the provided official pages, mark as not supported."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_season, all_official, add_ins)

    # 4) Reservation system (critical)
    node_res = evaluator.add_leaf(
        id=f"cg{idx}_reservation_system",
        desc="Campground accepts advance reservations through Recreation.gov or similar system (not first-come-first-served only), with URL reference provided",
        parent=cg_parent,
        critical=True
    )
    res_sources = rec_urls if rec_urls else all_official
    claim = (
        "This campground accepts advance reservations via Recreation.gov or a comparable official reservation system. "
        "It is not first-come-first-served only."
    )
    add_ins = (
        "Look for 'Reservations', 'Book now', 'Site availability', or similar indicators. "
        "If the page indicates reservations can be made in advance (e.g., via Recreation.gov), pass. "
        "If the campground is strictly first-come-first-served with no advance reservation option, fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_res, res_sources, add_ins)

    # 5) Flush toilets (critical)
    node_flush = evaluator.add_leaf(
        id=f"cg{idx}_flush_toilets",
        desc="Campground has flush toilets (not vault toilets) with URL reference confirming this facility",
        parent=cg_parent,
        critical=True
    )
    claim = "The campground has flush toilets (not just vault/pit toilets)."
    add_ins = (
        "Check the amenities/facilities section for 'Flush toilets'. If only 'Vault' or 'Pit' toilets are listed and "
        "no 'Flush', then fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_flush, all_official, add_ins)

    # 6) Potable water (critical)
    node_water = evaluator.add_leaf(
        id=f"cg{idx}_potable_water",
        desc="Campground has potable drinking water available with URL reference confirming availability",
        parent=cg_parent,
        critical=True
    )
    claim = "The campground provides potable drinking water."
    add_ins = (
        "Look for 'Potable water', 'Drinking water available', or similar in amenities/facilities. "
        "If the page only mentions non-potable water or does not mention potable/drinking water, fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_water, all_official, add_ins)

    # 7) Pets allowed (critical)
    node_pets = evaluator.add_leaf(
        id=f"cg{idx}_pet_policy",
        desc="Campground allows pets (dogs) with URL reference confirming pet policy",
        parent=cg_parent,
        critical=True
    )
    claim = "Pets (dogs) are allowed at the campground (subject to NPS rules)."
    add_ins = (
        "Look for pet policy statements on the official page(s). If pets are prohibited, or no indication is given, fail. "
        "It's acceptable if pets are allowed with restrictions (e.g., leash requirements)."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_pets, all_official, add_ins)

    # 8) ADA-accessible campsite (critical)
    node_ada = evaluator.add_leaf(
        id=f"cg{idx}_ada_accessible",
        desc="Campground has at least one ADA-accessible campsite with URL reference confirming accessibility",
        parent=cg_parent,
        critical=True
    )
    claim = (
        "The campground has at least one ADA-accessible campsite (e.g., 'accessible site(s)', wheelchair-accessible). "
        "Accessible restrooms alone do not satisfy this requirement; there must be at least one accessible campsite."
    )
    add_ins = (
        "Check for 'Accessible site(s)', 'ADA', wheelchair symbol, or explicit listing of accessible campsites. "
        "If the page does not clearly indicate accessible campsites (not only bathrooms or facilities), fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_ada, all_official, add_ins)

    # 9) RV capacity ≥ 30 ft at some sites (critical)
    node_rv = evaluator.add_leaf(
        id=f"cg{idx}_rv_capacity",
        desc="Campground can accommodate RVs of at least 30 feet in length at some sites with URL reference confirming RV specifications",
        parent=cg_parent,
        critical=True
    )
    claim = (
        "At least some sites in the campground can accommodate RVs of length 30 feet or more (e.g., max vehicle length ≥ 30 ft)."
    )
    add_ins = (
        "Use 'Max vehicle length' or 'RV length' indicators. If any loop or site supports ≥ 30 ft, pass. "
        "If the maximum vehicle length everywhere is < 30 ft, fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_rv, all_official, add_ins)

    # 10) Within 10 miles by road of a park visitor center (critical)
    node_vc = evaluator.add_leaf(
        id=f"cg{idx}_visitor_center_proximity",
        desc="Campground is located within 10 miles (by road) of a park visitor center with URL reference confirming location",
        parent=cg_parent,
        critical=True
    )
    claim = (
        "The campground is within 10 miles by road of a visitor center for the same national park."
    )
    add_ins = (
        "Use the provided official pages (campground, park visitor center pages, directions) to judge proximity. "
        "Explicit mileage or directions on NPS pages are strong evidence. "
        "If road distance cannot be reasonably confirmed from the provided pages (e.g., only straight-line distance or "
        "no distance context), fail."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_vc, all_official, add_ins)

    # 11) Dump station available at campground or elsewhere in same park (critical)
    node_dump = evaluator.add_leaf(
        id=f"cg{idx}_dump_station",
        desc="Campground has RV dump station on-site or elsewhere within the same national park with URL reference confirming availability",
        parent=cg_parent,
        critical=True
    )
    claim = (
        "An RV dump station is available either at this campground or elsewhere within the same national park."
    )
    add_ins = (
        "Look for 'Dump station', 'Sanitary dump', or similar. If the dump station is not at the campground but is "
        "available elsewhere in the same park (per NPS), that satisfies the requirement."
    )
    await verify_with_urls_or_fail(evaluator, claim, node_dump, all_official, add_ins)


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
    """
    Evaluate an answer for the '4 qualifying NPS campgrounds' task and return a structured result.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four campgrounds evaluated in parallel; each has critical leaves
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

    # Extract structured campground proposals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    # Normalize to exactly 4 items (pad if fewer, truncate if more)
    items: List[CampgroundItem] = list(extracted.campgrounds or [])
    if len(items) < 4:
        items = items + [CampgroundItem() for _ in range(4 - len(items))]
    else:
        items = items[:4]

    # Compute uniqueness across park units for the 4 selected items
    unique_flags = compute_uniqueness_flags(items)

    # Build verification subtrees for each of the 4 campgrounds
    for i in range(4):
        await verify_campground(
            evaluator=evaluator,
            parent_node=root,
            item=items[i],
            index_zero_based=i,
            all_items=items,
            unique_flags=unique_flags
        )

    # Return standardized summary with the verification tree and scores
    return evaluator.get_summary()