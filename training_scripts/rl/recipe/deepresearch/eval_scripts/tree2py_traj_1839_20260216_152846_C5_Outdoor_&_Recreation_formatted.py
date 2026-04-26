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
TASK_ID = "ct_pet_ada_trail_camping"
TASK_DESCRIPTION = (
    "I'm planning a camping trip to Connecticut and need to find three state parks or state forests that "
    "accommodate all members of my group. Specifically, I need locations where: "
    "(1) Pets are allowed at the campground itself (not just in the general park area), "
    "(2) ADA accessible camping facilities are available, and "
    "(3) Hiking trails are accessible from or near the campground. "
    "For each of the three parks or forests you identify, please provide: "
    "(1) The official name of the park or state forest, "
    "(2) Confirmation that pets are allowed at the campground sites, "
    "(3) Confirmation that ADA accessible camping facilities exist, "
    "(4) Confirmation that hiking trails are available, and "
    "(5) Reference URL(s) from official Connecticut state sources (such as portal.ct.gov/DEEP, ctparks.com, "
    "or connecticutstateparks.reserveamerica.com) that verify each of these features. "
    "Note: Many Connecticut state parks prohibit pets in campgrounds, so please ensure your selections "
    "specifically allow pets at camping sites, not just in day-use areas."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    name: Optional[str] = None
    pet_sources: List[str] = Field(default_factory=list)
    ada_sources: List[str] = Field(default_factory=list)
    trails_sources: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract up to the first three Connecticut state parks or state forests that the answer proposes as meeting
    ALL of the following: (1) pets allowed at the campground (sites or campground area itself), (2) ADA accessible
    camping facilities available, and (3) hiking trails accessible from or near the campground.

    For each identified park/forest, return a JSON object with:
    - name: the official park or state forest name as written in the answer
    - pet_sources: a list of all URLs in the answer that specifically support the pet policy at the CAMPGROUND
                   (i.e., pets allowed at campsites), not just general park areas
    - ada_sources: a list of all URLs in the answer that specifically support the presence of ADA accessible
                   camping facilities (e.g., accessible campsites, accessible restrooms/bathhouse at the campground)
    - trails_sources: a list of all URLs in the answer that specifically support that hiking trails are accessible
                      from or near the campground area

    IMPORTANT URL FILTERING RULES:
    - Include ONLY official Connecticut state sources among the following domains:
      • portal.ct.gov with path containing '/DEEP' (case-insensitive)
      • ctparks.com
      • connecticutstateparks.reserveamerica.com
    - If the answer contains sources outside of these domains, ignore them for the lists.
    - Always return full URLs including http/https scheme. If a URL is missing a scheme, prepend http://.
    - Remove duplicates. Keep reasonable URLs even if they include tracking parameters.

    If the answer lists more than three parks/forests, keep only the first three by order of appearance.
    If fewer than three are present, return only those available.

    If any field is missing, return null or an empty list as appropriate.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_url(u: str) -> Optional[str]:
    if not u or not isinstance(u, str):
        return None
    u = u.strip()
    if not u:
        return None
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "http://" + u
    try:
        parsed = urlparse(u)
        if not parsed.netloc:
            return None
        return u
    except Exception:
        return None


def is_official_ct_url(u: str) -> bool:
    nu = _normalize_url(u)
    if not nu:
        return False
    p = urlparse(nu)
    host = (p.netloc or "").lower()
    path = (p.path or "").lower()
    if host == "portal.ct.gov" and "deep" in path:
        return True
    if host.endswith("ctparks.com"):
        return True
    if host == "connecticutstateparks.reserveamerica.com":
        return True
    return False


def filter_official_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls or []:
        nu = _normalize_url(u)
        if not nu:
            continue
        if not is_official_ct_url(nu):
            continue
        if nu in seen:
            continue
        seen.add(nu)
        out.append(nu)
    return out


def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


# --------------------------------------------------------------------------- #
# Verification logic per park                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkItem,
    park_index: int,
) -> None:
    idx = park_index
    ord_name = ordinal(idx)
    park_name = park.name or f"Park/Forest #{idx+1}"

    park_node = evaluator.add_parallel(
        id=f"Park_{idx+1}",
        desc=f"{ord_name} Connecticut state park or forest meeting all requirements",
        parent=parent_node,
        critical=False
    )

    # Prepare filtered official sources for each category
    pet_urls = filter_official_urls(park.pet_sources)
    ada_urls = filter_official_urls(park.ada_sources)
    trails_urls = filter_official_urls(park.trails_sources)

    # -------------------- Pet-friendly camping -----------------------------
    pet_main = evaluator.add_sequential(
        id=f"Park_{idx+1}_Pet_Friendly_Camping",
        desc=("Verification that the identified park or forest explicitly allows pets at campground sites "
              "(not just in day-use areas), supported by at least one official Connecticut state source."),
        parent=park_node,
        critical=True
    )
    # Existence of official CT sources for pet policy at campground
    evaluator.add_custom_node(
        result=len(pet_urls) > 0,
        id=f"park_{idx+1}_pet_sources_official",
        desc="Official Connecticut source(s) provided for campground pet policy",
        parent=pet_main,
        critical=True
    )
    # Claim verification by URLs
    pet_leaf = evaluator.add_leaf(
        id=f"park_{idx+1}_pet_supported",
        desc="Campground allows pets at the campsites (not just in day-use areas)",
        parent=pet_main,
        critical=True
    )
    pet_claim = (
        f"The campground at {park_name} allows pets at the campsites (i.e., pets are permitted in the campground "
        f"itself, not only in day-use areas)."
    )
    await evaluator.verify(
        claim=pet_claim,
        node=pet_leaf,
        sources=pet_urls,
        additional_instruction=(
            "Verify the campground pet policy specifically. The page must indicate pets are allowed at campsites/"
            "in the campground. If the page indicates pets are prohibited in campgrounds (but allowed in day-use "
            "areas) or provides no campground-specific pet policy, the claim is not supported. ReserveAmerica "
            "attribute 'Pets Allowed' for campsites counts as support."
        )
    )

    # -------------------- ADA accessible camping facilities ----------------
    ada_main = evaluator.add_sequential(
        id=f"Park_{idx+1}_ADA_Accessible_Facilities",
        desc=("Verification that the identified park provides ADA accessible camping facilities (e.g., "
              "accessible campsites and/or accessible restrooms/bathhouse), supported by at least one official "
              "Connecticut state source."),
        parent=park_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(ada_urls) > 0,
        id=f"park_{idx+1}_ada_sources_official",
        desc="Official Connecticut source(s) provided for ADA accessible camping facilities",
        parent=ada_main,
        critical=True
    )
    ada_leaf = evaluator.add_leaf(
        id=f"park_{idx+1}_ada_supported",
        desc="Campground provides ADA accessible camping facilities",
        parent=ada_main,
        critical=True
    )
    ada_claim = (
        f"The campground at {park_name} provides ADA accessible camping facilities such as accessible campsites and/or "
        f"accessible restrooms/bathhouse."
    )
    await evaluator.verify(
        claim=ada_claim,
        node=ada_leaf,
        sources=ada_urls,
        additional_instruction=(
            "Accept clear indications of ADA/accessible campsites or accessible restrooms/bathhouse at the campground. "
            "On ReserveAmerica, 'ADA Access' attribute counts as support. If accessibility is only referenced for "
            "day-use facilities without relevance to the campground, do not support."
        )
    )

    # -------------------- Hiking trail access near/from campground ----------
    trails_main = evaluator.add_sequential(
        id=f"Park_{idx+1}_Hiking_Trail_Access",
        desc=("Verification that hiking trails are available and accessible from or near the campground area, "
              "supported by at least one official Connecticut state source."),
        parent=park_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(trails_urls) > 0,
        id=f"park_{idx+1}_trails_sources_official",
        desc="Official Connecticut source(s) provided for hiking trails accessible from/near campground",
        parent=trails_main,
        critical=True
    )
    trails_leaf = evaluator.add_leaf(
        id=f"park_{idx+1}_trails_supported",
        desc="Hiking trails are accessible from or near the campground",
        parent=trails_main,
        critical=True
    )
    trails_claim = (
        f"Hiking trails are accessible from or near the campground at {park_name} (e.g., trails within short walking "
        f"distance or directly connected to the campground area)."
    )
    await evaluator.verify(
        claim=trails_claim,
        node=trails_leaf,
        sources=trails_urls,
        additional_instruction=(
            "Support the claim only if the official page indicates hiking trails are at or near the campground area "
            "(e.g., trailheads within or adjacent to the campground, or the campground is situated within a park that "
            "states trails are accessible from the campground). If trails are only somewhere in the park without a "
            "reasonable indication they are accessible from or near the campground, do not support."
        )
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
    Evaluate an answer for the Connecticut pet-friendly + ADA accessible + trails camping task.
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

    # Extract structured park info
    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    parks = list(extracted.parks) if extracted and extracted.parks else []
    # Keep only first three; if fewer, pad with empty items
    parks = parks[:3]
    while len(parks) < 3:
        parks.append(ParkItem())

    # Add a top-level grouping node to mirror rubric
    top_node = evaluator.add_parallel(
        id="Connecticut_Pet_Friendly_ADA_Accessible_Camping_Parks",
        desc=("Evaluation of three Connecticut state parks or state forests that allow pets at campgrounds, "
              "provide ADA accessible facilities, and offer hiking trail access"),
        parent=root,
        critical=False
    )

    # Add custom info for transparency
    summary_sources_info: List[Dict[str, Any]] = []
    for i, p in enumerate(parks):
        summary_sources_info.append({
            "park_index": i + 1,
            "name": p.name,
            "pet_sources_official_count": len(filter_official_urls(p.pet_sources)),
            "ada_sources_official_count": len(filter_official_urls(p.ada_sources)),
            "trails_sources_official_count": len(filter_official_urls(p.trails_sources)),
        })
    evaluator.add_custom_info({"parks_sources_overview": summary_sources_info}, info_type="debug", info_name="sources_overview")

    # Verify each of the three parks/forests
    for i in range(3):
        await verify_single_park(evaluator, top_node, parks[i], i)

    return evaluator.get_summary()