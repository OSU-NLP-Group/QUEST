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
TASK_ID = "ca_family_parks"
TASK_DESCRIPTION = (
    "Identify at least two national parks in California that meet all of the following criteria for a family trip: "
    "(1) The park must have lodging facilities located inside the park boundaries (not just nearby gateway towns), "
    "(2) The park must operate a free shuttle bus system for visitor transportation, "
    "(3) The park must have at least one visitor center, and "
    "(4) The park must offer a Junior Ranger program for children. "
    "For each park you identify, provide: the park name, a reference URL confirming the availability of in-park lodging, "
    "a reference URL confirming the free shuttle system, a reference URL confirming the visitor center(s), and a "
    "reference URL confirming the Junior Ranger program."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkEntry(BaseModel):
    name: Optional[str] = None
    lodging_urls: List[str] = Field(default_factory=list)
    shuttle_urls: List[str] = Field(default_factory=list)
    visitor_center_urls: List[str] = Field(default_factory=list)
    junior_ranger_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract all park entries mentioned in the answer. For each park entry, extract the following fields:

    - name: The full park name as stated in the answer.
    - lodging_urls: A list of URLs that the answer cites to confirm lodging located inside the park boundaries.
    - shuttle_urls: A list of URLs that the answer cites to confirm the park operates a free shuttle bus system for visitors (inside the park).
    - visitor_center_urls: A list of URLs that the answer cites to confirm at least one visitor center exists in the park.
    - junior_ranger_urls: A list of URLs that the answer cites to confirm the park offers a Junior Ranger program.

    Rules:
    - Extract only URLs explicitly present in the answer. Do not invent any URLs.
    - Accept URLs in plain format or markdown links; include the actual target URL.
    - If the answer provides exactly one URL for a field, return it in a one-element list.
    - If no URL is provided for a field, return an empty list for that field.
    - If no park entries are provided, return {"parks": []}.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_merge(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for u in lst:
            if not u:
                continue
            val = u.strip()
            if val and val not in seen:
                seen.add(val)
                merged.append(val)
    return merged


def _has_nonempty_string(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification logic per park                                                 #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkEntry,
    index: int,
) -> None:
    """
    Build verification nodes for a single park and run verifications.
    """
    park_node = evaluator.add_parallel(
        id=f"park_{index}",
        desc=f"Evaluation of the {'first' if index == 1 else 'second' if index == 2 else f'#{index + 1}'} identified park against all required constraints and documentation requirements.",
        parent=parent_node,
        critical=False,  # Allow partial credit per park
    )

    # 1) Name provided (critical)
    name_provided_node = evaluator.add_custom_node(
        result=_has_nonempty_string(park.name),
        id=f"park_{index}_name_provided",
        desc=f"Provides the name of the {'first' if index == 0 else 'second'} park.",
        parent=park_node,
        critical=True
    )

    # Aggregate all provided URLs for cross-checks
    all_urls = _unique_merge(
        park.lodging_urls,
        park.shuttle_urls,
        park.visitor_center_urls,
        park.junior_ranger_urls
    )
    park_name = park.name or ""

    # 2) Park in California (critical)
    in_ca_node = evaluator.add_leaf(
        id=f"park_{index}_in_california",
        desc=f"The {'first' if index == 0 else 'second'} park is located in California.",
        parent=park_node,
        critical=True
    )
    ca_claim = f"The park named '{park_name}' is located in the state of California."
    ca_instruction = (
        "Use the provided official pages to confirm the state location. "
        "Look for explicit mentions like 'California' or recognizable California locations. "
        "If none of the URLs mention California or the park location, mark as not supported."
    )

    # 3) Park is a U.S. National Park (critical)
    is_np_node = evaluator.add_leaf(
        id=f"park_{index}_is_national_park",
        desc=f"The {'first' if index == 0 else 'second'} park is a U.S. National Park (not another designation).",
        parent=park_node,
        critical=True
    )
    np_claim = (
        f"'{park_name}' is formally designated as a U.S. National Park (an NPS unit of type 'National Park'), "
        "not a different designation like National Monument, National Seashore, or National Recreation Area."
    )
    np_instruction = (
        "Verify the official designation using the pages. Accept 'National Park' and combined formulations such as "
        "'Sequoia and Kings Canyon National Parks'. Do not accept other unit types."
    )

    # 4) In-park lodging present (critical)
    lodging_node = evaluator.add_leaf(
        id=f"park_{index}_in_park_lodging",
        desc=f"The {'first' if index == 0 else 'second'} park has lodging facilities located inside park boundaries (not only in gateway towns).",
        parent=park_node,
        critical=True
    )
    lodging_claim = (
        f"Official sources confirm that there is lodging located inside the boundaries of {park_name} "
        "(e.g., hotels/lodges/cabins within the park, not merely nearby towns)."
    )
    lodging_instruction = (
        "Confirm the lodging is inside park boundaries. Pages should explicitly state on-park or in-park locations. "
        "Concessioner pages are acceptable if they are the official, authorized concessioner operating in the park."
    )

    # 4a) Lodging URL is official (critical)
    lodging_official_node = evaluator.add_leaf(
        id=f"park_{index}_lodging_url_official",
        desc=f"Provides an official-source reference URL that confirms in-park lodging for the {'first' if index == 0 else 'second'} park.",
        parent=park_node,
        critical=True
    )
    lodging_official_claim = (
        f"This lodging webpage for {park_name} appears to be an official source (NPS.gov or an officially authorized park concessioner), "
        "not a third-party blog or aggregator."
    )
    lodging_official_instruction = (
        "Judge by the webpage content and domain branding. Official sources include NPS.gov or clearly marked official park "
        "concessioners (e.g., Yosemite Hospitality/Aramark, Delaware North, Xanterra where applicable). "
        "If the page is a blog or commercial aggregator without official designation, mark as not official."
    )

    # 5) Free shuttle present (critical)
    shuttle_node = evaluator.add_leaf(
        id=f"park_{index}_free_shuttle",
        desc=f"The {'first' if index == 0 else 'second'} park operates a free shuttle bus system for visitor transportation.",
        parent=park_node,
        critical=True
    )
    shuttle_claim = (
        f"Official sources confirm that {park_name} operates a free shuttle bus system for visitors inside the park."
    )
    shuttle_instruction = (
        "Confirm the shuttle service is free for rides inside the park. If a page only describes paid transit to the park "
        "(e.g., from a gateway town), that does not satisfy this requirement."
    )

    # 5a) Shuttle URL is official (critical)
    shuttle_official_node = evaluator.add_leaf(
        id=f"park_{index}_shuttle_url_official",
        desc=f"Provides an official-source reference URL that confirms the free shuttle system for the {'first' if index == 0 else 'second'} park.",
        parent=park_node,
        critical=True
    )
    shuttle_official_claim = (
        f"This shuttle webpage for {park_name} is an official source (NPS.gov or park-operated/authorized information), "
        "not a third-party blog or unverified commercial site."
    )
    shuttle_official_instruction = (
        "Use the page content and domain to determine official status. Prefer NPS.gov or official park pages. "
        "Reject unverified third-party blogs or generic travel aggregators."
    )

    # 6) Visitor center present (critical)
    vc_node = evaluator.add_leaf(
        id=f"park_{index}_visitor_center",
        desc=f"The {'first' if index == 0 else 'second'} park has at least one visitor center.",
        parent=park_node,
        critical=True
    )
    vc_claim = (
        f"Official sources confirm that {park_name} has at least one visitor center."
    )
    vc_instruction = (
        "Confirm presence of at least one visitor center via official pages. Names of centers or locations should be listed."
    )

    # 6a) Visitor center URL is official (critical)
    vc_official_node = evaluator.add_leaf(
        id=f"park_{index}_visitor_center_url_official",
        desc=f"Provides an official-source reference URL that confirms visitor center(s) for the {'first' if index == 0 else 'second'} park.",
        parent=park_node,
        critical=True
    )
    vc_official_claim = (
        f"This visitor center webpage for {park_name} is an official source (NPS.gov or park-operated), "
        "not a third-party blog or unofficial aggregator."
    )
    vc_official_instruction = (
        "Prefer NPS.gov or clearly official park pages. If the page appears to be non-official or generic, mark as not official."
    )

    # 7) Junior Ranger program present (critical)
    jr_node = evaluator.add_leaf(
        id=f"park_{index}_junior_ranger",
        desc=f"The {'first' if index == 0 else 'second'} park offers a Junior Ranger program.",
        parent=park_node,
        critical=True
    )
    jr_claim = (
        f"Official sources confirm that {park_name} offers a Junior Ranger program for children."
    )
    jr_instruction = (
        "Confirm that the park runs a Junior Ranger program via official pages (NPS or official park program page)."
    )

    # 7a) Junior Ranger URL is official (critical)
    jr_official_node = evaluator.add_leaf(
        id=f"park_{index}_junior_ranger_url_official",
        desc=f"Provides an official-source reference URL that confirms the Junior Ranger program for the {'first' if index == 0 else 'second'} park.",
        parent=park_node,
        critical=True
    )
    jr_official_claim = (
        f"This Junior Ranger webpage for {park_name} is an official source (NPS.gov or park-operated), "
        "not a third-party blog or unofficial site."
    )
    jr_official_instruction = (
        "Prefer NPS.gov or clearly official park pages. Reject pages without clear official provenance."
    )

    # Prepare batch verifications
    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = [
        (ca_claim, all_urls if all_urls else None, in_ca_node, ca_instruction),
        (np_claim, all_urls if all_urls else None, is_np_node, np_instruction),
        (lodging_claim, park.lodging_urls if park.lodging_urls else None, lodging_node, lodging_instruction),
        (lodging_official_claim, park.lodging_urls if park.lodging_urls else None, lodging_official_node, lodging_official_instruction),
        (shuttle_claim, park.shuttle_urls if park.shuttle_urls else None, shuttle_node, shuttle_instruction),
        (shuttle_official_claim, park.shuttle_urls if park.shuttle_urls else None, shuttle_official_node, shuttle_official_instruction),
        (vc_claim, park.visitor_center_urls if park.visitor_center_urls else None, vc_node, vc_instruction),
        (vc_official_claim, park.visitor_center_urls if park.visitor_center_urls else None, vc_official_node, vc_official_instruction),
        (jr_claim, park.junior_ranger_urls if park.junior_ranger_urls else None, jr_node, jr_instruction),
        (jr_official_claim, park.junior_ranger_urls if park.junior_ranger_urls else None, jr_official_node, jr_official_instruction),
    ]

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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California National Parks family trip requirements.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent evaluation of parks
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

    # IMPORTANT: Root node must be non-critical to allow non-critical children, per framework constraints
    root.critical = False

    # Extract parks data from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Pad or trim to ensure we have exactly two parks for evaluation
    parks: List[ParkEntry] = list(extraction.parks)
    if len(parks) < 2:
        # Pad with empty placeholders
        while len(parks) < 2:
            parks.append(ParkEntry())
    else:
        parks = parks[:2]

    # Root-level critical gate: At least two distinct parks identified
    distinct_names = [p.name.strip().lower() for p in extraction.parks if _has_nonempty_string(p.name)]
    distinct_count = len(set(distinct_names))
    evaluator.add_custom_node(
        result=distinct_count >= 2,
        id="at_least_two_parks_identified",
        desc="Response identifies at least two distinct national parks (i.e., two park entries).",
        parent=root,
        critical=True
    )

    # Build verification for park 1 and park 2
    await verify_park(evaluator, root, parks[0], 0)
    await verify_park(evaluator, root, parks[1], 1)

    # Return summary
    return evaluator.get_summary()