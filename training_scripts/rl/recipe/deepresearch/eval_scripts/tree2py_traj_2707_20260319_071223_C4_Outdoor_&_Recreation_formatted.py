import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohio_accessible_parks_odnr"
TASK_DESCRIPTION = """
I am planning to organize accessible outdoor recreation events in Ohio and need to identify suitable state park locations. Find four official Ohio state parks (managed by the Ohio Department of Natural Resources) that each provide comprehensive accessible facilities for visitors with mobility disabilities.

For each park, the park must have ALL of the following:
1. At least one accessible picnic shelter or accessible picnic area with wheelchair-accessible tables
2. Accessible restroom facilities that meet ADA requirements
3. Either an accessible playground OR an accessible trail (at least one of these two)

For each of the four parks you identify, provide:
- The official name of the state park
- A brief description of the accessible picnic facilities available (shelters or picnic areas)
- A brief description of the accessible restroom facilities
- A brief description of either the accessible playground or accessible trail (specify which)
- A direct link to the park's official page on the Ohio Department of Natural Resources website (ohiodnr.gov) or the official Ohio State Parks website where these accessible facilities are documented
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkItem(BaseModel):
    """Structured info for one park as claimed by the answer."""
    official_name: Optional[str] = None
    picnic_desc: Optional[str] = None
    restroom_desc: Optional[str] = None
    recreation_type: Optional[str] = None  # expected values: "playground" or "trail" (case-insensitive). If unspecified, null.
    recreation_desc: Optional[str] = None
    official_url: Optional[str] = None  # a direct ODNR/Ohio State Parks link for the park page that documents accessibility
    documentation_urls: List[str] = Field(default_factory=list)  # any additional official ODNR/Ohio State Parks URLs cited for this park
    accessible_route_clearance_statement: Optional[str] = None  # if the answer claims 36-inch route clearance (or similar), capture the text


class ParksExtraction(BaseModel):
    """List of parks extracted from the answer."""
    parks: List[ParkItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract all distinct Ohio state parks mentioned in the answer that the user is proposing (or listing) as suitable locations. Do not invent any parks. Return them in the order they appear.

    For each park, extract the following fields strictly from the answer text:
    - official_name: The official park name as given (e.g., "Hocking Hills State Park"). If name is missing, set null.
    - picnic_desc: A brief description (1–2 sentences or phrases) of accessible picnic facilities (shelter or picnic areas with wheelchair-accessible tables) exactly as stated or faithfully summarized from the answer. If not mentioned, set null.
    - restroom_desc: A brief description of accessible restroom facilities as stated or summarized from the answer. If not mentioned, set null.
    - recreation_type: "playground" or "trail" (lowercase) if the answer clearly specifies which accessible amenity is provided; otherwise null.
    - recreation_desc: A brief description for the accessible recreation amenity (playground OR trail) as stated in the answer. If not mentioned, set null.
    - official_url: A single direct link (URL) to the park's official page on the Ohio Department of Natural Resources / official Ohio State Parks site that the answer provides for documenting accessibility. Only include if the URL is explicitly present in the answer and is an ODNR/Ohio State Parks official domain (must contain 'ohiodnr.gov'); otherwise set null.
    - documentation_urls: A list of any other ODNR/Ohio State Parks official URLs explicitly cited in the answer that document the park’s accessible facilities. Only include URLs that contain 'ohiodnr.gov'. Exclude non-official sources.
    - accessible_route_clearance_statement: If the answer explicitly mentions that accessible facilities are on accessible routes with a minimum clearance (e.g., "36 inches"), capture that exact statement or a faithful short quote. Otherwise null.

    Additional extraction rules:
    - If the answer lists more than four parks, extract them all (we will select later). If fewer than four, extract whatever is present.
    - Deduplicate parks by normalized name (case-insensitive with leading/trailing spaces removed). Keep the first occurrence.
    - Do NOT infer or assume missing fields. Use null for any field not present in the answer.
    - For URLs: extract only fully-formed URLs. For markdown links, extract the actual URL target.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def is_official_odnr_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return "ohiodnr.gov" in u  # official ODNR/Ohio State Parks pages


def gather_sources(park: ParkItem) -> List[str]:
    """Return preferred evidence URL list (official only)."""
    urls: List[str] = []
    if park.official_url and is_official_odnr_url(park.official_url):
        urls.append(park.official_url)
    for u in park.documentation_urls:
        if is_official_odnr_url(u):
            urls.append(u)
    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def detect_recreation_type(park: ParkItem) -> Optional[str]:
    if park.recreation_type:
        t = park.recreation_type.strip().lower()
        if t in ("playground", "trail"):
            return t
    # Try to infer from description if type missing
    desc = (park.recreation_desc or "").strip().lower()
    if "playground" in desc:
        return "playground"
    if "trail" in desc or "path" in desc or "all-abilities" in desc or "all abilities" in desc:
        return "trail"
    return None


# --------------------------------------------------------------------------- #
# Verification per-park                                                       #
# --------------------------------------------------------------------------- #
async def verify_one_park(evaluator: Evaluator, parent_node, park: ParkItem, idx: int) -> None:
    """
    Build verification sub-tree and run checks for a single park (index 1-based in descriptions).
    """
    park_num = idx + 1

    # Parent node for this park (non-critical to allow partial credit across parks)
    park_node = evaluator.add_parallel(
        id=f"park_{park_num}",
        desc=f"Park {park_num} evaluation (meets constraints and includes required fields and official documentation link).",
        parent=parent_node,
        critical=False
    )

    # 1) Official name provided (existence check) - critical within this park
    has_name = bool(park.official_name and park.official_name.strip())
    evaluator.add_custom_node(
        result=has_name,
        id=f"park_{park_num}_official_name",
        desc="Provides the official name of the state park.",
        parent=park_node,
        critical=True
    )

    # 2) Have an official ODNR/Ohio State Parks URL present (gating for downstream source-grounded checks)
    #    This node is not explicitly in the rubric, but serves to enforce source-grounding prerequisites.
    official_urls = gather_sources(park)
    evaluator.add_custom_node(
        result=len(official_urls) > 0,
        id=f"park_{park_num}_official_url_present",
        desc="Has at least one official ODNR/Ohio State Parks link (ohiodnr.gov) cited for this park.",
        parent=park_node,
        critical=True
    )

    # 3) ODNR-managed verification (critical)
    odnr_managed_node = evaluator.add_leaf(
        id=f"park_{park_num}_odnr_managed",
        desc="Park is an official Ohio state park managed by the Ohio Department of Natural Resources (ODNR).",
        parent=park_node,
        critical=True
    )
    odnr_claim = f"{park.official_name or 'This park'} is an official Ohio state park managed by the Ohio Department of Natural Resources (ODNR)."
    await evaluator.verify(
        claim=odnr_claim,
        node=odnr_managed_node,
        sources=official_urls,
        additional_instruction="Confirm the page is an official ODNR/Ohio State Parks page for this specific park and that the park is an Ohio State Park managed by ODNR (Ohio Department of Natural Resources / Ohio State Parks & Watercraft)."
    )

    # 4) Accessible picnic facilities (critical)
    picnic_node = evaluator.add_leaf(
        id=f"park_{park_num}_picnic_facilities_and_description",
        desc="Describes at least one accessible picnic shelter or accessible picnic area with wheelchair-accessible tables meeting the specified ADA clear floor area requirement (48 inches by 30 inches).",
        parent=park_node,
        critical=True
    )
    picnic_claim = f"{park.official_name or 'This park'} provides at least one accessible picnic shelter or an accessible picnic area with wheelchair-accessible tables."
    await evaluator.verify(
        claim=picnic_claim,
        node=picnic_node,
        sources=official_urls,
        additional_instruction="Check the official page(s) for terms like 'accessible picnic area', 'accessible picnic shelter', 'wheelchair-accessible tables', or similar. If explicit dimensions are not stated, treat a clear statement of accessibility as meeting ADA intent."
    )

    # 5) Accessible restroom facilities (critical)
    rest_node = evaluator.add_leaf(
        id=f"park_{park_num}_restrooms_and_description",
        desc="Describes accessible restroom facilities meeting the specified ADA stall dimension/fixtures requirement (minimum 60 inches wide by 56–59 inches deep stalls with accessible fixtures).",
        parent=park_node,
        critical=True
    )
    rest_claim = f"{park.official_name or 'This park'} provides accessible restroom facilities that meet ADA requirements."
    await evaluator.verify(
        claim=rest_claim,
        node=rest_node,
        sources=official_urls,
        additional_instruction="Look for 'accessible restrooms', 'ADA-compliant restrooms', or equivalent phrasing on the official ODNR/Ohio State Parks page(s). Numeric stall dimensions may not be listed; a clear accessible/ADA restroom statement suffices."
    )

    # 6) Recreation amenity: accessible playground OR accessible trail (critical)
    rec_type = detect_recreation_type(park)
    rec_node = evaluator.add_leaf(
        id=f"park_{park_num}_recreation_amenity_and_description",
        desc="Provides either an accessible playground OR an accessible trail (at least one) and specifies which one is used, with a brief description.",
        parent=park_node,
        critical=True
    )
    if rec_type == "playground":
        rec_claim = f"{park.official_name or 'This park'} has an accessible playground."
        rec_hint = "Verify the official page(s) mention an accessible or inclusive/universal playground."
    elif rec_type == "trail":
        rec_claim = f"{park.official_name or 'This park'} has at least one accessible trail."
        rec_hint = "Verify the official page(s) mention an accessible trail (e.g., paved/all-abilities/ADA-accessible path)."
    else:
        # If unspecified, accept either playground or trail as long as the official page shows at least one of them
        rec_claim = f"{park.official_name or 'This park'} provides at least one of the following: an accessible playground OR an accessible trail."
        rec_hint = "Pass if the official page(s) clearly show at least one of: accessible playground or accessible trail."
    await evaluator.verify(
        claim=rec_claim,
        node=rec_node,
        sources=official_urls,
        additional_instruction=rec_hint + " Allow reasonable synonyms and phrasing."
    )

    # 7) Accessible route clearance statement (non-critical; evidence may rarely state exact inches)
    route_node = evaluator.add_leaf(
        id=f"park_{park_num}_accessible_route_clearance",
        desc="States that the accessible facilities are located on accessible routes with the specified minimum clearance (36 inches).",
        parent=park_node,
        critical=False  # Adjusted to non-critical due to low likelihood of explicit inch-width documentation on ODNR pages
    )
    route_claim = f"The accessible facilities at {park.official_name or 'this park'} are located on accessible routes with a minimum 36-inch clearance."
    await evaluator.verify(
        claim=route_claim,
        node=route_node,
        sources=official_urls,
        additional_instruction="Only mark as supported if the official page(s) explicitly mention accessible routes/path widths or ADA route compliance implying the clearance. If not mentioned, judge as not supported."
    )

    # 8) Official link documents the accessible features (critical)
    link_doc_node = evaluator.add_leaf(
        id=f"park_{park_num}_official_link_documents_features",
        desc="Provides a direct link to an official ODNR/Ohio State Parks webpage (ohiodnr.gov or official Ohio State Parks site) that documents the accessible facilities described for this park.",
        parent=park_node,
        critical=True
    )
    # Mention at least one of the accessible items so the judge checks for documentation
    sample_feature = "accessible picnic area/shelter or accessible restrooms"
    if rec_type == "playground":
        sample_feature = "accessible picnic area/shelter or accessible restrooms or accessible playground"
    elif rec_type == "trail":
        sample_feature = "accessible picnic area/shelter or accessible restrooms or accessible trail"
    link_doc_claim = f"This official ODNR/Ohio State Parks page for {park.official_name or 'this park'} documents the park's accessible facilities, including at least one of: {sample_feature}."
    await evaluator.verify(
        claim=link_doc_claim,
        node=link_doc_node,
        sources=official_urls,
        additional_instruction="First, ensure the URL is on ohiodnr.gov (official). Then verify the page explicitly documents at least one of the listed accessible features. Minor wording differences are acceptable."
    )


# --------------------------------------------------------------------------- #
# Global checks                                                               #
# --------------------------------------------------------------------------- #
def exactly_four_distinct_names(extracted: ParksExtraction) -> bool:
    names = [normalize_name(p.official_name) for p in extracted.parks if normalize_name(p.official_name)]
    unique_names = list(dict.fromkeys(names))  # preserve order, dedupe
    return len(unique_names) == 4


def get_first_four_parks(extracted: ParksExtraction) -> List[ParkItem]:
    """Return the first four distinct parks by normalized name; pad with empty items if fewer."""
    seen = set()
    selected: List[ParkItem] = []
    for p in extracted.parks:
        n = normalize_name(p.official_name)
        if not n:
            # keep item if we still need padding later; but skip for distinctness
            continue
        if n in seen:
            continue
        selected.append(p)
        seen.add(n)
        if len(selected) == 4:
            break
    # If fewer than 4, pad with empty placeholders
    while len(selected) < 4:
        selected.append(ParkItem())
    return selected


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
    Evaluate an answer for the ODNR accessible parks task.
    """
    # Initialize evaluator with a parallel root so per-park scoring can be aggregated
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

    # 1) Extract structured info from the answer
    extracted_parks = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction"
    )

    # 2) Global critical check: exactly four distinct parks provided in the answer
    evaluator.add_custom_node(
        result=exactly_four_distinct_names(extracted_parks),
        id="global_exactly_four_distinct_parks",
        desc="The response provides exactly four parks and they are all distinct (non-duplicate).",
        parent=root,
        critical=True
    )

    # 3) Build per-park verification under a non-critical group to allow partial credit across parks
    parks_group = evaluator.add_parallel(
        id="parks_group",
        desc="Per-park verification for four parks (each must meet accessibility constraints and have official documentation).",
        parent=root,
        critical=False
    )

    # Select first four distinct parks (pad if fewer)
    parks_to_check = get_first_four_parks(extracted_parks)

    # 4) Verify parks
    for idx, park in enumerate(parks_to_check):
        await verify_one_park(evaluator, parks_group, park, idx)

    # 5) Return structured evaluation summary
    return evaluator.get_summary()