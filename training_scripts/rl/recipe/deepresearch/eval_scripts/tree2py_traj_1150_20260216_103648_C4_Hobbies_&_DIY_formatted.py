import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "christmas_eve_2025_store_hours_us"
TASK_DESCRIPTION = (
    "On Christmas Eve 2025 (December 24, 2025), a DIY hobbyist in the United States plans to shop at multiple stores "
    "for craft and home improvement supplies. They want to visit all national retail stores that close at or before "
    "6:00 PM to maximize their shopping time. Among the following six national retail chains: Hobby Lobby, Michaels, "
    "JoAnn Fabrics, Home Depot, Lowe's, and Dollar General, identify which stores have Christmas Eve 2025 closing "
    "times at or before 6:00 PM. List these stores in chronological order from earliest closing time to latest closing "
    "time, and provide the exact closing time for each store."
)

EXPECTED_CLOSINGS: Dict[str, str] = {
    "JoAnn Fabrics": "4:30 PM",
    "Home Depot": "5:00 PM",
    "Hobby Lobby": "5:30 PM",
    "Lowe's": "6:00 PM",
    "Michaels": "6:00 PM",
}
ALL_CHAIN_NAMES = ["Hobby Lobby", "Michaels", "JoAnn Fabrics", "Home Depot", "Lowe's", "Dollar General"]
EXPECTED_STORES_SET = set(EXPECTED_CLOSINGS.keys())
DOLLAR_GENERAL = "Dollar General"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StoreItem(BaseModel):
    store_name: Optional[str] = None
    closing_time: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StoresExtraction(BaseModel):
    stores: List[StoreItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_stores() -> str:
    return """
    Extract the list of national retail chains that the answer claims close at or before 6:00 PM on Christmas Eve 2025 (December 24, 2025) in the United States.
    Preserve the order exactly as presented in the answer.

    Only consider stores among the following six chains:
    - Hobby Lobby
    - Michaels
    - JoAnn Fabrics (may be spelled as JOANN, Jo-Ann, JOANN Fabrics, etc.)
    - Home Depot (may be spelled as The Home Depot)
    - Lowe's (may appear as Lowes or Lowe’s)
    - Dollar General

    For each listed store in the answer, extract:
    - store_name: Normalize to one of the six chain names above. If a variant (e.g., "JOANN") appears, return the normalized canonical name (e.g., "JoAnn Fabrics").
    - closing_time: The exact text of the closing time for Christmas Eve 2025 as stated (e.g., "4:30 PM", "5 PM", "6:00 pm").
    - sources: All URLs that the answer explicitly cites as evidence for that store's hours. Include only valid URLs. If none are provided for that store, return an empty list.

    Return a JSON object:
    {
      "stores": [
        {"store_name": "...", "closing_time": "...", "sources": ["...", "..."]},
        ...
      ]
    }

    Notes:
    - Do not invent or infer any URLs; only extract those explicitly present in the answer text.
    - Keep the store order the same as in the answer.
    - If the answer mentions any of the six stores but does not provide a closing time, set closing_time to null.
    - If the answer lists other stores not in the six specified chains, ignore them.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
_SYNONYMS = {
    "joann": "JoAnn Fabrics",
    "joann fabrics": "JoAnn Fabrics",
    "jo-ann": "JoAnn Fabrics",
    "jo-ann fabrics": "JoAnn Fabrics",
    "joann fabrics and crafts": "JoAnn Fabrics",
    "joann's": "JoAnn Fabrics",
    "the home depot": "Home Depot",
    "home depot": "Home Depot",
    "hobby lobby": "Hobby Lobby",
    "lowes": "Lowe's",
    "lowe's": "Lowe's",
    "lowe’s": "Lowe's",
    "michaels": "Michaels",
    "michaels stores": "Michaels",
    "dollar general": "Dollar General",
}

def canonicalize_store_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    if s in _SYNONYMS:
        return _SYNONYMS[s]
    # Try to map simple cases (e.g., "the home depot")
    s_nopunct = re.sub(r"[^a-z0-9\s']", "", s).strip()
    if s_nopunct in _SYNONYMS:
        return _SYNONYMS[s_nopunct]
    # Final fallback: direct title-case if exact in the six chains
    for chain in ALL_CHAIN_NAMES:
        if s_nopunct == re.sub(r"[^a-z0-9\s']", "", chain.lower()):
            return chain
    return None

def find_store_item(extraction: StoresExtraction, canonical_store: str) -> Optional[StoreItem]:
    for item in extraction.stores:
        can = canonicalize_store_name(item.store_name)
        if can == canonical_store:
            return item
    return None

def parse_time_to_minutes(time_str: Optional[str]) -> Optional[int]:
    if not time_str:
        return None
    s = time_str.strip().lower()
    s = s.replace(".", "")  # p.m. -> pm
    s = s.replace(" ", "")
    # must contain am/pm
    ampm = None
    if "am" in s:
        ampm = "am"
        s = s.split("am")[0]
    elif "pm" in s:
        ampm = "pm"
        s = s.split("pm")[0]
    else:
        return None
    # now s is the numeric part (e.g., "430", "4:30", "5", "600")
    s = s.replace("：", ":")  # handle full-width colon
    s = s.replace(".", ":")
    h = None
    m = 0
    if ":" in s:
        parts = s.split(":", 1)
        if parts[0].isdigit():
            h = int(parts[0])
        else:
            return None
        if parts[1].isdigit():
            m = int(parts[1])
        else:
            return None
    else:
        if s.isdigit():
            if len(s) <= 2:
                h = int(s)
                m = 0
            elif len(s) == 3:
                h = int(s[:-2])
                m = int(s[-2:])
            elif len(s) == 4:
                h = int(s[:-2])
                m = int(s[-2:])
            else:
                return None
        else:
            return None
    if h < 1 or h > 12 or m < 0 or m > 59:
        return None
    # convert to minutes
    h24 = h % 12
    if ampm == "pm":
        h24 += 12
    return h24 * 60 + m

def is_non_decreasing(seq: List[int]) -> bool:
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_store_closing(
    evaluator: Evaluator,
    parent_node,
    extraction: StoresExtraction,
    canonical_store: str,
    expected_time: str,
    node_id: str,
    node_desc: str,
) -> None:
    """
    Verify a single store's closing time against cited sources (if any).
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False  # As per rubric: non-critical
    )

    item = find_store_item(extraction, canonical_store)
    sources_list = item.sources if item and item.sources else []

    claim = (
        f"On Christmas Eve 2025 (December 24, 2025) in the United States, the national retail chain {canonical_store} "
        f"closes at {expected_time} local time (i.e., at or before 6:00 PM)."
    )
    add_ins = (
        "Verify that at least one provided URL explicitly supports the claimed Christmas Eve 2025 closing time for the chain. "
        "Accept corporate announcements, official store or corporate pages, and credible national news releases that clearly state the chain’s 2025 Christmas Eve hours. "
        "The page should mention 'Christmas Eve' (Dec 24) and the specific closing time. "
        "Pages that only say 'hours vary by location' without giving the concrete closing time should not be considered supportive."
    )

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the answer for identifying national retail chains that close at or before 6:00 PM
    on Christmas Eve 2025 among the specified six chains, checking correctness, ordering, and exclusions.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # rubric root is parallel
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

    # Create a top-level rubric node to mirror the provided rubric tree
    main_node = evaluator.add_parallel(
        id="Stores_Closing_At_Or_Before_6PM",
        desc="Complete and accurate identification of all stores closing at or before 6:00 PM on Christmas Eve 2025, listed in chronological order by closing time",
        parent=root,
        critical=False
    )

    # Extract structured store list from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_stores(),
        template_class=StoresExtraction,
        extraction_name="stores_extraction"
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth(
        gt_info={"expected_closing_times": EXPECTED_CLOSINGS, "considered_chains": ALL_CHAIN_NAMES},
        gt_type="expected_store_hours"
    )

    # Verification leaves for each store (non-critical)
    await verify_store_closing(
        evaluator,
        main_node,
        extracted,
        canonical_store="JoAnn Fabrics",
        expected_time=EXPECTED_CLOSINGS["JoAnn Fabrics"],
        node_id="JoAnn_Fabrics_Identification",
        node_desc="JoAnn Fabrics is correctly identified as closing at 4:30 PM on Christmas Eve 2025"
    )

    await verify_store_closing(
        evaluator,
        main_node,
        extracted,
        canonical_store="Home Depot",
        expected_time=EXPECTED_CLOSINGS["Home Depot"],
        node_id="Home_Depot_Identification",
        node_desc="Home Depot is correctly identified as closing at 5:00 PM on Christmas Eve 2025"
    )

    await verify_store_closing(
        evaluator,
        main_node,
        extracted,
        canonical_store="Hobby Lobby",
        expected_time=EXPECTED_CLOSINGS["Hobby Lobby"],
        node_id="Hobby_Lobby_Identification",
        node_desc="Hobby Lobby is correctly identified as closing at 5:30 PM on Christmas Eve 2025"
    )

    await verify_store_closing(
        evaluator,
        main_node,
        extracted,
        canonical_store="Lowe's",
        expected_time=EXPECTED_CLOSINGS["Lowe's"],
        node_id="Lowes_Identification",
        node_desc="Lowe's is correctly identified as closing at 6:00 PM on Christmas Eve 2025"
    )

    await verify_store_closing(
        evaluator,
        main_node,
        extracted,
        canonical_store="Michaels",
        expected_time=EXPECTED_CLOSINGS["Michaels"],
        node_id="Michaels_Identification",
        node_desc="Michaels is correctly identified as closing at 6:00 PM on Christmas Eve 2025"
    )

    # Critical: No incorrect inclusions (e.g., Dollar General) in the list
    included_canonical = [canonicalize_store_name(it.store_name) for it in extracted.stores if it.store_name]
    included_canonical = [c for c in included_canonical if c is not None]
    # Fail if Dollar General is included or any included store is outside the expected <=6pm set
    no_incorrect_inclusions = (DOLLAR_GENERAL not in included_canonical) and all(
        name in EXPECTED_STORES_SET for name in included_canonical
    )
    evaluator.add_custom_node(
        result=no_incorrect_inclusions,
        id="No_Incorrect_Inclusions",
        desc="No stores that close after 6:00 PM (such as Dollar General) are incorrectly included in the list",
        parent=main_node,
        critical=True
    )

    # Critical: Chronological order check for identified stores by their (claimed) closing times
    # Only consider recognized stores among the expected set, in the extraction order
    recognized_times: List[int] = []
    recognized_ok = True
    for it in extracted.stores:
        can = canonicalize_store_name(it.store_name)
        if can and can in EXPECTED_STORES_SET:
            minutes = parse_time_to_minutes(it.closing_time)
            if minutes is None:
                recognized_ok = False
                break
            recognized_times.append(minutes)
    chronological_ok = recognized_ok and (len(recognized_times) <= 1 or is_non_decreasing(recognized_times))

    evaluator.add_custom_node(
        result=chronological_ok,
        id="Chronological_Order",
        desc="All identified stores are listed in chronological order by their closing time (earliest to latest)",
        parent=main_node,
        critical=True
    )

    # Return evaluation summary
    return evaluator.get_summary()