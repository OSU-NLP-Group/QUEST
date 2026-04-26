import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bottlerock_2025_headliners"
TASK_DESCRIPTION = (
    "For the BottleRock Napa Valley 2025 music festival taking place May 23-25, 2025 at the Napa Valley Expo in Napa, "
    "California, identify all three headlining artists. For each headliner, provide: "
    "(1) Confirmation of their status as a headliner at BottleRock Napa Valley 2025, "
    "(2) Their specific performance date (May 23, 24, or 25, 2025), "
    "(3) Their home state or city of origin, "
    "(4) Any album they released in 2024 or 2025 (if applicable), "
    "(5) Whether they received nominations for the 67th Annual Grammy Awards (Feb 2, 2025) and categories if so, "
    "(6) Whether they are performing at any other California music festival in 2025 (and which festival if so), "
    "(7) A reference URL confirming their headliner status at BottleRock Napa Valley 2025."
)

ALLOWED_DATE_SUBSTRINGS = ["May 23", "May 24", "May 25"]
REQUIRED_YEAR = "2025"
VALID_ALBUM_YEARS = {"2024", "2025"}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HeadlinerItem(BaseModel):
    # Core identity
    name: Optional[str] = None

    # Headliner status evidence (reference URLs)
    headliner_reference_urls: List[str] = Field(default_factory=list)

    # Performance date + optional dedicated sources
    performance_date: Optional[str] = None
    performance_date_urls: List[str] = Field(default_factory=list)

    # Origin + optional sources
    origin: Optional[str] = None
    origin_urls: List[str] = Field(default_factory=list)

    # Recent album (2024/2025) or explicit none + optional sources
    recent_album_title: Optional[str] = None
    recent_album_release_year: Optional[str] = None
    recent_album_none: Optional[bool] = None
    recent_album_urls: List[str] = Field(default_factory=list)

    # 67th Grammys (Feb 2, 2025) status + categories + optional sources
    grammy_status: Optional[str] = None  # expected values from answer: "yes", "no" (if explicitly stated). If unspecified -> null
    grammy_categories: List[str] = Field(default_factory=list)
    grammy_urls: List[str] = Field(default_factory=list)

    # Other CA festival 2025 + optional sources
    other_ca_festival: Optional[str] = None  # a festival name like "Coachella" or explicit "none" if stated by the answer
    other_ca_festival_urls: List[str] = Field(default_factory=list)


class HeadlinersExtraction(BaseModel):
    headliners: List[HeadlinerItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_headliners() -> str:
    return """
    Extract up to three BottleRock Napa Valley 2025 headliners from the answer. If more than three are presented, extract only the first three. 
    For each headliner, extract the following fields exactly as presented in the answer:

    REQUIRED FIELDS PER HEADLINER:
    - name: The headliner artist's name.
    - headliner_reference_urls: An array of URL(s) that the answer cites to support that the artist is a BottleRock Napa Valley 2025 headliner. If none are given, return an empty array.
    - performance_date: The artist’s performance date as stated (e.g., "May 23, 2025", or a reasonable variant that clearly means May 23, 2025).
    - performance_date_urls: An array of URL(s) cited in the answer for the performance date. If none, return an empty array.
    - origin: The home state or city of origin of the artist, as stated in the answer.
    - origin_urls: An array of URL(s) cited for origin. If none, return an empty array.
    - recent_album_title: The title of any album the answer claims the artist released in 2024 or 2025. If the answer explicitly claims there is no such album, set this to null.
    - recent_album_release_year: The year (e.g., "2024" or "2025") for the album if one is given; otherwise null.
    - recent_album_none: A boolean indicating whether the answer explicitly says there was no album in 2024 or 2025. 
                         Use true only if the answer explicitly claims none; use false if an album is mentioned; use null if the answer does not say.
    - recent_album_urls: An array of URL(s) cited for the album claim. If none, return an empty array.
    - grammy_status: For the 67th Annual Grammy Awards, set to "yes" if the answer says the artist received nominations, 
                     set to "no" if the answer explicitly states there were no nominations; otherwise null if unspecified.
    - grammy_categories: An array listing categories named in the answer if grammy_status is "yes"; otherwise an empty array.
    - grammy_urls: An array of URL(s) cited for the Grammy information. If none, return an empty array.
    - other_ca_festival: If the answer says the artist is playing another California music festival in 2025, provide the festival name (e.g., "Coachella"). 
                         If the answer explicitly says none, set this field to "none". If the answer is silent, set to null.
    - other_ca_festival_urls: An array of URL(s) cited for the other-festival claim. If none, return an empty array.

    IMPORTANT:
    - Only extract information explicitly present in the answer. Do not invent or infer.
    - For URLs, extract exactly the URLs present in the answer (plain or markdown). If not present, return an empty array for that URL list field.
    - If a field is not mentioned, return null (or an empty array for URL lists).
    - Maintain the original phrasing and casing of names and titles.

    Return a JSON with this schema:
    {
      "headliners": [
        {
          "name": ...,
          "headliner_reference_urls": [...],
          "performance_date": ...,
          "performance_date_urls": [...],
          "origin": ...,
          "origin_urls": [...],
          "recent_album_title": ...,
          "recent_album_release_year": ...,
          "recent_album_none": ...,
          "recent_album_urls": [...],
          "grammy_status": ...,
          "grammy_categories": [...],
          "grammy_urls": [...],
          "other_ca_festival": ...,
          "other_ca_festival_urls": [...]
        }
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _filter_valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if u.startswith("http://") or u.startswith("https://"):
            out.append(u)
        else:
            # Allow protocol-less URLs to pass (Extractor may normalize, but safeguard here)
            if u.startswith("www."):
                out.append("http://" + u)
    return out


def _is_allowed_perf_date(date_str: Optional[str]) -> bool:
    if not date_str:
        return False
    s = date_str.strip()
    if REQUIRED_YEAR not in s:
        return False
    return any(day in s for day in ALLOWED_DATE_SUBSTRINGS)


def _status_is_yes(s: Optional[str]) -> bool:
    if not s:
        return False
    return s.strip().lower() in {"yes", "y", "true", "nominated", "received nominations"}


def _status_is_no(s: Optional[str]) -> bool:
    if not s:
        return False
    return s.strip().lower() in {"no", "n", "false", "none", "not nominated", "no nominations"}


def _text_is_none(s: Optional[str]) -> bool:
    if s is None:
        return False
    return s.strip().lower() in {"none", "n/a", "no", "not applicable", "no album", "none mentioned"}


# --------------------------------------------------------------------------- #
# Verification logic per headliner                                            #
# --------------------------------------------------------------------------- #
async def verify_single_headliner(
    evaluator: Evaluator,
    parent_node,
    headliner: HeadlinerItem,
    idx: int,
    seen_names: Set[str],
) -> None:
    display_idx = idx + 1
    node = evaluator.add_parallel(
        id=f"headliner_{display_idx}",
        desc=f"Information about the {display_idx}{'st' if display_idx==1 else ('nd' if display_idx==2 else 'rd')} headlining artist",
        parent=parent_node,
        critical=False,
    )

    # -- Name provided ---------------------------------------------------- #
    name = (headliner.name or "").strip()
    name_provided = evaluator.add_custom_node(
        result=bool(name),
        id=f"headliner_{idx}_Headliner_Name",
        desc="Provides a clearly identified headliner artist name (factually accurate).",
        parent=node,
        critical=True
    )

    # -- Name distinct for headliners after the first --------------------- #
    if idx > 0:
        distinct = True
        lname = name.lower()
        for prev in seen_names:
            if prev.lower() == lname:
                distinct = False
                break
        evaluator.add_custom_node(
            result=bool(name) and distinct,
            id=f"headliner_{idx}_Headliner_Name_Distinct",
            desc="Headliner name is distinct from previously listed headliners.",
            parent=node,
            critical=True
        )
    if name:
        seen_names.add(name)

    # -- Reference URL existence (critical gating for status) ------------- #
    ref_urls = _filter_valid_urls(headliner.headliner_reference_urls)
    evaluator.add_custom_node(
        result=len(ref_urls) > 0,
        id=f"headliner_{idx}_Reference_URL",
        desc="Provides a reference URL supporting the claim that the artist is a BottleRock Napa Valley 2025 headliner.",
        parent=node,
        critical=True
    )

    # -- Headliner status verification (via URLs) ------------------------- #
    status_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Headliner_Status",
        desc="Confirms the artist is a headliner at BottleRock Napa Valley 2025 (factually accurate).",
        parent=node,
        critical=True
    )
    status_claim = f"{name} is a headliner at BottleRock Napa Valley 2025."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=ref_urls,
        additional_instruction=(
            "Verify the headliner claim using the cited URL(s). Prefer official BottleRock sources, credible press releases, "
            "or established music media outlets. The page should clearly indicate that this artist is a 'headliner' for the "
            "2025 BottleRock Napa Valley festival."
        )
    )

    # -- Performance date (format + verification) ------------------------- #
    perf_date = (headliner.performance_date or "").strip()
    evaluator.add_custom_node(
        result=_is_allowed_perf_date(perf_date),
        id=f"headliner_{idx}_Performance_Date_Format",
        desc="Performance date is specified and is one of May 23, May 24, or May 25, 2025.",
        parent=node,
        critical=True
    )
    perf_urls = _filter_valid_urls(headliner.performance_date_urls) or ref_urls
    perf_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Performance_Date",
        desc="Provides the artist’s specific performance date and it is one of May 23, May 24, or May 25, 2025 (factually accurate).",
        parent=node,
        critical=True
    )
    perf_claim = f"{name} will perform on {perf_date} at BottleRock Napa Valley 2025."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_leaf,
        sources=perf_urls,
        additional_instruction=(
            "Confirm the exact day (May 23, 24, or 25, 2025) of the artist's BottleRock performance. "
            "Accept reasonable variants in date formatting (e.g., 'Friday, May 23, 2025')."
        )
    )

    # -- Artist origin (existence + verification) ------------------------- #
    origin = (headliner.origin or "").strip()
    evaluator.add_custom_node(
        result=bool(origin),
        id=f"headliner_{idx}_Artist_Origin_Provided",
        desc="Provides the artist’s home state or city of origin (present).",
        parent=node,
        critical=True
    )
    origin_urls = _filter_valid_urls(headliner.origin_urls)
    origin_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Artist_Origin",
        desc="Provides the artist’s home state or city of origin (factually accurate).",
        parent=node,
        critical=True
    )
    origin_claim = f"{name} is from {origin}."
    await evaluator.verify(
        claim=origin_claim,
        node=origin_leaf,
        sources=origin_urls if origin_urls else None,
        additional_instruction=(
            "Verify the artist's origin using cited sources such as official sites, credible biographies, or reliable media. "
            "Allow minor variants (e.g., city vs. state vs. city+state)."
        )
    )

    # -- Recent album 2024/2025 or explicit none -------------------------- #
    album_title = (headliner.recent_album_title or "").strip() if headliner.recent_album_title else None
    album_year = (headliner.recent_album_release_year or "").strip() if headliner.recent_album_release_year else None
    album_none = bool(headliner.recent_album_none) if headliner.recent_album_none is not None else None
    album_urls = _filter_valid_urls(headliner.recent_album_urls)

    recent_album_specified = False
    if album_none is True:
        recent_album_specified = True
    elif album_title and album_year and (album_year in VALID_ALBUM_YEARS):
        recent_album_specified = True

    evaluator.add_custom_node(
        result=recent_album_specified,
        id=f"headliner_{idx}_Recent_Album_Specified",
        desc="Identifies any album released in 2024 or 2025 or explicitly states none.",
        parent=node,
        critical=True
    )

    recent_album_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Recent_Album",
        desc="Identifies any album released by the artist in 2024 or 2025, or explicitly states that no such album exists (factually accurate).",
        parent=node,
        critical=True
    )

    if album_none is True:
        album_claim = f"{name} did not release any album in 2024 or 2025."
        await evaluator.verify(
            claim=album_claim,
            node=recent_album_leaf,
            sources=album_urls if album_urls else None,
            additional_instruction=(
                "If URLs are provided (e.g., official discography or credible databases), check that no album release is listed in 2024 or 2025. "
                "If no URLs are provided, base the judgment on the statement as presented."
            )
        )
    else:
        # Expecting a concrete album and valid year
        album_claim = f"{name} released an album titled '{album_title}' in {album_year}."
        await evaluator.verify(
            claim=album_claim,
            node=recent_album_leaf,
            sources=album_urls if album_urls else None,
            additional_instruction=(
                "Verify that the cited album is indeed an official album release in 2024 or 2025 (not merely a single or EP), "
                "using the provided URLs (official announcements, credible music databases, or reputable media)."
            )
        )

    # -- 67th Grammys nominations (existence + verification) -------------- #
    grammy_status = (headliner.grammy_status or "").strip().lower() if headliner.grammy_status else None
    grammy_categories = headliner.grammy_categories or []
    grammy_urls = _filter_valid_urls(headliner.grammy_urls)

    grammy_info_provided = False
    if _status_is_yes(grammy_status):
        grammy_info_provided = len(grammy_categories) > 0
    elif _status_is_no(grammy_status):
        grammy_info_provided = True
    else:
        grammy_info_provided = False

    evaluator.add_custom_node(
        result=grammy_info_provided,
        id=f"headliner_{idx}_Grammy_Info_Provided",
        desc="States whether the artist received nominations for the 67th Annual Grammy Awards and categories if applicable.",
        parent=node,
        critical=True
    )

    grammy_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Grammy_Nomination_2025",
        desc="States whether the artist received nominations for the 67th Annual Grammy Awards; if yes, lists the category/categories; if no, explicitly states none (factually accurate).",
        parent=node,
        critical=True
    )

    if _status_is_yes(grammy_status):
        cat_text = ", ".join(grammy_categories) if grammy_categories else "unspecified categories"
        grammy_claim = f"{name} received nominations for the 67th Annual Grammy Awards in the following categories: {cat_text}."
        await evaluator.verify(
            claim=grammy_claim,
            node=grammy_leaf,
            sources=grammy_urls if grammy_urls else None,
            additional_instruction=(
                "Confirm nominations for the 67th Annual Grammy Awards (ceremony Feb 2, 2025). "
                "Prefer official Grammy sources or highly credible media. Verify categories if provided."
            )
        )
    else:
        grammy_claim = f"{name} did not receive any nominations for the 67th Annual Grammy Awards."
        await evaluator.verify(
            claim=grammy_claim,
            node=grammy_leaf,
            sources=grammy_urls if grammy_urls else None,
            additional_instruction=(
                "If URLs are provided (e.g., official Grammy listings), verify that this artist does not appear among nominees. "
                "If none are provided, base judgment on the statement as presented."
            )
        )

    # -- Other CA festival in 2025 (existence + verification) ------------- #
    other_fest = (headliner.other_ca_festival or "").strip() if headliner.other_ca_festival else None
    other_fest_urls = _filter_valid_urls(headliner.other_ca_festival_urls)

    other_info_provided = False
    if other_fest is None:
        other_info_provided = False
    elif _text_is_none(other_fest):
        other_info_provided = True
    else:
        other_info_provided = True  # festival named

    evaluator.add_custom_node(
        result=other_info_provided,
        id=f"headliner_{idx}_Other_CA_Festival_Provided",
        desc="States whether the artist is performing at any other California music festival in 2025 (or explicitly none).",
        parent=node,
        critical=True
    )

    other_leaf = evaluator.add_leaf(
        id=f"headliner_{idx}_Other_CA_Festival_2025",
        desc="States whether the artist is performing at any other California music festival in 2025; if yes, names the festival; if no, explicitly states none (factually accurate).",
        parent=node,
        critical=True
    )

    if other_fest and not _text_is_none(other_fest):
        other_claim = f"{name} is performing at {other_fest} in California in 2025."
        await evaluator.verify(
            claim=other_claim,
            node=other_leaf,
            sources=other_fest_urls if other_fest_urls else None,
            additional_instruction=(
                "Verify that the named festival is held in California and that the artist is on the 2025 lineup. "
                "Prefer official festival lineups or credible media coverage."
            )
        )
    else:
        other_claim = f"{name} is not performing at any other California music festival in 2025."
        await evaluator.verify(
            claim=other_claim,
            node=other_leaf,
            sources=other_fest_urls if other_fest_urls else None,
            additional_instruction=(
                "If URLs include a comprehensive 2025 tour schedule or official listings that support the negative claim, use them. "
                "If not available, base the judgment on the statement as presented."
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
    Evaluate an answer for the BottleRock Napa Valley 2025 headliners task using the Mind2Web2 framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root per rubric
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

    # Extract up to 3 headliners
    extracted = await evaluator.extract(
        prompt=prompt_extract_headliners(),
        template_class=HeadlinersExtraction,
        extraction_name="extracted_headliners"
    )

    headliners: List[HeadlinerItem] = list(extracted.headliners or [])[:3]
    while len(headliners) < 3:
        headliners.append(HeadlinerItem())

    # Build verification tree per headliner (parallel children)
    seen_names: Set[str] = set()
    parent_node = evaluator.add_parallel(
        id="BottleRock_Napa_Valley_2025_Headliners",
        desc="Identify the three BottleRock Napa Valley 2025 headlining artists and provide the required attributes for each.",
        parent=root,
        critical=False
    )

    for i in range(3):
        await verify_single_headliner(evaluator, parent_node, headliners[i], i, seen_names)

    return evaluator.get_summary()