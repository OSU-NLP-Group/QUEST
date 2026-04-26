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
TASK_ID = "national_dog_show_2025"
TASK_DESCRIPTION = """Research the 2025 National Dog Show Presented by Purina that was broadcast on NBC on Thanksgiving Day. Provide the following information:

1. The broadcast date (in MM/DD/YYYY format)
2. The broadcast time window (specify start and end times in 12-hour format, e.g., "12:00 PM to 2:00 PM", and mention the time zone or that it's "local time")
3. For the Best in Show winner:
   - The dog's registered name (the formal AKC-registered name with kennel prefixes/suffixes, not just the call name)
   - The dog's breed
   - The handler's full name
   - The handler's home state
4. For the Best in Show winner, identify at least one other major 2025 dog show where this dog won a group competition (meaning the dog placed first in one of the seven groups at that show). Provide:
   - The official name of that dog show
   - The month in which that show was held (e.g., "October")
5. For the Reserve Best in Show winner:
   - The dog's name (either the registered name or the commonly used show name)
   - The dog's breed
   - The handler's full name
   - The handler's home state

For each piece of information, provide at least one supporting URL from an official source (e.g., National Dog Show website, NBC Sports, American Kennel Club, Purina Pro Club) or reputable news outlet.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DogShowExtraction(BaseModel):
    # Broadcast info
    broadcast_date: Optional[str] = None
    broadcast_date_sources: List[str] = Field(default_factory=list)

    broadcast_time_window: Optional[str] = None
    broadcast_time_timezone_or_local: Optional[str] = None  # e.g., "ET", "local time"
    broadcast_time_sources: List[str] = Field(default_factory=list)

    # Best in Show (BIS) details
    bis_registered_name: Optional[str] = None
    bis_registered_name_sources: List[str] = Field(default_factory=list)

    bis_breed: Optional[str] = None
    bis_breed_sources: List[str] = Field(default_factory=list)

    bis_handler_name: Optional[str] = None
    bis_handler_name_sources: List[str] = Field(default_factory=list)

    bis_handler_state: Optional[str] = None
    bis_handler_state_sources: List[str] = Field(default_factory=list)

    # BIS: Additional 2025 major show with group win
    additional_show_name: Optional[str] = None
    additional_show_name_sources: List[str] = Field(default_factory=list)

    additional_show_major_claim_sources: List[str] = Field(default_factory=list)  # URLs that call it "major/premier"
    additional_show_month: Optional[str] = None
    additional_show_month_sources: List[str] = Field(default_factory=list)

    additional_show_group_win_sources: List[str] = Field(default_factory=list)

    # Reserve Best in Show (RBIS) details
    rbis_name: Optional[str] = None
    rbis_name_sources: List[str] = Field(default_factory=list)

    rbis_breed: Optional[str] = None
    rbis_breed_sources: List[str] = Field(default_factory=list)

    rbis_handler_name: Optional[str] = None
    rbis_handler_name_sources: List[str] = Field(default_factory=list)

    rbis_handler_state: Optional[str] = None
    rbis_handler_state_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dog_show() -> str:
    return """
Extract the required information about the 2025 National Dog Show Presented by Purina from the provided answer. Return a single JSON object matching the template fields exactly. Follow these rules:

GENERAL RULES
- Do not invent information; extract only what is explicitly in the answer.
- For each piece of information, list all supporting URLs mentioned in the answer that directly support that specific piece.
- If a field is missing in the answer, set it to null (for strings) or an empty array (for URL lists).
- Only extract URLs that are explicitly present in the answer (plain URLs or in markdown).

FIELDS TO EXTRACT
1) broadcast_date: The broadcast date, ideally in MM/DD/YYYY format (e.g., "11/27/2025") if the answer presents it that way. If the answer uses another format (e.g., "November 27, 2025"), extract exactly as written.
2) broadcast_date_sources: All URLs in the answer that support the broadcast date.

3) broadcast_time_window: The broadcast time window as written, e.g., "12:00 PM to 2:00 PM". Include only what's written.
4) broadcast_time_timezone_or_local: The time zone used (e.g., "ET", "Eastern Time", "PT") OR a phrase like "local time" if the answer states that. If multiple are mentioned, extract the primary one used with the time window.
5) broadcast_time_sources: All URLs in the answer that support the time window and the time zone or local-time convention.

Best in Show (BIS) Winner:
6) bis_registered_name: The dog's formal registered name as presented (with titles/kennel prefixes if included).
7) bis_registered_name_sources: All URLs that support this registered name.

8) bis_breed: The breed as presented for the BIS winner.
9) bis_breed_sources: All URLs that support the breed.

10) bis_handler_name: The handler's full name as presented.
11) bis_handler_name_sources: All URLs that support the handler's name.

12) bis_handler_state: The handler's home state as presented (e.g., "North Carolina").
13) bis_handler_state_sources: All URLs that support the handler's state.

Additional Major 2025 Show where the BIS dog won a GROUP (Group 1) in 2025:
14) additional_show_name: The official show name as presented in the answer.
15) additional_show_name_sources: All URLs that support the official show name.

16) additional_show_major_claim_sources: URLs provided in the answer that explicitly describe this show as "major", "premier", "top-tier", "marquee", or equivalent. If none are provided, return an empty array.

17) additional_show_month: The month (e.g., "October") in which that 2025 show took place, as presented in the answer.
18) additional_show_month_sources: All URLs that support the month/date/timeframe of the 2025 show.

19) additional_show_group_win_sources: All URLs that support that the BIS dog won a group competition (Group 1) at that 2025 show.

Reserve Best in Show (RBIS) Winner:
20) rbis_name: The RBIS dog's name (registered name or commonly used show name) as presented.
21) rbis_name_sources: All URLs that support this RBIS name.

22) rbis_breed: The RBIS breed as presented.
23) rbis_breed_sources: All URLs that support the RBIS breed.

24) rbis_handler_name: The RBIS handler's full name as presented.
25) rbis_handler_name_sources: All URLs that support the RBIS handler's name.

26) rbis_handler_state: The RBIS handler's home state as presented.
27) rbis_handler_state_sources: All URLs that support the RBIS handler's state.

NOTES
- If the answer mentions multiple additional shows, extract only the first one that is clearly identified as a major show with a group win in 2025.
- Maintain exact casing and punctuation for names as written in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december"
}
_TZ_KEYWORDS = {"et", "est", "edt", "eastern", "ct", "cst", "cdt", "central", "mt", "mst", "mdt", "mountain",
                "pt", "pst", "pdt", "pacific", "local time", "your local time", "in all time zones"}
_TITLE_TOKENS = {"CH", "GCH", "GCHB", "GCHG", "BIS", "MBIS", "RBIS", "BISS", "NBISS", "MRBIS", "GRCH", "AM", "CAN", "UKC"}


def _norm(s: Optional[str]) -> str:
    return "" if s is None else s.strip()


def _lower(s: Optional[str]) -> str:
    return _norm(s).lower()


def _sanitize_alnum_lower(s: Optional[str]) -> str:
    s = _norm(s)
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _check_date_is_nov_27_2025(date_str: Optional[str]) -> bool:
    s = _lower(date_str)
    if not s:
        return False
    # textual form
    if ("november" in s or "nov" in s) and "27" in s and "2025" in s:
        return True
    # numeric forms
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s)
    if m:
        mm, dd, yy = m.groups()
        try:
            mm_i, dd_i, yy_i = int(mm), int(dd), int(yy) if len(yy) == 4 else (2000 + int(yy))
            return mm_i == 11 and dd_i == 27 and yy_i == 2025
        except Exception:
            return False
    return False


def _check_mmddyyyy_format(date_str: Optional[str]) -> bool:
    s = _norm(date_str)
    return bool(re.fullmatch(r"\s*(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/20\d{2}\s*", s))


def _check_time_window_12_to_2_pm(window: Optional[str]) -> bool:
    s = _lower(window)
    if not s:
        return False
    # Accept variations like "12 PM - 2 PM", "12:00 PM to 2:00 PM", "12 – 2 p.m."
    # Must indicate 12 and 2 and PM
    has_12 = bool(re.search(r"\b12(:?00)?\s*p\.?m\.?\b", s)) or bool(re.search(r"\b12\s*p\.?m\.?\b", s))
    has_2 = bool(re.search(r"\b2(:?00)?\s*p\.?m\.?\b", s)) or bool(re.search(r"\b2\s*p\.?m\.?\b", s))
    return has_12 and has_2


def _check_12h_format(window: Optional[str]) -> bool:
    s = _lower(window)
    if not s:
        return False
    # Must include AM/PM markers
    return bool(re.search(r"\b(a\.?m\.?|p\.?m\.?)\b", s))


def _check_tz_or_local_mentioned(tz_str: Optional[str], window: Optional[str]) -> bool:
    combined = f"{_lower(tz_str)} {_lower(window)}"
    return any(kw in combined for kw in _TZ_KEYWORDS)


def _check_registered_name_contains_required(name: Optional[str]) -> bool:
    # Constraint substring (as provided by rubric)
    required = "Prairewind's Sxongs Of Summer At La Neige"
    a = _sanitize_alnum_lower(name)
    b = _sanitize_alnum_lower(required)
    return b in a if b else False


def _check_name_includes_titles_and_kennel(name: Optional[str]) -> bool:
    s = _norm(name)
    if not s:
        return False
    tokens = [re.sub(r"[^\w]", "", t) for t in s.split()]
    # Condition 1: Contains common AKC title tokens
    cond1 = any(t.upper() in _TITLE_TOKENS for t in tokens if t)
    # Condition 2: Contains connectors often in reg names with multiple words
    sl = s.lower()
    cond2 = (" of " in sl or " at " in sl or " from " in sl) and (len(tokens) >= 4)
    # Condition 3: Contains kennel-like apostrophe (’ or ')
    cond3 = ("'" in s) or ("’" in s)
    return cond1 or cond2 or cond3


def _check_breed_is_belgian_sheepdog(breed: Optional[str]) -> bool:
    sl = _lower(breed)
    return ("belgian" in sl and "sheepdog" in sl) if sl else False


def _check_tokens_in_name(name: Optional[str], must_tokens: List[str]) -> bool:
    sl = _lower(name)
    return all(tok in sl for tok in must_tokens) if sl else False


def _check_state(value: Optional[str], expected: str, abbrev: Optional[str] = None) -> bool:
    sl = _lower(value)
    if not sl:
        return False
    exp = expected.lower()
    if exp in sl:
        return True
    return abbrev.lower() in sl if abbrev else False


def _check_month_valid(month: Optional[str]) -> bool:
    return _lower(month) in _MONTHS if month else False


def _first_non_empty_name(*names: Optional[str]) -> str:
    for n in names:
        if _norm(n):
            return _norm(n)
    return ""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_broadcast_date(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="broadcast_date",
        desc="Provide the broadcast date in MM/DD/YYYY format with a supporting URL.",
        parent=parent,
        critical=True,
    )

    # Value check (must be Nov 27, 2025)
    evaluator.add_custom_node(
        result=_check_date_is_nov_27_2025(data.broadcast_date),
        id="broadcast_date_value",
        desc="Broadcast date matches the constraint (November 27, 2025).",
        parent=node,
        critical=True,
    )

    # Format check (MM/DD/YYYY)
    evaluator.add_custom_node(
        result=_check_mmddyyyy_format(data.broadcast_date),
        id="broadcast_date_format",
        desc="Broadcast date is presented in MM/DD/YYYY format.",
        parent=node,
        critical=True,
    )

    # Source support
    leaf = evaluator.add_leaf(
        id="broadcast_date_source_url",
        desc="At least one official/reputable URL supports the broadcast date.",
        parent=node,
        critical=True,
    )
    claim = "NBC broadcast of the 2025 National Dog Show occurred on November 27, 2025 (Thanksgiving Day)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.broadcast_date_sources,
        additional_instruction="Verify the page explicitly mentions the 2025 National Dog Show airing on November 27, 2025. Accept official or reputable outlets only.",
    )


async def build_broadcast_time(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="broadcast_time",
        desc="Provide the broadcast time window (start/end) in 12-hour format and specify the time zone or indicate 'local time', with a supporting URL.",
        parent=parent,
        critical=True,
    )

    # Time window 12:00 PM to 2:00 PM
    evaluator.add_custom_node(
        result=_check_time_window_12_to_2_pm(data.broadcast_time_window),
        id="broadcast_time_window_value",
        desc="Time window matches the constraint (12:00 PM to 2:00 PM).",
        parent=node,
        critical=True,
    )

    # 12-hour format
    evaluator.add_custom_node(
        result=_check_12h_format(data.broadcast_time_window),
        id="broadcast_time_12_hour_format",
        desc="Time window is expressed in 12-hour format (includes AM/PM).",
        parent=node,
        critical=True,
    )

    # Time zone or 'local time' mention
    evaluator.add_custom_node(
        result=_check_tz_or_local_mentioned(data.broadcast_time_timezone_or_local, data.broadcast_time_window),
        id="broadcast_time_tz_or_local",
        desc="Answer specifies the time zone or explicitly indicates 'local time' as required.",
        parent=node,
        critical=True,
    )

    # Source support for time and timezone/local-time
    tz_phrase = _first_non_empty_name(data.broadcast_time_timezone_or_local, "local time")
    leaf = evaluator.add_leaf(
        id="broadcast_time_source_url",
        desc="At least one official/reputable URL supports the broadcast time window and the stated timezone/local-time convention.",
        parent=node,
        critical=True,
    )
    claim = f"The 2025 National Dog Show on NBC aired from 12:00 PM to 2:00 PM ({tz_phrase})."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.broadcast_time_sources,
        additional_instruction="Verify that the page shows a 12:00 PM to 2:00 PM broadcast window and mentions the same time zone or explicitly notes 'local time'.",
    )


async def build_bis_registered_name(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="bis_dog_registered_name",
        desc="Provide the dog's formal AKC-registered name (not just the call name) and support it with a URL.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_registered_name_contains_required(data.bis_registered_name),
        id="bis_registered_name_contains_required",
        desc="Registered name contains \"Prairewind's Sxongs Of Summer At La Neige\" (per constraints).",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_name_includes_titles_and_kennel(data.bis_registered_name),
        id="bis_registered_name_includes_titles",
        desc="Registered name is presented with AKC titles and kennel prefix/suffix elements (per constraints), not merely the call name.",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="bis_registered_name_source_url",
        desc="At least one official/reputable URL supports the registered name as provided.",
        parent=node,
        critical=True,
    )
    rn = _first_non_empty_name(data.bis_registered_name)
    claim = f"The 2025 National Dog Show Best in Show dog's registered name is '{rn}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.bis_registered_name_sources,
        additional_instruction="Verify that the page shows the dog's formal registered name (with titles/kennel elements if applicable), not just the call name.",
    )


async def build_bis_breed(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="bis_dog_breed",
        desc="Provide the dog's breed and support it with a URL.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_breed_is_belgian_sheepdog(data.bis_breed),
        id="bis_breed_accuracy",
        desc="Breed is Belgian Sheepdog (per constraints).",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="bis_breed_source_url",
        desc="At least one official/reputable URL supports the breed identification.",
        parent=node,
        critical=True,
    )
    claim = "The 2025 National Dog Show Best in Show winner's breed is Belgian Sheepdog."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.bis_breed_sources,
        additional_instruction="Verify that the cited page explicitly identifies the Best in Show winner as a Belgian Sheepdog.",
    )


async def build_bis_handler_name(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="bis_handler_name",
        desc="Provide the handler's full name and support it with a URL.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_tokens_in_name(data.bis_handler_name, ["daniel", "martin"]),
        id="bis_handler_name_accuracy",
        desc="Handler is Daniel Martin (per constraints).",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="bis_handler_name_source_url",
        desc="At least one official/reputable URL supports the handler's name.",
        parent=node,
        critical=True,
    )
    hn = _first_non_empty_name(data.bis_handler_name)
    claim = f"The Best in Show dog's handler is {hn}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.bis_handler_name_sources,
        additional_instruction="Verify that the page explicitly names the handler for the Best in Show winner.",
    )


async def build_bis_handler_state(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="bis_handler_state",
        desc="Provide the handler's home state and support it with a URL.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_state(data.bis_handler_state, "North Carolina", "NC"),
        id="bis_handler_state_accuracy",
        desc="Handler's home state is North Carolina (per constraints).",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="bis_handler_state_source_url",
        desc="At least one official/reputable URL supports the handler's home state/location.",
        parent=node,
        critical=True,
    )
    hn = _first_non_empty_name(data.bis_handler_name)
    claim = f"The handler {hn if hn else 'the BIS handler'} is from North Carolina."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.bis_handler_state_sources,
        additional_instruction="Verify that the page explicitly states the handler's home state (North Carolina).",
    )


async def build_bis_additional_show(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="bis_additional_show_group_win_2025",
        desc="Identify at least one other major 2025 dog show where the BIS dog won a group competition; provide official show name and month, with supporting URLs for each piece.",
        parent=parent,
        critical=True,
    )

    # Official name
    name_node = evaluator.add_parallel(
        id="additional_show_official_name",
        desc="Provide the official name of the additional 2025 dog show, with a supporting URL.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(_norm(data.additional_show_name)),
        id="additional_show_name_present_and_specific",
        desc="An unambiguous official show name is provided.",
        parent=name_node,
        critical=True,
    )

    leaf_name = evaluator.add_leaf(
        id="additional_show_name_source_url",
        desc="At least one official/reputable URL supports the additional show’s official name.",
        parent=name_node,
        critical=True,
    )
    show_name = _first_non_empty_name(data.additional_show_name)
    claim_name = f"The official name of the 2025 additional show is '{show_name}'."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=data.additional_show_name_sources,
        additional_instruction="Verify that the page clearly shows the official name of the show as provided.",
    )

    # Major show claim
    major_node = evaluator.add_parallel(
        id="additional_show_major_claim_and_source",
        desc="The answer explicitly describes the cited show as a 'major' dog show and provides a supporting URL whose text explicitly characterizes it as major/premier/top-tier (or equivalent).",
        parent=node,
        critical=True,
    )

    major_claim_leaf = evaluator.add_leaf(
        id="major_show_claim_present",
        desc="Answer explicitly claims the additional show is 'major' (or equivalent phrasing).",
        parent=major_node,
        critical=True,
    )
    claim_major_text = f"The answer explicitly describes the show '{show_name}' as 'major', 'premier', 'top-tier', 'prestigious', 'marquee', or an equivalent phrase."
    await evaluator.verify(
        claim=claim_major_text,
        node=major_claim_leaf,
        sources=None,
        additional_instruction="Check the answer content for explicit phrasing labeling the show as major/premier/top-tier (or equivalent).",
    )

    major_src_leaf = evaluator.add_leaf(
        id="major_show_source_url",
        desc="At least one official/reputable URL is provided where the show is explicitly described as major/premier/top-tier (or equivalent).",
        parent=major_node,
        critical=True,
    )
    claim_major_src = f"The page explicitly characterizes the show '{show_name}' as major/premier/top-tier (or equivalent)."
    await evaluator.verify(
        claim=claim_major_src,
        node=major_src_leaf,
        sources=data.additional_show_major_claim_sources,
        additional_instruction="Look for explicit descriptors like 'major', 'premier', 'top-tier', 'prestigious', or similar in the page text.",
    )

    # Month
    month_node = evaluator.add_parallel(
        id="additional_show_month",
        desc="Provide the month when the additional show was held in 2025, with a supporting URL.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_month_valid(data.additional_show_month),
        id="additional_show_month_present_and_valid",
        desc="A calendar month (e.g., 'October') is provided and corresponds to the cited 2025 event timing.",
        parent=month_node,
        critical=True,
    )

    leaf_month = evaluator.add_leaf(
        id="additional_show_month_source_url",
        desc="At least one official/reputable URL supports the month/date of the additional show in 2025.",
        parent=month_node,
        critical=True,
    )
    month_name = _first_non_empty_name(data.additional_show_month)
    claim_month = f"The 2025 '{show_name}' show took place in {month_name} 2025."
    await evaluator.verify(
        claim=claim_month,
        node=leaf_month,
        sources=data.additional_show_month_sources,
        additional_instruction="Verify that the page indicates the 2025 show's timing/month as stated.",
    )

    # Group win evidence
    group_node = evaluator.add_parallel(
        id="additional_show_group_win_evidence",
        desc="Provide evidence (via URL) that the BIS dog won a group competition (placed first in one of the seven groups) at the named 2025 show.",
        parent=node,
        critical=True,
    )

    group_claim_leaf = evaluator.add_leaf(
        id="group_win_claim_present",
        desc="Answer explicitly states the dog won a group (Group 1 placement) at the named show.",
        parent=group_node,
        critical=True,
    )
    claim_group_text = f"The answer explicitly states that the Best in Show dog won a group (Group 1) at '{show_name}' in 2025."
    await evaluator.verify(
        claim=claim_group_text,
        node=group_claim_leaf,
        sources=None,
        additional_instruction="Scan the answer text for an explicit statement that the BIS dog won a Group (Group 1) at the named show.",
    )

    group_src_leaf = evaluator.add_leaf(
        id="group_win_source_url",
        desc="At least one official/reputable URL supports that the dog won a group competition at that show in 2025.",
        parent=group_node,
        critical=True,
    )
    dog_name = _first_non_empty_name(data.bis_registered_name, "the BIS dog")
    claim_group_src = f"The dog {dog_name} won a group (Group 1) at the 2025 show '{show_name}'."
    await evaluator.verify(
        claim=claim_group_src,
        node=group_src_leaf,
        sources=data.additional_show_group_win_sources,
        additional_instruction="Verify that the page clearly indicates a Group 1 win for the named dog at the named 2025 show.",
    )


async def build_bis(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="best_in_show_winner",
        desc="Provide all required Best in Show winner details with supporting URLs for each piece.",
        parent=parent,
        critical=True,
    )

    await build_bis_registered_name(evaluator, node, data)
    await build_bis_breed(evaluator, node, data)
    await build_bis_handler_name(evaluator, node, data)
    await build_bis_handler_state(evaluator, node, data)
    await build_bis_additional_show(evaluator, node, data)


async def build_rbis(evaluator: Evaluator, parent, data: DogShowExtraction):
    node = evaluator.add_parallel(
        id="reserve_best_in_show_winner",
        desc="Provide all required Reserve Best in Show winner details with supporting URLs for each piece.",
        parent=parent,
        critical=True,
    )

    # RBIS name
    name_node = evaluator.add_parallel(
        id="rbis_dog_name",
        desc="Provide the RBIS dog's name (registered or commonly used show name) and support it with a URL.",
        parent=node,
        critical=True,
    )

    # Accuracy according to constraints: "George (or registered name consistent...)"
    # We implement a tolerant check: pass if the provided name contains 'george' or appears to be a registered show name (has title tokens/connectors).
    name_is_george_or_reg = (_lower(data.rbis_name).find("george") != -1) or _check_name_includes_titles_and_kennel(data.rbis_name)
    evaluator.add_custom_node(
        result=name_is_george_or_reg,
        id="rbis_name_accuracy",
        desc="Name correctly identifies the RBIS winner as George (or a registered name consistent with the RBIS winner) per constraints.",
        parent=name_node,
        critical=True,
    )

    leaf_name = evaluator.add_leaf(
        id="rbis_name_source_url",
        desc="At least one official/reputable URL supports the RBIS dog's name.",
        parent=name_node,
        critical=True,
    )
    rbis_n = _first_non_empty_name(data.rbis_name)
    claim_rbis_name = f"The 2025 National Dog Show Reserve Best in Show winner is named {rbis_n} (registered or show name)."
    await evaluator.verify(
        claim=claim_rbis_name,
        node=leaf_name,
        sources=data.rbis_name_sources,
        additional_instruction="Verify the page names the Reserve Best in Show (RBIS) dog as stated.",
    )

    # RBIS breed
    breed_node = evaluator.add_parallel(
        id="rbis_dog_breed",
        desc="Provide the RBIS dog's breed and support it with a URL.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_lower(data.rbis_breed).find("american") != -1 and _lower(data.rbis_breed).find("foxhound") != -1,
        id="rbis_breed_accuracy",
        desc="Breed is American Foxhound (per constraints).",
        parent=breed_node,
        critical=True,
    )

    leaf_breed = evaluator.add_leaf(
        id="rbis_breed_source_url",
        desc="At least one official/reputable URL supports the RBIS dog's breed.",
        parent=breed_node,
        critical=True,
    )
    claim_rbis_breed = "The Reserve Best in Show winner's breed is American Foxhound."
    await evaluator.verify(
        claim=claim_rbis_breed,
        node=leaf_breed,
        sources=data.rbis_breed_sources,
        additional_instruction="Verify the page explicitly identifies the RBIS dog as an American Foxhound.",
    )

    # RBIS handler name
    handler_name_node = evaluator.add_parallel(
        id="rbis_handler_name",
        desc="Provide the RBIS handler's full name and support it with a URL.",
        parent=node,
        critical=True,
    )

    # Accept either "Tristen Lawrence" or "Tristen Miller"
    hn = _lower(data.rbis_handler_name)
    ok_handler = (hn.find("tristen") != -1 and (hn.find("lawrence") != -1 or hn.find("miller") != -1)) if hn else False
    evaluator.add_custom_node(
        result=bool(ok_handler),
        id="rbis_handler_name_accuracy",
        desc="Handler is Tristen Lawrence or Tristen Miller (per constraints).",
        parent=handler_name_node,
        critical=True,
    )

    leaf_rbis_handler = evaluator.add_leaf(
        id="rbis_handler_name_source_url",
        desc="At least one official/reputable URL supports the RBIS handler's name.",
        parent=handler_name_node,
        critical=True,
    )
    rbis_hn = _first_non_empty_name(data.rbis_handler_name)
    claim_rbis_handler = f"The Reserve Best in Show handler is {rbis_hn}."
    await evaluator.verify(
        claim=claim_rbis_handler,
        node=leaf_rbis_handler,
        sources=data.rbis_handler_name_sources,
        additional_instruction="Verify the page names the RBIS handler as stated.",
    )

    # RBIS handler state
    handler_state_node = evaluator.add_parallel(
        id="rbis_handler_state",
        desc="Provide the RBIS handler's home state and support it with a URL.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_check_state(data.rbis_handler_state, "Maryland", "MD"),
        id="rbis_handler_state_accuracy",
        desc="Handler's home state is Maryland (per constraints).",
        parent=handler_state_node,
        critical=True,
    )

    leaf_rbis_state = evaluator.add_leaf(
        id="rbis_handler_state_source_url",
        desc="At least one official/reputable URL supports the RBIS handler's home state/location.",
        parent=handler_state_node,
        critical=True,
    )
    claim_rbis_state = f"The RBIS handler {rbis_hn if rbis_hn else 'the RBIS handler'} is from Maryland."
    await evaluator.verify(
        claim=claim_rbis_state,
        node=leaf_rbis_state,
        sources=data.rbis_handler_state_sources,
        additional_instruction="Verify the page states that the RBIS handler's home state is Maryland.",
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
) -> Dict[str, Any]:
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

    # Optional: record ground-truth constraints for transparency
    evaluator.add_ground_truth({
        "expected_broadcast_date": "11/27/2025 (Thanksgiving Day)",
        "expected_time_window": "12:00 PM to 2:00 PM (with explicit timezone or 'local time')",
        "bis_constraints": {
            "registered_name_contains": "Prairewind's Sxongs Of Summer At La Neige",
            "breed": "Belgian Sheepdog",
            "handler_name": "Daniel Martin",
            "handler_state": "North Carolina",
        },
        "rbis_constraints": {
            "name": "George (or registered name consistent with RBIS winner)",
            "breed": "American Foxhound",
            "handler_name": "Tristen Lawrence or Tristen Miller",
            "handler_state": "Maryland",
        }
    }, gt_type="ground_truth")

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_dog_show(),
        template_class=DogShowExtraction,
        extraction_name="extracted_dog_show_info",
    )

    # Build a top-level critical node mirroring the rubric root (root from Evaluator is non-critical by design)
    top = evaluator.add_parallel(
        id="national_dog_show_2025_research",
        desc="Complete and accurate research of the 2025 National Dog Show Presented by Purina, including broadcast information and winner details, with supporting URLs for each required piece.",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_broadcast_date(evaluator, top, extracted)
    await build_broadcast_time(evaluator, top, extracted)
    await build_bis(evaluator, top, extracted)
    await build_rbis(evaluator, top, extracted)

    # Return summary
    return evaluator.get_summary()