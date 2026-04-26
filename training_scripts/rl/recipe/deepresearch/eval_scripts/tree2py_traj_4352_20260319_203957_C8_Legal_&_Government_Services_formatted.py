import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "congress_119_bills_enacted"
TASK_DESCRIPTION = (
    "Identify four pieces of legislation from the 119th Congress (2025-2026) that have successfully "
    "completed the entire legislative process by passing both the House of Representatives and the Senate "
    "and receiving presidential action (signed into law or allowed to become law). For each of the four bills, "
    "provide: 1) Bill Number and Title, 2) Congress.gov Link, 3) House Passage Details (date in MM/DD/YYYY, "
    "final passage vote Yeas–Nays, and roll call number), 4) Senate Passage Details (date in MM/DD/YYYY, final "
    "passage vote Yeas–Nays, roll call number, and cloture handling with ≥60 votes if required), and 5) Presidential "
    "Action (date in MM/DD/YYYY and action type). Verify using official government sources (Congress.gov, "
    "House Clerk, Senate.gov, etc.). Ensure documented votes are for final passage (not procedural/committee)."
)

ALLOWED_OFFICIAL_DOMAINS = [
    "congress.gov",
    "clerk.house.gov",
    "house.gov",
    "senate.gov",
    "govinfo.gov",
    "whitehouse.gov",
    "federalregister.gov",
]


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class HousePassage(BaseModel):
    date: Optional[str] = None  # MM/DD/YYYY as presented in the answer
    vote_count: Optional[str] = None  # "Yeas-Nays", e.g., "310-118"
    roll_call: Optional[str] = None  # e.g., "Roll No. 123"
    urls: List[str] = Field(default_factory=list)  # House Clerk vote page(s) or related official links


class SenatePassage(BaseModel):
    date: Optional[str] = None
    vote_count: Optional[str] = None
    roll_call: Optional[str] = None
    cloture_required: Optional[bool] = None  # True/False if debate required cloture
    cloture_vote: Optional[str] = None  # if required, vote count string, e.g., "65-35"
    cloture_roll_call: Optional[str] = None  # optional roll call identifier for the cloture vote
    urls: List[str] = Field(default_factory=list)  # Senate roll call page(s) or related official links


class PresidentialAction(BaseModel):
    date: Optional[str] = None  # MM/DD/YYYY
    action_type: Optional[str] = None  # "signed into law" or "became law without signature"
    urls: List[str] = Field(default_factory=list)  # Congress.gov action tab, White House, etc.


class LegislativeItem(BaseModel):
    bill_number: Optional[str] = None  # e.g., "H.R.1234" or "S.567"
    bill_title: Optional[str] = None
    congress_gov_url: Optional[str] = None
    house: Optional[HousePassage] = None
    senate: Optional[SenatePassage] = None
    pres_action: Optional[PresidentialAction] = None
    other_sources: List[str] = Field(default_factory=list)  # any additional official references


class LegislativeExtraction(BaseModel):
    items: List[LegislativeItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_legislation() -> str:
    return """
    Extract up to four legislative items from the answer that satisfy ALL of the following:
    – They are from the 119th Congress (2025–2026).
    – They passed both the House and Senate.
    – They received presidential action (signed into law or became law without signature).

    For each item, return an object with fields:
    - bill_number: string in the form "H.R.####" or "S.####" (allow optional space after the period).
    - bill_title: the complete official title as presented in the answer (aim to use the official title, not a short title).
    - congress_gov_url: a direct URL to the bill's page on Congress.gov (not a search/home page).
    - house: {
        "date": House final passage date in the answer (MM/DD/YYYY format in the answer),
        "vote_count": final passage vote count (in "Yeas-Nays" format, e.g., "310-118"),
        "roll_call": final passage roll call number string as in the answer (e.g., "Roll No. 123", "#123"),
        "urls": array of official links for verifying the House passage (e.g., clerk.house.gov vote page(s) or Congress.gov action tab)
      }
    - senate: {
        "date": Senate final passage date in the answer (MM/DD/YYYY format in the answer),
        "vote_count": final passage vote count (in "Yeas-Nays" format),
        "roll_call": final passage roll call number string,
        "cloture_required": boolean (true if a cloture motion was required to end debate; false if not required),
        "cloture_vote": the cloture vote count string if cloture_required is true (e.g., "65-35"), otherwise null,
        "cloture_roll_call": roll call identifier string for cloture vote if available (optional),
        "urls": array of official links for verifying the Senate passage and cloture (e.g., senate.gov roll call pages, Congress.gov)
      }
    - pres_action: {
        "date": presidential action/enactment date in the answer (MM/DD/YYYY format),
        "action_type": either "signed into law" or "became law without signature",
        "urls": array of official links (e.g., Congress.gov Actions tab, WhiteHouse.gov statement, govinfo.gov, Federal Register)
      }
    - other_sources: array of any additional official sources mentioned in the answer (leave empty if none).

    IMPORTANT:
    - Extract only what is explicitly present in the answer; do not invent or infer.
    - If any field is missing in the answer, set it to null (or an empty array for URL lists).
    - Prefer official government URLs where available: congress.gov, clerk.house.gov, senate.gov, govinfo.gov, whitehouse.gov, federalregister.gov.
    - Dates you return should be EXACTLY as formatted in the answer (ideally MM/DD/YYYY), not reformatted by you.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_http_url(u: Optional[str]) -> bool:
    return isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not _is_http_url(u):
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def collect_official_urls(item: LegislativeItem) -> List[str]:
    urls: List[str] = []
    if item.congress_gov_url:
        urls.append(item.congress_gov_url)
    if item.house and item.house.urls:
        urls.extend(item.house.urls)
    if item.senate and item.senate.urls:
        urls.extend(item.senate.urls)
    if item.pres_action and item.pres_action.urls:
        urls.extend(item.pres_action.urls)
    if item.other_sources:
        urls.extend(item.other_sources)
    return _unique_urls(urls)


def collect_house_urls(item: LegislativeItem) -> List[str]:
    urls: List[str] = []
    if item.house and item.house.urls:
        urls.extend(item.house.urls)
    if item.congress_gov_url:
        urls.append(item.congress_gov_url)
    return _unique_urls(urls)


def collect_senate_urls(item: LegislativeItem) -> List[str]:
    urls: List[str] = []
    if item.senate and item.senate.urls:
        urls.extend(item.senate.urls)
    if item.congress_gov_url:
        urls.append(item.congress_gov_url)
    return _unique_urls(urls)


def collect_pres_urls(item: LegislativeItem) -> List[str]:
    urls: List[str] = []
    if item.pres_action and item.pres_action.urls:
        urls.extend(item.pres_action.urls)
    if item.congress_gov_url:
        urls.append(item.congress_gov_url)
    return _unique_urls(urls)


def bill_number_format_ok(bill_number: Optional[str]) -> bool:
    if not bill_number or not isinstance(bill_number, str):
        return False
    # Accept H.R.#### or S.#### with optional space after the period, digits length 1-5
    pattern = r"^(H\.R\.|S\.)\s?\d{1,5}$"
    return re.match(pattern, bill_number.strip()) is not None


def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification for one legislative item                                       #
# --------------------------------------------------------------------------- #
async def verify_one_item(evaluator: Evaluator, parent_node, item: LegislativeItem, idx: int) -> None:
    # Create the item parent node (parallel; non-critical to allow partial credit per item)
    item_node = evaluator.add_parallel(
        id=f"Legislative_Item_{idx+1}",
        desc=f"Item {idx+1} (one qualifying bill) with all required fields.",
        parent=parent_node,
        critical=False
    )

    # 1) Congress_119 (critical leaf)
    congress_leaf = evaluator.add_leaf(
        id=f"Congress_119_{idx+1}",
        desc="Bill is from the 119th Congress (2025–2026).",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This Congress.gov page indicates that the bill {_safe(item.bill_number) or 'shown here'} is from the 119th Congress (2025–2026).",
        node=congress_leaf,
        sources=item.congress_gov_url,
        additional_instruction="Confirm that the page clearly shows '119th Congress' for this bill. Do not accept pages for prior Congresses."
    )

    # 2) Bill_Number (critical; format check)
    bn_ok = bill_number_format_ok(item.bill_number)
    evaluator.add_custom_node(
        result=bn_ok,
        id=f"Bill_Number_{idx+1}",
        desc="Bill number provided in required format (H.R.#### or S.####).",
        parent=item_node,
        critical=True
    )

    # 3) Bill_Title (critical; verify against Congress.gov)
    title_leaf = evaluator.add_leaf(
        id=f"Bill_Title_{idx+1}",
        desc="Complete official bill title is provided.",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official title of {_safe(item.bill_number) or 'this bill'} is exactly: \"{_safe(item.bill_title)}\".",
        node=title_leaf,
        sources=item.congress_gov_url,
        additional_instruction=(
            "Compare the provided title to the 'Official Title' on Congress.gov for this bill. "
            "Allow only trivial whitespace or punctuation differences; the title must be substantively identical and complete "
            "(not merely a short title)."
        )
    )

    # 4) CongressGov_Link (critical; verify direct page)
    link_leaf = evaluator.add_leaf(
        id=f"CongressGov_Link_{idx+1}",
        desc="Direct URL to the bill’s Congress.gov page is provided.",
        parent=item_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This URL is a direct Congress.gov bill page for {_safe(item.bill_number) or 'this bill'}, not a search or home page.",
        node=link_leaf,
        sources=item.congress_gov_url,
        additional_instruction="The page should be the canonical Congress.gov bill page for the 119th Congress, not a search results page."
    )

    # 5) House_Passage (parent; critical)
    house_parent = evaluator.add_parallel(
        id=f"House_Passage_{idx+1}",
        desc="House final passage details are provided (final passage, not procedural/committee).",
        parent=item_node,
        critical=True
    )

    # 5.1) House_Date (critical leaf)
    house_date_leaf = evaluator.add_leaf(
        id=f"House_Date_{idx+1}",
        desc="House final passage date is provided in MM/DD/YYYY format.",
        parent=house_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The House of Representatives' final passage date for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.house.date)} (as provided in MM/DD/YYYY format). This is the final passage vote date, "
            f"not a procedural or committee vote."
        ),
        node=house_date_leaf,
        sources=collect_house_urls(item),
        additional_instruction=(
            "Use the House Clerk roll call page or Congress.gov 'All Actions' to confirm the final passage date. "
            "Ensure that the referenced vote is 'On Passage' or equivalent final-passage label, not a procedural vote. "
            "Also check that the string provided in the answer is in MM/DD/YYYY format."
        )
    )

    # 5.2) House_VoteCount (critical leaf)
    house_votes_leaf = evaluator.add_leaf(
        id=f"House_VoteCount_{idx+1}",
        desc="House final passage vote count is provided in Yeas-Nays format.",
        parent=house_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The final passage vote count in the House for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.house.vote_count)} (Yeas–Nays)."
        ),
        node=house_votes_leaf,
        sources=collect_house_urls(item),
        additional_instruction=(
            "Confirm the YEAS and NAYS totals for the House final passage vote (not procedural). "
            "Accept minor formatting variations (e.g., spaces around the dash)."
        )
    )

    # 5.3) House_RollCall (critical leaf)
    house_rc_leaf = evaluator.add_leaf(
        id=f"House_RollCall_{idx+1}",
        desc="House final passage roll call number is provided.",
        parent=house_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The House final passage roll call number for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.house.roll_call)}."
        ),
        node=house_rc_leaf,
        sources=collect_house_urls(item),
        additional_instruction=(
            "Verify on clerk.house.gov (preferred) or Congress.gov that the referenced roll call number corresponds "
            "to the final passage vote. Allow reasonable formatting variants such as 'Roll No. 123' vs '#123'."
        )
    )

    # 6) Senate_Passage (parent; critical)
    senate_parent = evaluator.add_parallel(
        id=f"Senate_Passage_{idx+1}",
        desc="Senate final passage details are provided (final passage, not procedural).",
        parent=item_node,
        critical=True
    )

    # 6.1) Senate_Date (critical leaf)
    senate_date_leaf = evaluator.add_leaf(
        id=f"Senate_Date_{idx+1}",
        desc="Senate final passage date is provided in MM/DD/YYYY format.",
        parent=senate_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The Senate final passage date for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.senate.date)} (as provided in MM/DD/YYYY format). This refers to the final passage vote."
        ),
        node=senate_date_leaf,
        sources=collect_senate_urls(item),
        additional_instruction=(
            "Confirm via senate.gov roll call page or Congress.gov that this is the Senate final passage date "
            "(not a procedural/cloture date). Also check that the date string in the answer is in MM/DD/YYYY."
        )
    )

    # 6.2) Senate_VoteCount (critical leaf)
    senate_votes_leaf = evaluator.add_leaf(
        id=f"Senate_VoteCount_{idx+1}",
        desc="Senate final passage vote count is provided in Yeas-Nays format.",
        parent=senate_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The final passage vote count in the Senate for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.senate.vote_count)} (Yeas–Nays)."
        ),
        node=senate_votes_leaf,
        sources=collect_senate_urls(item),
        additional_instruction=(
            "Confirm the YEAS and NAYS totals for the Senate final passage vote (not procedural or cloture). "
            "Accept minor formatting variations in the dash."
        )
    )

    # 6.3) Senate_RollCall (critical leaf)
    senate_rc_leaf = evaluator.add_leaf(
        id=f"Senate_RollCall_{idx+1}",
        desc="Senate final passage roll call number is provided.",
        parent=senate_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The Senate final passage roll call identifier/number for {_safe(item.bill_number) or 'this bill'} "
            f"was {_safe(item.senate.roll_call)}."
        ),
        node=senate_rc_leaf,
        sources=collect_senate_urls(item),
        additional_instruction=(
            "Verify on senate.gov (preferred) or Congress.gov that the referenced roll call identifier/number "
            "matches the final passage vote (not a cloture or procedural vote). Allow formatting differences."
        )
    )

    # 6.4) Senate_Cloture (critical leaf)
    senate_cloture_leaf = evaluator.add_leaf(
        id=f"Senate_Cloture_{idx+1}",
        desc="If cloture was required to end debate, this is indicated and the cloture vote is confirmed to have at least 60 votes; otherwise, correctly indicate cloture was not required.",
        parent=senate_parent,
        critical=True
    )
    if item.senate and item.senate.cloture_required:
        cloture_vote_txt = _safe(item.senate.cloture_vote)
        claim_cloture = (
            f"A cloture motion was required in the Senate for {_safe(item.bill_number) or 'this bill'}, "
            f"and the cloture vote received at least 60 affirmative votes (e.g., 'Yeas' ≥ 60). "
            f"The recorded cloture vote total was {cloture_vote_txt}."
        )
    else:
        claim_cloture = (
            f"No cloture motion was required to end debate in the Senate for the final passage of "
            f"{_safe(item.bill_number) or 'this bill'}."
        )
    await evaluator.verify(
        claim=claim_cloture,
        node=senate_cloture_leaf,
        sources=collect_senate_urls(item),
        additional_instruction=(
            "If cloture occurred, verify pages labeled 'Motion to Invoke Cloture' (or similar) show 'Cloture Invoked' "
            "with at least 60 YEAs. If there is no cloture, confirm no such cloture invocation is present for this bill's passage."
        )
    )

    # 7) Presidential_Action (parent; critical)
    pres_parent = evaluator.add_parallel(
        id=f"Presidential_Action_{idx+1}",
        desc="Presidential action details are provided (showing enactment: signed into law or became law without signature).",
        parent=item_node,
        critical=True
    )

    # 7.1) Pres_Action_Date (critical leaf)
    pres_date_leaf = evaluator.add_leaf(
        id=f"Pres_Action_Date_{idx+1}",
        desc="Presidential action date is provided in MM/DD/YYYY format.",
        parent=pres_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The bill {_safe(item.bill_number) or 'in question'} became law on {_safe(item.pres_action.date)} "
            f"(date presented in MM/DD/YYYY)."
        ),
        node=pres_date_leaf,
        sources=collect_pres_urls(item),
        additional_instruction=(
            "Confirm enactment date via Congress.gov 'Became Law' entry, WhiteHouse.gov statement, or govinfo.gov. "
            "Ensure the date provided in the answer uses MM/DD/YYYY formatting."
        )
    )

    # 7.2) Pres_Action_Type (critical leaf)
    pres_type_leaf = evaluator.add_leaf(
        id=f"Pres_Action_Type_{idx+1}",
        desc="Presidential action type is specified as either 'signed into law' or 'became law without signature'.",
        parent=pres_parent,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The presidential action type for {_safe(item.bill_number) or 'this bill'} "
            f"was '{_safe(item.pres_action.action_type)}'."
        ),
        node=pres_type_leaf,
        sources=collect_pres_urls(item),
        additional_instruction="Confirm whether the bill was signed by the President or became law without signature."
    )

    # 8) Official_Verifiability (critical leaf)
    official_leaf = evaluator.add_leaf(
        id=f"Official_Verifiability_{idx+1}",
        desc="All provided information is verifiable via official government sources (Congress.gov, House Clerk site, Senate.gov, and/or Federal Register as applicable).",
        parent=item_node,
        critical=True
    )
    # Build a concise composite claim summarizing key facts to be cross-checked
    claim_parts = []
    if item.bill_number:
        claim_parts.append(f"Bill: {item.bill_number}")
    if item.bill_title:
        claim_parts.append(f"Title: {item.bill_title}")
    if item.house:
        claim_parts.append(
            f"House: date {_safe(item.house.date)}, votes {_safe(item.house.vote_count)}, roll {_safe(item.house.roll_call)}"
        )
    if item.senate:
        cloture_txt = (
            "cloture required with ≥60 YEAs" if item.senate.cloture_required else "cloture not required"
        )
        if item.senate.cloture_required and item.senate.cloture_vote:
            cloture_txt = f"cloture vote {_safe(item.senate.cloture_vote)} (≥60 YEAs)"
        claim_parts.append(
            f"Senate: date {_safe(item.senate.date)}, votes {_safe(item.senate.vote_count)}, roll {_safe(item.senate.roll_call)}, {cloture_txt}"
        )
    if item.pres_action:
        claim_parts.append(
            f"Presidential action: {_safe(item.pres_action.action_type)} on {_safe(item.pres_action.date)}"
        )
    composite_claim = (
        "The following details from the answer are all supported by the provided official government URLs: "
        + "; ".join(claim_parts)
        + "."
    )
    await evaluator.verify(
        claim=composite_claim,
        node=official_leaf,
        sources=collect_official_urls(item),
        additional_instruction=(
            "You may use any of the provided URLs (Congress.gov, House Clerk, Senate.gov, govinfo.gov, WhiteHouse.gov, "
            "Federal Register) to confirm each detail. If any critical detail cannot be confirmed by official sources, "
            "mark this verification as not supported."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root is parallel per rubric)
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

    # Extract structured items
    extracted: LegislativeExtraction = await evaluator.extract(
        prompt=prompt_extract_legislation(),
        template_class=LegislativeExtraction,
        extraction_name="legislative_extraction",
    )

    # Normalize to exactly 4 items: take first 4; pad with empty if fewer
    items: List[LegislativeItem] = list(extracted.items[:4])
    while len(items) < 4:
        items.append(LegislativeItem())

    # Build verification tree per item
    for i in range(4):
        await verify_one_item(evaluator, root, items[i], i)

    # Return result summary
    return evaluator.get_summary()