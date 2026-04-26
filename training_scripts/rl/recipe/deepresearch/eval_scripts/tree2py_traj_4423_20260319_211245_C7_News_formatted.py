import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "march_2026_us_political_military_news"
TASK_DESCRIPTION = """
Research recent major U.S. political and military news events from March 2026 and provide the following specific information:

1. Regarding the Homeland Security Secretary change:
   - The exact date when Kristi Noem was fired as Homeland Security Secretary
   - The name of the person nominated to replace her
   - The previous government position held by the replacement nominee
   - The date of the replacement nominee's Senate confirmation hearing

2. Regarding the Texas congressional primary:
   - The names of the two candidates who advanced to a Democratic primary runoff for Texas' 18th Congressional District
   - The date of the primary election that led to this runoff
   - The scheduled date of the runoff election

3. Regarding U.S. military operations against Iran:
   - The official name of the U.S. military operation
   - The date when Iran struck the U.S. Navy's Fifth Fleet base in Bahrain
   - The specific city where the Fifth Fleet base is located

4. Regarding President Trump's meeting with Japan's Prime Minister:
   - The date of the meeting
   - The full name of the Japanese Prime Minister
   - Confirmation that Trump made a reference to Pearl Harbor during discussions about why allies weren't informed about Iran operations

5. Regarding the Federal Reserve:
   - The date of the Federal Reserve's March interest rate decision meeting
   - The interest rate range that was maintained
   - The vote count for the decision to hold rates steady

For each piece of information, provide a reference URL from a credible news source that supports your answer.
"""

# Expected values per rubric (used for matching + source-backed verification)
EXPECTED = {
    "noem_firing_date": "March 5, 2026",
    "replacement_nominee_name": "Markwayne Mullin",
    "replacement_previous_position": "U.S. Senator from Oklahoma",
    "confirmation_hearing_date": "March 18, 2026",

    "runoff_candidates": ["Al Green", "Christian Menefee"],
    "primary_election_date": "March 4, 2026",
    "runoff_election_date": "May 26, 2026",

    "operation_name": "Operation Epic Fury",
    "bahrain_strike_date": "February 28, 2026",
    "fifth_fleet_base_city": "Manama, Bahrain",

    "meeting_date": "March 19, 2026",
    "japanese_pm_full_name": "Sanae Takaichi",
    "pearl_harbor_reference": "yes",  # Expect affirmative confirmation

    "fed_meeting_date": "March 18, 2026",
    "held_rate_range": "3.5%–3.75%",
    "vote_count": "11–1",
}

# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class HomelandSecurityChange(BaseModel):
    noem_firing_date: Optional[str] = None
    noem_firing_date_sources: List[str] = Field(default_factory=list)

    replacement_nominee_name: Optional[str] = None
    replacement_nominee_name_sources: List[str] = Field(default_factory=list)

    replacement_previous_position: Optional[str] = None
    replacement_previous_position_sources: List[str] = Field(default_factory=list)

    confirmation_hearing_date: Optional[str] = None
    confirmation_hearing_date_sources: List[str] = Field(default_factory=list)


class TexasRunoff(BaseModel):
    runoff_candidates: List[str] = Field(default_factory=list)
    runoff_candidates_sources: List[str] = Field(default_factory=list)

    primary_election_date: Optional[str] = None
    primary_election_date_sources: List[str] = Field(default_factory=list)

    runoff_election_date: Optional[str] = None
    runoff_election_date_sources: List[str] = Field(default_factory=list)


class USMilitaryIran(BaseModel):
    operation_name: Optional[str] = None
    operation_name_sources: List[str] = Field(default_factory=list)

    bahrain_strike_date: Optional[str] = None
    bahrain_strike_date_sources: List[str] = Field(default_factory=list)

    fifth_fleet_base_city: Optional[str] = None
    fifth_fleet_base_city_sources: List[str] = Field(default_factory=list)


class TrumpJapanMeeting(BaseModel):
    meeting_date: Optional[str] = None
    meeting_date_sources: List[str] = Field(default_factory=list)

    japanese_pm_full_name: Optional[str] = None
    japanese_pm_full_name_sources: List[str] = Field(default_factory=list)

    pearl_harbor_reference: Optional[str] = None  # e.g., "yes", "no", or a confirming statement
    pearl_harbor_reference_sources: List[str] = Field(default_factory=list)


class FederalReserveDecision(BaseModel):
    fed_meeting_date: Optional[str] = None
    fed_meeting_date_sources: List[str] = Field(default_factory=list)

    held_rate_range: Optional[str] = None
    held_rate_range_sources: List[str] = Field(default_factory=list)

    vote_count: Optional[str] = None
    vote_count_sources: List[str] = Field(default_factory=list)


class March2026NewsExtraction(BaseModel):
    homeland_security: Optional[HomelandSecurityChange] = None
    texas_runoff: Optional[TexasRunoff] = None
    military_iran: Optional[USMilitaryIran] = None
    trump_japan: Optional[TrumpJapanMeeting] = None
    fed_decision: Optional[FederalReserveDecision] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_march2026_news() -> str:
    return """
Extract the requested March 2026 U.S. political and military news details as they appear in the provided answer. For every requested piece, also extract all reference URLs cited in the answer that specifically support that piece.

Return a JSON object following this exact schema:

{
  "homeland_security": {
    "noem_firing_date": string|null,
    "noem_firing_date_sources": string[],

    "replacement_nominee_name": string|null,
    "replacement_nominee_name_sources": string[],

    "replacement_previous_position": string|null,
    "replacement_previous_position_sources": string[],

    "confirmation_hearing_date": string|null,
    "confirmation_hearing_date_sources": string[]
  },
  "texas_runoff": {
    "runoff_candidates": string[],   // exactly two names if provided
    "runoff_candidates_sources": string[],

    "primary_election_date": string|null,
    "primary_election_date_sources": string[],

    "runoff_election_date": string|null,
    "runoff_election_date_sources": string[]
  },
  "military_iran": {
    "operation_name": string|null,
    "operation_name_sources": string[],

    "bahrain_strike_date": string|null,
    "bahrain_strike_date_sources": string[],

    "fifth_fleet_base_city": string|null,
    "fifth_fleet_base_city_sources": string[]
  },
  "trump_japan": {
    "meeting_date": string|null,
    "meeting_date_sources": string[],

    "japanese_pm_full_name": string|null,
    "japanese_pm_full_name_sources": string[],

    "pearl_harbor_reference": string|null,   // e.g., 'yes', 'no', or a short confirming phrase from the answer
    "pearl_harbor_reference_sources": string[]
  },
  "fed_decision": {
    "fed_meeting_date": string|null,
    "fed_meeting_date_sources": string[],

    "held_rate_range": string|null,
    "held_rate_range_sources": string[],

    "vote_count": string|null,
    "vote_count_sources": string[]
  }
}

Instructions:
- Extract only what is explicitly stated in the answer.
- For all *_sources fields, extract the actual URLs (plain, markdown, or similar). If none are provided, return an empty list.
- If a field is missing, set it to null (or [] for arrays).
- Normalize URLs: ensure they include http:// or https:// if missing.
- Do not invent information or URLs.
"""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and len(u.strip()) > 0]


def _all_unique_urls(ext: March2026NewsExtraction) -> List[str]:
    urls: Set[str] = set()

    if ext.homeland_security:
        urls.update(_norm_urls(ext.homeland_security.noem_firing_date_sources))
        urls.update(_norm_urls(ext.homeland_security.replacement_nominee_name_sources))
        urls.update(_norm_urls(ext.homeland_security.replacement_previous_position_sources))
        urls.update(_norm_urls(ext.homeland_security.confirmation_hearing_date_sources))

    if ext.texas_runoff:
        urls.update(_norm_urls(ext.texas_runoff.runoff_candidates_sources))
        urls.update(_norm_urls(ext.texas_runoff.primary_election_date_sources))
        urls.update(_norm_urls(ext.texas_runoff.runoff_election_date_sources))

    if ext.military_iran:
        urls.update(_norm_urls(ext.military_iran.operation_name_sources))
        urls.update(_norm_urls(ext.military_iran.bahrain_strike_date_sources))
        urls.update(_norm_urls(ext.military_iran.fifth_fleet_base_city_sources))

    if ext.trump_japan:
        urls.update(_norm_urls(ext.trump_japan.meeting_date_sources))
        urls.update(_norm_urls(ext.trump_japan.japanese_pm_full_name_sources))
        urls.update(_norm_urls(ext.trump_japan.pearl_harbor_reference_sources))

    if ext.fed_decision:
        urls.update(_norm_urls(ext.fed_decision.fed_meeting_date_sources))
        urls.update(_norm_urls(ext.fed_decision.held_rate_range_sources))
        urls.update(_norm_urls(ext.fed_decision.vote_count_sources))

    return sorted(urls)


def _pieces_sources(ext: March2026NewsExtraction) -> List[Dict[str, Any]]:
    pieces: List[Dict[str, Any]] = []

    hs = ext.homeland_security or HomelandSecurityChange()
    pieces.extend([
        {"id": "hs_noem_firing_date", "desc": "URL(s) for Noem firing date provided", "urls": _norm_urls(hs.noem_firing_date_sources)},
        {"id": "hs_replacement_nominee_name", "desc": "URL(s) for replacement nominee name provided", "urls": _norm_urls(hs.replacement_nominee_name_sources)},
        {"id": "hs_replacement_previous_position", "desc": "URL(s) for replacement nominee previous position provided", "urls": _norm_urls(hs.replacement_previous_position_sources)},
        {"id": "hs_confirmation_hearing_date", "desc": "URL(s) for nominee's confirmation hearing date provided", "urls": _norm_urls(hs.confirmation_hearing_date_sources)},
    ])

    tx = ext.texas_runoff or TexasRunoff()
    pieces.extend([
        {"id": "tx_runoff_candidates", "desc": "URL(s) for TX-18 runoff candidates provided", "urls": _norm_urls(tx.runoff_candidates_sources)},
        {"id": "tx_primary_election_date", "desc": "URL(s) for TX-18 primary election date provided", "urls": _norm_urls(tx.primary_election_date_sources)},
        {"id": "tx_runoff_election_date", "desc": "URL(s) for TX-18 runoff election date provided", "urls": _norm_urls(tx.runoff_election_date_sources)},
    ])

    ir = ext.military_iran or USMilitaryIran()
    pieces.extend([
        {"id": "ir_operation_name", "desc": "URL(s) for operation name provided", "urls": _norm_urls(ir.operation_name_sources)},
        {"id": "ir_bahrain_strike_date", "desc": "URL(s) for Bahrain strike date provided", "urls": _norm_urls(ir.bahrain_strike_date_sources)},
        {"id": "ir_fifth_fleet_city", "desc": "URL(s) for Fifth Fleet base city provided", "urls": _norm_urls(ir.fifth_fleet_base_city_sources)},
    ])

    jp = ext.trump_japan or TrumpJapanMeeting()
    pieces.extend([
        {"id": "jp_meeting_date", "desc": "URL(s) for Trump–Japan PM meeting date provided", "urls": _norm_urls(jp.meeting_date_sources)},
        {"id": "jp_pm_full_name", "desc": "URL(s) for Japanese PM full name provided", "urls": _norm_urls(jp.japanese_pm_full_name_sources)},
        {"id": "jp_pearl_harbor_reference", "desc": "URL(s) for Pearl Harbor reference provided", "urls": _norm_urls(jp.pearl_harbor_reference_sources)},
    ])

    fd = ext.fed_decision or FederalReserveDecision()
    pieces.extend([
        {"id": "fd_meeting_date", "desc": "URL(s) for Fed March decision meeting date provided", "urls": _norm_urls(fd.fed_meeting_date_sources)},
        {"id": "fd_rate_range", "desc": "URL(s) for held rate range provided", "urls": _norm_urls(fd.held_rate_range_sources)},
        {"id": "fd_vote_count", "desc": "URL(s) for vote count provided", "urls": _norm_urls(fd.vote_count_sources)},
    ])

    return pieces


# -----------------------------------------------------------------------------
# Verification builders
# -----------------------------------------------------------------------------
async def build_citations_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    # Citations (critical)
    citations_node = evaluator.add_parallel(
        id="Citations",
        desc="Citations meet the question requirement for supporting each requested piece of information.",
        parent=root,
        critical=True
    )

    # URL presence for each requested piece (expanded to multiple critical leaves)
    presence_parent = evaluator.add_parallel(
        id="URL_For_Each_Requested_Piece",
        desc="Provides at least one reference URL supporting each requested piece of information.",
        parent=citations_node,
        critical=True
    )

    for piece in _pieces_sources(ext):
        result = len(piece["urls"]) > 0
        evaluator.add_custom_node(
            result=result,
            id=f"url_presence_{piece['id']}",
            desc=piece["desc"],
            parent=presence_parent,
            critical=True
        )

    # Credible news source URLs - one leaf per unique URL
    credible_parent = evaluator.add_parallel(
        id="Credible_News_Source_URLs",
        desc="Provided reference URLs are from credible news sources (or official .gov/.mil/central bank sites).",
        parent=citations_node,
        critical=True
    )

    credibility_instruction = (
        "Judge whether this URL is a credible source: Prefer professionally edited news outlets "
        "(e.g., AP, Reuters, Bloomberg, WSJ, NYT, Washington Post, major broadcast networks), "
        "major reputable regional newspapers, wire services, or official/government sites (.gov, .mil, federalreserve.gov, defense.gov, whitehouse.gov). "
        "Academic or well-known international outlets (BBC, FT, The Guardian) are acceptable. "
        "Do NOT consider social media posts, personal blogs, unvetted aggregators, forums, or user-generated content as credible. "
        "Return Correct only if it's clearly a credible/official source."
    )

    all_urls = _all_unique_urls(ext)
    claims_and_sources = []
    for i, url in enumerate(all_urls):
        node = evaluator.add_leaf(
            id=f"credible_url_{i+1}",
            desc=f"URL is from a credible news/official source: {url}",
            parent=credible_parent,
            critical=True
        )
        claim = "This URL points to a credible/official news source as defined in the instruction."
        claims_and_sources.append((claim, url, node, credibility_instruction))

    if claims_and_sources:
        await evaluator.batch_verify(claims_and_sources)


async def build_homeland_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Homeland_Security_Secretary_Change",
        desc="Details about the Homeland Security Secretary change match the constraints.",
        parent=root,
        critical=True
    )

    hs = ext.homeland_security or HomelandSecurityChange()

    # 1) Noem Firing Date
    # Match check
    match_node = evaluator.add_leaf(
        id="Noem_Firing_Date_Match",
        desc="Answer's Noem firing date matches expected March 5, 2026.",
        parent=node,
        critical=True
    )
    extracted_date = hs.noem_firing_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_date}' refers to the same calendar date as '{EXPECTED['noem_firing_date']}'. "
              f"Allow formats like 'Mar. 5, 2026' or '03/05/2026'.",
        node=match_node,
    )
    # Source support
    support_node = evaluator.add_leaf(
        id="Noem_Firing_Date",
        desc="Kristi Noem firing date is March 5, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Kristi Noem was fired as Homeland Security Secretary on March 5, 2026.",
        node=support_node,
        sources=_norm_urls(hs.noem_firing_date_sources),
        additional_instruction="Confirm the article explicitly states that Noem was fired on March 5, 2026."
    )

    # 2) Replacement Nominee Name
    match_node = evaluator.add_leaf(
        id="Replacement_Nominee_Name_Match",
        desc="Answer's replacement nominee name matches expected 'Markwayne Mullin'.",
        parent=node,
        critical=True
    )
    extracted_name = hs.replacement_nominee_name or ""
    await evaluator.verify(
        claim=f"The name '{extracted_name}' refers to the same person as '{EXPECTED['replacement_nominee_name']}'. "
              f"Allow minor variations (case, middle initials).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Replacement_Nominee_Name",
        desc="Replacement nominee is Markwayne Mullin.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The person nominated to replace Kristi Noem as Homeland Security Secretary was Markwayne Mullin.",
        node=support_node,
        sources=_norm_urls(hs.replacement_nominee_name_sources),
        additional_instruction="The source should clearly identify the nominee as Markwayne Mullin."
    )

    # 3) Replacement Previous Position
    match_node = evaluator.add_leaf(
        id="Replacement_Previous_Position_Match",
        desc="Answer's previous position for Mullin matches 'U.S. Senator from Oklahoma'.",
        parent=node,
        critical=True
    )
    extracted_pos = hs.replacement_previous_position or ""
    await evaluator.verify(
        claim=f"The role description '{extracted_pos}' is equivalent to 'U.S. Senator from Oklahoma' "
              f"(synonyms like 'United States Senator representing Oklahoma' are acceptable).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Replacement_Previous_Position",
        desc="Markwayne Mullin previously served as a Senator from Oklahoma.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Markwayne Mullin previously served as a U.S. Senator representing Oklahoma.",
        node=support_node,
        sources=_norm_urls(hs.replacement_previous_position_sources),
        additional_instruction="The article should explicitly state he served as a U.S. Senator from Oklahoma."
    )

    # 4) Confirmation Hearing Date
    match_node = evaluator.add_leaf(
        id="Confirmation_Hearing_Date_Match",
        desc="Answer's confirmation hearing date matches expected March 18, 2026.",
        parent=node,
        critical=True
    )
    extracted_hearing = hs.confirmation_hearing_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_hearing}' refers to the same calendar date as '{EXPECTED['confirmation_hearing_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Confirmation_Hearing_Date",
        desc="Markwayne Mullin confirmation hearing date is March 18, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Senate confirmation hearing for Markwayne Mullin as Homeland Security Secretary was on March 18, 2026.",
        node=support_node,
        sources=_norm_urls(hs.confirmation_hearing_date_sources),
        additional_instruction="The source should clearly state the hearing was held on March 18, 2026."
    )


async def build_texas_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Texas_18_Democratic_Runoff",
        desc="Details about the TX-18 Democratic primary runoff match the constraints.",
        parent=root,
        critical=True
    )

    tx = ext.texas_runoff or TexasRunoff()

    # Runoff candidates
    match_node = evaluator.add_leaf(
        id="Runoff_Candidates_Match",
        desc="Answer's TX-18 runoff candidates match 'Al Green' and 'Christian Menefee' (order-insensitive).",
        parent=node,
        critical=True
    )
    extracted_list = tx.runoff_candidates or []
    await evaluator.verify(
        claim=f"The extracted candidate list {extracted_list} matches exactly the pair "
              f"['{EXPECTED['runoff_candidates'][0]}', '{EXPECTED['runoff_candidates'][1]}'] "
              f"(order not important; allow minor name variants).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Runoff_Candidates",
        desc="Candidates advancing to the runoff are Al Green and Christian Menefee.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The two candidates who advanced to the Democratic primary runoff for Texas' 18th Congressional District were Al Green and Christian Menefee.",
        node=support_node,
        sources=_norm_urls(tx.runoff_candidates_sources),
        additional_instruction="The source should clearly state both Al Green and Christian Menefee advanced to the runoff."
    )

    # Primary election date
    match_node = evaluator.add_leaf(
        id="Primary_Election_Date_Match",
        desc="Answer's TX-18 primary election date matches March 4, 2026.",
        parent=node,
        critical=True
    )
    extracted_primary = tx.primary_election_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_primary}' is the same calendar date as '{EXPECTED['primary_election_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Primary_Election_Date",
        desc="Primary election date is March 4, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The primary election that led to the TX-18 runoff took place on March 4, 2026.",
        node=support_node,
        sources=_norm_urls(tx.primary_election_date_sources),
        additional_instruction="The source should explicitly state the March 4, 2026 primary date."
    )

    # Runoff election date
    match_node = evaluator.add_leaf(
        id="Runoff_Election_Date_Match",
        desc="Answer's TX-18 runoff election date matches May 26, 2026.",
        parent=node,
        critical=True
    )
    extracted_runoff = tx.runoff_election_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_runoff}' is the same calendar date as '{EXPECTED['runoff_election_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Runoff_Election_Date",
        desc="Runoff election date is May 26, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The TX-18 Democratic primary runoff election was scheduled for May 26, 2026.",
        node=support_node,
        sources=_norm_urls(tx.runoff_election_date_sources),
        additional_instruction="The source should clearly state the runoff election date is May 26, 2026."
    )


async def build_military_iran_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    node = evaluator.add_parallel(
        id="US_Military_Operations_Against_Iran",
        desc="Details about U.S. military operations against Iran match the constraints.",
        parent=root,
        critical=True
    )

    ir = ext.military_iran or USMilitaryIran()

    # Operation name
    match_node = evaluator.add_leaf(
        id="Operation_Name_Match",
        desc="Answer's operation name matches 'Operation Epic Fury'.",
        parent=node,
        critical=True
    )
    extracted_op = ir.operation_name or ""
    await evaluator.verify(
        claim=f"The extracted operation name '{extracted_op}' equals 'Operation Epic Fury' (allow minor punctuation/case variants).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Operation_Name",
        desc="Official operation name is 'Operation Epic Fury'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The official name of the U.S. military operation against Iran was 'Operation Epic Fury'.",
        node=support_node,
        sources=_norm_urls(ir.operation_name_sources),
        additional_instruction="The article should explicitly refer to the operation as 'Operation Epic Fury'."
    )

    # Bahrain strike date
    match_node = evaluator.add_leaf(
        id="Bahrain_Strike_Date_Match",
        desc="Answer's Bahrain strike date matches February 28, 2026.",
        parent=node,
        critical=True
    )
    extracted_strike = ir.bahrain_strike_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_strike}' is the same calendar date as '{EXPECTED['bahrain_strike_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Bahrain_Strike_Date",
        desc="Iran struck the U.S. Navy Fifth Fleet base in Bahrain on February 28, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Iran struck the U.S. Navy Fifth Fleet base in Bahrain on February 28, 2026.",
        node=support_node,
        sources=_norm_urls(ir.bahrain_strike_date_sources),
        additional_instruction="The source should clearly state the strike date as February 28, 2026."
    )

    # Fifth Fleet base city
    match_node = evaluator.add_leaf(
        id="Fifth_Fleet_Base_City_Match",
        desc="Answer's Fifth Fleet base city matches Manama, Bahrain.",
        parent=node,
        critical=True
    )
    extracted_city = ir.fifth_fleet_base_city or ""
    await evaluator.verify(
        claim=f"The location '{extracted_city}' refers to 'Manama, Bahrain' (allow minor formatting).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Fifth_Fleet_Base_City",
        desc="Fifth Fleet base city is Manama, Bahrain.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The U.S. Navy Fifth Fleet is based in Manama, Bahrain.",
        node=support_node,
        sources=_norm_urls(ir.fifth_fleet_base_city_sources),
        additional_instruction="Confirm the base is located in Manama, Bahrain."
    )


async def build_trump_japan_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Trump_Meeting_With_Japanese_PM",
        desc="Details about Trump’s meeting with Japan’s Prime Minister match the constraints.",
        parent=root,
        critical=True
    )

    jp = ext.trump_japan or TrumpJapanMeeting()

    # Meeting date
    match_node = evaluator.add_leaf(
        id="Meeting_Date_Match",
        desc="Answer's meeting date matches March 19, 2026.",
        parent=node,
        critical=True
    )
    extracted_meet = jp.meeting_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_meet}' is the same calendar date as '{EXPECTED['meeting_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Meeting_Date",
        desc="Meeting date is March 19, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="President Trump met with Japan's Prime Minister on March 19, 2026.",
        node=support_node,
        sources=_norm_urls(jp.meeting_date_sources),
        additional_instruction="The article should clearly state a March 19, 2026 meeting date."
    )

    # Japanese PM full name
    match_node = evaluator.add_leaf(
        id="Japanese_PM_Full_Name_Match",
        desc="Answer's Japanese PM full name matches 'Sanae Takaichi'.",
        parent=node,
        critical=True
    )
    extracted_pm = jp.japanese_pm_full_name or ""
    await evaluator.verify(
        claim=f"The name '{extracted_pm}' refers to the same person as '{EXPECTED['japanese_pm_full_name']}'. "
              f"Allow minor variants (order/case).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Japanese_PM_Full_Name",
        desc="Japanese Prime Minister is Sanae Takaichi.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Japanese Prime Minister's full name is Sanae Takaichi.",
        node=support_node,
        sources=_norm_urls(jp.japanese_pm_full_name_sources),
        additional_instruction="The article should identify the Japanese Prime Minister as Sanae Takaichi."
    )

    # Pearl Harbor reference confirmation
    match_node = evaluator.add_leaf(
        id="Pearl_Harbor_Reference_Match",
        desc="Answer asserts that Trump referenced Pearl Harbor in the described context.",
        parent=node,
        critical=True
    )
    extracted_ref = jp.pearl_harbor_reference or ""
    await evaluator.verify(
        claim=f"The statement '{extracted_ref}' indicates that Trump referenced Pearl Harbor (affirmative). "
              f"Interpret 'yes/true/affirmative' or clearly affirmative phrasing as confirmation.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Pearl_Harbor_Reference",
        desc="Trump referenced Pearl Harbor in the described context.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="During discussions about why allies weren't informed about Iran operations, President Trump referenced Pearl Harbor.",
        node=support_node,
        sources=_norm_urls(jp.pearl_harbor_reference_sources),
        additional_instruction="The article should explicitly mention Trump referencing 'Pearl Harbor' in this context."
    )


async def build_fed_branch(evaluator: Evaluator, root, ext: March2026NewsExtraction) -> None:
    node = evaluator.add_parallel(
        id="Federal_Reserve_March_Decision",
        desc="Details about the Federal Reserve March interest rate decision match the constraints.",
        parent=root,
        critical=True
    )

    fd = ext.fed_decision or FederalReserveDecision()

    # Fed meeting date
    match_node = evaluator.add_leaf(
        id="Fed_Meeting_Date_Match",
        desc="Answer's Fed decision meeting date matches March 18, 2026.",
        parent=node,
        critical=True
    )
    extracted_fed_date = fd.fed_meeting_date or ""
    await evaluator.verify(
        claim=f"The date string '{extracted_fed_date}' is the same calendar date as '{EXPECTED['fed_meeting_date']}'.",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Fed_Meeting_Date",
        desc="Fed decision meeting date is March 18, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Federal Reserve's March interest rate decision meeting occurred on March 18, 2026.",
        node=support_node,
        sources=_norm_urls(fd.fed_meeting_date_sources),
        additional_instruction="Accept official FOMC calendar/statement or credible reporting explicitly stating March 18, 2026."
    )

    # Held rate range
    match_node = evaluator.add_leaf(
        id="Held_Rate_Range_Match",
        desc="Answer's held rate range matches 3.5%–3.75%.",
        parent=node,
        critical=True
    )
    extracted_range = fd.held_rate_range or ""
    await evaluator.verify(
        claim=f"The stated range '{extracted_range}' is equivalent to '3.5% to 3.75%' "
              f"(accept '3.5%–3.75%', '3.5%-3.75%', '3.50%-3.75%' and similar).",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Held_Rate_Range",
        desc="Maintained interest rate range is 3.5%–3.75%.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Federal Reserve maintained the target range at 3.5% to 3.75%.",
        node=support_node,
        sources=_norm_urls(fd.held_rate_range_sources),
        additional_instruction="Prefer the Fed's official statement or credible outlets explicitly stating this range."
    )

    # Vote count
    match_node = evaluator.add_leaf(
        id="Vote_Count_Match",
        desc="Answer's vote count matches 11–1.",
        parent=node,
        critical=True
    )
    extracted_vote = fd.vote_count or ""
    await evaluator.verify(
        claim=f"The vote count '{extracted_vote}' equals '11–1' (allow '11-1' or 'eleven to one').",
        node=match_node,
    )
    support_node = evaluator.add_leaf(
        id="Vote_Count",
        desc="Vote count to hold rates steady is 11–1.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The decision to hold rates steady passed with a vote of 11–1.",
        node=support_node,
        sources=_norm_urls(fd.vote_count_sources),
        additional_instruction="Confirm the exact vote split was 11–1."
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    # Initialize evaluator (root is non-critical by framework, so make all top branches critical)
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_march2026_news(),
        template_class=March2026NewsExtraction,
        extraction_name="march_2026_news_extraction",
    )

    # Record expected values as ground truth in the summary
    evaluator.add_ground_truth(
        {
            "expected": {
                "Homeland_Security_Secretary_Change": {
                    "noem_firing_date": EXPECTED["noem_firing_date"],
                    "replacement_nominee_name": EXPECTED["replacement_nominee_name"],
                    "replacement_previous_position": EXPECTED["replacement_previous_position"],
                    "confirmation_hearing_date": EXPECTED["confirmation_hearing_date"],
                },
                "Texas_18_Democratic_Runoff": {
                    "runoff_candidates": EXPECTED["runoff_candidates"],
                    "primary_election_date": EXPECTED["primary_election_date"],
                    "runoff_election_date": EXPECTED["runoff_election_date"],
                },
                "US_Military_Operations_Against_Iran": {
                    "operation_name": EXPECTED["operation_name"],
                    "bahrain_strike_date": EXPECTED["bahrain_strike_date"],
                    "fifth_fleet_base_city": EXPECTED["fifth_fleet_base_city"],
                },
                "Trump_Meeting_With_Japanese_PM": {
                    "meeting_date": EXPECTED["meeting_date"],
                    "japanese_pm_full_name": EXPECTED["japanese_pm_full_name"],
                    "pearl_harbor_reference": EXPECTED["pearl_harbor_reference"],
                },
                "Federal_Reserve_March_Decision": {
                    "fed_meeting_date": EXPECTED["fed_meeting_date"],
                    "held_rate_range": EXPECTED["held_rate_range"],
                    "vote_count": EXPECTED["vote_count"],
                },
            }
        },
        gt_type="ground_truth_expected_values"
    )

    # Build verification tree
    # 1) Citations branch first (so its critical results can gate others)
    await build_citations_branch(evaluator, root, extraction)

    # 2) Content verification branches
    await build_homeland_branch(evaluator, root, extraction)
    await build_texas_branch(evaluator, root, extraction)
    await build_military_iran_branch(evaluator, root, extraction)
    await build_trump_japan_branch(evaluator, root, extraction)
    await build_fed_branch(evaluator, root, extraction)

    # Return final structured summary
    return evaluator.get_summary()