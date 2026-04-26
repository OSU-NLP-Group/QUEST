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
TASK_ID = "trump_2025_2026_events"
TASK_DESCRIPTION = """
I am researching major Trump administration events and nominations from 2025-2026 for a political science project. Please provide detailed, verified information about the following four events:

1. Kevin Warsh's Federal Reserve Chair Nomination:
   - The date when President Trump announced the nomination
   - The date when the formal nomination was transmitted to the Senate
   - The name of the Senate committee reviewing the nomination
   - The date when Jerome Powell's term as Fed Chair ends

2. Doug Collins's Confirmation as VA Secretary:
   - The date when Doug Collins was confirmed by the Senate
   - The Senate vote count (yes-no) for his confirmation
   - The date of his confirmation hearing
   - His position number as VA Secretary (e.g., "Xth Secretary")

3. The Rescissions Act of 2025 (CPB Funding Cuts):
   - The date and vote count for the House of Representatives vote
   - The date and vote count for the Senate vote
   - The specific amount of funding rescinded from the Corporation for Public Broadcasting (CPB)
   - The date when CPB's board of directors voted to dissolve the organization

4. Operation Midnight Hammer (Iran Strikes):
   - The date when the operation was conducted
   - The type and number of primary bomber aircraft used
   - The air base from which the bombers departed
   - The type and total number of the primary bunker-buster bombs dropped
   - Whether this was the first combat use of that bomb type
   - The names of the three Iranian nuclear facilities targeted

For each piece of information, please provide a supporting URL reference from a credible news source, government website, or reliable publication that verifies the information.
"""

# Expected values encoded from the rubric (treated as ground truth targets for checking)
EXPECTED = {
    "warsh": {
        "announcement_date": "January 30, 2026",
        "transmission_date": "March 4, 2026",
        "committee_name": "Senate Banking, Housing, and Urban Affairs Committee",
        "powell_term_end": "May 15, 2026",
    },
    "collins": {
        "confirmation_date": "February 4, 2025",
        "vote_count": "77-23",
        "hearing_date": "January 21, 2025",
        "position_number": "12th",
    },
    "rescissions": {
        "house_vote_date": "June 12, 2025",
        "house_vote_count": "214-212",
        "senate_vote_date": "July 17, 2025",
        "senate_vote_count": "52-47",
        "cpb_amount": "$1.1 billion",
        "cpb_dissolution_date": "January 5, 2026",
    },
    "midnight_hammer": {
        "operation_date": "June 22, 2025",
        "aircraft_type": "B-2 Spirit stealth bombers",
        "aircraft_count": "7",
        "base_location": "Whiteman Air Force Base",
        "munition_name": "GBU-57A/B Massive Ordnance Penetrator (MOP)",
        "munition_count": "14",
        "first_combat_use": "Yes",
        "facilities": [
            "Fordow Uranium Enrichment Plant",
            "Natanz Nuclear Facility",
            "Isfahan Nuclear Technology Center",
        ],
    }
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SimpleField(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ListField(BaseModel):
    values: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class WarshNomination(BaseModel):
    announcement_date: Optional[SimpleField] = None
    senate_transmission_date: Optional[SimpleField] = None
    committee: Optional[SimpleField] = None
    powell_term_end: Optional[SimpleField] = None


class CollinsConfirmation(BaseModel):
    confirmation_date: Optional[SimpleField] = None
    vote_count: Optional[SimpleField] = None
    hearing_date: Optional[SimpleField] = None
    position_number: Optional[SimpleField] = None


class RescissionsActInfo(BaseModel):
    house_vote_date: Optional[SimpleField] = None
    house_vote_count: Optional[SimpleField] = None
    senate_vote_date: Optional[SimpleField] = None
    senate_vote_count: Optional[SimpleField] = None
    cpb_funding_amount: Optional[SimpleField] = None
    cpb_dissolution_date: Optional[SimpleField] = None


class OperationMidnightHammer(BaseModel):
    operation_date: Optional[SimpleField] = None
    aircraft_type: Optional[SimpleField] = None
    aircraft_count: Optional[SimpleField] = None
    base_location: Optional[SimpleField] = None
    munition_name: Optional[SimpleField] = None
    munition_count: Optional[SimpleField] = None
    first_combat_use: Optional[SimpleField] = None
    target_facilities: Optional[ListField] = None


class ProjectExtraction(BaseModel):
    kevin_warsh_nomination: Optional[WarshNomination] = None
    doug_collins_confirmation: Optional[CollinsConfirmation] = None
    rescissions_act_2025: Optional[RescissionsActInfo] = None
    operation_midnight_hammer: Optional[OperationMidnightHammer] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the requested facts from the answer. For each fact, also extract the specific URL(s) cited in the answer that directly support that fact.

Return a JSON object with these top-level keys:
- kevin_warsh_nomination: {
    announcement_date: { value: string|null, urls: string[] },
    senate_transmission_date: { value: string|null, urls: string[] },
    committee: { value: string|null, urls: string[] },
    powell_term_end: { value: string|null, urls: string[] }
  }
- doug_collins_confirmation: {
    confirmation_date: { value: string|null, urls: string[] },
    vote_count: { value: string|null, urls: string[] },
    hearing_date: { value: string|null, urls: string[] },
    position_number: { value: string|null, urls: string[] }
  }
- rescissions_act_2025: {
    house_vote_date: { value: string|null, urls: string[] },
    house_vote_count: { value: string|null, urls: string[] },
    senate_vote_date: { value: string|null, urls: string[] },
    senate_vote_count: { value: string|null, urls: string[] },
    cpb_funding_amount: { value: string|null, urls: string[] },
    cpb_dissolution_date: { value: string|null, urls: string[] }
  }
- operation_midnight_hammer: {
    operation_date: { value: string|null, urls: string[] },
    aircraft_type: { value: string|null, urls: string[] },
    aircraft_count: { value: string|null, urls: string[] },
    base_location: { value: string|null, urls: string[] },
    munition_name: { value: string|null, urls: string[] },
    munition_count: { value: string|null, urls: string[] },
    first_combat_use: { value: string|null, urls: string[] },
    target_facilities: { values: string[], urls: string[] }
  }

Rules:
- Extract only what is explicitly stated in the answer text.
- For each urls array, include all URLs (plain or markdown links) explicitly provided for that specific fact. If none, return [].
- Keep dates and numbers as free-form strings as written (e.g., "Jan. 30, 2026" or "77–23").
- If a field is not present in the answer, set its value to null (or empty list for values) and urls to [].
"""


# --------------------------------------------------------------------------- #
# Helper for URL-backed leaf creation                                         #
# --------------------------------------------------------------------------- #
async def add_url_support_or_fail(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    parent,
    critical: bool = True,
    additional_instruction: str = ""
):
    """
    Add a leaf that verifies the claim using provided URLs.
    If no URLs are provided, record a failed custom node (to enforce source-grounding).
    """
    urls = urls or []
    if len(urls) == 0:
        evaluator.add_custom_node(
            result=False,
            id=id,
            desc=f"{desc} (failed: no URLs provided in the answer)",
            parent=parent,
            critical=critical
        )
        return

    leaf = evaluator.add_leaf(
        id=id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction or "Verify strictly using the provided URL(s). If the page does not explicitly support the statement, judge as not supported."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_kevin_warsh(evaluator: Evaluator, parent, data: Optional[WarshNomination]):
    node = evaluator.add_parallel(
        id="kevin_warsh_nomination",
        desc="Information about Kevin Warsh's nomination as Federal Reserve Chair",
        parent=parent,
        critical=True
    )
    data = data or WarshNomination()

    # 1) Announcement date (sequential: value then URL)
    seq1 = evaluator.add_sequential(
        id="nomination_announcement_date",
        desc="The date when President Trump announced Kevin Warsh's nomination as Fed Chair",
        parent=node,
        critical=True
    )
    # value
    v1 = evaluator.add_leaf(
        id="announcement_date_value",
        desc=f"The announcement date is {EXPECTED['warsh']['announcement_date']}",
        parent=seq1,
        critical=True
    )
    announced = data.announcement_date.value if data.announcement_date else None
    await evaluator.verify(
        claim=f"The answer states the announcement date as '{announced}'. The correct announcement date is '{EXPECTED['warsh']['announcement_date']}'. Judge if they refer to the same calendar date (allow minor formatting/abbreviations).",
        node=v1,
        additional_instruction="Allow equivalent date formats (e.g., Jan. 30, 2026 vs January 30, 2026). Treat them as the same day if equivalent."
    )
    # url
    await add_url_support_or_fail(
        evaluator,
        id="announcement_date_url",
        desc="URL reference supporting the announcement date",
        claim=f"President Trump announced Kevin Warsh's nomination as Federal Reserve Chair on {EXPECTED['warsh']['announcement_date']}.",
        urls=(data.announcement_date.urls if data.announcement_date else []),
        parent=seq1,
        critical=True,
        additional_instruction="Accept official White House releases, major outlets, or government sources."
    )

    # 2) Transmission date
    seq2 = evaluator.add_sequential(
        id="senate_transmission_date",
        desc="The date when the formal nomination was transmitted to the Senate",
        parent=node,
        critical=True
    )
    v2 = evaluator.add_leaf(
        id="transmission_date_value",
        desc=f"The transmission date is {EXPECTED['warsh']['transmission_date']}",
        parent=seq2,
        critical=True
    )
    tx = data.senate_transmission_date.value if data.senate_transmission_date else None
    await evaluator.verify(
        claim=f"The answer states the formal nomination transmission date as '{tx}', which should be '{EXPECTED['warsh']['transmission_date']}'. Judge if they are the same date.",
        node=v2,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="transmission_date_url",
        desc="URL reference supporting the transmission date",
        claim=f"The formal nomination of Kevin Warsh as Federal Reserve Chair was transmitted to the U.S. Senate on {EXPECTED['warsh']['transmission_date']}.",
        urls=(data.senate_transmission_date.urls if data.senate_transmission_date else []),
        parent=seq2,
        critical=True,
        additional_instruction="Prefer Congress/Senate official records or the White House."
    )

    # 3) Committee review
    seq3 = evaluator.add_sequential(
        id="committee_review",
        desc="The Senate committee responsible for reviewing the nomination",
        parent=node,
        critical=True
    )
    v3 = evaluator.add_leaf(
        id="committee_name",
        desc=f"The committee is {EXPECTED['warsh']['committee_name']}",
        parent=seq3,
        critical=True
    )
    comm = data.committee.value if data.committee else None
    await evaluator.verify(
        claim=f"The answer states the reviewing Senate committee as '{comm}'. The correct committee is '{EXPECTED['warsh']['committee_name']}'. Judge if these refer to the same committee (allow minor name variants like including 'Committee on').",
        node=v3,
        additional_instruction="Consider 'Senate Committee on Banking, Housing, and Urban Affairs' equivalent to 'Senate Banking, Housing, and Urban Affairs Committee'."
    )
    await add_url_support_or_fail(
        evaluator,
        id="committee_url",
        desc="URL reference supporting the committee assignment",
        claim=f"Kevin Warsh's nomination as Fed Chair was referred to the {EXPECTED['warsh']['committee_name']}.",
        urls=(data.committee.urls if data.committee else []),
        parent=seq3,
        critical=True
    )

    # 4) Powell term end
    seq4 = evaluator.add_sequential(
        id="predecessor_term_end",
        desc="The date when Jerome Powell's term as Fed Chair ends",
        parent=node,
        critical=True
    )
    v4 = evaluator.add_leaf(
        id="powell_term_end_date",
        desc=f"Jerome Powell's term as Chair ends on {EXPECTED['warsh']['powell_term_end']}",
        parent=seq4,
        critical=True
    )
    powell = data.powell_term_end.value if data.powell_term_end else None
    await evaluator.verify(
        claim=f"The answer states Powell's term end date as '{powell}', which should be '{EXPECTED['warsh']['powell_term_end']}'. Judge if they refer to the same calendar date.",
        node=v4,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="powell_term_url",
        desc="URL reference supporting Powell's term end date",
        claim=f"Jerome Powell's term as Chair of the Federal Reserve ends on {EXPECTED['warsh']['powell_term_end']}.",
        urls=(data.powell_term_end.urls if data.powell_term_end else []),
        parent=seq4,
        critical=True,
        additional_instruction="Prefer Federal Reserve official pages or major reputable sources."
    )


async def verify_doug_collins(evaluator: Evaluator, parent, data: Optional[CollinsConfirmation]):
    node = evaluator.add_parallel(
        id="doug_collins_confirmation",
        desc="Information about Doug Collins's confirmation as VA Secretary",
        parent=parent,
        critical=True
    )
    data = data or CollinsConfirmation()

    # 1) Confirmation date
    s1 = evaluator.add_sequential(
        id="confirmation_date",
        desc="The date when Doug Collins was confirmed by the Senate",
        parent=node,
        critical=True
    )
    v1 = evaluator.add_leaf(
        id="confirmation_date_value",
        desc=f"The confirmation date is {EXPECTED['collins']['confirmation_date']}",
        parent=s1,
        critical=True
    )
    cd = data.confirmation_date.value if data.confirmation_date else None
    await evaluator.verify(
        claim=f"The answer states the confirmation date as '{cd}', which should be '{EXPECTED['collins']['confirmation_date']}'. Judge if they are the same date.",
        node=v1,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="confirmation_date_url",
        desc="URL reference supporting the confirmation date",
        claim=f"Doug Collins was confirmed by the U.S. Senate as Secretary of Veterans Affairs on {EXPECTED['collins']['confirmation_date']}.",
        urls=(data.confirmation_date.urls if data.confirmation_date else []),
        parent=s1,
        critical=True,
        additional_instruction="Prefer Senate roll call pages or major reputable outlets."
    )

    # 2) Vote count
    s2 = evaluator.add_sequential(
        id="confirmation_vote",
        desc="The Senate vote count for Doug Collins's confirmation",
        parent=node,
        critical=True
    )
    v2 = evaluator.add_leaf(
        id="vote_count",
        desc=f"The vote was {EXPECTED['collins']['vote_count']} in favor",
        parent=s2,
        critical=True
    )
    vc = data.vote_count.value if data.vote_count else None
    await evaluator.verify(
        claim=f"The answer states the Senate vote count as '{vc}', which should be '{EXPECTED['collins']['vote_count']}'. Judge if they are equivalent (normalize punctuation like en-dashes vs hyphens).",
        node=v2,
        additional_instruction="Normalize formats like '77–23', '77 to 23', or '77-23' as equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="vote_url",
        desc="URL reference supporting the vote count",
        claim=f"The Senate confirmed Doug Collins by a {EXPECTED['collins']['vote_count']} vote.",
        urls=(data.vote_count.urls if data.vote_count else []),
        parent=s2,
        critical=True
    )

    # 3) Hearing date
    s3 = evaluator.add_sequential(
        id="hearing_date",
        desc="The date of Doug Collins's confirmation hearing",
        parent=node,
        critical=True
    )
    v3 = evaluator.add_leaf(
        id="hearing_date_value",
        desc=f"The hearing was held on {EXPECTED['collins']['hearing_date']}",
        parent=s3,
        critical=True
    )
    hd = data.hearing_date.value if data.hearing_date else None
    await evaluator.verify(
        claim=f"The answer states the hearing date as '{hd}', which should be '{EXPECTED['collins']['hearing_date']}'. Judge if they are the same date.",
        node=v3,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="hearing_date_url",
        desc="URL reference supporting the hearing date",
        claim=f"Doug Collins's confirmation hearing was held on {EXPECTED['collins']['hearing_date']}.",
        urls=(data.hearing_date.urls if data.hearing_date else []),
        parent=s3,
        critical=True,
        additional_instruction="Prefer Senate/Veterans Affairs Committee schedule, or C-SPAN, or major reputable outlets."
    )

    # 4) VA position number
    s4 = evaluator.add_sequential(
        id="va_position_number",
        desc="Doug Collins's position number as VA Secretary",
        parent=node,
        critical=True
    )
    v4 = evaluator.add_leaf(
        id="position_number",
        desc=f"Doug Collins is the {EXPECTED['collins']['position_number']} Secretary of Veterans Affairs",
        parent=s4,
        critical=True
    )
    pn = data.position_number.value if data.position_number else None
    await evaluator.verify(
        claim=f"The answer states the VA Secretary position order as '{pn}', which should be '{EXPECTED['collins']['position_number']}'. Judge if they are equivalent (e.g., '12th', 'twelfth').",
        node=v4,
        additional_instruction="Accept ordinal word/number equivalents."
    )
    await add_url_support_or_fail(
        evaluator,
        id="position_url",
        desc="URL reference supporting the position number",
        claim=f"Doug Collins is the {EXPECTED['collins']['position_number']} Secretary of Veterans Affairs.",
        urls=(data.position_number.urls if data.position_number else []),
        parent=s4,
        critical=True
    )


async def verify_rescissions_act(evaluator: Evaluator, parent, data: Optional[RescissionsActInfo]):
    node = evaluator.add_parallel(
        id="rescissions_act",
        desc="Information about the Rescissions Act of 2025 that cut CPB funding",
        parent=parent,
        critical=True
    )
    data = data or RescissionsActInfo()

    # House vote (parallel: date + count)
    house = evaluator.add_parallel(
        id="house_vote",
        desc="Details of the House vote on the Rescissions Act",
        parent=node,
        critical=True
    )
    # House vote date (sequential)
    h_date = evaluator.add_sequential(
        id="house_vote_date",
        desc="The date of the House vote",
        parent=house,
        critical=True
    )
    hvd_leaf = evaluator.add_leaf(
        id="house_date_value",
        desc=f"The House vote occurred on {EXPECTED['rescissions']['house_vote_date']}",
        parent=h_date,
        critical=True
    )
    hvd = data.house_vote_date.value if data.house_vote_date else None
    await evaluator.verify(
        claim=f"The answer states the House vote date as '{hvd}', which should be '{EXPECTED['rescissions']['house_vote_date']}'. Judge if they are the same date.",
        node=hvd_leaf,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="house_date_url",
        desc="URL reference supporting the House vote date",
        claim=f"The U.S. House of Representatives voted on the Rescissions Act on {EXPECTED['rescissions']['house_vote_date']}.",
        urls=(data.house_vote_date.urls if data.house_vote_date else []),
        parent=h_date,
        critical=True,
        additional_instruction="Prefer House Clerk records, Congress.gov, or reputable outlets."
    )

    # House vote count (sequential)
    h_count = evaluator.add_sequential(
        id="house_vote_count",
        desc="The House vote count",
        parent=house,
        critical=True
    )
    hvc_leaf = evaluator.add_leaf(
        id="house_count_value",
        desc=f"The House vote was {EXPECTED['rescissions']['house_vote_count']} in favor",
        parent=h_count,
        critical=True
    )
    hvc = data.house_vote_count.value if data.house_vote_count else None
    await evaluator.verify(
        claim=f"The answer states the House vote count as '{hvc}', which should be '{EXPECTED['rescissions']['house_vote_count']}'. Judge if they are equivalent (normalize punctuation).",
        node=hvc_leaf,
        additional_instruction="Normalize formats like '214–212' vs '214-212'."
    )
    await add_url_support_or_fail(
        evaluator,
        id="house_count_url",
        desc="URL reference supporting the House vote count",
        claim=f"The House passed the Rescissions Act by a {EXPECTED['rescissions']['house_vote_count']} vote.",
        urls=(data.house_vote_count.urls if data.house_vote_count else []),
        parent=h_count,
        critical=True
    )

    # Senate vote (parallel: date + count)
    senate = evaluator.add_parallel(
        id="senate_vote",
        desc="Details of the Senate vote on the Rescissions Act",
        parent=node,
        critical=True
    )
    # Senate vote date
    s_date = evaluator.add_sequential(
        id="senate_vote_date",
        desc="The date of the Senate vote",
        parent=senate,
        critical=True
    )
    svd_leaf = evaluator.add_leaf(
        id="senate_date_value",
        desc=f"The Senate vote occurred on {EXPECTED['rescissions']['senate_vote_date']}",
        parent=s_date,
        critical=True
    )
    svd = data.senate_vote_date.value if data.senate_vote_date else None
    await evaluator.verify(
        claim=f"The answer states the Senate vote date as '{svd}', which should be '{EXPECTED['rescissions']['senate_vote_date']}'. Judge if they are the same date.",
        node=svd_leaf,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="senate_date_url",
        desc="URL reference supporting the Senate vote date",
        claim=f"The U.S. Senate voted on the Rescissions Act on {EXPECTED['rescissions']['senate_vote_date']}.",
        urls=(data.senate_vote_date.urls if data.senate_vote_date else []),
        parent=s_date,
        critical=True,
        additional_instruction="Prefer Senate records or reputable outlets."
    )

    # Senate vote count
    s_count = evaluator.add_sequential(
        id="senate_vote_count",
        desc="The Senate vote count",
        parent=senate,
        critical=True
    )
    svc_leaf = evaluator.add_leaf(
        id="senate_count_value",
        desc=f"The Senate vote was {EXPECTED['rescissions']['senate_vote_count']} in favor",
        parent=s_count,
        critical=True
    )
    svc = data.senate_vote_count.value if data.senate_vote_count else None
    await evaluator.verify(
        claim=f"The answer states the Senate vote count as '{svc}', which should be '{EXPECTED['rescissions']['senate_vote_count']}'. Judge equivalence.",
        node=svc_leaf,
        additional_instruction="Normalize punctuation in counts."
    )
    await add_url_support_or_fail(
        evaluator,
        id="senate_count_url",
        desc="URL reference supporting the Senate vote count",
        claim=f"The Senate passed the Rescissions Act by a {EXPECTED['rescissions']['senate_vote_count']} vote.",
        urls=(data.senate_vote_count.urls if data.senate_vote_count else []),
        parent=s_count,
        critical=True
    )

    # CPB amount rescinded
    cpb_amount = evaluator.add_sequential(
        id="cpb_funding_amount",
        desc="The amount of CPB funding rescinded",
        parent=node,
        critical=True
    )
    cpb_leaf = evaluator.add_leaf(
        id="cpb_amount_value",
        desc=f"The amount rescinded from CPB was {EXPECTED['rescissions']['cpb_amount']}",
        parent=cpb_amount,
        critical=True
    )
    cpb_val = data.cpb_funding_amount.value if data.cpb_funding_amount else None
    await evaluator.verify(
        claim=f"The answer states the CPB rescission amount as '{cpb_val}', which should be '{EXPECTED['rescissions']['cpb_amount']}'. Judge if they are equivalent (allow currency format variants).",
        node=cpb_leaf,
        additional_instruction="Treat '$1.1B', '1.1 billion dollars', and '$1.1 billion' as equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="cpb_amount_url",
        desc="URL reference supporting the CPB funding amount",
        claim=f"The Rescissions Act rescinded {EXPECTED['rescissions']['cpb_amount']} from the Corporation for Public Broadcasting (CPB).",
        urls=(data.cpb_funding_amount.urls if data.cpb_funding_amount else []),
        parent=cpb_amount,
        critical=True
    )

    # CPB board dissolution date
    cpb_diss = evaluator.add_sequential(
        id="cpb_dissolution_date",
        desc="The date when CPB's board voted to dissolve the organization",
        parent=node,
        critical=True
    )
    cpb_dis_leaf = evaluator.add_leaf(
        id="dissolution_date_value",
        desc=f"CPB's board voted to dissolve on {EXPECTED['rescissions']['cpb_dissolution_date']}",
        parent=cpb_diss,
        critical=True
    )
    cpb_dis = data.cpb_dissolution_date.value if data.cpb_dissolution_date else None
    await evaluator.verify(
        claim=f"The answer states the CPB board dissolution vote date as '{cpb_dis}', which should be '{EXPECTED['rescissions']['cpb_dissolution_date']}'. Judge date equivalence.",
        node=cpb_dis_leaf,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="dissolution_date_url",
        desc="URL reference supporting the dissolution date",
        claim=f"The CPB board voted to dissolve the organization on {EXPECTED['rescissions']['cpb_dissolution_date']}.",
        urls=(data.cpb_dissolution_date.urls if data.cpb_dissolution_date else []),
        parent=cpb_diss,
        critical=True
    )


async def verify_operation_midnight_hammer(evaluator: Evaluator, parent, data: Optional[OperationMidnightHammer]):
    node = evaluator.add_parallel(
        id="operation_midnight_hammer",
        desc="Information about Operation Midnight Hammer, the U.S. strikes on Iranian nuclear facilities",
        parent=parent,
        critical=True
    )
    data = data or OperationMidnightHammer()

    # 1) Operation date
    s1 = evaluator.add_sequential(
        id="operation_date",
        desc="The date when Operation Midnight Hammer was conducted",
        parent=node,
        critical=True
    )
    v1 = evaluator.add_leaf(
        id="date_value",
        desc=f"The operation occurred on {EXPECTED['midnight_hammer']['operation_date']}",
        parent=s1,
        critical=True
    )
    op_date = data.operation_date.value if data.operation_date else None
    await evaluator.verify(
        claim=f"The answer states the operation date as '{op_date}', which should be '{EXPECTED['midnight_hammer']['operation_date']}'. Judge date equivalence.",
        node=v1,
        additional_instruction="Allow equivalent date formats."
    )
    await add_url_support_or_fail(
        evaluator,
        id="date_url",
        desc="URL reference supporting the operation date",
        claim=f"Operation Midnight Hammer occurred on {EXPECTED['midnight_hammer']['operation_date']}.",
        urls=(data.operation_date.urls if data.operation_date else []),
        parent=s1,
        critical=True,
        additional_instruction="Prefer DoD statements or major reputable outlets."
    )

    # 2) Aircraft details (parallel)
    ac = evaluator.add_parallel(
        id="aircraft_details",
        desc="Details about the primary aircraft used in the operation",
        parent=node,
        critical=True
    )
    # type
    ac_type = evaluator.add_sequential(
        id="aircraft_type",
        desc="The type of bomber aircraft used",
        parent=ac,
        critical=True
    )
    ac_type_leaf = evaluator.add_leaf(
        id="type_value",
        desc="B-2 Spirit stealth bombers were used",
        parent=ac_type,
        critical=True
    )
    act = data.aircraft_type.value if data.aircraft_type else None
    await evaluator.verify(
        claim=f"The answer states the bomber type as '{act}', which should be equivalent to '{EXPECTED['midnight_hammer']['aircraft_type']}'. Judge if they refer to the same platform.",
        node=ac_type_leaf,
        additional_instruction="Minor wording differences acceptable if clearly B-2 Spirit stealth bombers."
    )
    await add_url_support_or_fail(
        evaluator,
        id="type_url",
        desc="URL reference supporting the aircraft type",
        claim="B-2 Spirit stealth bombers were the primary bomber aircraft used in Operation Midnight Hammer.",
        urls=(data.aircraft_type.urls if data.aircraft_type else []),
        parent=ac_type,
        critical=True
    )

    # count
    ac_count = evaluator.add_sequential(
        id="aircraft_count",
        desc="The number of B-2 bombers used",
        parent=ac,
        critical=True
    )
    ac_count_leaf = evaluator.add_leaf(
        id="count_value",
        desc="Seven B-2 bombers were used",
        parent=ac_count,
        critical=True
    )
    acc = data.aircraft_count.value if data.aircraft_count else None
    await evaluator.verify(
        claim=f"The answer states the number of B-2s as '{acc}', which should be '7' (seven). Judge numeric equivalence.",
        node=ac_count_leaf,
        additional_instruction="Treat '7' and 'seven' as equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="count_url",
        desc="URL reference supporting the aircraft count",
        claim="Seven B-2 bombers participated in Operation Midnight Hammer.",
        urls=(data.aircraft_count.urls if data.aircraft_count else []),
        parent=ac_count,
        critical=True
    )

    # base
    base = evaluator.add_sequential(
        id="base_location",
        desc="The air base from which the B-2s departed",
        parent=ac,
        critical=True
    )
    base_leaf = evaluator.add_leaf(
        id="base_value",
        desc="The bombers departed from Whiteman Air Force Base",
        parent=base,
        critical=True
    )
    b = data.base_location.value if data.base_location else None
    await evaluator.verify(
        claim=f"The answer states the departure base as '{b}', which should be '{EXPECTED['midnight_hammer']['base_location']}'. Judge equivalence.",
        node=base_leaf,
        additional_instruction="Allow 'Whiteman AFB' as equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="base_url",
        desc="URL reference supporting the base location",
        claim="The B-2 bombers departed from Whiteman Air Force Base for Operation Midnight Hammer.",
        urls=(data.base_location.urls if data.base_location else []),
        parent=base,
        critical=True
    )

    # 3) Munitions details (parallel)
    mun = evaluator.add_parallel(
        id="munitions_details",
        desc="Details about the primary munitions used",
        parent=node,
        critical=True
    )
    # type
    m_type = evaluator.add_sequential(
        id="munition_type",
        desc="The type of bomb used by the B-2s",
        parent=mun,
        critical=True
    )
    m_type_leaf = evaluator.add_leaf(
        id="munition_name",
        desc="GBU-57A/B MOP (Massive Ordnance Penetrator) bunker buster bombs were used",
        parent=m_type,
        critical=True
    )
    mt = data.munition_name.value if data.munition_name else None
    await evaluator.verify(
        claim=f"The answer states the munition as '{mt}'. The correct munition is '{EXPECTED['midnight_hammer']['munition_name']}'. Judge if they are the same munition.",
        node=m_type_leaf,
        additional_instruction="Allow minor naming variants for the GBU-57 'Massive Ordnance Penetrator (MOP)'."
    )
    await add_url_support_or_fail(
        evaluator,
        id="munition_url",
        desc="URL reference supporting the munition type",
        claim="The primary munition used was the GBU-57 Massive Ordnance Penetrator (MOP).",
        urls=(data.munition_name.urls if data.munition_name else []),
        parent=m_type,
        critical=True
    )

    # count
    m_count = evaluator.add_sequential(
        id="munition_count",
        desc="The total number of GBU-57 bombs dropped",
        parent=mun,
        critical=True
    )
    m_count_leaf = evaluator.add_leaf(
        id="bomb_count_value",
        desc="Fourteen GBU-57 bombs were dropped",
        parent=m_count,
        critical=True
    )
    mc = data.munition_count.value if data.munition_count else None
    await evaluator.verify(
        claim=f"The answer states the number of GBU-57 bombs as '{mc}', which should be '14'. Judge numeric equivalence.",
        node=m_count_leaf,
        additional_instruction="Treat number words and digits as equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="bomb_count_url",
        desc="URL reference supporting the bomb count",
        claim="Fourteen GBU-57 Massive Ordnance Penetrator bombs were dropped in Operation Midnight Hammer.",
        urls=(data.munition_count.urls if data.munition_count else []),
        parent=m_count,
        critical=True
    )

    # first combat use
    first_use = evaluator.add_sequential(
        id="first_combat_use",
        desc="Whether this was the first combat use of the GBU-57",
        parent=mun,
        critical=True
    )
    fu_leaf = evaluator.add_leaf(
        id="first_use_value",
        desc="This was the first combat use of the GBU-57 bomb",
        parent=first_use,
        critical=True
    )
    fu = data.first_combat_use.value if data.first_combat_use else None
    await evaluator.verify(
        claim=f"The answer states whether this was the first combat use as '{fu}'. The correct statement is 'Yes, this was the first combat use of the GBU-57.' Judge if equivalent.",
        node=fu_leaf,
        additional_instruction="Interpret 'Yes'/'True'/'first use' as equivalent; if explicitly 'No', not equivalent."
    )
    await add_url_support_or_fail(
        evaluator,
        id="first_use_url",
        desc="URL reference supporting the first combat use claim",
        claim="Operation Midnight Hammer marked the first combat use of the GBU-57 Massive Ordnance Penetrator.",
        urls=(data.first_combat_use.urls if data.first_combat_use else []),
        parent=first_use,
        critical=True
    )

    # 4) Target facilities
    targets = evaluator.add_sequential(
        id="target_facilities",
        desc="The three Iranian nuclear facilities targeted",
        parent=node,
        critical=True
    )
    tf_leaf = evaluator.add_leaf(
        id="three_facilities",
        desc="The three facilities are Fordow Uranium Enrichment Plant, Natanz Nuclear Facility, and Isfahan Nuclear Technology Center",
        parent=targets,
        critical=True
    )
    tf_vals = (data.target_facilities.values if data.target_facilities else [])
    await evaluator.verify(
        claim=f"The answer lists the targeted facilities as {tf_vals}. The correct three are {EXPECTED['midnight_hammer']['facilities']}. Judge if the sets match allowing reasonable naming variants/synonyms (e.g., 'Fordow' vs 'Fordo', 'Isfahan Nuclear Technology Center' vs 'Isfahan Uranium Conversion Facility').",
        node=tf_leaf,
        additional_instruction="Focus on semantic equivalence of the three sites."
    )
    await add_url_support_or_fail(
        evaluator,
        id="facilities_url",
        desc="URL reference supporting the target facilities",
        claim=f"The three Iranian nuclear facilities targeted were {', '.join(EXPECTED['midnight_hammer']['facilities'])}.",
        urls=(data.target_facilities.urls if data.target_facilities else []),
        parent=targets,
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2025-2026 Trump administration events task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates four independent event groups
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

    # Extract all structured information in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ProjectExtraction,
        extraction_name="project_extraction"
    )

    # Record "ground truth" expectations from rubric for transparency
    evaluator.add_ground_truth({
        "expected_values": EXPECTED
    })

    # Build verification tree according to rubric
    # Top-level children are all critical to reflect rubric's emphasis
    await verify_kevin_warsh(
        evaluator,
        parent=root,
        data=extracted.kevin_warsh_nomination if extracted else None
    )
    await verify_doug_collins(
        evaluator,
        parent=root,
        data=extracted.doug_collins_confirmation if extracted else None
    )
    await verify_rescissions_act(
        evaluator,
        parent=root,
        data=extracted.rescissions_act_2025 if extracted else None
    )
    await verify_operation_midnight_hammer(
        evaluator,
        parent=root,
        data=extracted.operation_midnight_hammer if extracted else None
    )

    # Return evaluator summary
    return evaluator.get_summary()