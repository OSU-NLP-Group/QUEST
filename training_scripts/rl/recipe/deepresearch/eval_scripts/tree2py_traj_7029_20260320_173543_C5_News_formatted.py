import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_primary_runoff_2026"
TASK_DESCRIPTION = (
    "I am a Texas voter who participated in the Republican primary on March 3, 2026, and I want to participate in "
    "the upcoming primary runoff election in May 2026. Please provide me with: "
    "(1) all critical dates and deadlines I need to know, including the election date, registration deadline, mail ballot "
    "application deadline, early voting period, and polling hours on Election Day; "
    "(2) the names of candidates in at least three statewide runoff races, including the Republican U.S. Senate race, "
    "Republican Attorney General race, and Democratic Lieutenant Governor race; and "
    "(3) the key voting rules and requirements, including which party's runoff I'm eligible to vote in based on my March "
    "primary participation, what photo ID types are accepted for in-person voting, who qualifies to vote by mail in Texas, "
    "and the deadline for returning mail ballots."
)

# Ground truth expectations for 2026 Texas May primary runoff
ELECTION_DATE_EXPECTED = "May 26, 2026"
POLLING_HOURS_EXPECTED = "7 a.m. to 7 p.m."
REGISTRATION_DEADLINE_EXPECTED = "April 27, 2026"
MAIL_APP_DEADLINE_EXPECTED_DATE = "May 15, 2026"
EARLY_VOTING_PERIOD_EXPECTED = "May 18–22, 2026"  # Allow variants like "May 18 to May 22, 2026"

EXPECTED_RACES = {
    "republican_us_senate": {"candidate_1": "John Cornyn", "candidate_2": "Ken Paxton"},
    "republican_attorney_general": {"candidate_1": "Mayes Middleton", "candidate_2": "Chip Roy"},
    "democratic_lieutenant_governor": {"candidate_1": "Vikki Goodwin", "candidate_2": "Marcos Vélez"},
}

EXPECTED_ID_TYPES = [
    "Texas driver's license",
    "Texas election identification certificate",
    "Texas personal ID card",
    "Texas handgun license",
    "U.S. military ID with photo",
    "U.S. citizenship certificate with photo",
    "U.S. passport",
]

EXPECTED_VBM_ELIGIBILITY = [
    "Age 65 or older",
    "Out of the county during the entire voting period",
    "Sickness or disability",
    "Confined in jail but otherwise eligible",
    "Expected to give birth within three weeks of Election Day",
]

MAIL_BALLOT_RETURN_RULE_EXPECTED = (
    "Mail ballot must be received by May 26, 2026, OR if postmarked by 7 p.m. on May 26, 2026, it must be received by 5 p.m. on May 27, 2026."
)

# --------------------------------------------------------------------------- #
# Pydantic data models for extraction                                         #
# --------------------------------------------------------------------------- #
class DateField(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CriticalDates(BaseModel):
    election_date: Optional[DateField] = None
    polling_hours: Optional[DateField] = None
    registration_deadline: Optional[DateField] = None
    mail_ballot_application_deadline: Optional[DateField] = None
    early_voting_period: Optional[DateField] = None


class RaceCandidates(BaseModel):
    candidate_1: Optional[str] = None
    candidate_2: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Races(BaseModel):
    republican_us_senate: Optional[RaceCandidates] = None
    republican_attorney_general: Optional[RaceCandidates] = None
    democratic_lieutenant_governor: Optional[RaceCandidates] = None


class RulesField(BaseModel):
    text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Rules(BaseModel):
    party_runoff_eligibility: Optional[RulesField] = None
    accepted_photo_id_types: List[str] = Field(default_factory=list)
    accepted_photo_id_sources: List[str] = Field(default_factory=list)
    vote_by_mail_eligibility: List[str] = Field(default_factory=list)
    vote_by_mail_sources: List[str] = Field(default_factory=list)
    mail_ballot_return_deadline: Optional[RulesField] = None


class RunoffInfoExtraction(BaseModel):
    dates: Optional[CriticalDates] = None
    races: Optional[Races] = None
    rules: Optional[Rules] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_runoff_info() -> str:
    return """
Extract the specific information the answer provides for the May 2026 Texas primary runoff. Only extract what is explicitly stated in the answer text. Do not infer or invent.

Return a JSON object following this schema:

- dates:
  - election_date: { value: string|null, sources: string[] }   // the runoff election date (e.g., "May 26, 2026")
  - polling_hours: { value: string|null, sources: string[] }   // election day hours (e.g., "7 a.m. to 7 p.m.")
  - registration_deadline: { value: string|null, sources: string[] } // voter registration/address change deadline
  - mail_ballot_application_deadline: { value: string|null, sources: string[] } // last day to apply; include receipt-vs-postmark detail if given
  - early_voting_period: { value: string|null, sources: string[] } // date range (e.g., "May 18–22, 2026")

- races:
  - republican_us_senate: { candidate_1: string|null, candidate_2: string|null, sources: string[] }
  - republican_attorney_general: { candidate_1: string|null, candidate_2: string|null, sources: string[] }
  - democratic_lieutenant_governor: { candidate_1: string|null, candidate_2: string|null, sources: string[] }

- rules:
  - party_runoff_eligibility: { text: string|null, sources: string[] } // rule about voting in the same party's runoff
  - accepted_photo_id_types: string[] // enumerate exactly the ID types listed in the answer (do not add or remove)
  - accepted_photo_id_sources: string[] // URLs supporting the ID list; collect all URLs the answer cites for this topic
  - vote_by_mail_eligibility: string[] // each eligibility category as a separate item, exactly as stated in the answer
  - vote_by_mail_sources: string[] // URLs supporting vote-by-mail eligibility
  - mail_ballot_return_deadline: { text: string|null, sources: string[] } // the return deadline rule including postmark/receipt timing

Source extraction rules:
- Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.
- If a URL is missing http/https, prepend http://
- If no source is provided for a field, return an empty array.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _safe_sources(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if isinstance(u, str) and len(u.strip()) > 0]


# ---------------------- DATES & DEADLINES VERIFICATION --------------------- #
async def verify_dates(evaluator: Evaluator, parent, ext: RunoffInfoExtraction) -> None:
    dates_node = evaluator.add_parallel(
        id="Critical_Dates_and_Deadlines",
        desc="Provide all required critical dates/deadlines and hours for the May 2026 Texas primary runoff.",
        parent=parent,
        critical=True,
    )

    # Election Date
    el_seq = evaluator.add_sequential(
        id="Election_Date",
        desc="States the runoff election date is May 26, 2026.",
        parent=dates_node,
        critical=True,
    )
    el_val = evaluator.add_leaf(
        id="Election_Date_value_correct",
        desc="Answer states the runoff election date is May 26, 2026.",
        parent=el_seq,
        critical=True,
    )
    stated = (ext.dates.election_date.value if ext and ext.dates and ext.dates.election_date else None) or ""
    await evaluator.verify(
        claim=f"The answer explicitly states the Texas primary runoff election date as '{stated}', and it matches '{ELECTION_DATE_EXPECTED}'.",
        node=el_val,
        additional_instruction="Allow minor formatting variants (e.g., 'May 26, 2026' vs 'May 26th, 2026').",
    )
    el_src = evaluator.add_leaf(
        id="Election_Date_source_supported",
        desc="Cited sources support that the runoff election date is May 26, 2026.",
        parent=el_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Texas 2026 primary runoff election date is {ELECTION_DATE_EXPECTED}.",
        node=el_src,
        sources=_safe_sources(ext.dates.election_date.sources if ext and ext.dates and ext.dates.election_date else []),
        additional_instruction="Verify the page explicitly indicates the May 2026 Texas primary runoff occurs on May 26, 2026.",
    )

    # Polling Hours
    ph_seq = evaluator.add_sequential(
        id="Polling_Hours_Election_Day",
        desc="States polling hours on Election Day are 7 a.m. to 7 p.m.",
        parent=dates_node,
        critical=True,
    )
    ph_val = evaluator.add_leaf(
        id="Polling_Hours_value_correct",
        desc="Answer states polling hours on Election Day are 7 a.m. to 7 p.m.",
        parent=ph_seq,
        critical=True,
    )
    ph_stated = (ext.dates.polling_hours.value if ext and ext.dates and ext.dates.polling_hours else None) or ""
    await evaluator.verify(
        claim=f"The answer explicitly states polling hours on Election Day as '{ph_stated}', and it matches '{POLLING_HOURS_EXPECTED}'.",
        node=ph_val,
        additional_instruction="Accept reasonable variants such as '7 AM - 7 PM', '7:00 a.m. to 7:00 p.m.', or en-dash instead of 'to'.",
    )
    ph_src = evaluator.add_leaf(
        id="Polling_Hours_source_supported",
        desc="Cited sources support polling hours are 7 a.m. to 7 p.m.",
        parent=ph_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="Election Day polling hours in Texas are 7 a.m. to 7 p.m.",
        node=ph_src,
        sources=_safe_sources(ext.dates.polling_hours.sources if ext and ext.dates and ext.dates.polling_hours else []),
        additional_instruction="Verify the page explicitly shows the Texas Election Day polling hours as 7 a.m. to 7 p.m.",
    )

    # Registration Deadline
    rd_seq = evaluator.add_sequential(
        id="Voter_Registration_Address_Change_Deadline",
        desc="States the voter registration/address change deadline is April 27, 2026.",
        parent=dates_node,
        critical=True,
    )
    rd_val = evaluator.add_leaf(
        id="Registration_Deadline_value_correct",
        desc="Answer states the voter registration/address change deadline is April 27, 2026.",
        parent=rd_seq,
        critical=True,
    )
    rd_stated = (ext.dates.registration_deadline.value if ext and ext.dates and ext.dates.registration_deadline else None) or ""
    await evaluator.verify(
        claim=f"The answer explicitly states the voter registration or address change deadline as '{rd_stated}', and it matches '{REGISTRATION_DEADLINE_EXPECTED}'.",
        node=rd_val,
        additional_instruction="Allow minor formatting variants; this is the last day to register or update address for the May 26, 2026 runoff.",
    )
    rd_src = evaluator.add_leaf(
        id="Registration_Deadline_source_supported",
        desc="Cited sources support registration/address change deadline is April 27, 2026.",
        parent=rd_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The voter registration/address change deadline for the May 26, 2026 Texas primary runoff is {REGISTRATION_DEADLINE_EXPECTED}.",
        node=rd_src,
        sources=_safe_sources(ext.dates.registration_deadline.sources if ext and ext.dates and ext.dates.registration_deadline else []),
        additional_instruction="Confirm the page states April 27, 2026 as the registration/address change deadline for this election.",
    )

    # Mail Ballot Application Deadline (received, not postmarked)
    mbad_seq = evaluator.add_sequential(
        id="Mail_Ballot_Application_Deadline",
        desc="States the last day to apply for a mail ballot is May 15, 2026 and that the application must be received (not merely postmarked).",
        parent=dates_node,
        critical=True,
    )
    mbad_val = evaluator.add_leaf(
        id="Mail_Ballot_App_Deadline_value_correct",
        desc="Answer states last day to apply is May 15, 2026 and 'received' not merely postmarked.",
        parent=mbad_seq,
        critical=True,
    )
    mb_stated = (ext.dates.mail_ballot_application_deadline.value if ext and ext.dates and ext.dates.mail_ballot_application_deadline else None) or ""
    await evaluator.verify(
        claim=(
            f"The answer states the last day to apply for a ballot by mail is '{mb_stated}', and that date is '{MAIL_APP_DEADLINE_EXPECTED_DATE}', "
            "and the answer explicitly indicates the application must be 'received' by that date (not merely postmarked)."
        ),
        node=mbad_val,
        additional_instruction="Check both the date and that the 'received, not postmarked' requirement is clearly stated.",
    )
    mbad_src = evaluator.add_leaf(
        id="Mail_Ballot_App_Deadline_source_supported",
        desc="Cited sources support mail ballot application deadline: received by May 15, 2026 (not postmarked).",
        parent=mbad_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            f"The last day to apply for a ballot by mail for the May 26, 2026 Texas primary runoff is {MAIL_APP_DEADLINE_EXPECTED_DATE}, "
            "and the application must be received by that date (not merely postmarked)."
        ),
        node=mbad_src,
        sources=_safe_sources(ext.dates.mail_ballot_application_deadline.sources if ext and ext.dates and ext.dates.mail_ballot_application_deadline else []),
        additional_instruction="The page should explicitly indicate 'received by' language, not 'postmarked'.",
    )

    # Early Voting Period
    ev_seq = evaluator.add_sequential(
        id="Early_Voting_Period",
        desc="States early voting runs May 18–22, 2026.",
        parent=dates_node,
        critical=True,
    )
    ev_val = evaluator.add_leaf(
        id="Early_Voting_value_correct",
        desc="Answer states early voting runs May 18–22, 2026.",
        parent=ev_seq,
        critical=True,
    )
    ev_stated = (ext.dates.early_voting_period.value if ext and ext.dates and ext.dates.early_voting_period else None) or ""
    await evaluator.verify(
        claim=f"The answer states the early voting period as '{ev_stated}', and it matches 'May 18–22, 2026'.",
        node=ev_val,
        additional_instruction="Allow variants like 'May 18 to May 22, 2026' or using hyphen/dash.",
    )
    ev_src = evaluator.add_leaf(
        id="Early_Voting_source_supported",
        desc="Cited sources support early voting runs May 18–22, 2026.",
        parent=ev_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="Early voting for the May 26, 2026 Texas primary runoff runs from May 18, 2026 through May 22, 2026.",
        node=ev_src,
        sources=_safe_sources(ext.dates.early_voting_period.sources if ext and ext.dates and ext.dates.early_voting_period else []),
        additional_instruction="Verify the page clearly displays the early voting start and end dates.",
    )


# --------------------- RACES & CANDIDATES VERIFICATION --------------------- #
async def _verify_race(
    evaluator: Evaluator,
    parent,
    node_id: str,
    node_desc: str,
    extracted: Optional[RaceCandidates],
    expected_a: str,
    expected_b: str,
):
    race_seq = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent,
        critical=True,
    )

    # Value correctness (names listed match expected; order-insensitive)
    val_node = evaluator.add_leaf(
        id=f"{node_id}_names_correct",
        desc=f"Answer lists exactly these two candidates (order-insensitive): {expected_a} and {expected_b}.",
        parent=race_seq,
        critical=True,
    )
    stated_1 = (extracted.candidate_1 if extracted else "") or ""
    stated_2 = (extracted.candidate_2 if extracted else "") or ""
    await evaluator.verify(
        claim=(
            f"The answer names the two runoff candidates for this race as '{stated_1}' and '{stated_2}', "
            f"and those two names are exactly '{expected_a}' and '{expected_b}' (order doesn't matter)."
        ),
        node=val_node,
        additional_instruction="Permit minor punctuation/diacritics or middle names; evaluate order-insensitively.",
    )

    # Sources support
    src_node = evaluator.add_leaf(
        id=f"{node_id}_sources_supported",
        desc="Cited sources support both named candidates are in the specified runoff.",
        parent=race_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The two runoff candidates are {expected_a} and {expected_b}.",
        node=src_node,
        sources=_safe_sources(extracted.sources if extracted else []),
        additional_instruction="The page(s) should clearly indicate these two individuals are the runoff candidates for the specified 2026 Texas race.",
    )


async def verify_races(evaluator: Evaluator, parent, ext: RunoffInfoExtraction) -> None:
    races_node = evaluator.add_parallel(
        id="Statewide_Runoff_Races_and_Candidates",
        desc="Provide candidate names for the specified statewide runoff races.",
        parent=parent,
        critical=True,
    )

    # Republican U.S. Senate
    await _verify_race(
        evaluator,
        races_node,
        "Republican_US_Senate_Runoff_Candidates",
        "Names both candidates in the Republican U.S. Senate runoff: John Cornyn vs Ken Paxton.",
        extracted=ext.races.republican_us_senate if ext and ext.races else None,
        expected_a=EXPECTED_RACES["republican_us_senate"]["candidate_1"],
        expected_b=EXPECTED_RACES["republican_us_senate"]["candidate_2"],
    )

    # Republican Attorney General
    await _verify_race(
        evaluator,
        races_node,
        "Republican_Attorney_General_Runoff_Candidates",
        "Names both candidates in the Republican Attorney General runoff: Mayes Middleton vs Chip Roy.",
        extracted=ext.races.republican_attorney_general if ext and ext.races else None,
        expected_a=EXPECTED_RACES["republican_attorney_general"]["candidate_1"],
        expected_b=EXPECTED_RACES["republican_attorney_general"]["candidate_2"],
    )

    # Democratic Lieutenant Governor
    await _verify_race(
        evaluator,
        races_node,
        "Democratic_Lieutenant_Governor_Runoff_Candidates",
        "Names both candidates in the Democratic Lieutenant Governor runoff: Vikki Goodwin vs Marcos Vélez.",
        extracted=ext.races.democratic_lieutenant_governor if ext and ext.races else None,
        expected_a=EXPECTED_RACES["democratic_lieutenant_governor"]["candidate_1"],
        expected_b=EXPECTED_RACES["democratic_lieutenant_governor"]["candidate_2"],
    )


# --------------------- RULES & REQUIREMENTS VERIFICATION ------------------- #
async def verify_rules(evaluator: Evaluator, parent, ext: RunoffInfoExtraction) -> None:
    rules_node = evaluator.add_parallel(
        id="Voting_Rules_and_Requirements",
        desc="Provide key runoff voting rules and requirements requested.",
        parent=parent,
        critical=True,
    )

    # Party runoff eligibility (same party rule)
    pr_seq = evaluator.add_sequential(
        id="Party_Runoff_Eligibility_Rule",
        desc="States that a voter who participated in a party's March 3 primary may only vote in that same party's runoff.",
        parent=rules_node,
        critical=True,
    )
    pr_val = evaluator.add_leaf(
        id="Party_Runoff_Eligibility_value_correct",
        desc="Answer states the same-party runoff eligibility rule.",
        parent=pr_seq,
        critical=True,
    )
    pr_text = (ext.rules.party_runoff_eligibility.text if ext and ext.rules and ext.rules.party_runoff_eligibility else "") or ""
    await evaluator.verify(
        claim="The answer clearly states that if a voter participated in a party's March 3 primary, they can only vote in that same party's May runoff.",
        node=pr_val,
        additional_instruction="Check the answer text for this exact rule; mention of the user's Republican participation is context but not required for this check.",
    )
    pr_src = evaluator.add_leaf(
        id="Party_Runoff_Eligibility_sources_supported",
        desc="Cited sources support the same-party runoff eligibility rule.",
        parent=pr_seq,
        critical=True,
    )
    await evaluator.verify(
        claim="Texas runoff voting rule: a voter who voted in a party's primary may vote only in that same party's runoff.",
        node=pr_src,
        sources=_safe_sources(ext.rules.party_runoff_eligibility.sources if ext and ext.rules and ext.rules.party_runoff_eligibility else []),
        additional_instruction="Verify the page states the same-party runoff restriction.",
    )

    # Accepted photo ID types (7 types)
    id_seq = evaluator.add_sequential(
        id="Accepted_Photo_ID_Types",
        desc="Correctly lists the seven accepted photo ID types for in-person voting.",
        parent=rules_node,
        critical=True,
    )
    id_val = evaluator.add_leaf(
        id="Accepted_ID_value_correct",
        desc="Answer lists exactly the seven accepted photo ID types (allowing naming variants).",
        parent=id_seq,
        critical=True,
    )
    stated_ids = ext.rules.accepted_photo_id_types if ext and ext.rules else []
    await evaluator.verify(
        claim=(
            f"The answer's accepted ID list {stated_ids} corresponds exactly to these seven types (order irrelevant, allow synonyms): "
            f"{EXPECTED_ID_TYPES}."
        ),
        node=id_val,
        additional_instruction=(
            "Treat 'Texas License to Carry (handgun license)' as equivalent to 'Texas handgun license'. "
            "Treat 'Election Identification Certificate' as 'Texas election identification certificate'. "
            "Allow minor wording differences but do not allow extra or missing categories."
        ),
    )
    id_src = evaluator.add_leaf(
        id="Accepted_ID_sources_supported",
        desc="Cited sources support the list of seven accepted photo ID types.",
        parent=id_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "Texas in-person voting requires one of these seven photo IDs: "
            "Texas driver's license; Texas election identification certificate; Texas personal ID card; "
            "Texas handgun license; U.S. military ID with photo; U.S. citizenship certificate with photo; U.S. passport."
        ),
        node=id_src,
        sources=_safe_sources(ext.rules.accepted_photo_id_sources if ext and ext.rules else []),
        additional_instruction="The page(s) should enumerate these exact seven categories (allowing minor naming variants).",
    )

    # Vote by mail eligibility categories
    vbm_seq = evaluator.add_sequential(
        id="Vote_By_Mail_Eligibility",
        desc="Correctly states who qualifies to vote by mail in Texas.",
        parent=rules_node,
        critical=True,
    )
    vbm_val = evaluator.add_leaf(
        id="VBM_Eligibility_value_correct",
        desc="Answer lists all required VBM eligibility categories (no extras, no missing).",
        parent=vbm_seq,
        critical=True,
    )
    stated_vbm = ext.rules.vote_by_mail_eligibility if ext and ext.rules else []
    await evaluator.verify(
        claim=(
            f"The answer's vote-by-mail eligibility list {stated_vbm} contains exactly these categories (order irrelevant, allow small wording differences): "
            f"{EXPECTED_VBM_ELIGIBILITY}."
        ),
        node=vbm_val,
        additional_instruction="Confirm all five categories are present and there are no extra unrelated categories.",
    )
    vbm_src = evaluator.add_leaf(
        id="VBM_Eligibility_sources_supported",
        desc="Cited sources support the VBM eligibility categories.",
        parent=vbm_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "Texas allows voting by mail if the voter is 65 or older; out of the county for the entire voting period; "
            "has a sickness or disability; is confined in jail but otherwise eligible; or is expected to give birth within three weeks of Election Day."
        ),
        node=vbm_src,
        sources=_safe_sources(ext.rules.vote_by_mail_sources if ext and ext.rules else []),
        additional_instruction="The page(s) should clearly enumerate the eligibility categories (allowing minor wording differences).",
    )

    # Mail ballot return deadline rule
    mr_seq = evaluator.add_sequential(
        id="Mail_Ballot_Return_Deadline",
        desc="Correctly states the mail ballot return deadline rule for the May 2026 runoff.",
        parent=rules_node,
        critical=True,
    )
    mr_val = evaluator.add_leaf(
        id="Mail_Return_value_correct",
        desc="Answer states return-by deadlines correctly (received by May 26; postmark by 7 p.m. May 26 -> received by 5 p.m. May 27).",
        parent=mr_seq,
        critical=True,
    )
    mr_text = (ext.rules.mail_ballot_return_deadline.text if ext and ext.rules and ext.rules.mail_ballot_return_deadline else "") or ""
    await evaluator.verify(
        claim=(
            "The answer correctly states the mail ballot return deadline rule as: received by May 26, 2026, OR if postmarked by 7 p.m. on May 26, 2026, "
            "then received by 5 p.m. on May 27, 2026."
        ),
        node=mr_val,
        additional_instruction="Check that both the received-by-Election-Day and the postmark-then-receipt-next-day by 5 p.m. conditions are present.",
    )
    mr_src = evaluator.add_leaf(
        id="Mail_Return_sources_supported",
        desc="Cited sources support the mail ballot return deadline rule.",
        parent=mr_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=MAIL_BALLOT_RETURN_RULE_EXPECTED,
        node=mr_src,
        sources=_safe_sources(ext.rules.mail_ballot_return_deadline.sources if ext and ext.rules and ext.rules.mail_ballot_return_deadline else []),
        additional_instruction="Verify both conditions (received by Election Day OR timely postmark with next-day receipt by 5 p.m.) are stated.",
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
) -> Dict:
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_runoff_info(),
        template_class=RunoffInfoExtraction,
        extraction_name="runoff_info_extraction",
    )

    # Add Ground Truth / expected info for traceability
    evaluator.add_ground_truth(
        {
            "expected_dates": {
                "election_date": ELECTION_DATE_EXPECTED,
                "polling_hours": POLLING_HOURS_EXPECTED,
                "registration_deadline": REGISTRATION_DEADLINE_EXPECTED,
                "mail_ballot_application_deadline": f"{MAIL_APP_DEADLINE_EXPECTED_DATE} (received, not postmarked)",
                "early_voting_period": EARLY_VOTING_PERIOD_EXPECTED,
            },
            "expected_races": EXPECTED_RACES,
            "expected_rules": {
                "party_runoff_rule": "Must vote in the same party’s runoff as the party primary you participated in.",
                "accepted_photo_id_types": EXPECTED_ID_TYPES,
                "vote_by_mail_eligibility": EXPECTED_VBM_ELIGIBILITY,
                "mail_ballot_return_deadline": MAIL_BALLOT_RETURN_RULE_EXPECTED,
            },
        },
        gt_type="ground_truth",
    )

    # Build main critical node under root
    main = evaluator.add_parallel(
        id="Texas_Primary_Runoff_Information",
        desc="Provide required information for a Texas voter about the May 2026 primary runoff, covering (1) critical dates/deadlines, (2) candidates in specified statewide runoff races, and (3) key voting rules/requirements.",
        parent=root,
        critical=True,
    )

    # Verify subparts
    await verify_dates(evaluator, main, extracted)
    await verify_races(evaluator, main, extracted)
    await verify_rules(evaluator, main, extracted)

    return evaluator.get_summary()