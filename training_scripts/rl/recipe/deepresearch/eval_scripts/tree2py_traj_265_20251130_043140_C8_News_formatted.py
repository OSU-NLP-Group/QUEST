import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "news_briefing_2025_11_24_28"
TASK_DESCRIPTION = (
    "Compile a detailed news briefing report on four significant international events that occurred between November 24-28, 2025. "
    "For each event, provide the specific factual details requested below, along with a reference URL from a credible news source.\n\n"
    "Event 1 - West African Military Crisis: Identify the West African country that experienced a military coup on November 26, 2025, just one day before scheduled election results were to be announced. Provide: "
    "(a) the name of the country, (b) the exact date of the coup, (c) the name and rank of the military officer who led the coup, (d) the name of the president who was arrested, "
    "(e) the name of the general who was proclaimed head of the military government on November 27, (f) the name of the person appointed as prime minister on November 28, "
    "(g) the international organization that suspended the country's membership on November 27, and (h) a reference URL.\n\n"
    "Event 2 - U.S. Diplomatic Statement on European NATO Ally: Identify the European NATO member country for which the U.S. Secretary of State issued a congratulatory statement on November 28, 2025, marking the country's Independence Day. "
    "Provide: (a) the name of the country, (b) the date of the statement, (c) the name of the U.S. Secretary of State who issued it, (d) the occasion being commemorated, "
    "(e) mention of at least two key cooperation themes highlighted in the statement (defense alliance and security cooperation areas), and (f) a reference URL.\n\n"
    "Event 3 - Eastern European Political Resignation: Identify the high-ranking official from an Eastern European country who resigned on November 28, 2025, following a corruption scandal in which their home was raided. "
    "This official had been leading their country's delegation in peace negotiations. Provide: (a) the name of the country, (b) the name of the official, (c) their position/title, (d) the resignation date, "
    "(e) the reason for resignation, (f) mention of the home raid detail, and (g) a reference URL.\n\n"
    "Event 4 - U.S. Gubernatorial Race Announcement: Identify the 30-year-old investment firm CEO who announced their candidacy for a U.S. state's 2026 gubernatorial race on November 24, 2025. "
    "Provide: (a) the state, (b) the candidate's full name, (c) the announcement date, (d) the candidate's age, (e) their professional background, (f) the election year, and (g) a reference URL."
)


# Ground truth expectations from rubric (recorded for reference)
GT_EVENT_1 = {
    "country": "Guinea-Bissau",
    "coup_date": "November 26, 2025",
    "coup_leader": "Brigadier General Dinis Incanha",
    "arrested_president": "Umaro Sissoco Embaló",
    "military_government_head": "General Horta Inta-A Na Man",
    "military_government_head_date": "November 27, 2025",
    "prime_minister": "Ilídio Vieira Té",
    "prime_minister_date": "November 28, 2025",
    "organization": "ECOWAS"
}
GT_EVENT_2 = {
    "country": "Albania",
    "statement_date": "November 28, 2025",
    "issuing_official": "Marco Rubio",
    "issuing_official_title": "U.S. Secretary of State",
    "occasion": "Albania's Independence Day",
    "themes": ["NATO / mutual defense", "cybersecurity cooperation"]
}
GT_EVENT_3 = {
    "country": "Ukraine",
    "official_name": "Andriy Yermak",
    "position_title": "chief of staff",
    "resignation_date": "November 28, 2025",
    "reason": "corruption scandal",
    "home_raid_detail": "home was raided",
    "peace_negotiations_role": "leading the country's delegation in peace negotiations"
}
GT_EVENT_4 = {
    "state": "Florida",
    "candidate_full_name": "James Fishback",
    "announcement_date": "November 24, 2025",
    "candidate_age": "30",
    "professional_background": "investment firm CEO",
    "election_year": "2026"
}


# ----------------------------- Data Models ---------------------------------- #
class Event1Details(BaseModel):
    country: Optional[str] = None
    coup_date: Optional[str] = None
    coup_leader_name: Optional[str] = None
    coup_leader_rank: Optional[str] = None
    arrested_president: Optional[str] = None
    military_government_head: Optional[str] = None
    military_government_head_date: Optional[str] = None
    prime_minister: Optional[str] = None
    prime_minister_date: Optional[str] = None
    international_org: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Event2Details(BaseModel):
    country: Optional[str] = None
    statement_date: Optional[str] = None
    issuing_official_name: Optional[str] = None
    issuing_official_title: Optional[str] = None
    occasion: Optional[str] = None
    theme_nato_mutual_defense: Optional[str] = None
    theme_cybersecurity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Event3Details(BaseModel):
    country: Optional[str] = None
    official_name: Optional[str] = None
    position_title: Optional[str] = None
    resignation_date: Optional[str] = None
    reason_for_resignation: Optional[str] = None
    home_raid_detail: Optional[str] = None
    peace_negotiations_role: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Event4Details(BaseModel):
    state: Optional[str] = None
    candidate_full_name: Optional[str] = None
    announcement_date: Optional[str] = None
    candidate_age: Optional[str] = None
    professional_background: Optional[str] = None
    election_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class BriefingExtraction(BaseModel):
    west_africa_military_crisis: Optional[Event1Details] = None
    us_diplomatic_statement: Optional[Event2Details] = None
    eastern_europe_resignation: Optional[Event3Details] = None
    us_governor_race_announcement: Optional[Event4Details] = None


# ------------------------- Extraction Prompt -------------------------------- #
def prompt_extract_briefing() -> str:
    return (
        "Extract structured details for four events (Nov 24–28, 2025) exactly as presented in the answer. "
        "Return a JSON object with keys: 'west_africa_military_crisis', 'us_diplomatic_statement', "
        "'eastern_europe_resignation', 'us_governor_race_announcement'. For each key, extract the following:\n\n"
        "west_africa_military_crisis:\n"
        "- country: West African country that experienced the military coup\n"
        "- coup_date: exact coup date\n"
        "- coup_leader_name: name of the coup leader\n"
        "- coup_leader_rank: rank of the coup leader (e.g., Brigadier General)\n"
        "- arrested_president: name of the arrested president\n"
        "- military_government_head: name of the general proclaimed head of the military government (Nov 27)\n"
        "- military_government_head_date: the proclamation date\n"
        "- prime_minister: name of the person appointed prime minister (Nov 28)\n"
        "- prime_minister_date: the appointment date\n"
        "- international_org: name of the organization that suspended membership (Nov 27)\n"
        "- reference_urls: all URLs the answer cites for this event\n\n"
        "us_diplomatic_statement:\n"
        "- country: European NATO member country addressed by the statement\n"
        "- statement_date: date of the statement\n"
        "- issuing_official_name: name of the U.S. Secretary of State who issued it\n"
        "- issuing_official_title: the title (e.g., U.S. Secretary of State)\n"
        "- occasion: the occasion commemorated (e.g., Independence Day)\n"
        "- theme_nato_mutual_defense: phrase or sentence in the answer mentioning NATO/mutual defense cooperation\n"
        "- theme_cybersecurity: phrase or sentence mentioning cybersecurity cooperation\n"
        "- reference_urls: all URLs the answer cites for this event\n\n"
        "eastern_europe_resignation:\n"
        "- country: the Eastern European country\n"
        "- official_name: name of the official who resigned\n"
        "- position_title: the official's position/title\n"
        "- resignation_date: date of resignation\n"
        "- reason_for_resignation: description (e.g., corruption scandal)\n"
        "- home_raid_detail: phrase noting the home raid\n"
        "- peace_negotiations_role: phrase noting leading the country's delegation in peace negotiations\n"
        "- reference_urls: all URLs the answer cites for this event\n\n"
        "us_governor_race_announcement:\n"
        "- state: U.S. state\n"
        "- candidate_full_name: full name of the candidate\n"
        "- announcement_date: date of the announcement\n"
        "- candidate_age: age stated in the answer\n"
        "- professional_background: role (e.g., investment firm CEO)\n"
        "- election_year: the election year\n"
        "- reference_urls: all URLs the answer cites for this event\n\n"
        "Rules:\n"
        "1) Extract only what is explicitly stated in the answer. If a field is missing, set it to null. "
        "2) For reference_urls, extract actual URLs (including markdown links), one per array item. "
        "3) Do not infer or invent any details."
    )


# ------------------------- Helper Functions --------------------------------- #
def _safe_str(v: Optional[str]) -> str:
    return v if (v is not None and str(v).strip() != "") else "NULL"


def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if urls else []


# ----------------------- Verification Functions ----------------------------- #
async def verify_event_1(
    evaluator: Evaluator,
    parent: VerificationNode,
    ev1: Optional[Event1Details]
) -> None:
    # Create Event 1 node (parallel aggregation, non-critical at event level)
    event_node = evaluator.add_parallel(
        id="west_africa_military_crisis",
        desc="Event 1: West African military coup details",
        parent=parent,
        critical=False
    )

    urls = _urls_or_empty(ev1.reference_urls if ev1 else [])

    # Credible reference URL check (critical)
    cred_leaf = evaluator.add_leaf(
        id="credible_reference_url",
        desc="Provides at least one reference URL from a credible news source for Event 1",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is a credible news or official organization source.",
        node=cred_leaf,
        sources=urls,
        additional_instruction=(
            "Judge credibility by domain reputation (e.g., AP, Reuters, BBC, Bloomberg, NYTimes, WSJ, CNN, Al Jazeera, The Guardian) "
            "or official government/intergovernmental websites (.gov, state.gov, un.org, europa.eu, nato.int, ecowas.org). "
            "If there are no URLs or all are non-credible, return Incorrect."
        )
    )

    # Country identification (critical) – match expected country
    country_leaf = evaluator.add_leaf(
        id="country_identification",
        desc="Identifies Guinea-Bissau as the country",
        parent=event_node,
        critical=True
    )
    extracted_country = _safe_str(ev1.country if ev1 else None)
    await evaluator.verify(
        claim=f"The answer identifies the country as '{extracted_country}', and it matches 'Guinea-Bissau'.",
        node=country_leaf,
        additional_instruction="Treat case-insensitivity and minor diacritic variations as acceptable. If the extracted value is NULL or not matching, return Incorrect."
    )

    # Coup date (critical) – verify answer's date with sources and match expected
    coup_date_leaf = evaluator.add_leaf(
        id="coup_date",
        desc="States the coup date as November 26, 2025",
        parent=event_node,
        critical=True
    )
    extracted_coup_date = _safe_str(ev1.coup_date if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, the coup date is '{extracted_coup_date}'. The coup occurred on November 26, 2025.",
        node=coup_date_leaf,
        sources=urls,
        additional_instruction="Allow date format variants (e.g., 26 November 2025). If the extracted value is NULL or not November 26, 2025, return Incorrect."
    )

    # Coup leader name and rank (critical)
    leader_leaf = evaluator.add_leaf(
        id="coup_leader_name_and_rank",
        desc="Identifies Brigadier General Dinis Incanha as the coup leader (including rank)",
        parent=event_node,
        critical=True
    )
    leader_rank = _safe_str(ev1.coup_leader_rank if ev1 else None)
    leader_name = _safe_str(ev1.coup_leader_name if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, the coup leader is '{leader_rank} {leader_name}'. Sources should confirm Brigadier General Dinis Incanha led the coup.",
        node=leader_leaf,
        sources=urls,
        additional_instruction="Allow minor spelling/diacritic variations. If extracted is NULL or does not match, return Incorrect."
    )

    # Arrested president (critical)
    arrested_leaf = evaluator.add_leaf(
        id="arrested_president",
        desc="Identifies President Umaro Sissoco Embaló as the arrested president",
        parent=event_node,
        critical=True
    )
    extracted_pres = _safe_str(ev1.arrested_president if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, the arrested president was '{extracted_pres}'. Sources should confirm it was Umaro Sissoco Embaló.",
        node=arrested_leaf,
        sources=urls,
        additional_instruction="Allow minor spelling/diacritic variations. If extracted is NULL or does not match, return Incorrect."
    )

    # Military government head (critical) – includes date
    head_leaf = evaluator.add_leaf(
        id="military_government_head",
        desc="Identifies General Horta Inta-A Na Man as proclaimed head of the military government on November 27, 2025",
        parent=event_node,
        critical=True
    )
    head_name = _safe_str(ev1.military_government_head if ev1 else None)
    head_date = _safe_str(ev1.military_government_head_date if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, on '{head_date}', '{head_name}' was proclaimed head of the military government. Sources should confirm General Horta Inta-A Na Man was proclaimed on November 27, 2025.",
        node=head_leaf,
        sources=urls,
        additional_instruction="If extracted values are NULL or the date/person do not match the fact (Nov 27, 2025; Horta Inta-A Na Man), return Incorrect."
    )

    # Prime minister appointment (critical) – includes date
    pm_leaf = evaluator.add_leaf(
        id="prime_minister",
        desc="Identifies Ilídio Vieira Té as appointed prime minister on November 28, 2025",
        parent=event_node,
        critical=True
    )
    pm_name = _safe_str(ev1.prime_minister if ev1 else None)
    pm_date = _safe_str(ev1.prime_minister_date if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, on '{pm_date}', '{pm_name}' was appointed prime minister. Sources should confirm Ilídio Vieira Té was appointed on November 28, 2025.",
        node=pm_leaf,
        sources=urls,
        additional_instruction="If extracted values are NULL or do not match the person/date, return Incorrect."
    )

    # International organization response (critical)
    org_leaf = evaluator.add_leaf(
        id="international_organization_response",
        desc="Identifies ECOWAS as the international organization that suspended the country's membership on November 27, 2025",
        parent=event_node,
        critical=True
    )
    org_name = _safe_str(ev1.international_org if ev1 else None)
    await evaluator.verify(
        claim=f"According to the answer, the organization was '{org_name}' and it suspended the country's membership on November 27, 2025. Sources should confirm ECOWAS suspension on that date.",
        node=org_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or does not match ECOWAS and the date, return Incorrect."
    )


async def verify_event_2(
    evaluator: Evaluator,
    parent: VerificationNode,
    ev2: Optional[Event2Details]
) -> None:
    event_node = evaluator.add_parallel(
        id="us_diplomatic_statement",
        desc="Event 2: U.S. Secretary of State congratulatory statement for a European NATO ally",
        parent=parent,
        critical=False
    )

    urls = _urls_or_empty(ev2.reference_urls if ev2 else [])

    # Credible reference URL (critical)
    cred_leaf = evaluator.add_leaf(
        id="credible_reference_url",
        desc="Provides at least one reference URL from a credible news source for Event 2",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is a credible news source or an official U.S. government site.",
        node=cred_leaf,
        sources=urls,
        additional_instruction="Accept reputable media or official .gov domains (e.g., state.gov). If none or non-credible, return Incorrect."
    )

    # Subject country (critical) – match expected
    subj_leaf = evaluator.add_leaf(
        id="subject_country",
        desc="Identifies Albania as the country",
        parent=event_node,
        critical=True
    )
    subj_country = _safe_str(ev2.country if ev2 else None)
    await evaluator.verify(
        claim=f"The answer identifies the subject country as '{subj_country}', and it matches 'Albania'.",
        node=subj_leaf,
        additional_instruction="If extracted is NULL or not 'Albania' (case-insensitive allowed), return Incorrect."
    )

    # Statement date (critical) – verify with sources and match expected
    date_leaf = evaluator.add_leaf(
        id="statement_date",
        desc="States the statement date as November 28, 2025",
        parent=event_node,
        critical=True
    )
    stmt_date = _safe_str(ev2.statement_date if ev2 else None)
    await evaluator.verify(
        claim=f"According to the answer, the statement date is '{stmt_date}'. Sources should confirm the statement was issued on November 28, 2025.",
        node=date_leaf,
        sources=urls,
        additional_instruction="Allow formatting variants. If extracted is NULL or not November 28, 2025, return Incorrect."
    )

    # Issuing official (critical)
    issu_leaf = evaluator.add_leaf(
        id="issuing_official",
        desc="Identifies Marco Rubio as the U.S. Secretary of State who issued the statement",
        parent=event_node,
        critical=True
    )
    issu_name = _safe_str(ev2.issuing_official_name if ev2 else None)
    issu_title = _safe_str(ev2.issuing_official_title if ev2 else None)
    await evaluator.verify(
        claim=f"According to the answer, the statement was issued by '{issu_name}' in the role '{issu_title}'. Sources should confirm it was issued by Marco Rubio as U.S. Secretary of State.",
        node=issu_leaf,
        sources=urls,
        additional_instruction="If extracted values are NULL or the source shows a different official/title, return Incorrect."
    )

    # Occasion (critical)
    occ_leaf = evaluator.add_leaf(
        id="occasion",
        desc="Identifies Albania's Independence Day as the occasion being commemorated",
        parent=event_node,
        critical=True
    )
    occ_text = _safe_str(ev2.occasion if ev2 else None)
    await evaluator.verify(
        claim=f"According to the answer, the occasion was '{occ_text}'. Sources should confirm it commemorates Albania's Independence Day.",
        node=occ_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or does not match, return Incorrect."
    )

    # Themes – NATO/mutual defense (critical)
    nato_leaf = evaluator.add_leaf(
        id="theme_nato_mutual_defense",
        desc="Mentions NATO / mutual defense as a key cooperation theme highlighted in the statement",
        parent=event_node,
        critical=True
    )
    nato_text = _safe_str(ev2.theme_nato_mutual_defense if ev2 else None)
    await evaluator.verify(
        claim=f"According to the answer, the statement highlights NATO/mutual defense cooperation: '{nato_text}'. The source should show this theme.",
        node=nato_leaf,
        sources=urls,
        additional_instruction="Look for mentions of NATO/alliance/mutual defense in the statement. If extracted is NULL or the theme isn't present, return Incorrect."
    )

    # Themes – cybersecurity (critical)
    cyber_leaf = evaluator.add_leaf(
        id="theme_cybersecurity",
        desc="Mentions cybersecurity cooperation as a key cooperation theme highlighted in the statement",
        parent=event_node,
        critical=True
    )
    cyber_text = _safe_str(ev2.theme_cybersecurity if ev2 else None)
    await evaluator.verify(
        claim=f"According to the answer, the statement highlights cybersecurity cooperation: '{cyber_text}'. The source should show this theme.",
        node=cyber_leaf,
        sources=urls,
        additional_instruction="Look for mentions of cybersecurity/cyber cooperation. If extracted is NULL or the theme isn't present, return Incorrect."
    )


async def verify_event_3(
    evaluator: Evaluator,
    parent: VerificationNode,
    ev3: Optional[Event3Details]
) -> None:
    event_node = evaluator.add_parallel(
        id="eastern_europe_resignation",
        desc="Event 3: Eastern European high-ranking official resignation after corruption scandal and home raid",
        parent=parent,
        critical=False
    )

    urls = _urls_or_empty(ev3.reference_urls if ev3 else [])

    # Credible reference URL (critical)
    cred_leaf = evaluator.add_leaf(
        id="credible_reference_url",
        desc="Provides at least one reference URL from a credible news source for Event 3",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is a credible news source or official site.",
        node=cred_leaf,
        sources=urls,
        additional_instruction="Accept reputable media or official domains. If no URLs or non-credible, return Incorrect."
    )

    # Country (critical) – match expected
    country_leaf = evaluator.add_leaf(
        id="country",
        desc="Identifies Ukraine as the country",
        parent=event_node,
        critical=True
    )
    e_country = _safe_str(ev3.country if ev3 else None)
    await evaluator.verify(
        claim=f"The answer identifies the country as '{e_country}', and it matches 'Ukraine'.",
        node=country_leaf,
        additional_instruction="If extracted is NULL or not 'Ukraine', return Incorrect."
    )

    # Official name (critical) – match expected
    name_leaf = evaluator.add_leaf(
        id="official_name",
        desc="Identifies Andriy Yermak as the official who resigned",
        parent=event_node,
        critical=True
    )
    e_name = _safe_str(ev3.official_name if ev3 else None)
    await evaluator.verify(
        claim=f"The answer identifies the official as '{e_name}', and it matches 'Andriy Yermak'.",
        node=name_leaf,
        additional_instruction="Allow minor variations. If extracted is NULL or not matching, return Incorrect."
    )

    # Position/title (critical) – match expected (chief of staff)
    pos_leaf = evaluator.add_leaf(
        id="position_title",
        desc="States the official's position/title as chief of staff",
        parent=event_node,
        critical=True
    )
    e_pos = _safe_str(ev3.position_title if ev3 else None)
    await evaluator.verify(
        claim=f"The answer states the position/title as '{e_pos}', and it matches 'chief of staff'.",
        node=pos_leaf,
        additional_instruction="Case-insensitive. If extracted is NULL or not matching, return Incorrect."
    )

    # Resignation date (critical) – verify with sources and match expected
    date_leaf = evaluator.add_leaf(
        id="resignation_date",
        desc="States the resignation date as November 28, 2025",
        parent=event_node,
        critical=True
    )
    e_date = _safe_str(ev3.resignation_date if ev3 else None)
    await evaluator.verify(
        claim=f"According to the answer, the resignation date is '{e_date}'. Sources should confirm it was November 28, 2025.",
        node=date_leaf,
        sources=urls,
        additional_instruction="Allow format variants. If extracted is NULL or not November 28, 2025, return Incorrect."
    )

    # Reason for resignation (critical) – corruption scandal
    reason_leaf = evaluator.add_leaf(
        id="reason_for_resignation",
        desc="Gives the reason for resignation as a corruption scandal",
        parent=event_node,
        critical=True
    )
    e_reason = _safe_str(ev3.reason_for_resignation if ev3 else None)
    await evaluator.verify(
        claim=f"According to the answer, the reason for resignation is '{e_reason}'. Sources should confirm it is due to a corruption scandal.",
        node=reason_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or does not match corruption scandal, return Incorrect."
    )

    # Home raid detail (critical)
    raid_leaf = evaluator.add_leaf(
        id="home_raid_detail",
        desc="Mentions that the official's home was raided",
        parent=event_node,
        critical=True
    )
    e_raid = _safe_str(ev3.home_raid_detail if ev3 else None)
    await evaluator.verify(
        claim=f"According to the answer, it mentions a home raid: '{e_raid}'. Sources should confirm the official's home was raided.",
        node=raid_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or the raid is not supported by sources, return Incorrect."
    )

    # Peace negotiations role (critical)
    peace_leaf = evaluator.add_leaf(
        id="peace_negotiations_role",
        desc="Mentions that the official had been leading the country's delegation in peace negotiations",
        parent=event_node,
        critical=True
    )
    e_peace = _safe_str(ev3.peace_negotiations_role if ev3 else None)
    await evaluator.verify(
        claim=f"According to the answer, it mentions the official had been leading the country's delegation in peace negotiations: '{e_peace}'. Sources should confirm this role.",
        node=peace_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or the role isn't supported, return Incorrect."
    )


async def verify_event_4(
    evaluator: Evaluator,
    parent: VerificationNode,
    ev4: Optional[Event4Details]
) -> None:
    event_node = evaluator.add_parallel(
        id="us_governor_race_announcement",
        desc="Event 4: U.S. gubernatorial race candidacy announcement",
        parent=parent,
        critical=False
    )

    urls = _urls_or_empty(ev4.reference_urls if ev4 else [])

    # Credible reference URL (critical)
    cred_leaf = evaluator.add_leaf(
        id="credible_reference_url",
        desc="Provides at least one reference URL from a credible news source for Event 4",
        parent=event_node,
        critical=True
    )
    await evaluator.verify(
        claim="At least one of the provided URLs is a credible news source or official site.",
        node=cred_leaf,
        sources=urls,
        additional_instruction="Accept reputable media or official domains. If no URLs or non-credible, return Incorrect."
    )

    # State (critical) – match expected
    state_leaf = evaluator.add_leaf(
        id="state",
        desc="Identifies Florida as the state",
        parent=event_node,
        critical=True
    )
    e_state = _safe_str(ev4.state if ev4 else None)
    await evaluator.verify(
        claim=f"The answer identifies the state as '{e_state}', and it matches 'Florida'.",
        node=state_leaf,
        additional_instruction="If extracted is NULL or not 'Florida', return Incorrect."
    )

    # Candidate full name (critical) – match expected
    name_leaf = evaluator.add_leaf(
        id="candidate_full_name",
        desc="Identifies James Fishback as the candidate",
        parent=event_node,
        critical=True
    )
    e_name = _safe_str(ev4.candidate_full_name if ev4 else None)
    await evaluator.verify(
        claim=f"The answer identifies the candidate as '{e_name}', and it matches 'James Fishback'.",
        node=name_leaf,
        additional_instruction="Allow minor variations. If extracted is NULL or not matching, return Incorrect."
    )

    # Announcement date (critical) – verify with sources and match expected
    ann_leaf = evaluator.add_leaf(
        id="announcement_date",
        desc="States the announcement date as November 24, 2025",
        parent=event_node,
        critical=True
    )
    e_ann = _safe_str(ev4.announcement_date if ev4 else None)
    await evaluator.verify(
        claim=f"According to the answer, the announcement date is '{e_ann}'. Sources should confirm it was November 24, 2025.",
        node=ann_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or not November 24, 2025, return Incorrect."
    )

    # Candidate age (critical)
    age_leaf = evaluator.add_leaf(
        id="candidate_age",
        desc="States the candidate's age as 30 years old",
        parent=event_node,
        critical=True
    )
    e_age = _safe_str(ev4.candidate_age if ev4 else None)
    await evaluator.verify(
        claim=f"According to the answer, the candidate's age is '{e_age}'. Sources should confirm the candidate is 30 years old.",
        node=age_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or not 30 (allow '30' or '30 years old'), return Incorrect."
    )

    # Professional background (critical)
    prof_leaf = evaluator.add_leaf(
        id="professional_background",
        desc="Describes the candidate as an investment firm CEO (professional background)",
        parent=event_node,
        critical=True
    )
    e_prof = _safe_str(ev4.professional_background if ev4 else None)
    await evaluator.verify(
        claim=f"According to the answer, the candidate's professional background is '{e_prof}'. Sources should confirm the candidate is an investment firm CEO.",
        node=prof_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or not matching investment firm CEO, return Incorrect."
    )

    # Election year (critical)
    year_leaf = evaluator.add_leaf(
        id="election_year",
        desc="States the election year as 2026",
        parent=event_node,
        critical=True
    )
    e_year = _safe_str(ev4.election_year if ev4 else None)
    await evaluator.verify(
        claim=f"According to the answer, the election year is '{e_year}'. Sources should confirm the race is for 2026.",
        node=year_leaf,
        sources=urls,
        additional_instruction="If extracted is NULL or not 2026, return Incorrect."
    )


# -------------------------- Main Evaluation --------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
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
        default_model=model
    )

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_briefing(),
        template_class=BriefingExtraction,
        extraction_name="briefing_extraction"
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({"event_1_expected": GT_EVENT_1}, gt_type="ground_truth_event_1")
    evaluator.add_ground_truth({"event_2_expected": GT_EVENT_2}, gt_type="ground_truth_event_2")
    evaluator.add_ground_truth({"event_3_expected": GT_EVENT_3}, gt_type="ground_truth_event_3")
    evaluator.add_ground_truth({"event_4_expected": GT_EVENT_4}, gt_type="ground_truth_event_4")

    # Build and verify each event subtree
    await verify_event_1(evaluator, root, extraction.west_africa_military_crisis)
    await verify_event_2(evaluator, root, extraction.us_diplomatic_statement)
    await verify_event_3(evaluator, root, extraction.eastern_europe_resignation)
    await verify_event_4(evaluator, root, extraction.us_governor_race_announcement)

    return evaluator.get_summary()