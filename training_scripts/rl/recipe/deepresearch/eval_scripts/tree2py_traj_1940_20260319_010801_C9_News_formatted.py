import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ven_crisis_2026_idents"
TASK_DESCRIPTION = """
In the context of the 2026 Venezuela crisis and related U.S. political developments, identify four individuals who each meet the following specific criteria:

Individual A: A Venezuelan military officer who was appointed as Defense Minister on March 18, 2026, replacing Vladimir Padrino Lopez. This person was born in November 1960, graduated from a military academy in 1982, and previously served as director of SEBIN (the Bolivarian National Intelligence Service).

Individual B: The person who became Venezuela's interim president in early January 2026 (specifically on January 5, 2026) following the capture and removal of Nicolas Maduro. This person had served as vice president since 2018 and met with CIA Director John Ratcliffe in Caracas in mid-January 2026 (specifically on January 16, 2026).

Individual C: The U.S. Senator from New Hampshire who serves as Ranking Member of the Senate Foreign Relations Committee. This person made public statements in January 2026 condemning or expressing concern about the U.S. military action in Venezuela through official committee channels, and participated in a Senate Foreign Relations Committee hearing on U.S. Policy Towards Venezuela on January 28, 2026.

Individual D: A Republican political candidate who competed in the Texas 31st Congressional District primary election on March 3, 2026. This person finished in second place with approximately 10-11% of the vote (while incumbent John Carter won the primary), and had been barred from attending Williamson County GOP events in January 2026 following an altercation or incident with incumbent John Carter.

For each individual, provide their full name and include reference URLs that verify the stated criteria.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IndividualA(BaseModel):
    name: Optional[str] = None
    # Appointment
    appointment_date: Optional[str] = None  # e.g., "March 18, 2026"
    appointment_position_title: Optional[str] = None  # e.g., "Defense Minister" / "Minister of Defense"
    appointment_replaced: Optional[str] = None  # e.g., "Vladimir Padrino Lopez"
    appointment_urls: List[str] = Field(default_factory=list)
    # Biographical
    birth_month: Optional[str] = None  # e.g., "November"
    birth_year: Optional[str] = None  # e.g., "1960"
    military_academy_graduation_year: Optional[str] = None  # e.g., "1982"
    bio_urls: List[str] = Field(default_factory=list)
    # Previous role
    sebin_director_role_text: Optional[str] = None
    role_urls: List[str] = Field(default_factory=list)


class IndividualB(BaseModel):
    name: Optional[str] = None
    # Presidential transition
    interim_assumption_date: Optional[str] = None  # e.g., "January 5, 2026"
    transition_context: Optional[str] = None  # text mentioning capture/removal of Maduro
    transition_urls: List[str] = Field(default_factory=list)
    # Previous position
    vice_president_since: Optional[str] = None  # e.g., "2018"
    previous_position_urls: List[str] = Field(default_factory=list)
    # CIA meeting
    cia_meeting_date: Optional[str] = None  # e.g., "January 16, 2026"
    cia_meeting_location: Optional[str] = None  # e.g., "Caracas"
    cia_meeting_urls: List[str] = Field(default_factory=list)


class IndividualC(BaseModel):
    name: Optional[str] = None
    # Senate position
    is_senator_text: Optional[str] = None
    state: Optional[str] = None  # "New Hampshire"
    committee_role: Optional[str] = None  # "Ranking Member" of SFRC
    position_urls: List[str] = Field(default_factory=list)
    # Venezuela statements
    statement_urls: List[str] = Field(default_factory=list)
    # Committee hearing
    hearing_urls: List[str] = Field(default_factory=list)


class IndividualD(BaseModel):
    name: Optional[str] = None
    # Primary election participation
    district: Optional[str] = None  # "Texas 31st Congressional District"
    primary_date: Optional[str] = None  # e.g., "March 3, 2026"
    party: Optional[str] = None  # "Republican"
    election_urls: List[str] = Field(default_factory=list)
    # Results
    placement: Optional[str] = None  # "second"
    vote_percentage: Optional[str] = None  # "10-11%"
    incumbent: Optional[str] = None  # "John Carter"
    results_urls: List[str] = Field(default_factory=list)
    # Incident and party ban
    incident_ban_text: Optional[str] = None
    incident_with: Optional[str] = None  # "John Carter"
    incident_date: Optional[str] = None  # e.g., "January 2026"
    incident_urls: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    individual_a: Optional[IndividualA] = None
    individual_b: Optional[IndividualB] = None
    individual_c: Optional[IndividualC] = None
    individual_d: Optional[IndividualD] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_task_data() -> str:
    return """
    Extract from the provided answer the four requested individuals (A–D) and the specific verification details and URLs for each criterion. Return JSON with the exact schema below. Follow these strict rules:
    - Do not invent or infer any information or URLs that are not explicitly present in the answer text.
    - Include only URLs explicitly mentioned in the answer (plain URLs or markdown links). If a category has no URLs in the answer, return an empty list for that field.
    - Keep dates and titles as they appear in the answer text.
    - For all name and role/title fields, return them as strings exactly as stated in the answer.

    JSON schema to output:

    {
      "individual_a": {
        "name": null or string,
        "appointment_date": null or string,
        "appointment_position_title": null or string,
        "appointment_replaced": null or string,
        "appointment_urls": [array of strings],
        "birth_month": null or string,
        "birth_year": null or string,
        "military_academy_graduation_year": null or string,
        "bio_urls": [array of strings],
        "sebin_director_role_text": null or string,
        "role_urls": [array of strings]
      },
      "individual_b": {
        "name": null or string,
        "interim_assumption_date": null or string,
        "transition_context": null or string,
        "transition_urls": [array of strings],
        "vice_president_since": null or string,
        "previous_position_urls": [array of strings],
        "cia_meeting_date": null or string,
        "cia_meeting_location": null or string,
        "cia_meeting_urls": [array of strings]
      },
      "individual_c": {
        "name": null or string,
        "is_senator_text": null or string,
        "state": null or string,
        "committee_role": null or string,
        "position_urls": [array of strings],
        "statement_urls": [array of strings],
        "hearing_urls": [array of strings]
      },
      "individual_d": {
        "name": null or string,
        "district": null or string,
        "primary_date": null or string,
        "party": null or string,
        "election_urls": [array of strings],
        "placement": null or string,
        "vote_percentage": null or string,
        "incumbent": null or string,
        "results_urls": [array of strings],
        "incident_ban_text": null or string,
        "incident_with": null or string,
        "incident_date": null or string,
        "incident_urls": [array of strings]
      }
    }

    Notes:
    - appointment_urls should contain URLs that substantiate the appointment (date/position/replacement).
    - bio_urls should substantiate birth month/year and military academy graduation year.
    - role_urls should substantiate SEBIN directorship.
    - transition_urls should substantiate the interim presidency assumption and date/context.
    - previous_position_urls should substantiate vice presidency and since-2018 detail.
    - cia_meeting_urls should substantiate the CIA Director meeting, location (Caracas), and date.
    - position_urls should substantiate Senate status, state, and committee role (Ranking Member).
    - statement_urls should substantiate Venezuela military action statements in Jan 2026 via official SFRC channels.
    - hearing_urls should substantiate participation in the Jan 28, 2026 SFRC hearing on U.S. Policy Towards Venezuela.
    - election_urls should substantiate participation in the TX-31 Republican primary on Mar 3, 2026.
    - results_urls should substantiate finishing second, ~10-11%, and that John Carter won.
    - incident_urls should substantiate the county GOP ban, altercation/incident with John Carter, and that it occurred in Jan 2026.
    """.strip()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Deduplicate and strip empties
    seen = set()
    out = []
    for u in urls:
        if isinstance(u, str):
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _safe_name(name: Optional[str], fallback: str = "the identified individual") -> str:
    return name.strip() if name else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_individual_a(evaluator: Evaluator, parent_node, a: Optional[IndividualA]) -> None:
    a = a or IndividualA()
    name = _safe_name(a.name, "the identified officer")
    a_node = evaluator.add_parallel(
        id="individual_a",
        desc="Correctly identify the Venezuelan military officer appointed as Defense Minister on March 18, 2026",
        parent=parent_node,
        critical=False
    )

    # Appointment details
    appt_node = evaluator.add_parallel(
        id="a_appointment_details",
        desc="Verify the appointment to Defense Minister on March 18, 2026",
        parent=a_node,
        critical=True
    )
    appt_urls = _sanitize_urls(a.appointment_urls)

    evaluator.add_custom_node(
        result=len(appt_urls) > 0,
        id="a_appointment_url",
        desc="Provide a URL reference confirming the appointment details",
        parent=appt_node,
        critical=True
    )

    appt_date_leaf = evaluator.add_leaf(
        id="a_appointment_date",
        desc="The appointment occurred on March 18, 2026",
        parent=appt_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was appointed as Defense Minister (Minister of Defense) of Venezuela on March 18, 2026.",
        node=appt_date_leaf,
        sources=appt_urls,
        additional_instruction="Accept 'Minister of Defense' and 'Defense Minister' as equivalent; Spanish 'Ministro de la Defensa' is acceptable. Verify explicit date March 18, 2026."
    )

    position_leaf = evaluator.add_leaf(
        id="a_position_title",
        desc="The position is Defense Minister (or Minister of Defense)",
        parent=appt_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In this referenced appointment, {name} was named Venezuela's Defense Minister (Minister of Defense).",
        node=position_leaf,
        sources=appt_urls,
        additional_instruction="Confirm that the appointment was to the post of Defense Minister / Minister of Defense."
    )

    replaced_leaf = evaluator.add_leaf(
        id="a_replacement_context",
        desc="This person replaced Vladimir Padrino Lopez",
        parent=appt_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The appointment of {name} replaced Vladimir Padrino Lopez as Defense Minister.",
        node=replaced_leaf,
        sources=appt_urls,
        additional_instruction="Allow 'López' vs 'Lopez' spelling variations."
    )

    # Biographical details
    bio_node = evaluator.add_parallel(
        id="a_biographical_details",
        desc="Verify biographical information matches the criteria",
        parent=a_node,
        critical=True
    )
    bio_urls = _sanitize_urls(a.bio_urls)

    evaluator.add_custom_node(
        result=len(bio_urls) > 0,
        id="a_biographical_url",
        desc="Provide a URL reference confirming biographical details",
        parent=bio_node,
        critical=True
    )

    birth_year_leaf = evaluator.add_leaf(
        id="a_birth_year",
        desc="Born in 1960",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was born in 1960.",
        node=birth_year_leaf,
        sources=bio_urls,
        additional_instruction="The page can show a full birthdate that includes the year 1960; that suffices."
    )

    birth_month_leaf = evaluator.add_leaf(
        id="a_birth_month",
        desc="Born in November",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was born in November.",
        node=birth_month_leaf,
        sources=bio_urls,
        additional_instruction="Accept reasonable month spelling variants and Spanish month names."
    )

    grad_leaf = evaluator.add_leaf(
        id="a_military_academy_graduation",
        desc="Graduated from military academy in 1982",
        parent=bio_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} graduated from a military academy in 1982.",
        node=grad_leaf,
        sources=bio_urls,
        additional_instruction="Look for graduation info such as 'Escuela Militar' or 'Academia Militar' with the year 1982."
    )

    # Previous role
    role_node = evaluator.add_parallel(
        id="a_previous_role",
        desc="Verify previous role as SEBIN director",
        parent=a_node,
        critical=True
    )
    role_urls = _sanitize_urls(a.role_urls)

    evaluator.add_custom_node(
        result=len(role_urls) > 0,
        id="a_role_url",
        desc="Provide a URL reference confirming previous role",
        parent=role_node,
        critical=True
    )

    sebin_leaf = evaluator.add_leaf(
        id="a_sebin_director",
        desc="Previously served as director of SEBIN (Bolivarian National Intelligence Service)",
        parent=role_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} previously served as director of SEBIN (Bolivarian National Intelligence Service).",
        node=sebin_leaf,
        sources=role_urls,
        additional_instruction="Accept Spanish references to 'SEBIN' and 'Servicio Bolivariano de Inteligencia Nacional'."
    )


async def verify_individual_b(evaluator: Evaluator, parent_node, b: Optional[IndividualB]) -> None:
    b = b or IndividualB()
    name = _safe_name(b.name, "the identified interim leader")
    b_node = evaluator.add_parallel(
        id="individual_b",
        desc="Correctly identify Venezuela's interim president who assumed office in January 2026",
        parent=parent_node,
        critical=False
    )

    # Presidential transition
    trans_node = evaluator.add_parallel(
        id="b_presidential_transition",
        desc="Verify assumption of interim presidency in January 2026",
        parent=b_node,
        critical=True
    )
    trans_urls = _sanitize_urls(b.transition_urls)

    evaluator.add_custom_node(
        result=len(trans_urls) > 0,
        id="b_transition_url",
        desc="Provide a URL reference confirming the presidential transition",
        parent=trans_node,
        critical=True
    )

    interim_leaf = evaluator.add_leaf(
        id="b_interim_president_status",
        desc="Became interim or acting president of Venezuela",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} became interim (acting) president of Venezuela.",
        node=interim_leaf,
        sources=trans_urls,
        additional_instruction="Allow synonyms 'acting president' or 'interim president'."
    )

    timing_leaf = evaluator.add_leaf(
        id="b_transition_timing",
        desc="Assumed office in early January 2026 (specifically January 5, 2026)",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} assumed the interim presidency on January 5, 2026.",
        node=timing_leaf,
        sources=trans_urls,
        additional_instruction="Confirm explicit date January 5, 2026; minor timezone phrasing acceptable if clearly Jan 5 local."
    )

    context_leaf = evaluator.add_leaf(
        id="b_transition_context",
        desc="Assumed office following the capture and removal of Nicolas Maduro",
        parent=trans_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} assumed office following the capture and removal of Nicolás Maduro.",
        node=context_leaf,
        sources=trans_urls,
        additional_instruction="Allow 'Lopez'/'López' variants and Spanish phrasing indicating capture/removal."
    )

    # Previous position
    prevpos_node = evaluator.add_parallel(
        id="b_previous_position",
        desc="Verify previous role as vice president",
        parent=b_node,
        critical=True
    )
    prevpos_urls = _sanitize_urls(b.previous_position_urls)

    evaluator.add_custom_node(
        result=len(prevpos_urls) > 0,
        id="b_previous_position_url",
        desc="Provide a URL reference confirming previous position",
        parent=prevpos_node,
        critical=True
    )

    vp_role_leaf = evaluator.add_leaf(
        id="b_vice_president_role",
        desc="Served as vice president under Maduro",
        parent=prevpos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} served as vice president under Nicolás Maduro.",
        node=vp_role_leaf,
        sources=prevpos_urls,
        additional_instruction="Spanish-language sources acceptable."
    )

    vp_since_leaf = evaluator.add_leaf(
        id="b_vice_president_since",
        desc="Held vice president position since 2018",
        parent=prevpos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} has served as vice president since 2018.",
        node=vp_since_leaf,
        sources=prevpos_urls,
        additional_instruction="The page may show a start year 2018; that suffices."
    )

    # CIA meeting
    cia_node = evaluator.add_parallel(
        id="b_cia_meeting",
        desc="Verify meeting with CIA Director in January 2026",
        parent=b_node,
        critical=True
    )
    cia_urls = _sanitize_urls(b.cia_meeting_urls)

    evaluator.add_custom_node(
        result=len(cia_urls) > 0,
        id="b_meeting_url",
        desc="Provide a URL reference confirming the CIA Director meeting",
        parent=cia_node,
        critical=True
    )

    met_leaf = evaluator.add_leaf(
        id="b_meeting_occurred",
        desc="Met with CIA Director John Ratcliffe",
        parent=cia_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} met with CIA Director John Ratcliffe.",
        node=met_leaf,
        sources=cia_urls,
        additional_instruction="Confirm that the meeting was with the CIA Director named John Ratcliffe."
    )

    loc_leaf = evaluator.add_leaf(
        id="b_meeting_location",
        desc="Meeting took place in Caracas",
        parent=cia_node,
        critical=True
    )
    await evaluator.verify(
        claim="The meeting took place in Caracas.",
        node=loc_leaf,
        sources=cia_urls,
        additional_instruction="Check the location is Caracas, Venezuela."
    )

    date_leaf = evaluator.add_leaf(
        id="b_meeting_timing",
        desc="Meeting occurred in mid-January 2026 (specifically January 16, 2026)",
        parent=cia_node,
        critical=True
    )
    await evaluator.verify(
        claim="The meeting occurred on January 16, 2026.",
        node=date_leaf,
        sources=cia_urls,
        additional_instruction="Explicit date Jan 16, 2026 should appear or be clearly indicated."
    )


async def verify_individual_c(evaluator: Evaluator, parent_node, c: Optional[IndividualC]) -> None:
    c = c or IndividualC()
    name = _safe_name(c.name, "the identified U.S. Senator")
    c_node = evaluator.add_parallel(
        id="individual_c",
        desc="Correctly identify the U.S. Senator who is Ranking Member of Senate Foreign Relations Committee",
        parent=parent_node,
        critical=False
    )

    # Senate position
    pos_node = evaluator.add_parallel(
        id="c_senate_position",
        desc="Verify Senate position and committee role",
        parent=c_node,
        critical=True
    )
    pos_urls = _sanitize_urls(c.position_urls)

    evaluator.add_custom_node(
        result=len(pos_urls) > 0,
        id="c_position_url",
        desc="Provide a URL reference confirming Senate position and committee role",
        parent=pos_node,
        critical=True
    )

    senator_leaf = evaluator.add_leaf(
        id="c_senator_status",
        desc="Serves as a U.S. Senator",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} serves as a United States Senator.",
        node=senator_leaf,
        sources=pos_urls,
        additional_instruction="Official Senate or committee pages preferred but any credible source acceptable."
    )

    state_leaf = evaluator.add_leaf(
        id="c_state_representation",
        desc="Represents New Hampshire",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} represents New Hampshire in the U.S. Senate.",
        node=state_leaf,
        sources=pos_urls,
        additional_instruction="Confirm the state is New Hampshire."
    )

    role_leaf = evaluator.add_leaf(
        id="c_committee_role",
        desc="Holds the position of Ranking Member on the Senate Foreign Relations Committee",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} is the Ranking Member of the Senate Foreign Relations Committee.",
        node=role_leaf,
        sources=pos_urls,
        additional_instruction="Allow variants such as 'Republican leader' or 'Ranking Member'."
    )

    # Venezuela statements
    stmt_node = evaluator.add_parallel(
        id="c_venezuela_statements",
        desc="Verify public statements regarding Venezuela military action",
        parent=c_node,
        critical=True
    )
    stmt_urls = _sanitize_urls(c.statement_urls)

    evaluator.add_custom_node(
        result=len(stmt_urls) > 0,
        id="c_statement_url",
        desc="Provide a URL reference confirming the Venezuela-related statements",
        parent=stmt_node,
        critical=True
    )

    stmt_exist_leaf = evaluator.add_leaf(
        id="c_statement_existence",
        desc="Made public statements condemning or expressing concern about the military action in Venezuela",
        parent=stmt_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In January 2026, {name} made public statements condemning or expressing concern about U.S. military action in Venezuela.",
        node=stmt_exist_leaf,
        sources=stmt_urls,
        additional_instruction="Focus on official committee channels such as SFRC website, press releases, or official committee social posts."
    )

    stmt_timing_leaf = evaluator.add_leaf(
        id="c_statement_timing",
        desc="Statements made in January 2026",
        parent=stmt_node,
        critical=True
    )
    await evaluator.verify(
        claim="The statements were made in January 2026.",
        node=stmt_timing_leaf,
        sources=stmt_urls,
        additional_instruction="Confirm the date/month is within January 2026."
    )

    official_leaf = evaluator.add_leaf(
        id="c_official_capacity",
        desc="Statements made through official Senate Foreign Relations Committee channels",
        parent=stmt_node,
        critical=True
    )
    await evaluator.verify(
        claim="These statements were issued through official Senate Foreign Relations Committee channels.",
        node=official_leaf,
        sources=stmt_urls,
        additional_instruction="Accept SFRC official website pages, official SFRC social accounts, or official committee press releases."
    )

    # Committee hearing
    hearing_node = evaluator.add_parallel(
        id="c_committee_hearing",
        desc="Verify participation in January 28, 2026 hearing on Venezuela",
        parent=c_node,
        critical=True
    )
    hearing_urls = _sanitize_urls(c.hearing_urls)

    evaluator.add_custom_node(
        result=len(hearing_urls) > 0,
        id="c_hearing_url",
        desc="Provide a URL reference confirming hearing participation",
        parent=hearing_node,
        critical=True
    )

    hearing_part_leaf = evaluator.add_leaf(
        id="c_hearing_participation",
        desc="Participated in Senate Foreign Relations Committee hearing on U.S. Policy Towards Venezuela",
        parent=hearing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} participated in a Senate Foreign Relations Committee hearing on 'U.S. Policy Towards Venezuela.'",
        node=hearing_part_leaf,
        sources=hearing_urls,
        additional_instruction="Look for agendas, notices, video, or transcripts showing participation."
    )

    hearing_date_leaf = evaluator.add_leaf(
        id="c_hearing_date",
        desc="Hearing occurred on January 28, 2026",
        parent=hearing_node,
        critical=True
    )
    await evaluator.verify(
        claim="The hearing occurred on January 28, 2026.",
        node=hearing_date_leaf,
        sources=hearing_urls,
        additional_instruction="Confirm explicit date Jan 28, 2026."
    )


async def verify_individual_d(evaluator: Evaluator, parent_node, d: Optional[IndividualD]) -> None:
    d = d or IndividualD()
    name = _safe_name(d.name, "the identified candidate")
    d_node = evaluator.add_parallel(
        id="individual_d",
        desc="Correctly identify the Republican candidate in Texas 31st District primary",
        parent=parent_node,
        critical=False
    )

    # Primary election participation
    primary_node = evaluator.add_parallel(
        id="d_primary_election",
        desc="Verify participation and results in Texas 31st District Republican primary",
        parent=d_node,
        critical=True
    )
    election_urls = _sanitize_urls(d.election_urls)

    evaluator.add_custom_node(
        result=len(election_urls) > 0,
        id="d_election_url",
        desc="Provide a URL reference confirming participation in the primary",
        parent=primary_node,
        critical=True
    )

    district_leaf = evaluator.add_leaf(
        id="d_district",
        desc="Competed in Texas 31st Congressional District",
        parent=primary_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} competed in the Republican primary for Texas's 31st Congressional District.",
        node=district_leaf,
        sources=election_urls,
        additional_instruction="Look for ballot or candidate listing pages that show TX-31 and party."
    )

    primary_date_leaf = evaluator.add_leaf(
        id="d_election_date",
        desc="Primary election held on March 3, 2026",
        parent=primary_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Texas 31st Congressional District Republican primary was held on March 3, 2026.",
        node=primary_date_leaf,
        sources=election_urls,
        additional_instruction="Confirm the election date."
    )

    party_leaf = evaluator.add_leaf(
        id="d_party_affiliation",
        desc="Ran as a Republican candidate",
        parent=primary_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} ran as a Republican.",
        node=party_leaf,
        sources=election_urls,
        additional_instruction="Confirm party affiliation is Republican."
    )

    # Election results
    results_node = evaluator.add_parallel(
        id="d_election_results",
        desc="Verify second-place finish with approximately 10-11% of vote",
        parent=d_node,
        critical=True
    )
    results_urls = _sanitize_urls(d.results_urls)

    evaluator.add_custom_node(
        result=len(results_urls) > 0,
        id="d_results_url",
        desc="Provide a URL reference confirming election results",
        parent=results_node,
        critical=True
    )

    placement_leaf = evaluator.add_leaf(
        id="d_placement",
        desc="Finished in second place",
        parent=results_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} finished in second place in the primary.",
        node=placement_leaf,
        sources=results_urls,
        additional_instruction="Confirm explicit second-place finish."
    )

    percent_leaf = evaluator.add_leaf(
        id="d_vote_percentage",
        desc="Received approximately 10-11% of the vote",
        parent=results_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} received approximately 10–11% of the vote.",
        node=percent_leaf,
        sources=results_urls,
        additional_instruction="Allow minor rounding differences within ~±1%."
    )

    incumbent_leaf = evaluator.add_leaf(
        id="d_incumbent_opponent",
        desc="Competed against incumbent John Carter who won the primary",
        parent=results_node,
        critical=True
    )
    await evaluator.verify(
        claim="Incumbent John Carter won the primary.",
        node=incumbent_leaf,
        sources=results_urls,
        additional_instruction="Confirm that John Carter is incumbent and winner."
    )

    # Party incident and ban
    incident_node = evaluator.add_parallel(
        id="d_party_incident",
        desc="Verify being barred from county party events following incident",
        parent=d_node,
        critical=True
    )
    incident_urls = _sanitize_urls(d.incident_urls)

    evaluator.add_custom_node(
        result=len(incident_urls) > 0,
        id="d_incident_url",
        desc="Provide a URL reference confirming the party incident and ban",
        parent=incident_node,
        critical=True
    )

    barred_leaf = evaluator.add_leaf(
        id="d_barred_from_events",
        desc="Was barred or banned from attending Williamson County GOP events",
        parent=incident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name} was barred or banned from attending Williamson County GOP events.",
        node=barred_leaf,
        sources=incident_urls,
        additional_instruction="Look for reports or party statements indicating a ban/bar from Williamson County GOP events."
    )

    altercation_leaf = evaluator.add_leaf(
        id="d_incident_with_incumbent",
        desc="The ban followed an altercation or incident with incumbent John Carter",
        parent=incident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The ban followed an altercation or incident between {name} and incumbent John Carter.",
        node=altercation_leaf,
        sources=incident_urls,
        additional_instruction="The page should state or clearly imply the incident/altercation with John Carter precipitated the ban."
    )

    incident_time_leaf = evaluator.add_leaf(
        id="d_incident_timing",
        desc="The incident occurred in January 2026",
        parent=incident_node,
        critical=True
    )
    await evaluator.verify(
        claim="The incident occurred in January 2026.",
        node=incident_time_leaf,
        sources=incident_urls,
        additional_instruction="Confirm month and year (January 2026) even if exact day is not specified."
    )


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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root set to non-critical to allow partial credit across individuals)
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_task_data(),
        template_class=TaskExtraction,
        extraction_name="extracted_individuals",
    )

    # Build and verify tree according to rubric
    await verify_individual_a(evaluator, root, extraction.individual_a)
    await verify_individual_b(evaluator, root, extraction.individual_b)
    await verify_individual_c(evaluator, root, extraction.individual_c)
    await verify_individual_d(evaluator, root, extraction.individual_d)

    # Return standard summary
    return evaluator.get_summary()