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
TASK_ID = "find_church_through_investigation_chain"
TASK_DESCRIPTION = """
Identify the Catholic church in Florida where a specific priest served during 1966-1967 by tracing through the following investigative journalism chain:

1. First, identify the investigation that won the 2025 Pulitzer Prize for Investigative Reporting (provide the investigation title and the news organization).

2. Then, identify the lead reporter or editor-in-charge who headed that Pulitzer Prize-winning investigation team.

3. Next, find the name of the newspaper where that lead reporter worked before joining Reuters.

4. At that previous newspaper, the reporter worked on a 2006 investigation about former U.S. Representative Mark Foley and a Catholic priest from Foley's childhood. Confirm this investigation exists.

5. Identify the full name of the Catholic priest who was the subject of that 2006 Mark Foley investigation.

6. Finally, identify the Catholic church in Florida where that priest served during 1966-1967 when Mark Foley was an altar boy there.

Provide: the church name, the city, and the state.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Step1PulitzerInfo(BaseModel):
    investigation_title: Optional[str] = None
    news_organization: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Step2LeadInfo(BaseModel):
    lead_name: Optional[str] = None
    role_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Step3PreviousNewspaperInfo(BaseModel):
    previous_newspaper_name: Optional[str] = None
    join_year_at_previous: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Step4Investigation2006Info(BaseModel):
    article_or_series_title: Optional[str] = None
    published_year: Optional[str] = None
    coauthors: List[str] = Field(default_factory=list)
    topic_summary: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Step5PriestInfo(BaseModel):
    priest_full_name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Step6ChurchInfo(BaseModel):
    church_name: Optional[str] = None
    church_city: Optional[str] = None
    church_state: Optional[str] = None
    service_years: Optional[str] = None
    mark_foley_altar_boy: Optional[bool] = None
    sources: List[str] = Field(default_factory=list)


class ChainExtraction(BaseModel):
    step1: Optional[Step1PulitzerInfo] = None
    step2: Optional[Step2LeadInfo] = None
    step3: Optional[Step3PreviousNewspaperInfo] = None
    step4: Optional[Step4Investigation2006Info] = None
    step5: Optional[Step5PriestInfo] = None
    step6: Optional[Step6ChurchInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_chain_info() -> str:
    return """
    Extract the information explicitly provided in the answer for the following investigative chain. Return null for any missing field. For each step, also extract all URLs explicitly cited in the answer that support that step (sources).

    STEP 1 (Pulitzer 2025 - Investigative Reporting winner):
    - investigation_title: Title/name of the Pulitzer-winning investigation (winner, not a finalist)
    - news_organization: The news organization credited with the award
    - sources: URLs in the answer that directly support the winner identification (e.g., Pulitzer site, official org post, credible coverage)

    STEP 2 (Lead reporter or editor-in-charge of that investigation):
    - lead_name: The full name of the lead or editor-in-charge who headed the team
    - role_description: How the role is described (e.g., "editor-in-charge", "led the team")
    - sources: URLs supporting that this person led/was editor-in-charge of the Pulitzer-winning investigation

    STEP 3 (Previous newspaper before Reuters + join year):
    - previous_newspaper_name: Name of the newspaper where the lead worked immediately before joining Reuters
    - join_year_at_previous: The year they joined that previous newspaper (if mentioned)
    - sources: URLs in the answer that support the employment timeline and joining year

    STEP 4 (2006 investigation at that previous newspaper about Mark Foley and a Catholic priest):
    - article_or_series_title: Title of the 2006 investigation article/series (if mentioned)
    - published_year: The publication year (should be 2006)
    - coauthors: List of any co-authors mentioned (include all names)
    - topic_summary: Brief summary from the answer of what/who the investigation was about
    - sources: URLs supporting the existence, timing (2006), topic, authorship at that previous newspaper

    STEP 5 (Priest identified in the 2006 investigation):
    - priest_full_name: Full name of the Catholic priest who was the subject of the 2006 investigation
    - sources: URLs supporting the priest identification and any interview/acknowledgment described

    STEP 6 (Florida church + 1966–1967 + altar boy):
    - church_name: Name of the Catholic church where the priest served during 1966-1967
    - church_city: City of that church
    - church_state: State of that church (should be Florida)
    - service_years: Text in the answer for the years the priest served there (e.g., "1966–1967")
    - mark_foley_altar_boy: true/false if the answer states Mark Foley was an altar boy at that church at that time
    - sources: URLs supporting the church identification, location, service years, and altar boy claim

    Notes:
    - Extract exactly what is in the answer. Do not invent or add anything not present.
    - Sources must be actual URLs present in the answer (including markdown links). If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_or_empty(lst: Optional[List[str]]) -> List[str]:
    return lst if isinstance(lst, list) else []


def _combine_sources(*args: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in args:
        for u in _list_or_empty(lst):
            if isinstance(u, str):
                if u not in seen:
                    seen.add(u)
                    combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_step1(
    evaluator: Evaluator,
    parent,
    s1: Optional[Step1PulitzerInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Step1_Identify_2025_Pulitzer_Winning_Investigation",
        desc="Identify the 2025 Pulitzer Prize for Investigative Reporting winning investigation (winner, not finalist), including title and news organization.",
        parent=parent,
        critical=False
    )

    title = (s1.investigation_title or "").strip() if s1 else ""
    org = (s1.news_organization or "").strip() if s1 else ""
    sources = _list_or_empty(s1.sources) if s1 else []

    # Pulitzer winner status (critical)
    leaf_win = evaluator.add_leaf(
        id="Pulitzer_Winner_Status_2025",
        desc="Verify the identified investigation is the WINNER (not merely a finalist) of the 2025 Pulitzer Prize for Investigative Reporting.",
        parent=node,
        critical=True
    )
    claim_win = (
        f"In 2025, the Pulitzer Prize for Investigative Reporting WINNER (not a finalist) "
        f"was the investigation titled '{title}' by {org}."
        if title and org else
        "In 2025, confirm which investigation is listed as the WINNER (not a finalist) for the Pulitzer Prize for Investigative Reporting."
    )
    await evaluator.verify(
        claim=claim_win,
        node=leaf_win,
        sources=sources,
        additional_instruction="Use official Pulitzer.org or equivalent authoritative coverage. The result must clearly indicate 'Winner' (not 'Finalist')."
    )

    # Investigation title (critical)
    leaf_title = evaluator.add_leaf(
        id="Investigation_Title",
        desc="Provide the title/name of the Pulitzer-winning investigation.",
        parent=node,
        critical=True
    )
    claim_title = (
        f"The Pulitzer-winning investigation's title is '{title}'."
        if title else
        "Identify and confirm the precise title/name of the Pulitzer-winning investigation."
    )
    await evaluator.verify(
        claim=claim_title,
        node=leaf_title,
        sources=sources,
        additional_instruction="Allow minor formatting variants. Confirm the page explicitly shows the investigation title associated with the 2025 Investigative Reporting winner."
    )

    # News organization (critical)
    leaf_org = evaluator.add_leaf(
        id="News_Organization",
        desc="Provide the news organization that produced/published the Pulitzer-winning investigation.",
        parent=node,
        critical=True
    )
    claim_org = (
        f"The news organization credited for the Pulitzer-winning investigation is {org}."
        if org else
        "Confirm the news organization credited with the Pulitzer-winning investigation."
    )
    await evaluator.verify(
        claim=claim_org,
        node=leaf_org,
        sources=sources,
        additional_instruction="Use the same authoritative sources; confirm the organization credited with the award."
    )


async def build_step2(
    evaluator: Evaluator,
    parent,
    s2: Optional[Step2LeadInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Step2_Identify_Lead_Reporter_Or_Editor_In_Charge",
        desc="Identify the lead reporter/editor-in-charge who headed the Pulitzer-winning investigation team.",
        parent=parent,
        critical=False
    )

    lead = (s2.lead_name or "").strip() if s2 else ""
    role_desc = (s2.role_description or "").strip() if s2 else ""
    sources = _list_or_empty(s2.sources) if s2 else []

    # Lead name (critical)
    leaf_name = evaluator.add_leaf(
        id="Lead_Reporter_Name",
        desc="Provide the full name of the lead reporter or editor-in-charge.",
        parent=node,
        critical=True
    )
    claim_name = (
        f"The lead (or editor-in-charge) of the Pulitzer-winning investigation team is {lead}."
        if lead else
        "Identify the person who served as lead or editor-in-charge of the Pulitzer-winning investigation team."
    )
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=sources,
        additional_instruction="The person must be clearly identified as having led or headed the winning investigation team."
    )

    # Lead role verification (critical)
    leaf_role = evaluator.add_leaf(
        id="Lead_Role_Verification",
        desc="Verify this person is identified as the editor-in-charge or lead of the Pulitzer-winning investigation team.",
        parent=node,
        critical=True
    )
    claim_role = (
        f"{lead} is explicitly identified as the editor-in-charge or lead of the Pulitzer-winning investigation team. {('Role detail: ' + role_desc) if role_desc else ''}".strip()
        if lead else
        "The identified person must be explicitly described as editor-in-charge or lead of the winning investigation team."
    )
    await evaluator.verify(
        claim=claim_role,
        node=leaf_role,
        sources=sources,
        additional_instruction="Look for language like 'led the team', 'editor-in-charge', 'head of the investigation'."
    )


async def build_step3(
    evaluator: Evaluator,
    parent,
    s2: Optional[Step2LeadInfo],
    s3: Optional[Step3PreviousNewspaperInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Step3_Identify_Previous_Newspaper_Before_Reuters",
        desc="Identify the newspaper where the lead reporter worked immediately before joining Reuters and verify join year.",
        parent=parent,
        critical=False
    )

    lead = (s2.lead_name or "").strip() if s2 else ""
    prev_paper = (s3.previous_newspaper_name or "").strip() if s3 else ""
    join_year_prev = (s3.join_year_at_previous or "").strip() if s3 else ""
    sources = _list_or_empty(s3.sources) if s3 else []

    # Previous newspaper name (critical)
    leaf_prev = evaluator.add_leaf(
        id="Previous_Newspaper_Name",
        desc="Provide the name of the newspaper where the lead reporter worked immediately before joining Reuters.",
        parent=node,
        critical=True
    )
    claim_prev = (
        f"Before joining Reuters, {lead} worked at the newspaper '{prev_paper}'."
        if lead and prev_paper else
        "Identify the newspaper where the lead reporter worked immediately prior to joining Reuters."
    )
    await evaluator.verify(
        claim=claim_prev,
        node=leaf_prev,
        sources=sources,
        additional_instruction="Confirm employment timeline indicates this was the job immediately prior to Reuters."
    )

    # Immediately before Reuters (critical)
    leaf_immediate = evaluator.add_leaf(
        id="Previous_Newspaper_Immediately_Before_Reuters",
        desc="Verify that this newspaper job was immediately before the reporter joined Reuters.",
        parent=node,
        critical=True
    )
    claim_immediate = (
        f"The job at '{prev_paper}' was immediately before {lead} joined Reuters."
        if prev_paper and lead else
        "Verify that the identified previous newspaper role was the position immediately before joining Reuters."
    )
    await evaluator.verify(
        claim=claim_immediate,
        node=leaf_immediate,
        sources=sources,
        additional_instruction="Look for phrasing like 'before joining Reuters', 'immediately prior to Reuters', or timeline listings showing no intervening employer."
    )

    # Joined previous newspaper in 2004 (critical)
    leaf_2004 = evaluator.add_leaf(
        id="Joined_Previous_Newspaper_In_2004",
        desc="Verify that the reporter joined the previous newspaper in 2004.",
        parent=node,
        critical=True
    )
    claim_2004 = (
        f"{lead} joined '{prev_paper}' in 2004."
        if prev_paper and lead else
        "Verify that the lead reporter joined the identified previous newspaper in 2004."
    )
    await evaluator.verify(
        claim=claim_2004,
        node=leaf_2004,
        sources=sources,
        additional_instruction="Confirm the year is 2004 on the employment history source. If the answer states another year, prefer authoritative bio/CV pages."
    )


async def build_step4(
    evaluator: Evaluator,
    parent,
    s2: Optional[Step2LeadInfo],
    s3: Optional[Step3PreviousNewspaperInfo],
    s4: Optional[Step4Investigation2006Info],
) -> None:
    node = evaluator.add_parallel(
        id="Step4_Verify_2006_Mark_Foley_Investigation_At_Previous_Newspaper",
        desc="Confirm the relevant 2006 investigation at the previous newspaper and its required properties.",
        parent=parent,
        critical=False
    )

    lead = (s2.lead_name or "").strip() if s2 else ""
    prev_paper = (s3.previous_newspaper_name or "").strip() if s3 else ""
    inv_title = (s4.article_or_series_title or "").strip() if s4 else ""
    pub_year = (s4.published_year or "").strip() if s4 else ""
    coauthors = [c.strip() for c in _list_or_empty(s4.coauthors)]
    sources = _list_or_empty(s4.sources) if s4 else []

    # Investigation exists (critical)
    leaf_exists = evaluator.add_leaf(
        id="Investigation_Exists",
        desc="Confirm the 2006 investigation exists at the previous newspaper.",
        parent=node,
        critical=True
    )
    claim_exists = (
        f"A 2006 investigation (article or series{f' titled {inv_title!r}' if inv_title else ''}) about Mark Foley and a Catholic priest was published by {prev_paper}."
        if prev_paper else
        "A 2006 investigation about Mark Foley and a Catholic priest was published by the identified previous newspaper."
    )
    await evaluator.verify(
        claim=claim_exists,
        node=leaf_exists,
        sources=sources,
        additional_instruction="The source should show the investigation is from the previous newspaper (not Reuters) and is a real published piece."
    )

    # Published in 2006 (critical)
    leaf_2006 = evaluator.add_leaf(
        id="Published_In_2006",
        desc="Verify the investigation was published in 2006.",
        parent=node,
        critical=True
    )
    claim_2006 = (
        f"The investigation was published in 2006{f' (reported year in answer: {pub_year}).' if pub_year else '.'}"
    )
    await evaluator.verify(
        claim=claim_2006,
        node=leaf_2006,
        sources=sources,
        additional_instruction="Look for the publication date/year on the article page or credible archive."
    )

    # Topic: Mark Foley + Catholic priest (critical)
    leaf_topic = evaluator.add_leaf(
        id="Topic_Is_Mark_Foley_And_Catholic_Priest",
        desc="Verify the investigation is specifically about former U.S. Representative Mark Foley and a Catholic priest.",
        parent=node,
        critical=True
    )
    claim_topic = "The investigation centers on former U.S. Representative Mark Foley and a Catholic priest from his childhood."
    await evaluator.verify(
        claim=claim_topic,
        node=leaf_topic,
        sources=sources,
        additional_instruction="Accept phrasing variants like 'series about Mark Foley and the priest who knew him as a boy'."
    )

    # Reporter worked on it (critical)
    leaf_reporter = evaluator.add_leaf(
        id="Reporter_Worked_On_Investigation",
        desc="Verify the identified lead reporter worked on this 2006 investigation at that previous newspaper.",
        parent=node,
        critical=True
    )
    claim_reporter = (
        f"{lead} worked on the 2006 investigation at {prev_paper}."
        if lead and prev_paper else
        "The identified lead reporter worked on the 2006 investigation at the previous newspaper."
    )
    await evaluator.verify(
        claim=claim_reporter,
        node=leaf_reporter,
        sources=sources,
        additional_instruction="Look for bylines or credits indicating this reporter authored/co-authored the investigation."
    )

    # Coauthored with Maurice Tamman (critical)
    leaf_tamman = evaluator.add_leaf(
        id="Coauthored_With_Maurice_Tamman",
        desc="Verify the investigation was co-authored with Maurice Tamman.",
        parent=node,
        critical=True
    )
    claim_tamman = "Maurice Tamman is credited as a co-author on the 2006 investigation."
    await evaluator.verify(
        claim=claim_tamman,
        node=leaf_tamman,
        sources=sources,
        additional_instruction="Explicitly check the byline/credits for 'Maurice Tamman'."
    )


async def build_step5(
    evaluator: Evaluator,
    parent,
    s4: Optional[Step4Investigation2006Info],
    s5: Optional[Step5PriestInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Step5_Identify_Priest_From_2006_Investigation",
        desc="Identify the priest referenced by the 2006 Mark Foley investigation and verify the priest meets the stated condition.",
        parent=parent,
        critical=False
    )

    priest = (s5.priest_full_name or "").strip() if s5 else ""
    sources = _combine_sources(_list_or_empty(s5.sources) if s5 else [], _list_or_empty(s4.sources) if s4 else [])

    # Priest full name (critical)
    leaf_priest_name = evaluator.add_leaf(
        id="Priest_Full_Name",
        desc="Provide the full name of the Catholic priest who was the subject of the 2006 Mark Foley investigation.",
        parent=node,
        critical=True
    )
    claim_priest_name = (
        f"The Catholic priest who was the subject of the 2006 Mark Foley investigation is {priest}."
        if priest else
        "Identify the full name of the Catholic priest who was the subject of the 2006 Mark Foley investigation."
    )
    await evaluator.verify(
        claim=claim_priest_name,
        node=leaf_priest_name,
        sources=sources,
        additional_instruction="Confirm that the investigation explicitly names the priest subject."
    )

    # Priest interviewed and acknowledged (critical)
    leaf_ack = evaluator.add_leaf(
        id="Priest_Interviewed_And_Acknowledged",
        desc="Verify the priest is the one who was interviewed and acknowledged in the 2006 investigation.",
        parent=node,
        critical=True
    )
    claim_ack = (
        f"In the 2006 investigation, {priest} was interviewed and acknowledged or discussed the matter in question."
        if priest else
        "The priest identified in the 2006 investigation was interviewed in the piece and acknowledged or discussed the matter."
    )
    await evaluator.verify(
        claim=claim_ack,
        node=leaf_ack,
        sources=sources,
        additional_instruction="Look for quotes or paraphrases indicating the priest was interviewed and acknowledged/admitted/discussed the facts."
    )


async def build_step6(
    evaluator: Evaluator,
    parent,
    s5: Optional[Step5PriestInfo],
    s6: Optional[Step6ChurchInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Step6_Identify_Florida_Church_And_Output_Required_Fields",
        desc="Identify the Florida Catholic church where the priest served in 1966–1967 and where Mark Foley was an altar boy during that service; provide church name, city, and state.",
        parent=parent,
        critical=False
    )

    priest = (s5.priest_full_name or "").strip() if s5 else ""
    church = (s6.church_name or "").strip() if s6 else ""
    city = (s6.church_city or "").strip() if s6 else ""
    state = (s6.church_state or "").strip() if s6 else ""
    years = (s6.service_years or "").strip() if s6 else ""
    sources = _list_or_empty(s6.sources) if s6 else []

    # Church name (critical)
    leaf_church_name = evaluator.add_leaf(
        id="Church_Name",
        desc="Provide the name of the Catholic church.",
        parent=node,
        critical=True
    )
    claim_church_name = (
        f"The church where {priest} served during 1966–1967 is named '{church}'."
        if church and priest else
        "Identify the name of the church where the priest served during 1966–1967."
    )
    await evaluator.verify(
        claim=claim_church_name,
        node=leaf_church_name,
        sources=sources,
        additional_instruction="Confirm the exact church name; minor variations in punctuation or 'St.' vs 'Saint' are acceptable."
    )

    # Church city (critical)
    leaf_city = evaluator.add_leaf(
        id="Church_City",
        desc="Provide the city where the church is located.",
        parent=node,
        critical=True
    )
    claim_city = (
        f"The church '{church}' is located in the city of {city}."
        if church and city else
        "Confirm the city in which the identified church is located."
    )
    await evaluator.verify(
        claim=claim_city,
        node=leaf_city,
        sources=sources,
        additional_instruction="Prefer official parish pages or credible news coverage listing the city."
    )

    # Church state is Florida (critical)
    leaf_state = evaluator.add_leaf(
        id="Church_State_Is_Florida",
        desc="Provide the state for the church and verify it is Florida.",
        parent=node,
        critical=True
    )
    claim_state = (
        f"The church '{church}' is located in the state of Florida."
        if church else
        "Confirm that the church is located in the state of Florida."
    )
    await evaluator.verify(
        claim=claim_state,
        node=leaf_state,
        sources=sources,
        additional_instruction="Accept 'FL' as Florida."
    )

    # Priest served 1966–1967 (critical)
    leaf_service = evaluator.add_leaf(
        id="Priest_Served_At_Church_1966_1967",
        desc="Verify the priest served at this church during 1966–1967.",
        parent=node,
        critical=True
    )
    claim_service = (
        f"{priest} served at '{church}' during 1966 and 1967{f' (as indicated by {years}).' if years else '.'}"
        if priest and church else
        "Verify that the priest served at the identified church during 1966–1967."
    )
    await evaluator.verify(
        claim=claim_service,
        node=leaf_service,
        sources=sources,
        additional_instruction="Explicitly check dates or pastoral assignments indicating the 1966–1967 period."
    )

    # Mark Foley altar boy at that church during service (critical)
    leaf_altar = evaluator.add_leaf(
        id="Mark_Foley_Altar_Boy_At_Church_During_Service",
        desc="Verify this is the church where Mark Foley was an altar boy during the priest's service there.",
        parent=node,
        critical=True
    )
    claim_altar = (
        f"Mark Foley was an altar boy at '{church}' during the period when {priest} served there."
        if church and priest else
        "Verify that Mark Foley was an altar boy at the identified church during the priest’s period of service."
    )
    await evaluator.verify(
        claim=claim_altar,
        node=leaf_altar,
        sources=sources,
        additional_instruction="Look for explicit mention tying Mark Foley as an altar boy to this specific church during the priest's tenure (1966–1967)."
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
    """
    Evaluate an answer for the investigative chain to identify the Florida church (name, city, state).
    """
    # Initialize evaluator with sequential aggregation at root to enforce the chain dependency.
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract all relevant structured info from the answer
    extracted: ChainExtraction = await evaluator.extract(
        prompt=prompt_extract_chain_info(),
        template_class=ChainExtraction,
        extraction_name="chain_extraction",
    )

    # Optionally record the final output fields for convenience
    if extracted and extracted.step6:
        evaluator.add_custom_info(
            info={
                "church_name": extracted.step6.church_name,
                "church_city": extracted.step6.church_city,
                "church_state": extracted.step6.church_state,
                "service_years": extracted.step6.service_years,
            },
            info_type="final_output",
            info_name="final_church_fields"
        )

    # Build verification tree following rubric (order matters due to sequential root)
    await build_step1(evaluator, root, extracted.step1)
    await build_step2(evaluator, root, extracted.step2)
    await build_step3(evaluator, root, extracted.step2, extracted.step3)
    await build_step4(evaluator, root, extracted.step2, extracted.step3, extracted.step4)
    await build_step5(evaluator, root, extracted.step4, extracted.step5)
    await build_step6(evaluator, root, extracted.step5, extracted.step6)

    # Return summary with verification tree and scores
    return evaluator.get_summary()