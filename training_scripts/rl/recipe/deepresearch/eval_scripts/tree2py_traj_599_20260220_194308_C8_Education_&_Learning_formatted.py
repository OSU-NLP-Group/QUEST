import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_leaders_5"
TASK_DESCRIPTION = (
    "Identify five education leaders currently serving in the United States who meet the specified career "
    "trajectory and achievement criteria. For each leader, provide their full name, current position title, "
    "and current institution/district name, along with reference URLs that verify the key criteria.\n\n"
    "Leader 1 - University President: Identify the individual who worked at McKinsey & Company for exactly 26 years "
    "before transitioning to higher education, served as a business school dean, was appointed as a university president "
    "in December 2025, took office on January 1, 2026, and currently serves as president of the University of Virginia.\n\n"
    "Leader 2 - Oregon Superintendent: Identify the individual who holds a master's degree in Education Policy from "
    "Harvard Graduate School of Education, started their current superintendent position on July 1, 2023, and serves "
    "as superintendent of the second-largest school district in Oregon.\n\n"
    "Leader 3 - Texas Superintendent: Identify the individual who has over 30 years of experience in public education, "
    "was named Region 10 Superintendent of the Year in 2020, was officially appointed to their current superintendent "
    "position in January 2024, previously worked in their current district for 6 years as an executive principal and "
    "assistant superintendent before becoming superintendent, and served as superintendent in at least two other Texas "
    "school districts before their current position.\n\n"
    "Leader 4 - College Football Coach (FCS to FBS): Identify the individual who led their FCS team to playoff "
    "appearances in three consecutive years (2022, 2023, and 2024), whose team reached the FCS playoff quarterfinals "
    "in both 2023 and 2024, and who was hired as a head coach at an FBS program on December 14, 2024.\n\n"
    "Leader 5 - College Football Coach (Ivy League): Identify the individual who graduated from Princeton University "
    "in 2006, was named as a head football coach in February 2024, came to their current position directly from "
    "Rutgers University, led their team to a shared Ivy League championship in their first season (2024), and previously "
    "served as associate head coach at Princeton University."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class LeaderEntry(BaseModel):
    name: Optional[str] = None
    current_title: Optional[str] = None
    current_org: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class LeadersExtraction(BaseModel):
    leader_1: Optional[LeaderEntry] = None
    leader_2: Optional[LeaderEntry] = None
    leader_3: Optional[LeaderEntry] = None
    leader_4: Optional[LeaderEntry] = None
    leader_5: Optional[LeaderEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_leaders() -> str:
    return """
Extract five leader profiles from the provided answer text, mapped to the five categories described in the task (Leader 1 through Leader 5). For each leader, extract:
- name: The full name of the leader.
- current_title: The leader's current position title (e.g., President, Superintendent, Head Football Coach).
- current_org: The current institution or district name (e.g., University of Virginia, [District Name], or [University/College]).
- reference_urls: A list of all URLs (including those embedded as Markdown links) that the answer cites as evidence for this leader.

Important instructions:
1) Only extract information explicitly stated in the answer. Do not infer or invent.
2) For URLs: extract the actual URL targets (from Markdown links or plain URLs). Include all relevant URLs.
3) Use null for missing name/current_title/current_org. Use an empty list for missing URLs.
4) Map the information to:
   - leader_1: University President (McKinsey 26 years → B-school dean → appointed Dec 2025 → took office Jan 1, 2026 → currently UVA President)
   - leader_2: Oregon Superintendent (HGSE Ed Policy master’s → start July 1, 2023 → superintendent of Oregon’s 2nd largest district)
   - leader_3: Texas Superintendent (30+ years exp → 2020 Region 10 Superintendent of the Year → appointed Jan 2024 → 6 years previously in current district as executive principal & assistant superintendent → superintendent in at least two other Texas districts)
   - leader_4: FCS→FBS Head Coach (FCS playoffs in 2022, 2023, 2024; quarterfinals in 2023 & 2024; hired to FBS head coach on Dec 14, 2024)
   - leader_5: Ivy League Head Coach (Princeton 2006 grad → hired Feb 2024 → came directly from Rutgers → shared Ivy title in 2024 → previously associate head coach at Princeton)

Return a JSON object with fields: leader_1, leader_2, leader_3, leader_4, leader_5, each being a LeaderEntry object.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(entry: Optional[LeaderEntry]) -> str:
    return entry.name if (entry and entry.name) else "the individual"


def _sources(entry: Optional[LeaderEntry]) -> List[str]:
    return entry.reference_urls if (entry and entry.reference_urls) else []


async def _add_and_verify(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    critical: bool = True,
    add_ins: Optional[str] = None,
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=add_ins or "None",
    )
    return leaf


# --------------------------------------------------------------------------- #
# Leader-specific verification builders                                       #
# --------------------------------------------------------------------------- #
async def verify_leader_1(evaluator: Evaluator, parent, entry: Optional[LeaderEntry]) -> None:
    """
    Leader 1: University President with McKinsey 26 years -> business school dean -> appointed Dec 2025 -> office Jan 1 2026 -> current UVA president.
    """
    leader_node = evaluator.add_parallel(
        id="leader_1",
        desc="Identify the university president who transitioned from consulting to business school dean to university president",
        parent=parent,
        critical=False
    )
    name = _safe_name(entry)
    urls = _sources(entry)

    common_ins = (
        "Use the provided source(s) only. Allow reasonable synonyms and abbreviations. "
        "Treat 'UVA' as 'University of Virginia'. If the page states 'effective January 1, 2026', it counts as taking office on that date. "
        "Minor name format variations are acceptable."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_consulting_career",
        "The individual worked at McKinsey & Company for exactly 26 years before transitioning to higher education",
        f"The person named {name} worked at McKinsey & Company for exactly 26 years and then moved into higher education (e.g., left McKinsey and took a university role).",
        urls,
        critical=True,
        add_ins=common_ins + " Look for an explicit '26 years' tenure at McKinsey and indicate that the shift was into academia."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_business_school_dean",
        "The individual served as a business school dean before becoming a university president",
        f"The person named {name} served as a business school dean prior to being appointed a university president.",
        urls,
        critical=True,
        add_ins=common_ins + " Accept 'dean of the business school' or equivalent phrasing."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_appointment_timing",
        "The individual was appointed as a university president in December 2025",
        f"The person named {name} was appointed as a university president in December 2025.",
        urls,
        critical=True,
        add_ins=common_ins + " If the page says 'appointed in December 2025' or similar (e.g., 'announced in December 2025'), accept it."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_office_start",
        "The individual took office as president on January 1, 2026",
        f"The person named {name} took office as university president on January 1, 2026.",
        urls,
        critical=True,
        add_ins=common_ins + " Accept 'took office January 1, 2026' or 'term began January 1, 2026' or 'effective January 1, 2026'."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_current_institution",
        "The individual currently serves as president of the University of Virginia",
        f"The person named {name} currently serves as president of the University of Virginia.",
        urls,
        critical=True,
        add_ins=common_ins + " 'currently' refers to the status stated on the page; 'UVA' equals 'University of Virginia'."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_1_reference_url_1",
        "Provide a reference URL that verifies the consulting background and current university president role",
        f"This page verifies both that {name} worked at McKinsey & Company for 26 years and that {name} is the current president of the University of Virginia.",
        urls,
        critical=True,
        add_ins=common_ins + " Passing requires a single provided page to support both statements together."
    )


async def verify_leader_2(evaluator: Evaluator, parent, entry: Optional[LeaderEntry]) -> None:
    """
    Leader 2: Oregon Superintendent (HGSE Education Policy master's; started July 1, 2023; superintendent of Oregon's second-largest district).
    """
    leader_node = evaluator.add_parallel(
        id="leader_2",
        desc="Identify the superintendent of Oregon's second-largest school district with Harvard education background",
        parent=parent,
        critical=False
    )
    name = _safe_name(entry)
    org = entry.current_org if (entry and entry.current_org) else "their district"
    urls = _sources(entry)

    common_ins = (
        "Use the provided sources only. Allow reasonable paraphrases. "
        "For the Harvard degree, it must specifically be a master's degree in Education Policy from HGSE. "
        "For district rank, accept phrasing like 'Oregon's second largest school district'."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_2_harvard_degree",
        "The individual holds a master's degree in Education Policy from Harvard Graduate School of Education",
        f"The person named {name} holds a master's degree in Education Policy from the Harvard Graduate School of Education.",
        urls,
        critical=True,
        add_ins=common_ins + " Acronyms like 'HGSE' are acceptable."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_2_start_date",
        "The individual started their current superintendent position on July 1, 2023",
        f"The person named {name} started their current superintendent position on July 1, 2023.",
        urls,
        critical=True,
        add_ins=common_ins + " Accept explicit references to a start date of July 1, 2023."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_2_district_rank",
        "The individual serves as superintendent of the second-largest school district in Oregon",
        f"The person named {name} is the superintendent of {org}, which is the second-largest school district in Oregon.",
        urls,
        critical=True,
        add_ins=common_ins + " Verify that the district is identified as the second largest in Oregon."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_2_reference_url_2",
        "Provide a reference URL that verifies the Harvard degree and superintendent appointment",
        f"This page verifies both that {name} holds a master's degree in Education Policy from HGSE and that {name} was appointed as superintendent (with a start date of July 1, 2023).",
        urls,
        critical=True,
        add_ins=common_ins + " Passing requires a single provided page to support both statements together."
    )


async def verify_leader_3(evaluator: Evaluator, parent, entry: Optional[LeaderEntry]) -> None:
    """
    Leader 3: Texas Superintendent with 30+ years, Region 10 SOTY 2020, appointed Jan 2024, prior 6 years in current district (exec principal & assistant superintendent), previously superintendent at 2+ other Texas districts.
    """
    leader_node = evaluator.add_parallel(
        id="leader_3",
        desc="Identify the Texas superintendent with 30+ years experience who returned to a district where they previously worked",
        parent=parent,
        critical=False
    )
    name = _safe_name(entry)
    org = entry.current_org if (entry and entry.current_org) else "their current district"
    urls = _sources(entry)

    common_ins = (
        "Use provided sources only. Allow reasonable paraphrase. "
        "Accept 'Region 10 Superintendent of the Year' phrasing including TASB/ESC references. "
        "For 'over 30 years', any explicit '30+ years' or 'more than 30 years' qualifies."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_experience_years",
        "The individual has over 30 years of experience in public education",
        f"The person named {name} has over 30 years of experience in public education.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_superintendent_award",
        "The individual was named Region 10 Superintendent of the Year in 2020",
        f"The person named {name} was named Region 10 Superintendent of the Year in 2020.",
        urls,
        critical=True,
        add_ins=common_ins + " 'Region 10' may appear as ESC Region 10 or TASB Region 10."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_appointment_date",
        "The individual was officially appointed to their current superintendent position in January 2024",
        f"The person named {name} was officially appointed to their current superintendent position in January 2024.",
        urls,
        critical=True,
        add_ins=common_ins + " Accept 'appointed in January 2024' or 'approved in January 2024'."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_prior_district_role",
        "The individual previously worked in their current district for 6 years as an executive principal and assistant superintendent before becoming superintendent",
        f"Before becoming superintendent, {name} worked in {org} for 6 years as an executive principal and assistant superintendent.",
        urls,
        critical=True,
        add_ins=common_ins + " The 6-year tenure should be explicit; titles may appear as 'executive principal' and 'assistant superintendent'."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_multiple_superintendencies",
        "The individual served as superintendent in at least two other Texas school districts before their current position",
        f"The person named {name} previously served as superintendent in at least two other Texas school districts before the current position.",
        urls,
        critical=True,
        add_ins=common_ins + " Look for mentions of superintendent roles at two or more Texas districts prior to the current role."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_3_reference_url_3",
        "Provide a reference URL that verifies the Region 10 award and career history",
        f"This page verifies that {name} was Region 10 Superintendent of the Year in 2020 and provides a summary of {name}'s career history.",
        urls,
        critical=True,
        add_ins=common_ins + " Passing requires one page to support both the Region 10 award and biographical/career history elements."
    )


async def verify_leader_4(evaluator: Evaluator, parent, entry: Optional[LeaderEntry]) -> None:
    """
    Leader 4: FCS to FBS head coach—FCS playoffs 2022/2023/2024; reached quarterfinals in 2023 & 2024; hired as FBS head coach on Dec 14, 2024.
    """
    leader_node = evaluator.add_parallel(
        id="leader_4",
        desc="Identify the college football coach who moved from FCS to FBS after three consecutive playoff appearances",
        parent=parent,
        critical=False
    )
    name = _safe_name(entry)
    urls = _sources(entry)

    common_ins = (
        "Use provided sources only. 'FCS' refers to NCAA Division I Football Championship Subdivision. "
        "Accept synonyms like 'national quarterfinals' for quarterfinal appearances."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_4_three_playoffs",
        "The individual led their FCS team to playoff appearances in three consecutive years: 2022, 2023, and 2024",
        f"The person named {name} led their FCS team to the playoffs in 2022, 2023, and 2024.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_4_quarterfinal_appearances",
        "The individual's team reached the FCS playoff quarterfinals in both 2023 and 2024",
        f"The team coached by {name} reached the FCS playoff quarterfinals in both 2023 and 2024.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_4_fbs_hire_date",
        "The individual was hired as a head coach at an FBS program on December 14, 2024",
        f"The person named {name} was hired as a head coach at an FBS program on December 14, 2024.",
        urls,
        critical=True,
        add_ins=common_ins + " Look for an official announcement or reputable report with the exact date."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_4_reference_url_4",
        "Provide a reference URL that verifies the playoff appearances and FBS hire",
        f"This page verifies the three consecutive FCS playoff appearances (2022, 2023, 2024) and the FBS head coach hire on December 14, 2024 for {name}.",
        urls,
        critical=True,
        add_ins=common_ins + " Passing requires one page supporting both the playoff streak and the FBS hire/date."
    )


async def verify_leader_5(evaluator: Evaluator, parent, entry: Optional[LeaderEntry]) -> None:
    """
    Leader 5: Ivy League head coach—Princeton 2006 graduate, named head coach Feb 2024, came directly from Rutgers, shared Ivy title in 2024, previously associate head coach at Princeton.
    """
    leader_node = evaluator.add_parallel(
        id="leader_5",
        desc="Identify the Ivy League football coach who is a Princeton alumnus and won a conference title in their first season",
        parent=parent,
        critical=False
    )
    name = _safe_name(entry)
    urls = _sources(entry)

    common_ins = (
        "Use provided sources only. Accept phrasing like 'co-champions' or 'shared the Ivy League title' for the 2024 championship. "
        "For the Princeton degree, the class year 2006 must be explicit."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_princeton_graduate",
        "The individual graduated from Princeton University in 2006",
        f"The person named {name} graduated from Princeton University in 2006.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_hire_announcement",
        "The individual was named as a head football coach in February 2024",
        f"The person named {name} was named as a head football coach in February 2024.",
        urls,
        critical=True,
        add_ins=common_ins + " Accept official announcements or reputable reports dated in February 2024."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_prior_position",
        "The individual came to their current position directly from Rutgers University",
        f"The person named {name} came to their current head coach position directly from Rutgers University.",
        urls,
        critical=True,
        add_ins=common_ins + " Look for explicit mention that the prior role was at Rutgers immediately before the current head-coach role."
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_first_season_success",
        "The individual led their team to a shared Ivy League championship in their first season (2024)",
        f"In 2024 (their first season), {name} led the team to a shared Ivy League championship.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_princeton_coaching",
        "The individual previously served as associate head coach at Princeton University",
        f"The person named {name} previously served as an associate head coach at Princeton University.",
        urls,
        critical=True,
        add_ins=common_ins
    )

    await _add_and_verify(
        evaluator,
        leader_node,
        "leader_5_reference_url_5",
        "Provide a reference URL that verifies the Princeton background and first-season championship",
        f"This page verifies that {name} is a 2006 Princeton graduate and that the team won a shared Ivy League championship in 2024 under {name}.",
        urls,
        critical=True,
        add_ins=common_ins + " Passing requires one page supporting both the Princeton 2006 alumni status and the 2024 shared Ivy title."
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
    Build and execute the evaluation for the five-education-leaders task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # Extract leaders information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_leaders(),
        template_class=LeadersExtraction,
        extraction_name="leaders_extraction"
    )

    # Build verification subtree for each leader
    await verify_leader_1(evaluator, root, extracted.leader_1 if extracted else None)
    await verify_leader_2(evaluator, root, extracted.leader_2 if extracted else None)
    await verify_leader_3(evaluator, root, extracted.leader_3 if extracted else None)
    await verify_leader_4(evaluator, root, extracted.leader_4 if extracted else None)
    await verify_leader_5(evaluator, root, extracted.leader_5 if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()