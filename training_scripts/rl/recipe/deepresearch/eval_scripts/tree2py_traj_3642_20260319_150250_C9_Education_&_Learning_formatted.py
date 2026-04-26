import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "college_sports_and_tx_superintendent_2024_2026"
TASK_DESCRIPTION = """
Between January 2024 and March 2026, several notable career transitions occurred in American collegiate athletics and public education leadership. Identify four specific individuals who made significant career moves during this period, each meeting the following detailed criteria:

Individual 1: A Division II football head coach who won at least four NCAA Division II national championships at a single institution and led that same institution to 11 consecutive NCAA Division II playoff appearances. This coach must still be actively serving at that institution as of March 2026, and the institution must compete in the Great Lakes Intercollegiate Athletic Conference (GLIAC).

Individual 2: A Division I men's basketball head coach who led a team to the NCAA Final Four in 2023 and subsequently accepted a head coaching position at a Big Ten Conference institution in March 2024. At their previous institution, this coach must have achieved a winning record exceeding 125 total wins over six seasons, and their previous school must have been Florida Atlantic University.

Individual 3: A head football coach at a Historically Black College or University (HBCU) who was officially appointed to their head coaching position in January 2026. This individual must have played quarterback at the same HBCU institution where they were appointed head coach, and they must hold their conference's (Mid-Eastern Athletic Conference - MEAC) all-time records for both career passing yards (exceeding 9,800 yards) and career passing touchdowns (exceeding 90 touchdowns) from their playing career.

Individual 4: A superintendent of a public school district in Texas who was appointed to the superintendent position in March 2022 and oversees a district with student enrollment exceeding 45,000 students as documented in the 2024-2025 school year. This superintendent must hold a doctoral degree (Ed.D. or Ph.D.) and their district must rank among the 10 largest public school districts in Texas by enrollment.

For each individual, provide:
- Full name
- Current institution or school district name
- Key supporting facts that verify they meet all specified criteria
- At least one reference URL from an official source (university athletics website, school district website, or credible news organization) that documents their appointment, achievements, or credentials
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Individual(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None  # School/University/Athletics Department or School District
    role_title: Optional[str] = None    # e.g., "Head Football Coach", "Head Men's Basketball Coach", "Superintendent"
    sport_or_domain: Optional[str] = None  # e.g., "football", "men's basketball", "K-12 education"
    division_level: Optional[str] = None    # e.g., "NCAA Division II", "NCAA Division I"
    conference: Optional[str] = None        # e.g., "GLIAC", "Big Ten", "MEAC"
    previous_org: Optional[str] = None      # e.g., "Florida Atlantic University"
    appointment_month_year: Optional[str] = None  # e.g., "March 2024", "January 2026"
    achievements: List[str] = Field(default_factory=list)
    wins_total_previous: Optional[str] = None   # e.g., "126 wins"
    seasons_count_previous: Optional[str] = None  # e.g., "6 seasons"
    final_four_year: Optional[str] = None
    championships_count: Optional[str] = None   # e.g., "4", "at least four"
    playoff_streak_desc: Optional[str] = None   # e.g., "11 straight NCAA DII playoff appearances"
    enrollment_2024_2025: Optional[str] = None  # e.g., "48,200"
    doctoral_degree: Optional[str] = None       # e.g., "Ed.D.", "Ph.D.", "Doctor of Education"
    urls: List[str] = Field(default_factory=list)


class IndividualsExtraction(BaseModel):
    individual_1: Optional[Individual] = None
    individual_2: Optional[Individual] = None
    individual_3: Optional[Individual] = None
    individual_4: Optional[Individual] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_individuals() -> str:
    return """
Extract structured information for exactly four individuals from the answer, mapping each to the specified category.
For each of the following, extract the fields listed below. If any field is missing, set it to null. Collect all URLs explicitly mentioned for that individual.

Common fields to extract for all individuals:
- name: full name of the person
- organization: current institution or school district
- role_title: e.g., "Head Football Coach", "Head Men's Basketball Coach", "Superintendent"
- sport_or_domain: e.g., "football", "men's basketball", "K-12 education"
- division_level: e.g., "NCAA Division II", "NCAA Division I" (for coaches)
- conference: e.g., "GLIAC", "Big Ten", "MEAC" (if applicable)
- previous_org: previous institution if mentioned
- appointment_month_year: e.g., "March 2024", "January 2026" (for the relevant move/appointment)
- achievements: list of short fact strings relevant to the criteria
- wins_total_previous: e.g., "126 wins" (if applicable)
- seasons_count_previous: e.g., "6 seasons" (if applicable)
- final_four_year: e.g., "2023" (if applicable)
- championships_count: e.g., "4", "at least four" (if applicable)
- playoff_streak_desc: e.g., "11 consecutive NCAA Division II playoff appearances" (if applicable)
- enrollment_2024_2025: e.g., "48,200" (if applicable to superintendent)
- doctoral_degree: e.g., "Ed.D.", "Ph.D." (if applicable to superintendent)
- urls: array of all URLs cited for this individual (official websites, credible news, etc.)

Return a JSON object:
{
  "individual_1": { ...fields above... },
  "individual_2": { ... },
  "individual_3": { ... },
  "individual_4": { ... }
}

Notes per individual:
- individual_1 (DII football head coach; GLIAC; ≥4 national titles at a single institution; 11 consecutive DII playoffs; active as of Mar 2026)
- individual_2 (DI men's basketball head coach; led 2023 Final Four; accepted Big Ten job in Mar 2024; prior school FAU; >125 wins over 6 seasons at FAU)
- individual_3 (HBCU head football coach; appointed Jan 2026; played QB at same HBCU; holds MEAC all-time records for career passing yards > 9,800 and TDs > 90)
- individual_4 (Texas public school district superintendent; appointed Mar 2022; enrollment >45,000 in 2024–2025; has doctoral degree; district top-10 by TX enrollment)

SPECIAL URL RULES:
- Only extract URLs explicitly present in the answer text. Include full URLs (http/https).
- If multiple URLs are given for an individual, include them all in the 'urls' list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _normalize_name(name: Optional[str]) -> str:
    return "".join(ch.lower() for ch in (name or "") if ch.isalnum())


def _safe_sources(item: Optional[Individual]) -> List[str]:
    if not item or not item.urls:
        return []
    # De-duplicate & strip
    seen = set()
    out = []
    for u in item.urls:
        us = (u or "").strip()
        if not us:
            continue
        if us not in seen:
            seen.add(us)
            out.append(us)
    return out


async def _verify_with_urls(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: List[str],
    add_ins: str = "None",
    extra_prereq_nodes: Optional[List] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=add_ins,
        extra_prerequisites=extra_prereq_nodes or [],
    )


# --------------------------------------------------------------------------- #
# Verification logic per individual                                           #
# --------------------------------------------------------------------------- #
async def verify_individual_1(evaluator: Evaluator, parent, item: Optional[Individual]) -> None:
    node = evaluator.add_parallel(
        id="individual_1",
        desc="Individual 1: Division II football head coach meeting championships/playoff/GLIAC/active-service criteria.",
        parent=parent,
        critical=False,
    )

    name_ok = evaluator.add_custom_node(
        result=bool(_norm(item.name)) if item else False,
        id="individual_1_name_provided",
        desc="Provides the individual’s full name.",
        parent=node,
        critical=True,
    )

    org_ok = evaluator.add_custom_node(
        result=bool(_norm(item.organization)) if item else False,
        id="individual_1_current_institution_provided",
        desc="Provides the coach’s current institution name.",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(item)
    ref_ok = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="individual_1_reference_url",
        desc="Provides ≥1 reference URL from an allowed source.",
        parent=node,
        critical=True,
    )

    # Serving within Jan 2024–Mar 2026
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_serving_within_jan_2024_to_mar_2026",
        desc="Individual was serving as a head coach at some point during Jan 2024–Mar 2026.",
        parent=node,
        critical=True,
        claim=f"Between January 2024 and March 2026, {_norm(item.name)} served as head football coach at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Accept if any reliable source shows this person as head coach in that timeframe; current roster or bio pages implying continuity into 2024–2026 also qualify.",
        extra_prereq_nodes=[ref_ok],
    )

    # Role and Division
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_role_and_division",
        desc="Individual is a Division II football head coach.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} is the head coach of the football team at {_norm(item.organization)} and that team competes in NCAA Division II.",
        sources=sources,
        add_ins="Look for wording like NCAA Division II, D2, or explicit institutional division classification on official pages.",
        extra_prereq_nodes=[ref_ok],
    )

    # ≥4 national titles at a single institution
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_single_institution_4_titles",
        desc="Coach won ≥4 NCAA Division II national championships at a single institution.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} has won at least four NCAA Division II national championships while coaching at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Verify that at least four national titles are credited to this coach at this same institution.",
        extra_prereq_nodes=[ref_ok],
    )

    # 11 consecutive playoff appearances
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_eleven_consecutive_playoffs",
        desc="That institution reached 11 consecutive NCAA Division II playoff appearances under the coach.",
        parent=node,
        critical=True,
        claim=f"Under {_norm(item.name)}, {_norm(item.organization)} made 11 consecutive NCAA Division II playoff appearances.",
        sources=sources,
        add_ins="Look for official records, season summaries, or media guides confirming an 11-year consecutive DII playoff streak.",
        extra_prereq_nodes=[ref_ok],
    )

    # GLIAC membership
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_gliac_membership",
        desc="The institution competes in the Great Lakes Intercollegiate Athletic Conference (GLIAC).",
        parent=node,
        critical=True,
        claim=f"{_norm(item.organization)} competes in the Great Lakes Intercollegiate Athletic Conference (GLIAC).",
        sources=sources,
        add_ins="Conference membership should be explicitly stated on official athletics or conference pages.",
        extra_prereq_nodes=[ref_ok],
    )

    # Active as of March 2026
    await _verify_with_urls(
        evaluator,
        node_id="individual_1_active_as_of_mar_2026",
        desc="Coach is still actively serving at that institution as of March 2026.",
        parent=node,
        critical=True,
        claim=f"As of March 2026, {_norm(item.name)} is still the head coach at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Accept if the most recent official source indicates the person is the current head coach during or near 2026.",
        extra_prereq_nodes=[ref_ok],
    )


async def verify_individual_2(evaluator: Evaluator, parent, item: Optional[Individual]) -> None:
    node = evaluator.add_parallel(
        id="individual_2",
        desc="Individual 2: Division I men’s basketball head coach with 2023 Final Four and March 2024 Big Ten hire; prior school FAU; >125 wins in 6 seasons at FAU.",
        parent=parent,
        critical=False,
    )

    name_ok = evaluator.add_custom_node(
        result=bool(_norm(item.name)) if item else False,
        id="individual_2_name_provided",
        desc="Provides the individual’s full name.",
        parent=node,
        critical=True,
    )

    org_ok = evaluator.add_custom_node(
        result=bool(_norm(item.organization)) if item else False,
        id="individual_2_current_institution_provided",
        desc="Provides the coach’s current institution name.",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(item)
    ref_ok = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="individual_2_reference_url",
        desc="Provides ≥1 reference URL from an allowed source (official institutional/athletics site, credible news organization, or credible sports database).",
        parent=node,
        critical=True,
    )

    # Serving within timeframe
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_serving_within_jan_2024_to_mar_2026",
        desc="Individual was serving as a head coach at some point during Jan 2024–Mar 2026.",
        parent=node,
        critical=True,
        claim=f"Between January 2024 and March 2026, {_norm(item.name)} served as a head coach at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Roster/bio/announcement pages in or after March 2024 qualify as evidence.",
        extra_prereq_nodes=[ref_ok],
    )

    # Role and Division
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_role_and_division",
        desc="Individual is a Division I men’s basketball head coach.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} is the head coach of the men's basketball team at {_norm(item.organization)}, which competes in NCAA Division I.",
        sources=sources,
        add_ins="Confirm NCAA Division I and men's basketball head coach role.",
        extra_prereq_nodes=[ref_ok],
    )

    # Final Four 2023
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_final_four_2023",
        desc="Coach led a team to the NCAA Final Four in 2023.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} led a team to the NCAA Men's Basketball Final Four in 2023.",
        sources=sources,
        add_ins="Look for 2023 NCAA tournament results showing a Final Four appearance under this coach.",
        extra_prereq_nodes=[ref_ok],
    )

    # Hired to Big Ten in March 2024
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_hired_big_ten_mar_2024",
        desc="Coach accepted a head coaching position at a Big Ten Conference institution in March 2024.",
        parent=node,
        critical=True,
        claim=f"In March 2024, {_norm(item.name)} accepted the head coaching position at {_norm(item.organization)}, a Big Ten Conference institution.",
        sources=sources,
        add_ins="The source should indicate March 2024 timing and Big Ten affiliation of the hiring institution.",
        extra_prereq_nodes=[ref_ok],
    )

    # Previous institution FAU
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_previous_institution_fau",
        desc="Coach’s previous institution was Florida Atlantic University (FAU).",
        parent=node,
        critical=True,
        claim=f"Before joining {_norm(item.organization)}, {_norm(item.name)} was the head coach at Florida Atlantic University (FAU).",
        sources=sources,
        add_ins="Confirm FAU as the prior head coaching position.",
        extra_prereq_nodes=[ref_ok],
    )

    # >125 wins over 6 seasons at FAU
    await _verify_with_urls(
        evaluator,
        node_id="individual_2_wins_exceed_125_over_6_seasons",
        desc="At FAU, coach recorded >125 total wins over six seasons.",
        parent=node,
        critical=True,
        claim=f"At Florida Atlantic University (FAU), {_norm(item.name)} recorded more than 125 total wins over six seasons.",
        sources=sources,
        add_ins="Look for explicit career record totals at FAU across six seasons; accept ≥126 as 'more than 125'.",
        extra_prereq_nodes=[ref_ok],
    )


async def verify_individual_3(evaluator: Evaluator, parent, item: Optional[Individual]) -> None:
    node = evaluator.add_parallel(
        id="individual_3",
        desc="Individual 3: HBCU head football coach appointed January 2026; played QB at same HBCU; holds MEAC all-time passing yards/TD records above thresholds.",
        parent=parent,
        critical=False,
    )

    name_ok = evaluator.add_custom_node(
        result=bool(_norm(item.name)) if item else False,
        id="individual_3_name_provided",
        desc="Provides the individual’s full name.",
        parent=node,
        critical=True,
    )

    inst_ok = evaluator.add_custom_node(
        result=bool(_norm(item.organization)) if item else False,
        id="individual_3_institution_provided",
        desc="Provides the HBCU institution name.",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(item)
    ref_ok = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="individual_3_reference_url",
        desc="Provides ≥1 reference URL from an allowed source (official institutional/athletics site, credible news organization, or credible sports database).",
        parent=node,
        critical=True,
    )

    # Serving within timeframe
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_serving_within_jan_2024_to_mar_2026",
        desc="Individual was serving as a head coach at some point during Jan 2024–Mar 2026.",
        parent=node,
        critical=True,
        claim=f"Between January 2024 and March 2026, {_norm(item.name)} served as head football coach at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Appointment in January 2026 counts as serving within the window.",
        extra_prereq_nodes=[ref_ok],
    )

    # Role: HBCU head football coach (verify head coach role; HBCU context supported by MEAC node)
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_role_hbcu_head_football_coach",
        desc="Individual is a head football coach at a Historically Black College or University (HBCU).",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} is the head football coach at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Confirm head football coach role on official site; HBCU status will be corroborated via MEAC membership.",
        extra_prereq_nodes=[ref_ok],
    )

    # Appointed January 2026
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_appointed_jan_2026",
        desc="Individual was officially appointed to the head coaching position in January 2026.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} was officially appointed head football coach at {_norm(item.organization)} in January 2026.",
        sources=sources,
        add_ins="Look for press releases or announcements dated January 2026.",
        extra_prereq_nodes=[ref_ok],
    )

    # Played QB at same school
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_played_qb_same_school",
        desc="Individual played quarterback at the same HBCU institution where they were appointed head coach.",
        parent=node,
        critical=True,
        claim=f"During his playing career, {_norm(item.name)} played quarterback at {_norm(item.organization)}.",
        sources=sources,
        add_ins="Look for bio pages or record books indicating the person played QB at the same school.",
        extra_prereq_nodes=[ref_ok],
    )

    # MEAC membership
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_meac_membership",
        desc="Institution competes in the Mid-Eastern Athletic Conference (MEAC).",
        parent=node,
        critical=True,
        claim=f"{_norm(item.organization)} competes in the Mid-Eastern Athletic Conference (MEAC).",
        sources=sources,
        add_ins="Conference membership should be explicitly stated on official athletics or conference pages.",
        extra_prereq_nodes=[ref_ok],
    )

    # MEAC all-time records thresholds
    await _verify_with_urls(
        evaluator,
        node_id="individual_3_meac_all_time_records_thresholds",
        desc="Individual holds MEAC all-time records for both career passing yards (>9,800) and career passing touchdowns (>90).",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} holds the MEAC all-time records for career passing yards (exceeding 9,800 yards) and career passing touchdowns (exceeding 90).",
        sources=sources,
        add_ins="Verify all-time conference records (not single-season). Totals must exceed 9,800 yards and 90 TDs.",
        extra_prereq_nodes=[ref_ok],
    )


async def verify_individual_4(evaluator: Evaluator, parent, item: Optional[Individual]) -> None:
    node = evaluator.add_parallel(
        id="individual_4",
        desc="Individual 4: Texas public school district superintendent appointed March 2022; district enrollment >45,000 in 2024–2025; doctorate; district top-10 by Texas enrollment.",
        parent=parent,
        critical=False,
    )

    name_ok = evaluator.add_custom_node(
        result=bool(_norm(item.name)) if item else False,
        id="individual_4_name_provided",
        desc="Provides the individual’s full name.",
        parent=node,
        critical=True,
    )

    dist_ok = evaluator.add_custom_node(
        result=bool(_norm(item.organization)) if item else False,
        id="individual_4_district_provided",
        desc="Provides the school district name.",
        parent=node,
        critical=True,
    )

    sources = _safe_sources(item)
    ref_ok = evaluator.add_custom_node(
        result=len(sources) > 0,
        id="individual_4_reference_url",
        desc="Provides ≥1 reference URL from an allowed source (official school district website or credible news organization).",
        parent=node,
        critical=True,
    )

    # Serving within timeframe
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_serving_within_jan_2024_to_mar_2026",
        desc="Individual was serving as superintendent at some point during Jan 2024–Mar 2026.",
        parent=node,
        critical=True,
        claim=f"Between January 2024 and March 2026, {_norm(item.name)} served as superintendent of {_norm(item.organization)}.",
        sources=sources,
        add_ins="District leadership pages current in 2024–2026 or news updates suffice.",
        extra_prereq_nodes=[ref_ok],
    )

    # Role and state
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_role_and_state",
        desc="Individual is a superintendent of a Texas public school district.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} is the superintendent of {_norm(item.organization)}, a Texas public school district.",
        sources=sources,
        add_ins="Confirm superintendent role and Texas location/state context.",
        extra_prereq_nodes=[ref_ok],
    )

    # Appointed March 2022
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_appointed_mar_2022",
        desc="Superintendent was appointed in March 2022.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} was appointed superintendent of {_norm(item.organization)} in March 2022.",
        sources=sources,
        add_ins="Look for official board minutes, press releases, or credible news from March 2022.",
        extra_prereq_nodes=[ref_ok],
    )

    # Enrollment > 45,000 in 2024–2025
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_enrollment_gt_45000_2024_2025",
        desc="District enrollment exceeds 45,000 students as documented in the 2024–2025 school year.",
        parent=node,
        critical=True,
        claim=f"During the 2024–2025 school year, {_norm(item.organization)} had student enrollment exceeding 45,000.",
        sources=sources,
        add_ins="Use official district or state education reports; accept explicit counts > 45,000 for 2024–2025.",
        extra_prereq_nodes=[ref_ok],
    )

    # Doctoral degree
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_doctoral_degree",
        desc="Superintendent holds a doctoral degree (Ed.D. or Ph.D.).",
        parent=node,
        critical=True,
        claim=f"{_norm(item.name)} holds a doctoral degree (Ed.D. or Ph.D. or equivalent).",
        sources=sources,
        add_ins="District bio or official CV indicating doctoral degree suffices.",
        extra_prereq_nodes=[ref_ok],
    )

    # Top-10 in Texas by enrollment
    await _verify_with_urls(
        evaluator,
        node_id="individual_4_top_10_texas_enrollment",
        desc="District ranks among the 10 largest public school districts in Texas by enrollment.",
        parent=node,
        critical=True,
        claim=f"{_norm(item.organization)} ranks among the 10 largest public school districts in Texas by enrollment.",
        sources=sources,
        add_ins="Look for Texas-wide rankings (state reports, recognized education orgs, or district citing state rank).",
        extra_prereq_nodes=[ref_ok],
    )


# --------------------------------------------------------------------------- #
# Root-level cardinality/distinctness check                                   #
# --------------------------------------------------------------------------- #
def add_distinct_individuals_check(evaluator: Evaluator, parent, extracted: IndividualsExtraction) -> None:
    names = [
        _normalize_name(extracted.individual_1.name if extracted.individual_1 else None),
        _normalize_name(extracted.individual_2.name if extracted.individual_2 else None),
        _normalize_name(extracted.individual_3.name if extracted.individual_3 else None),
        _normalize_name(extracted.individual_4.name if extracted.individual_4 else None),
    ]
    nonempty = [n for n in names if n]
    unique_count = len(set(nonempty))
    provides_four = (len(nonempty) == 4) and (unique_count == 4)
    evaluator.add_custom_node(
        result=provides_four,
        id="provides_four_distinct_individuals",
        desc="Response includes four distinct individuals corresponding to Individuals 1–4 (not fewer, not duplicates).",
        parent=parent,
        critical=True,
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
    # Initialize evaluator with a parallel root aggregation (to allow partial credit across the four individuals)
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

    # Extract structured info for four individuals
    extracted = await evaluator.extract(
        prompt=prompt_extract_individuals(),
        template_class=IndividualsExtraction,
        extraction_name="extracted_individuals",
    )

    # Root-level distinctness check
    add_distinct_individuals_check(evaluator, root, extracted)

    # Build subtrees and verifications for each individual (always create nodes; missing info will fail specific checks)
    await verify_individual_1(evaluator, root, extracted.individual_1 or Individual())
    await verify_individual_2(evaluator, root, extracted.individual_2 or Individual())
    await verify_individual_3(evaluator, root, extracted.individual_3 or Individual())
    await verify_individual_4(evaluator, root, extracted.individual_4 or Individual())

    # Return evaluation summary
    return evaluator.get_summary()