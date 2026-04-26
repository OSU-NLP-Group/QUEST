import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "q1_2026_federal_developments"
TASK_DESCRIPTION = """
Identify four significant federal government developments that were publicly announced during the first quarter of 2026 (January 1 through March 31, 2026), with each development representing a different policy domain as specified below. For each development, provide the exact announcement date, a description of the key decision or action, relevant specific details, the primary responsible official or governing body, and a reference URL from a credible news source.

The four required policy domains are:

1. Monetary Policy: A major Federal Reserve monetary policy decision or announcement (include the specific policy action, the voting breakdown, and the Federal Reserve official who announced it)

2. International Diplomacy: A significant bilateral diplomatic agreement or change in diplomatic relations involving the United States (include the countries involved, the nature of the change, and context about the previous state of relations)

3. Space Exploration: A major NASA mission development, milestone, or significant announcement (include the official mission name, the type of mission, and key mission characteristics such as crew size, duration, or mission objectives)

4. Law Enforcement and Justice Policy: A significant Department of Justice policy initiative, new division, or major organizational development (include the specific policy or organizational change, the focus area, and the primary government official who announced it)

For each of the four developments, the answer must include all requested details and be supported by at least one reference URL from a major news outlet confirming the information.
"""

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class MonetaryPolicyInfo(BaseModel):
    announcement_date: Optional[str] = None
    decision_content: Optional[str] = None
    vote_breakdown: Optional[str] = None
    decision_maker: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class InternationalDiplomacyInfo(BaseModel):
    announcement_date: Optional[str] = None
    countries_involved: List[str] = Field(default_factory=list)
    nature_agreement: Optional[str] = None
    previous_status: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SpaceExplorationInfo(BaseModel):
    event_date: Optional[str] = None
    mission_name: Optional[str] = None
    mission_type: Optional[str] = None
    mission_characteristics: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class LawEnforcementJusticeInfo(BaseModel):
    announcement_date: Optional[str] = None
    department_agency: Optional[str] = None
    policy_nature: Optional[str] = None
    primary_official: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AllDevelopmentsExtraction(BaseModel):
    monetary_policy: Optional[MonetaryPolicyInfo] = None
    international_diplomacy: Optional[InternationalDiplomacyInfo] = None
    space_exploration: Optional[SpaceExplorationInfo] = None
    law_enforcement_justice: Optional[LawEnforcementJusticeInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the four required developments from the answer, one for each domain: Monetary Policy (Federal Reserve), International Diplomacy (bilateral, involving the United States), Space Exploration (NASA), and Law Enforcement/Justice Policy (Department of Justice). For each, extract exactly the requested fields. If any field is missing in the answer, set it to null (for strings) or an empty array (for lists). If multiple items are provided for a domain, select the first clearly described and most prominent one.

Output JSON schema:
{
  "monetary_policy": {
    "announcement_date": string | null,        // The exact announcement date as written
    "decision_content": string | null,         // Specific policy action (e.g., held rates, cut, QT change)
    "vote_breakdown": string | null,           // Full vote breakdown text (e.g., "9–1" or "9-1 with X dissent")
    "decision_maker": string | null,           // Fed official who announced it (e.g., Jerome Powell)
    "reference_urls": string[]                 // All URLs cited; must be explicit in the answer
  },
  "international_diplomacy": {
    "announcement_date": string | null,
    "countries_involved": string[],            // List of country names as written in the answer
    "nature_agreement": string | null,         // Nature of agreement or change
    "previous_status": string | null,          // Prior state of relations or policy
    "reference_urls": string[]
  },
  "space_exploration": {
    "event_date": string | null,
    "mission_name": string | null,             // Official mission/program name
    "mission_type": string | null,             // Type (e.g., crewed lunar flyby, Mars rover)
    "mission_characteristics": string | null,  // Key details: crew size, duration, objectives, etc.
    "reference_urls": string[]
  },
  "law_enforcement_justice": {
    "announcement_date": string | null,
    "department_agency": string | null,        // DOJ or DOJ component leading the change
    "policy_nature": string | null,            // Policy/initiative/org change details
    "primary_official": string | null,         // Official who announced or leads it
    "reference_urls": string[]
  }
}

Rules:
- Extract only what appears in the answer verbatim. Do not invent data.
- For URLs: include only explicit URLs present in the answer text (plain, markdown, or otherwise). If none, return an empty array.
- Preserve formatting such as en-dashes or “to” in vote breakdown.
- Keep country names as they appear (e.g., "United States", "U.S.", "USA" – do not normalize).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls(sources: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (sources or []) if _nonempty_str(u)]


# --------------------------------------------------------------------------- #
# Verification builders per domain                                            #
# --------------------------------------------------------------------------- #
async def verify_monetary_policy(evaluator: Evaluator, root):
    info: MonetaryPolicyInfo = evaluator.find_node("mp_extraction_placeholder")  # not used; kept for clarity


async def verify_development_monetary(
    evaluator: Evaluator,
    parent,
    mp: Optional[MonetaryPolicyInfo],
):
    node = evaluator.add_parallel(
        id="Development_1_Monetary_Policy",
        desc="A major monetary policy decision or announcement made by the Federal Reserve",
        parent=parent,
        critical=False,
    )

    # Sources existence (critical precondition for URL-grounded checks)
    sources = _urls(mp.reference_urls if mp else [])
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="MP_URLs_Provided",
        desc="At least one reference URL is provided for the monetary policy development",
        parent=node,
        critical=True,
    )

    # MP_Reference_URL
    mp_ref = evaluator.add_leaf(
        id="MP_Reference_URL",
        desc="A reference URL from a credible news source confirming the monetary policy development",
        parent=node,
        critical=True,
    )
    claim_ref = (
        "This article is from a major credible news outlet and confirms the described Federal Reserve "
        f"monetary policy development (decision: '{(mp.decision_content or '').strip()}')."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=mp_ref,
        sources=sources,
        additional_instruction="Verify that the page is a major news outlet (e.g., AP, Reuters, WSJ, Bloomberg, NYT, WaPo, major TV networks, etc.) and that it reports the specified Fed decision.",
    )

    # MP_Announcement_Date
    mp_date = evaluator.add_leaf(
        id="MP_Announcement_Date",
        desc="The specific date the monetary policy decision was announced",
        parent=node,
        critical=True,
    )
    claim_date = (
        "The article explicitly states that the Federal Reserve's monetary policy decision was announced on "
        f"{mp.announcement_date or 'UNKNOWN DATE'} (allow ±1 day due to time zones)."
    )
    await evaluator.verify(
        claim=claim_date,
        node=mp_date,
        sources=sources,
        additional_instruction="Look for an explicit date of the Fed announcement in the article text or dateline; allow minor timezone/date-line shifts (±1 day).",
    )

    # MP_Date_In_Q1_2026
    mp_q1 = evaluator.add_leaf(
        id="MP_Date_In_Q1_2026",
        desc="The announcement date falls within Jan 1–Mar 31, 2026 (inclusive)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The date '{mp.announcement_date or 'UNKNOWN'}' falls between January 1, 2026 and March 31, 2026 inclusive.",
        node=mp_q1,
        sources=None,
        additional_instruction="Interpret common date formats (e.g., 'Jan 3, 2026', '2026-03-31'). If ambiguous or missing, this should be judged as incorrect.",
    )

    # MP_Decision_Content
    mp_decision = evaluator.add_leaf(
        id="MP_Decision_Content",
        desc="The specific monetary policy decision made (e.g., interest rate action)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article states the specific monetary policy action as: '{mp.decision_content or 'UNKNOWN'}'.",
        node=mp_decision,
        sources=sources,
        additional_instruction="Check that the text clearly identifies the action (e.g., rate hold/cut/hike, QT/QE adjustments, balance sheet moves).",
    )

    # MP_Vote_Breakdown
    mp_vote = evaluator.add_leaf(
        id="MP_Vote_Breakdown",
        desc="The voting breakdown of the Federal Reserve committee",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article includes the voting breakdown: '{mp.vote_breakdown or 'UNKNOWN'}'.",
        node=mp_vote,
        sources=sources,
        additional_instruction="Allow equivalent formatting (e.g., '9–1', '9-1', or 'nine to one'); verify the same meaning.",
    )

    # MP_Decision_Maker
    mp_maker = evaluator.add_leaf(
        id="MP_Decision_Maker",
        desc="The primary Federal Reserve official who announced the decision",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article identifies '{mp.decision_maker or 'UNKNOWN'}' as the primary Federal Reserve official who announced the decision.",
        node=mp_maker,
        sources=sources,
        additional_instruction="Look for explicit attribution (e.g., Chair Jerome Powell). Allow minor naming variations.",
    )


async def verify_development_diplomacy(
    evaluator: Evaluator,
    parent,
    di: Optional[InternationalDiplomacyInfo],
):
    node = evaluator.add_parallel(
        id="Development_2_International_Diplomacy",
        desc="A major international diplomatic agreement or policy change involving the United States",
        parent=parent,
        critical=False,
    )

    sources = _urls(di.reference_urls if di else [])
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="ID_URLs_Provided",
        desc="At least one reference URL is provided for the diplomatic development",
        parent=node,
        critical=True,
    )

    # ID_Reference_URL
    id_ref = evaluator.add_leaf(
        id="ID_Reference_URL",
        desc="A reference URL from a credible news source confirming the diplomatic development",
        parent=node,
        critical=True,
    )
    countries_str = ", ".join(di.countries_involved) if di and di.countries_involved else "UNKNOWN COUNTRIES"
    claim_ref = (
        f"This article is from a major credible news outlet and confirms a bilateral diplomatic development involving "
        f"{countries_str} (including the United States), described as '{(di.nature_agreement or '').strip()}'."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=id_ref,
        sources=sources,
        additional_instruction="Confirm the article is from a major news source and specifically covers the described bilateral change or agreement involving the U.S.",
    )

    # ID_Announcement_Date
    id_date = evaluator.add_leaf(
        id="ID_Announcement_Date",
        desc="The specific date the diplomatic development was announced",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article explicitly states the announcement date as {di.announcement_date or 'UNKNOWN'} (allow ±1 day).",
        node=id_date,
        sources=sources,
        additional_instruction="Look for explicit mention of the announcement date in text or dateline.",
    )

    # ID_Date_In_Q1_2026
    id_q1 = evaluator.add_leaf(
        id="ID_Date_In_Q1_2026",
        desc="The announcement date falls within Jan 1–Mar 31, 2026 (inclusive)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The date '{di.announcement_date or 'UNKNOWN'}' falls between January 1, 2026 and March 31, 2026 inclusive.",
        node=id_q1,
        sources=None,
        additional_instruction="Interpret common date formats. If missing or outside the window, mark incorrect.",
    )

    # ID_Countries_Involved
    id_countries = evaluator.add_leaf(
        id="ID_Countries_Involved",
        desc="The specific countries involved in the diplomatic development",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article shows that the countries involved are: {countries_str}.",
        node=id_countries,
        sources=sources,
        additional_instruction="Treat 'U.S.', 'United States', and 'USA' as equivalent. Ensure the listed countries match what's reported.",
    )

    # ID_Nature_Agreement
    id_nature = evaluator.add_leaf(
        id="ID_Nature_Agreement",
        desc="The nature of the diplomatic agreement or change (e.g., restoration of relations, new treaty)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article describes the diplomatic change as: '{di.nature_agreement or 'UNKNOWN'}'.",
        node=id_nature,
        sources=sources,
        additional_instruction="Verify the nature/type of change (restoration, normalization, new treaty, sanctions relief, etc.).",
    )

    # ID_Previous_Status
    id_prev = evaluator.add_leaf(
        id="ID_Previous_Status",
        desc="The previous state of relations or policy context",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article provides prior context as: '{di.previous_status or 'UNKNOWN'}'.",
        node=id_prev,
        sources=sources,
        additional_instruction="Look for explicit mention of the prior policy state or relationship status.",
    )


async def verify_development_space(
    evaluator: Evaluator,
    parent,
    se: Optional[SpaceExplorationInfo],
):
    node = evaluator.add_parallel(
        id="Development_3_Space_Exploration",
        desc="A major NASA mission milestone, announcement, or significant space program development",
        parent=parent,
        critical=False,
    )

    sources = _urls(se.reference_urls if se else [])
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="SE_URLs_Provided",
        desc="At least one reference URL is provided for the space exploration development",
        parent=node,
        critical=True,
    )

    # SE_Reference_URL
    se_ref = evaluator.add_leaf(
        id="SE_Reference_URL",
        desc="A reference URL from a credible news source confirming the space exploration development",
        parent=node,
        critical=True,
    )
    claim_ref = (
        f"This article is from a major credible news outlet and confirms a NASA mission/program development named "
        f"'{(se.mission_name or 'UNKNOWN').strip()}'."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=se_ref,
        sources=sources,
        additional_instruction="Confirm the outlet is credible and that it reports on the specified NASA mission/program development.",
    )

    # SE_Event_Date
    se_date = evaluator.add_leaf(
        id="SE_Event_Date",
        desc="The specific date of the space exploration milestone or announcement",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article explicitly states the mission milestone/announcement date as {se.event_date or 'UNKNOWN'} (allow ±1 day).",
        node=se_date,
        sources=sources,
        additional_instruction="Look for explicit date mention in the text or dateline.",
    )

    # SE_Date_In_Q1_2026
    se_q1 = evaluator.add_leaf(
        id="SE_Date_In_Q1_2026",
        desc="The event/announcement date falls within Jan 1–Mar 31, 2026 (inclusive)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The date '{se.event_date or 'UNKNOWN'}' falls between January 1, 2026 and March 31, 2026 inclusive.",
        node=se_q1,
        sources=None,
        additional_instruction="Interpret common date formats. If missing/out-of-range, incorrect.",
    )

    # SE_Mission_Name
    se_name = evaluator.add_leaf(
        id="SE_Mission_Name",
        desc="The official name of the space mission or program",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article states the mission/program name as '{se.mission_name or 'UNKNOWN'}'.",
        node=se_name,
        sources=sources,
        additional_instruction="Verify the official naming as reported.",
    )

    # SE_Mission_Type
    se_type = evaluator.add_leaf(
        id="SE_Mission_Type",
        desc="The type or nature of the mission (e.g., crewed lunar mission, Mars rover, ISS operation)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article describes the mission type as '{se.mission_type or 'UNKNOWN'}'.",
        node=se_type,
        sources=sources,
        additional_instruction="Check that the mission type matches the reporting.",
    )

    # SE_Mission_Characteristics
    se_char = evaluator.add_leaf(
        id="SE_Mission_Characteristics",
        desc="Key characteristics of the mission such as duration, crew size, or objectives",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article reports mission characteristics: '{se.mission_characteristics or 'UNKNOWN'}'.",
        node=se_char,
        sources=sources,
        additional_instruction="Look for details like crew size, duration, objectives, trajectory, etc. Allow paraphrase equivalence.",
    )


async def verify_development_lej(
    evaluator: Evaluator,
    parent,
    lj: Optional[LawEnforcementJusticeInfo],
):
    node = evaluator.add_parallel(
        id="Development_4_Law_Enforcement_Justice",
        desc="A major law enforcement policy change, new DOJ initiative, or significant justice department development",
        parent=parent,
        critical=False,
    )

    sources = _urls(lj.reference_urls if lj else [])
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id="LEJ_URLs_Provided",
        desc="At least one reference URL is provided for the law enforcement/justice development",
        parent=node,
        critical=True,
    )

    # LEJ_Reference_URL
    lej_ref = evaluator.add_leaf(
        id="LEJ_Reference_URL",
        desc="A reference URL from a credible news source confirming the law enforcement/justice development",
        parent=node,
        critical=True,
    )
    claim_ref = (
        "This article is from a major credible news outlet and confirms the Department of Justice policy/initiative or "
        f"organizational development described as '{(lj.policy_nature or 'UNKNOWN').strip()}'."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=lej_ref,
        sources=sources,
        additional_instruction="Confirm the outlet is credible and the article reports on the specified DOJ-related development.",
    )

    # LEJ_Announcement_Date
    lej_date = evaluator.add_leaf(
        id="LEJ_Announcement_Date",
        desc="The specific date the law enforcement or justice policy was announced",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article explicitly states the announcement date as {lj.announcement_date or 'UNKNOWN'} (allow ±1 day).",
        node=lej_date,
        sources=sources,
        additional_instruction="Look for the explicit announcement date.",
    )

    # LEJ_Date_In_Q1_2026
    lej_q1 = evaluator.add_leaf(
        id="LEJ_Date_In_Q1_2026",
        desc="The announcement date falls within Jan 1–Mar 31, 2026 (inclusive)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The date '{lj.announcement_date or 'UNKNOWN'}' falls between January 1, 2026 and March 31, 2026 inclusive.",
        node=lej_q1,
        sources=None,
        additional_instruction="Interpret common date formats. If missing/out-of-range, incorrect.",
    )

    # LEJ_Department_Agency
    lej_dept = evaluator.add_leaf(
        id="LEJ_Department_Agency",
        desc="The specific government department or agency responsible for the policy",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article states that the Department of Justice (or a DOJ component) is the primary responsible body, named as '{lj.department_agency or 'UNKNOWN'}'.",
        node=lej_dept,
        sources=sources,
        additional_instruction="Treat DOJ components (e.g., FBI, ATF, DEA, Antitrust Division) as part of DOJ.",
    )

    # LEJ_Policy_Nature
    lej_policy = evaluator.add_leaf(
        id="LEJ_Policy_Nature",
        desc="The nature of the policy, initiative, or organizational change",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article describes the DOJ policy/initiative/organizational change as '{lj.policy_nature or 'UNKNOWN'}'.",
        node=lej_policy,
        sources=sources,
        additional_instruction="Verify the specific nature/focus area is stated.",
    )

    # LEJ_Primary_Official
    lej_official = evaluator.add_leaf(
        id="LEJ_Primary_Official",
        desc="The primary government official who announced or is responsible for the development",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The article identifies '{lj.primary_official or 'UNKNOWN'}' as the primary official announcing or responsible.",
        node=lej_official,
        sources=sources,
        additional_instruction="Allow reasonable name variants (with/without middle initials).",
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
        strategy=AggregationStrategy.PARALLEL,  # Independent domains
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

    # Extract structured information for all four domains
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllDevelopmentsExtraction,
        extraction_name="all_developments",
    )

    # Build and verify each development domain
    await verify_development_monetary(
        evaluator,
        root,
        extracted.monetary_policy or MonetaryPolicyInfo(),
    )
    await verify_development_diplomacy(
        evaluator,
        root,
        extracted.international_diplomacy or InternationalDiplomacyInfo(),
    )
    await verify_development_space(
        evaluator,
        root,
        extracted.space_exploration or SpaceExplorationInfo(),
    )
    await verify_development_lej(
        evaluator,
        root,
        extracted.law_enforcement_justice or LawEnforcementJusticeInfo(),
    )

    # Return summary with verification tree
    return evaluator.get_summary()