import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sanctuary_uasi_states_2026"
TASK_DESCRIPTION = (
    "In response to the Trump administration's January 2026 announcement to suspend federal funding to sanctuary "
    "jurisdictions beginning February 1, 2026, several states have taken legal action to protect their homeland "
    "security grant funding while maintaining their sanctuary policies. Identify 4 US states that meet ALL of the "
    "following criteria: (1) The state is designated as a sanctuary jurisdiction on the DOJ's list (published or "
    "updated between April and October 2025); (2) The state has at least one urban area eligible for the Urban Area "
    "Security Initiative (UASI) funding in Fiscal Year 2025; (3) The state's attorney general filed a lawsuit against "
    "the Department of Homeland Security (DHS) and/or the Federal Emergency Management Agency (FEMA) challenging grant "
    "terminations or funding restrictions between September 2025 and February 2026; (4) The state is scheduled to hold "
    "a gubernatorial election on November 3, 2026. For each of the 4 states you identify, provide the following "
    "information with supporting URL references: state name, current governor's name and party affiliation, description "
    "of a specific sanctuary policy or executive action in that state, name of at least one UASI-eligible urban area "
    "within the state, name of the state's attorney general, and confirmation that the AG filed a lawsuit against "
    "DHS/FEMA regarding homeland security grants or immigration enforcement funding. All information must be verifiable "
    "through reliable sources dated between April 2025 and February 2026."
)

DATE_RANGE_START = "2025-04-01"
DATE_RANGE_END = "2026-02-28"
DOJ_DATE_RANGE_DESCRIPTION = "between April 1, 2025 and October 31, 2025"
LAWSUIT_DATE_RANGE_DESCRIPTION = "between September 1, 2025 and February 28, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateExtractionItem(BaseModel):
    state_name: Optional[str] = None

    governor_name: Optional[str] = None
    governor_party: Optional[str] = None
    governor_sources: List[str] = Field(default_factory=list)

    sanctuary_policy: Optional[str] = None
    sanctuary_sources: List[str] = Field(default_factory=list)

    doj_sanctuary_sources: List[str] = Field(default_factory=list)

    uasi_urban_areas: List[str] = Field(default_factory=list)
    uasi_sources: List[str] = Field(default_factory=list)

    ag_name: Optional[str] = None
    ag_lawsuit_description: Optional[str] = None
    ag_lawsuit_filed_date: Optional[str] = None
    ag_lawsuit_sources: List[str] = Field(default_factory=list)

    election_sources: List[str] = Field(default_factory=list)


class StatesExtraction(BaseModel):
    states: List[StateExtractionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract exactly the first four distinct US states described in the answer that allegedly meet the task criteria.
    For each of these four states, extract the following fields as a JSON list under key "states":
      - state_name: The US state's name.
      - governor_name: The current governor's name (as stated in the answer).
      - governor_party: The governor's party affiliation (e.g., Democratic, Republican).
      - governor_sources: A list of URLs provided in the answer that support the governor's name and party; use only URLs explicitly present in the answer.
      - sanctuary_policy: A short description of a specific sanctuary policy or executive action in the state.
      - sanctuary_sources: A list of URLs that support this sanctuary policy description; use only URLs explicitly present in the answer.
      - doj_sanctuary_sources: A list of URLs that support the claim that the state appears on a DOJ sanctuary jurisdictions list published or updated between April and October 2025; use only URLs explicitly present in the answer.
      - uasi_urban_areas: A list of the urban area names in that state claimed to be eligible for FY 2025 UASI (take them exactly as written in the answer; if multiple are listed, include them all).
      - uasi_sources: A list of URLs that support the FY 2025 UASI eligibility for at least one listed urban area; use only URLs explicitly present in the answer.
      - ag_name: The state's attorney general name (as stated in the answer).
      - ag_lawsuit_description: A short description of the lawsuit filed against DHS and/or FEMA regarding grant terminations/funding restrictions.
      - ag_lawsuit_filed_date: The filing date text as presented in the answer, if any (keep as a string; do not reformat).
      - ag_lawsuit_sources: A list of URLs that support the lawsuit claim and date range (Sep 2025 to Feb 2026); use only URLs explicitly present in the answer.
      - election_sources: A list of URLs that support that the state is scheduled to hold a gubernatorial election on Nov 3, 2026; use only URLs explicitly present in the answer.

    Rules:
    - Use only URLs explicitly mentioned in the answer (including markdown links).
    - If a field is missing, set it to null; if a URL list is missing, return an empty list.
    - Do not invent or infer URLs.
    - If the answer lists more than four states, extract only the first four.
    - Ensure the 'uasi_urban_areas' list contains the names as given in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def collect_all_sources(item: StateExtractionItem) -> List[str]:
    urls: List[str] = []
    urls.extend(item.governor_sources or [])
    urls.extend(item.sanctuary_sources or [])
    urls.extend(item.doj_sanctuary_sources or [])
    urls.extend(item.uasi_sources or [])
    urls.extend(item.ag_lawsuit_sources or [])
    urls.extend(item.election_sources or [])
    # Deduplicate preserving order
    seen: Set[str] = set()
    deduped = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def safe_first(lst: List[str]) -> str:
    return lst[0] if lst else ""


# --------------------------------------------------------------------------- #
# Verification per state                                                      #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    item: StateExtractionItem,
    index: int,
) -> None:
    """
    Build verification nodes for a single state and run checks.
    """
    # Create a parallel node for this state
    state_node = evaluator.add_parallel(
        id=f"state_{index+1}",
        desc=f"{index+1}st provided state: satisfies all constraints and includes all required details with valid sources."
             if index == 0 else (
                 f"{index+1}nd provided state: satisfies all constraints and includes all required details with valid sources."
                 if index == 1 else (
                     f"{index+1}rd provided state: satisfies all constraints and includes all required details with valid sources."
                     if index == 2 else
                     f"{index+1}th provided state: satisfies all constraints and includes all required details with valid sources."
                 )
             ),
        parent=parent_node,
        critical=False
    )

    state_name = item.state_name or ""

    # 1) State name provided (critical existence)
    evaluator.add_custom_node(
        result=bool(state_name.strip()),
        id=f"state_{index+1}_state_name_provided",
        desc="State name is provided and is a US state.",
        parent=state_node,
        critical=True
    )

    # 2) DOJ sanctuary designation (Apr–Oct 2025)
    doj_node = evaluator.add_leaf(
        id=f"state_{index+1}_doj_sanctuary_designation_apr_oct_2025",
        desc="State is designated as a sanctuary jurisdiction on the DOJ list published/updated between April and October 2025.",
        parent=state_node,
        critical=True
    )
    doj_claim = (
        f"{state_name} appears on a U.S. Department of Justice list of 'sanctuary jurisdictions' that was "
        f"published or updated {DOJ_DATE_RANGE_DESCRIPTION}."
    )
    await evaluator.verify(
        claim=doj_claim,
        node=doj_node,
        sources=item.doj_sanctuary_sources,
        additional_instruction=(
            f"Confirm that the page is explicitly a DOJ (justice.gov) or otherwise authoritative DOJ listing of "
            f"sanctuary jurisdictions. The page's publication or last-updated date must fall {DOJ_DATE_RANGE_DESCRIPTION}. "
            f"If the URL is not DOJ/official or the date is out of range, judge as not supported."
        )
    )

    # 3) Sanctuary policy described
    sanctuary_node = evaluator.add_leaf(
        id=f"state_{index+1}_sanctuary_policy_described",
        desc="A specific sanctuary policy or executive action in the state is described.",
        parent=state_node,
        critical=True
    )
    sanctuary_policy = item.sanctuary_policy or ""
    sanctuary_claim = f"In {state_name}, the following sanctuary policy or executive action exists: {sanctuary_policy}"
    await evaluator.verify(
        claim=sanctuary_claim,
        node=sanctuary_node,
        sources=item.sanctuary_sources,
        additional_instruction=(
            f"Verify that the policy description is accurately reflected by the cited source(s), and that each source "
            f"has a publication or last-updated date between {DATE_RANGE_START} and {DATE_RANGE_END}. "
            f"Prioritize official state or city sites (.gov), legislative documents, or reputable news."
        )
    )

    # 4) UASI FY2025 urban area named and verified
    uasi_node = evaluator.add_leaf(
        id=f"state_{index+1}_uasi_fy2025_urban_area_named_and_verified",
        desc="At least one urban area in the state is named and is verified as UASI-eligible for Fiscal Year 2025.",
        parent=state_node,
        critical=True
    )
    uasi_area = safe_first(item.uasi_urban_areas or [])
    uasi_claim = (
        f"'{uasi_area}' is eligible for FY 2025 Urban Area Security Initiative (UASI) funding and is located in or "
        f"serves {state_name}."
    )
    await evaluator.verify(
        claim=uasi_claim,
        node=uasi_node,
        sources=item.uasi_sources,
        additional_instruction=(
            "Confirm that the page(s) explicitly reference FY 2025 UASI eligibility and list the specified urban area. "
            "Accept DHS/FEMA pages, official state/local emergency management pages, or authoritative summaries. "
            f"The source date should fall between {DATE_RANGE_START} and {DATE_RANGE_END}."
        )
    )

    # 5) AG name provided (critical existence)
    evaluator.add_custom_node(
        result=bool((item.ag_name or "").strip()),
        id=f"state_{index+1}_ag_name_provided",
        desc="State attorney general name is provided.",
        parent=state_node,
        critical=True
    )

    # 6) AG lawsuit vs DHS/FEMA in date window
    ag_lawsuit_node = evaluator.add_leaf(
        id=f"state_{index+1}_ag_lawsuit_dhs_fema_sep2025_feb2026",
        desc="The attorney general filed a lawsuit against DHS and/or FEMA challenging grant terminations/funding restrictions (homeland security grants / immigration enforcement funding) and the filing date is between Sep 2025 and Feb 2026.",
        parent=state_node,
        critical=True
    )
    ag_name = item.ag_name or ""
    ag_lawsuit_claim = (
        f"{ag_name}, the attorney general of {state_name}, filed a lawsuit against DHS and/or FEMA challenging grant "
        f"terminations or funding restrictions related to homeland security grants or immigration enforcement funding. "
        f"The filing (or formal announcement) occurred {LAWSUIT_DATE_RANGE_DESCRIPTION}."
    )
    await evaluator.verify(
        claim=ag_lawsuit_claim,
        node=ag_lawsuit_node,
        sources=item.ag_lawsuit_sources,
        additional_instruction=(
            f"Verify that the described action is an actual lawsuit (not just a statement or letter) against DHS and/or FEMA, "
            f"focused on grant terminations or funding restrictions. Confirm the filing (or official announcement) date is "
            f"{LAWSUIT_DATE_RANGE_DESCRIPTION}. Prefer court dockets, AG press releases, or reputable news coverage."
        )
    )

    # 7) State scheduled gubernatorial election on Nov 3, 2026
    election_node = evaluator.add_leaf(
        id=f"state_{index+1}_gubernatorial_election_nov_3_2026",
        desc="State is scheduled to hold a gubernatorial election on November 3, 2026.",
        parent=state_node,
        critical=True
    )
    election_claim = f"{state_name} is scheduled to hold a gubernatorial election on November 3, 2026."
    await evaluator.verify(
        claim=election_claim,
        node=election_node,
        sources=item.election_sources,
        additional_instruction=(
            f"Confirm the election schedule specifically for governor on Nov 3, 2026. Accept authoritative sources like state "
            f"election calendars or reputable news. The source date should be between {DATE_RANGE_START} and {DATE_RANGE_END}."
        )
    )

    # 8) Current governor and party
    gov_info_node = evaluator.add_leaf(
        id=f"state_{index+1}_governor_name_and_party",
        desc="Current governor name and party affiliation are provided.",
        parent=state_node,
        critical=True
    )
    governor_name = item.governor_name or ""
    governor_party = item.governor_party or ""
    gov_claim = f"The current governor of {state_name} is {governor_name}, a member of the {governor_party} party."
    await evaluator.verify(
        claim=gov_claim,
        node=gov_info_node,
        sources=item.governor_sources,
        additional_instruction=(
            f"Verify the governor's name and party affiliation from the cited page(s). The page date should be between "
            f"{DATE_RANGE_START} and {DATE_RANGE_END}. Prefer official state (.gov) or reputable news sources."
        )
    )

    # 9) Sources overall reliability and date window (split into per-URL checks under a critical parallel node)
    sources_group = evaluator.add_parallel(
        id=f"state_{index+1}_sources_urls_reliable_and_dated",
        desc="Claims for this state are supported with URL references to reliable sources dated between April 2025 and February 2026.",
        parent=state_node,
        critical=True
    )

    all_urls = collect_all_sources(item)
    # Ensure at least one source exists
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"state_{index+1}_sources_present",
        desc="At least one supporting source URL is present for this state.",
        parent=sources_group,
        critical=True
    )

    # Limit number of per-URL checks to avoid excessive calls (e.g., first 10)
    urls_to_check = all_urls[:10]
    per_url_checks = []
    for k, url in enumerate(urls_to_check):
        leaf = evaluator.add_leaf(
            id=f"state_{index+1}_source_{k+1}_dated_reliable",
            desc=f"Source #{k+1} is reliable and dated in range.",
            parent=sources_group,
            critical=True
        )
        claim = (
            f"This source is from a reliable outlet (e.g., government .gov site, DOJ/DHS/FEMA, official state or city site, "
            f"or a reputable mainstream news outlet) and it shows a publication or last updated date between "
            f"{DATE_RANGE_START} and {DATE_RANGE_END}."
        )
        per_url_checks.append((claim, url, leaf, "Evaluate both reliability and date on the page (or its metadata)."))

    if per_url_checks:
        await evaluator.batch_verify(per_url_checks)


# --------------------------------------------------------------------------- #
# Root-level checks                                                           #
# --------------------------------------------------------------------------- #
def check_four_distinct_states(extraction: StatesExtraction) -> bool:
    names = []
    for item in extraction.states[:4]:
        if item and item.state_name:
            names.append(item.state_name.strip())
        else:
            names.append("")
    # Must be exactly four and all non-empty and distinct
    if len(names) != 4:
        return False
    if any(n == "" for n in names):
        return False
    return len(set(names)) == 4


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
    Evaluate an answer for the sanctuary/UASI states task.
    """
    # Initialize evaluator with PARALLEL root aggregation
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

    # Extract up to four states with details
    extraction = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction"
    )

    # Root critical check: Exactly four distinct states provided
    evaluator.add_custom_node(
        result=check_four_distinct_states(extraction),
        id="four_distinct_states_provided",
        desc="Exactly four distinct US states are provided (no duplicates).",
        parent=root,
        critical=True
    )

    # Prepare exactly four items (pad with empty if fewer)
    items: List[StateExtractionItem] = list(extraction.states[:4])
    while len(items) < 4:
        items.append(StateExtractionItem())

    # Build per-state verification trees
    for i in range(4):
        await verify_state(evaluator, root, items[i], i)

    # Return standard evaluation summary
    return evaluator.get_summary()