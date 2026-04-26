import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_governors_2026_reelection"
TASK_DESCRIPTION = """Identify two US governors who are running for reelection in the 2026 gubernatorial elections. The first governor must have a military service background and must have been the Democratic vice presidential nominee in 2024. The second governor must be a billionaire. For each governor, provide detailed attributes and a credible reference URL."""


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class GovernorMilitaryVP(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    running_for_reelection_2026: Optional[str] = None
    military_branch: Optional[str] = None
    highest_military_rank: Optional[str] = None
    years_of_military_service: Optional[str] = None
    vp_2024_nominee_confirmation: Optional[str] = None
    year_first_became_governor: Optional[str] = None
    term_number_seeking_2026: Optional[str] = None
    campaign_announcement_date: Optional[str] = None
    election_date_2026: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class GovernorBillionaire(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    running_for_reelection_2026: Optional[str] = None
    estimated_net_worth_in_billions: Optional[str] = None
    primary_source_of_wealth: Optional[str] = None
    year_first_became_governor: Optional[str] = None
    term_number_seeking_2026: Optional[str] = None
    campaign_announcement_date: Optional[str] = None
    election_date_2026: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class GovernorsExtraction(BaseModel):
    military_vp_governor: Optional[GovernorMilitaryVP] = None
    billionaire_governor: Optional[GovernorBillionaire] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_governors() -> str:
    return """
    Extract information for exactly two US governors running for reelection in 2026:
    1) A governor with military service background who was the 2024 Democratic vice presidential nominee.
    2) A governor who is a billionaire.

    For the military + 2024 VP nominee governor, extract:
    - name
    - state
    - running_for_reelection_2026 (text confirming they are running in 2026)
    - military_branch
    - highest_military_rank
    - years_of_military_service (number or textual description)
    - vp_2024_nominee_confirmation (text confirming 2024 Democratic VP nominee)
    - year_first_became_governor
    - term_number_seeking_2026 (e.g., "2nd term", "third term")
    - campaign_announcement_date (the date their 2026 reelection campaign was announced)
    - election_date_2026 (the claimed date for the 2026 gubernatorial election in their state)
    - reference_urls (list of URLs cited in the answer supporting the claims; only include explicit URLs)

    For the billionaire governor, extract:
    - name
    - state
    - running_for_reelection_2026 (text confirming they are running in 2026)
    - estimated_net_worth_in_billions (as a text/number string, e.g., "3.5", "$3.5B", "about 4 billion")
    - primary_source_of_wealth
    - year_first_became_governor
    - term_number_seeking_2026
    - campaign_announcement_date
    - election_date_2026
    - reference_urls (list of URLs cited in the answer supporting the claims; only include explicit URLs)

    Return a JSON with:
    {
      "military_vp_governor": { ...fields above... },
      "billionaire_governor": { ...fields above... }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - If any field is missing, set it to null (or [] for lists).
    - For URLs, include only valid explicit URLs provided in the answer (plain or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_billion_value(text: Optional[str]) -> Optional[float]:
    """
    Parse a string describing a net worth into a numeric value in billions.
    Examples handled:
      "3.5", "3.5B", "$3.5B", "about 4 billion", "1,200 million", "$1,500,000,000"
    Returns float in billions, or None if cannot parse.
    """
    if not text:
        return None
    s = text.strip().lower()
    s = s.replace(",", "").replace("usd", "").replace("us$", "").replace("$", "").strip()

    # Units detection
    unit_multiplier = 1.0  # default billions
    if "trillion" in s or "t" in s and s.endswith("t"):
        unit_multiplier = 1000.0
    elif "billion" in s or s.endswith("b"):
        unit_multiplier = 1.0
    elif "million" in s or s.endswith("m"):
        unit_multiplier = 0.001
    else:
        # If string contains very large plain number, infer units:
        # If >= 1e9 treat as dollars, convert to billions
        digits = re.findall(r"[-+]?\d*\.?\d+", s)
        if digits:
            try:
                val = float(digits[0])
                if val >= 1_000_000_000:
                    return val / 1_000_000_000.0
                elif val >= 1_000_000:
                    return val / 1_000_000_000.0  # assume dollars
                else:
                    return val  # likely already billions
            except:
                pass

    # Extract first number
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None
    try:
        num = float(m.group(1))
        # If suffix letter exists, adjust multiplier
        if s.endswith("b"):
            unit_multiplier = 1.0
        elif s.endswith("m"):
            unit_multiplier = 0.001
        elif s.endswith("t"):
            unit_multiplier = 1000.0
        return num * unit_multiplier
    except:
        return None


def non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_military_vp_governor(evaluator: Evaluator, parent_node, gov: Optional[GovernorMilitaryVP]) -> None:
    """
    Build verification subtree for the governor with military background and 2024 Democratic VP nominee status.
    All children under this subtree are critical to satisfy the rubric.
    """
    node = evaluator.add_parallel(
        id="gov_military_vp",
        desc="Governor #1: has military service background, was the 2024 Democratic VP nominee, and is running for reelection in 2026.",
        parent=parent_node,
        critical=True
    )

    # Basic existence/inputs
    name_exists = evaluator.add_custom_node(
        result=bool(gov and gov.name and gov.name.strip()),
        id="gov1_name_exists",
        desc="Provide the governor's name (identity).",
        parent=node,
        critical=True,
    )
    refs = non_empty_urls(gov.reference_urls if gov else None)
    refs_exist = evaluator.add_custom_node(
        result=bool(refs),
        id="gov1_reference_url_exists",
        desc="Reference URL is provided (at least one credible source URL).",
        parent=node,
        critical=True
    )

    name = gov.name if gov and gov.name else ""
    state = gov.state if gov and gov.state else ""

    # Governor and state verification
    leaf = evaluator.add_leaf(
        id="gov1_is_governor_state",
        desc="State which US state they serve as governor (must be a sitting US governor of that state).",
        parent=node,
        critical=True,
    )
    claim = f"{name} is the current governor of {state}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Verify that the person is the sitting governor of the specified US state. Allow minor naming variations (e.g., middle initials)."
    )

    # Running for reelection in 2026
    leaf = evaluator.add_leaf(
        id="gov1_running_2026",
        desc="Confirm the governor is running for reelection in the 2026 gubernatorial election.",
        parent=node,
        critical=True,
    )
    claim = f"{name} is running for reelection in the 2026 gubernatorial election."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Look for explicit announcements, filings, or credible reporting that the governor is running in 2026."
    )

    # Military branch
    branch = gov.military_branch if gov and gov.military_branch else ""
    leaf = evaluator.add_leaf(
        id="gov1_military_branch",
        desc="Specify the branch of military service.",
        parent=node,
        critical=True,
    )
    claim = f"{name} served in the {branch}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Accept closely equivalent phrasing (e.g., 'Army National Guard' vs. 'National Guard (Army)')."
    )

    # Highest rank
    rank = gov.highest_military_rank if gov and gov.highest_military_rank else ""
    leaf = evaluator.add_leaf(
        id="gov1_highest_rank",
        desc="Specify the highest military rank achieved.",
        parent=node,
        critical=True,
    )
    claim = f"{name}'s highest military rank was {rank}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Allow minor variants like abbreviations (e.g., 'SFC' vs 'Sergeant First Class')."
    )

    # Years of service
    years = gov.years_of_military_service if gov and gov.years_of_military_service else ""
    leaf = evaluator.add_leaf(
        id="gov1_years_service",
        desc="Provide the number of years served in the military.",
        parent=node,
        critical=True,
    )
    claim = f"{name} served in the military for {years} years."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Allow reasonable approximations (e.g., 'about 12 years', '1996–2005 ~ 9 years')."
    )

    # 2024 Democratic VP nominee confirmation
    vp_conf = gov.vp_2024_nominee_confirmation if gov and gov.vp_2024_nominee_confirmation else ""
    leaf = evaluator.add_leaf(
        id="gov1_vp_2024_nominee",
        desc="Confirm they were the 2024 Democratic vice presidential nominee.",
        parent=node,
        critical=True,
    )
    claim = f"{name} was the 2024 Democratic vice presidential nominee."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Verify explicit phrasing that the person was the 2024 Democratic VP nominee."
    )

    # Year first became governor
    first_year = gov.year_first_became_governor if gov and gov.year_first_became_governor else ""
    leaf = evaluator.add_leaf(
        id="gov1_first_governor_year",
        desc="Provide the year they first became governor.",
        parent=node,
        critical=True,
    )
    claim = f"{name} first became governor in {first_year}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="The page should indicate the initial term start year or the year first elected/sworn in."
    )

    # Term number seeking in 2026
    term_seek = gov.term_number_seeking_2026 if gov and gov.term_number_seeking_2026 else ""
    leaf = evaluator.add_leaf(
        id="gov1_term_number_2026",
        desc="State which term number they are seeking in 2026 (e.g., 2nd term, 3rd term).",
        parent=node,
        critical=True,
    )
    claim = f"In 2026, {name} is seeking their {term_seek} term as governor."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Accept ordinal variations like '2nd' vs 'second'."
    )

    # Campaign announcement date
    ann_date = gov.campaign_announcement_date if gov and gov.campaign_announcement_date else ""
    leaf = evaluator.add_leaf(
        id="gov1_campaign_announce_date",
        desc="Provide the date their reelection campaign was announced.",
        parent=node,
        critical=True,
    )
    claim = f"On {ann_date}, {name} announced their 2026 reelection campaign."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="The source should clearly indicate a reelection announcement or filing date for the 2026 race."
    )

    # 2026 election date (must be November 3, 2026)
    election_date = gov.election_date_2026 if gov and gov.election_date_2026 else ""
    leaf = evaluator.add_leaf(
        id="gov1_election_date_2026",
        desc="Provide the election date for their state's 2026 gubernatorial election; must be November 3, 2026 (per constraint).",
        parent=node,
        critical=True,
    )
    claim = f"The 2026 gubernatorial election in {state} will be held on November 3, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Verify the scheduled 2026 general election date for the state's gubernatorial race is Nov 3, 2026. Allow minor formatting differences like 'Nov. 3, 2026' or 'Tuesday, November 3, 2026'."
    )


async def verify_billionaire_governor(evaluator: Evaluator, parent_node, gov: Optional[GovernorBillionaire]) -> None:
    """
    Build verification subtree for the billionaire governor running in 2026.
    All children under this subtree are critical to satisfy the rubric.
    """
    node = evaluator.add_parallel(
        id="gov_billionaire",
        desc="Governor #2: is a billionaire and is running for reelection in the 2026 gubernatorial election.",
        parent=parent_node,
        critical=True
    )

    # Basic existence/inputs
    name_exists = evaluator.add_custom_node(
        result=bool(gov and gov.name and gov.name.strip()),
        id="gov2_name_exists",
        desc="Provide the governor's name (identity).",
        parent=node,
        critical=True,
    )
    refs = non_empty_urls(gov.reference_urls if gov else None)
    refs_exist = evaluator.add_custom_node(
        result=bool(refs),
        id="gov2_reference_url_exists",
        desc="Reference URL is provided (at least one credible source URL).",
        parent=node,
        critical=True
    )

    name = gov.name if gov and gov.name else ""
    state = gov.state if gov and gov.state else ""

    # Governor and state verification
    leaf = evaluator.add_leaf(
        id="gov2_is_governor_state",
        desc="State which US state they serve as governor (must be a sitting US governor of that state).",
        parent=node,
        critical=True,
    )
    claim = f"{name} is the current governor of {state}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Verify that the person is the sitting governor of the specified US state. Allow minor naming variations."
    )

    # Running for reelection in 2026
    leaf = evaluator.add_leaf(
        id="gov2_running_2026",
        desc="Confirm the governor is running for reelection in the 2026 gubernatorial election.",
        parent=node,
        critical=True,
    )
    claim = f"{name} is running for reelection in the 2026 gubernatorial election."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Look for explicit announcements, filings, or credible reporting that the governor is running in 2026."
    )

    # Estimated net worth in billions (supported by sources)
    networth_text = gov.estimated_net_worth_in_billions if gov and gov.estimated_net_worth_in_billions else ""
    leaf = evaluator.add_leaf(
        id="gov2_net_worth_estimate",
        desc="Provide the estimated net worth in billions of dollars.",
        parent=node,
        critical=True,
    )
    claim = f"{name}'s estimated net worth is approximately {networth_text} billion dollars."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Allow reasonable approximations and rounding. If the source states a range near the claimed value, consider it supported."
    )

    # Billionaire status verification (>= $1B)
    networth_bil = parse_billion_value(networth_text)
    billionaire_ok = networth_bil is not None and networth_bil >= 1.0
    evaluator.add_custom_node(
        result=billionaire_ok,
        id="gov2_billionaire_status_check",
        desc="Verify billionaire status: the estimated net worth must be at least $1 billion.",
        parent=node,
        critical=True
    )

    # Primary source of wealth
    wealth_source = gov.primary_source_of_wealth if gov and gov.primary_source_of_wealth else ""
    leaf = evaluator.add_leaf(
        id="gov2_primary_wealth_source",
        desc="Specify the primary source of wealth.",
        parent=node,
        critical=True,
    )
    claim = f"{name}'s primary source of wealth is {wealth_source}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Accept synonymous descriptions (e.g., 'technology entrepreneurship' vs 'tech company ownership')."
    )

    # Year first became governor
    first_year = gov.year_first_became_governor if gov and gov.year_first_became_governor else ""
    leaf = evaluator.add_leaf(
        id="gov2_first_governor_year",
        desc="Provide the year they first became governor.",
        parent=node,
        critical=True,
    )
    claim = f"{name} first became governor in {first_year}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="The page should indicate the initial term start year or the year first elected/sworn in."
    )

    # Term number seeking in 2026
    term_seek = gov.term_number_seeking_2026 if gov and gov.term_number_seeking_2026 else ""
    leaf = evaluator.add_leaf(
        id="gov2_term_number_2026",
        desc="State which term number they are seeking in 2026 (e.g., 2nd term, 3rd term).",
        parent=node,
        critical=True,
    )
    claim = f"In 2026, {name} is seeking their {term_seek} term as governor."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Accept ordinal variations like '2nd' vs 'second'."
    )

    # Campaign announcement date
    ann_date = gov.campaign_announcement_date if gov and gov.campaign_announcement_date else ""
    leaf = evaluator.add_leaf(
        id="gov2_campaign_announce_date",
        desc="Provide the date their reelection campaign was announced.",
        parent=node,
        critical=True,
    )
    claim = f"On {ann_date}, {name} announced their 2026 reelection campaign."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="The source should clearly indicate a reelection announcement or filing date for the 2026 race."
    )

    # 2026 election date (must be November 3, 2026)
    election_date = gov.election_date_2026 if gov and gov.election_date_2026 else ""
    leaf = evaluator.add_leaf(
        id="gov2_election_date_2026",
        desc="Provide the election date for their state's 2026 gubernatorial election; must be November 3, 2026 (per constraint).",
        parent=node,
        critical=True,
    )
    claim = f"The 2026 gubernatorial election in {state} will be held on November 3, 2026."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=refs,
        additional_instruction="Verify the scheduled 2026 general election date for the state's gubernatorial race is Nov 3, 2026. Allow minor formatting differences like 'Nov. 3, 2026' or 'Tuesday, November 3, 2026'."
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
    """
    Evaluate an answer for the task of identifying two US governors running for reelection in 2026,
    with specific constraints: (1) military background and 2024 Democratic VP nominee, (2) billionaire.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel per rubric
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_governors(),
        template_class=GovernorsExtraction,
        extraction_name="governors_extraction",
    )

    # Build two critical subtrees corresponding to the rubric’s two governors
    await verify_military_vp_governor(
        evaluator=evaluator,
        parent_node=root,
        gov=extracted.military_vp_governor,
    )

    await verify_billionaire_governor(
        evaluator=evaluator,
        parent_node=root,
        gov=extracted.billionaire_governor,
    )

    # Return standard evaluation summary
    return evaluator.get_summary()