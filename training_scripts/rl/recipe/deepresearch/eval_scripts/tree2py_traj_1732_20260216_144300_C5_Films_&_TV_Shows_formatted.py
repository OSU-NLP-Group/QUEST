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
TASK_ID = "comedy_series_identification"
TASK_DESCRIPTION = """
Identify a comedy television series that satisfies all of the following criteria: 
(1) The series was created by exactly three co-creators working together; 
(2) The series is produced through a production company that is owned or operated by at least two of these co-creators; 
(3) The series was produced by Universal Television; 
(4) The series premiered on HBO Max in 2021; 
(5) The series won the Outstanding Comedy Series award at the 76th Primetime Emmy Awards (held in 2024); 
(6) The lead actress of the series won the Outstanding Lead Actress in a Comedy Series Emmy Award; 
(7) At least one of the three co-creators also has a recurring or main acting role in the series itself; 
(8) At least one of the co-creators previously worked on the Comedy Central series Broad City which aired from 2014 to 2019; 
(9) The co-creators signed an overall deal with Warner Bros. Television Group in 2021. 
Provide the title of the series, the names of the three co-creators, the name of their production company, the premiere date, the name of the lead actress who won the Emmy, the name of the co-creator who also acts in the series, and the name of the co-creator who worked on Broad City. Include URL references for each major criterion.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SeriesExtraction(BaseModel):
    # Core identification
    title: Optional[str] = None

    # Creators
    co_creators: List[str] = Field(default_factory=list)
    creators_urls: List[str] = Field(default_factory=list)

    # Production company owned/operated by at least two co-creators
    production_company: Optional[str] = None
    production_company_urls: List[str] = Field(default_factory=list)

    # Universal Television involvement
    universal_tv_urls: List[str] = Field(default_factory=list)

    # Premiere details
    premiere_platform: Optional[str] = None
    premiere_date: Optional[str] = None
    premiere_urls: List[str] = Field(default_factory=list)

    # Outstanding Comedy Series at 76th Primetime Emmy Awards (2024)
    outstanding_comedy_series_urls: List[str] = Field(default_factory=list)

    # Lead actress Emmy
    lead_actress_name: Optional[str] = None
    lead_actress_urls: List[str] = Field(default_factory=list)

    # Co-creator acting role
    acting_cocreator_name: Optional[str] = None
    acting_cocreator_urls: List[str] = Field(default_factory=list)

    # Broad City connection
    broad_city_cocreator_name: Optional[str] = None
    broad_city_urls: List[str] = Field(default_factory=list)

    # Warner Bros. Television Group overall deal (2021)
    wbtv_deal_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series() -> str:
    return """
    Extract structured information about a single comedy television series described in the answer.
    Return a JSON object with the following fields. If a field is not present in the answer text, set it to null (for strings) or [] (for arrays).

    Required fields:
    - title: The series title.
    - co_creators: An array of exactly the three names credited as creators/co-creators of the series in the answer.
    - creators_urls: An array of URL(s) that explicitly document the three co-creators of the series.
    - production_company: The name of the production company through which the series is produced that is owned or operated by at least two of the co-creators.
    - production_company_urls: URL(s) that explicitly document (1) that the series is produced through this production company and/or (2) that at least two of the listed co-creators own or operate that company. Include all relevant URLs cited in the answer.
    - universal_tv_urls: URL(s) explicitly showing that Universal Television produced the series (e.g., production credits, press releases, trades).
    - premiere_platform: The platform on which the series premiered (e.g., "HBO Max", not the later rebrand "Max" if the premiere was in 2021).
    - premiere_date: The specific date of the series premiere (as written in the answer; do not normalize). If not provided, set null.
    - premiere_urls: URL(s) that document the platform and premiere date/location of the premiere.
    - outstanding_comedy_series_urls: URL(s) that explicitly show the series won Outstanding Comedy Series at the 76th Primetime Emmy Awards (2024 ceremony).
    - lead_actress_name: The name of the lead actress who won the Outstanding Lead Actress in a Comedy Series Emmy Award for this series.
    - lead_actress_urls: URL(s) that explicitly document the lead actress Emmy win for this series.
    - acting_cocreator_name: The name of one co-creator who also has a recurring or main acting role in the series.
    - acting_cocreator_urls: URL(s) that explicitly document that this co-creator acts in the series (recurring or main role).
    - broad_city_cocreator_name: The name of one co-creator who previously worked on Comedy Central's Broad City (2014–2019).
    - broad_city_urls: URL(s) that explicitly document this co-creator’s prior work on Broad City.
    - wbtv_deal_urls: URL(s) that explicitly document the co-creators signing an overall deal with Warner Bros. Television Group in 2021.

    URL rules:
    - Extract only explicit URLs appearing in the answer (plain or markdown). Do not invent URLs.
    - Include full URLs with protocol. Deduplicate.
    - Assign each URL to the most relevant field above; do not mix unrelated URLs.

    If the answer mentions multiple series, extract information for the main series that the answer uses to satisfy all criteria.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def join_names(names: List[str]) -> str:
    clean = [n for n in (names or []) if isinstance(n, str) and n.strip()]
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    return ", ".join(clean[:-1]) + f", and {clean[-1]}"


def unique_urls(*url_lists: List[str]) -> List[str]:
    seq: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst or []:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in seen:
                    seen.add(s)
                    seq.append(s)
    return seq


def series_label(data: SeriesExtraction) -> str:
    return f"'{data.title}'" if data.title else "the series discussed in the answer"


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_creator_team_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="creator_team",
        desc="The series must be created by exactly three co-creators working together",
        parent=parent,
        critical=False
    )

    # References existence (non-critical; used as prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.creators_urls) > 0,
        id="creator_names_reference",
        desc="Provide URL reference documenting the three co-creators",
        parent=node,
        critical=False
    )

    # Names provided (non-critical; used as prerequisite)
    provided = (len(data.co_creators) == 3) and all(isinstance(n, str) and n.strip() for n in data.co_creators)
    provided_node = evaluator.add_custom_node(
        result=provided,
        id="creator_names_provided",
        desc="Provide the names of all three co-creators",
        parent=node,
        critical=False
    )

    # Critical verification: exactly three co-creators
    leaf = evaluator.add_leaf(
        id="three_cocreators",
        desc="Verify that the series has exactly three credited co-creators",
        parent=node,
        critical=True
    )
    creators_str = join_names(data.co_creators)
    claim = f"The series {series_label(data)} was created by exactly three co-creators: {creators_str}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.creators_urls,
        additional_instruction="Confirm that exactly three creators are credited for the series. Allow minor name formatting variations. The page(s) should explicitly list three creators matching or equivalent to the provided names.",
        extra_prerequisites=[ref_node, provided_node]
    )


async def build_production_company_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="production_company",
        desc="The series must be produced through a production company owned by at least two of the co-creators",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.production_company_urls) > 0,
        id="production_company_reference",
        desc="Provide URL reference documenting the production company and its ownership",
        parent=node,
        critical=False
    )

    # Company name provided (non-critical; prerequisite)
    name_ok = isinstance(data.production_company, str) and data.production_company.strip() != ""
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id="company_name_provided",
        desc="Provide the name of the production company",
        parent=node,
        critical=False
    )

    # Critical verification: owned/operated by at least two co-creators and used for series
    leaf = evaluator.add_leaf(
        id="creator_owned_company",
        desc="Verify that at least two co-creators own or operate the production company",
        parent=node,
        critical=True
    )
    creators_str = join_names(data.co_creators)
    company = data.production_company or "the production company"
    claim = (
        f"The series {series_label(data)} is produced through {company}, and at least two of its co-creators "
        f"({creators_str}) own or operate this company."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.production_company_urls,
        additional_instruction="Check that the cited sources explicitly indicate (1) the series is produced through the named company, and (2) at least two of the co-creators own or operate that company. Trades, official bios, or company pages are acceptable. Minor name variants are okay.",
        extra_prerequisites=[ref_node, name_node]
    )


async def build_studio_partnership_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="studio_partnership",
        desc="The series must be produced by Universal Television",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.universal_tv_urls) > 0,
        id="studio_reference",
        desc="Provide URL reference documenting Universal Television's involvement",
        parent=node,
        critical=False
    )

    # Critical verification: Universal Television producer
    leaf = evaluator.add_leaf(
        id="universal_tv_producer",
        desc="Verify that Universal Television is listed as the production studio",
        parent=node,
        critical=True
    )
    claim = f"Universal Television is credited as a production studio for {series_label(data)}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.universal_tv_urls,
        additional_instruction="Confirm that Universal Television (not UCP or other Universal labels) is credited for the series in production credits or official materials.",
        extra_prerequisites=[ref_node]
    )


async def build_premiere_details_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="premiere_details",
        desc="The series must have premiered on HBO Max in 2021",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.premiere_urls) > 0,
        id="premiere_reference",
        desc="Provide URL reference documenting the premiere details",
        parent=node,
        critical=False
    )

    # Provide premiere date (non-critical info)
    date_ok = isinstance(data.premiere_date, str) and data.premiere_date.strip() != ""
    evaluator.add_custom_node(
        result=date_ok,
        id="premiere_date_provided",
        desc="Provide the specific premiere date",
        parent=node,
        critical=False
    )

    # Critical: Premiered on HBO Max
    leaf_platform = evaluator.add_leaf(
        id="hbo_max_platform",
        desc="Verify that the series premiered on HBO Max",
        parent=node,
        critical=True
    )
    claim_platform = f"The series {series_label(data)} premiered on HBO Max."
    await evaluator.verify(
        claim=claim_platform,
        node=leaf_platform,
        sources=data.premiere_urls,
        additional_instruction="Verify that the platform at the time of premiere (in 2021) is 'HBO Max' (not the later rebrand 'Max').",
        extra_prerequisites=[ref_node]
    )

    # Critical: Premiered in 2021
    leaf_year = evaluator.add_leaf(
        id="year_2021_premiere",
        desc="Verify that the series premiered in 2021",
        parent=node,
        critical=True
    )
    claim_year = f"The series {series_label(data)} premiered in 2021."
    await evaluator.verify(
        claim=claim_year,
        node=leaf_year,
        sources=data.premiere_urls,
        additional_instruction="Confirm the initial premiere year is 2021. If a specific date is given, it must fall in 2021.",
        extra_prerequisites=[ref_node]
    )


async def build_emmy_comedy_series_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="emmy_comedy_series_win",
        desc="The series must have won Outstanding Comedy Series at the 76th Primetime Emmy Awards",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.outstanding_comedy_series_urls) > 0,
        id="comedy_series_emmy_reference",
        desc="Provide URL reference documenting the Outstanding Comedy Series Emmy win",
        parent=node,
        critical=False
    )

    # Critical: won Outstanding Comedy Series
    leaf_win = evaluator.add_leaf(
        id="comedy_series_win",
        desc="Verify that the series won the Outstanding Comedy Series Emmy",
        parent=node,
        critical=True
    )
    claim_win = f"The series {series_label(data)} won the Outstanding Comedy Series award."
    await evaluator.verify(
        claim=claim_win,
        node=leaf_win,
        sources=data.outstanding_comedy_series_urls,
        additional_instruction="Confirm a 'win' (not just a nomination) for Outstanding Comedy Series.",
        extra_prerequisites=[ref_node]
    )

    # Critical: at the 76th (2024 ceremony)
    leaf_76th = evaluator.add_leaf(
        id="76th_emmy_awards",
        desc="Verify that the win occurred at the 76th Primetime Emmy Awards (2024 ceremony)",
        parent=node,
        critical=True
    )
    claim_76th = f"The Outstanding Comedy Series win for {series_label(data)} occurred at the 76th Primetime Emmy Awards in 2024."
    await evaluator.verify(
        claim=claim_76th,
        node=leaf_76th,
        sources=data.outstanding_comedy_series_urls,
        additional_instruction="Confirm that the cited win is explicitly tied to the 76th Primetime Emmy Awards (held in 2024).",
        extra_prerequisites=[ref_node]
    )


async def build_lead_actress_emmy_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="lead_actress_emmy",
        desc="The lead actress of the series must have won Outstanding Lead Actress in a Comedy Series Emmy",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.lead_actress_urls) > 0,
        id="lead_actress_emmy_reference",
        desc="Provide URL reference documenting the lead actress Emmy win",
        parent=node,
        critical=False
    )

    # Actress name provided (non-critical; prerequisite)
    name_ok = isinstance(data.lead_actress_name, str) and data.lead_actress_name.strip() != ""
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id="actress_name_provided",
        desc="Provide the name of the lead actress who won",
        parent=node,
        critical=False
    )

    # Critical: Lead actress won Outstanding Lead Actress in a Comedy Series
    leaf = evaluator.add_leaf(
        id="lead_actress_win",
        desc="Verify that the series' lead actress won the Outstanding Lead Actress in a Comedy Series Emmy",
        parent=node,
        critical=True
    )
    actress = data.lead_actress_name or "the lead actress"
    claim = f"{actress} won the Outstanding Lead Actress in a Comedy Series Emmy Award for {series_label(data)}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.lead_actress_urls,
        additional_instruction="Confirm that the named actress won (not just nominated) the Outstanding Lead Actress in a Comedy Series for this series. Minor name variants acceptable.",
        extra_prerequisites=[ref_node, name_node]
    )


async def build_creator_acting_role_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="creator_acting_role",
        desc="At least one of the three co-creators must also have a recurring or main acting role in the series",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.acting_cocreator_urls) > 0,
        id="creator_acting_reference",
        desc="Provide URL reference documenting the co-creator's acting role",
        parent=node,
        critical=False
    )

    # Acting co-creator name provided (non-critical; prerequisite)
    name_ok = isinstance(data.acting_cocreator_name, str) and data.acting_cocreator_name.strip() != ""
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id="acting_creator_name",
        desc="Provide the name of the co-creator who also acts in the series",
        parent=node,
        critical=False
    )

    # Critical: at least one co-creator acts in the series
    leaf = evaluator.add_leaf(
        id="creator_acts_in_series",
        desc="Verify that at least one co-creator also acts in the series",
        parent=node,
        critical=True
    )
    creators_str = join_names(data.co_creators)
    actor_creator = data.acting_cocreator_name or "one of the co-creators"
    claim = (
        f"At least one of the co-creators ({creators_str}) also has a recurring or main acting role in {series_label(data)}, "
        f"specifically {actor_creator}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.acting_cocreator_urls,
        additional_instruction="Confirm that the named co-creator appears in the series in a recurring or main capacity (not just a cameo).",
        extra_prerequisites=[ref_node, name_node]
    )


async def build_broad_city_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="broad_city_connection",
        desc="At least one co-creator must have previously worked on Comedy Central's Broad City (2014-2019)",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.broad_city_urls) > 0,
        id="broad_city_reference",
        desc="Provide URL reference documenting the co-creator's work on Broad City",
        parent=node,
        critical=False
    )

    # Broad City creator name provided (non-critical; prerequisite)
    name_ok = isinstance(data.broad_city_cocreator_name, str) and data.broad_city_cocreator_name.strip() != ""
    name_node = evaluator.add_custom_node(
        result=name_ok,
        id="broad_city_creator_name",
        desc="Provide the name of the co-creator who worked on Broad City",
        parent=node,
        critical=False
    )

    # Critical: a co-creator worked on Broad City
    leaf = evaluator.add_leaf(
        id="broad_city_prior_work",
        desc="Verify that at least one co-creator worked on Broad City",
        parent=node,
        critical=True
    )
    creators_str = join_names(data.co_creators)
    bc_name = data.broad_city_cocreator_name or "one of the co-creators"
    claim = (
        f"{bc_name}, one of the co-creators of {series_label(data)}, previously worked on Comedy Central's Broad City (2014–2019)."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.broad_city_urls,
        additional_instruction="Confirm that the cited person is both a co-creator of the series and has a credited role on Broad City (any season 2014–2019).",
        extra_prerequisites=[ref_node, name_node]
    )


async def build_wbtv_deal_nodes(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="warner_bros_deal",
        desc="The co-creators must have signed an overall deal with Warner Bros. Television Group in 2021",
        parent=parent,
        critical=False
    )

    # Reference existence (non-critical; prerequisite)
    ref_node = evaluator.add_custom_node(
        result=len(data.wbtv_deal_urls) > 0,
        id="wbtv_deal_reference",
        desc="Provide URL reference documenting the Warner Bros. Television Group deal signed in 2021",
        parent=node,
        critical=False
    )

    # Critical: overall deal in 2021
    leaf = evaluator.add_leaf(
        id="wbtv_deal_2021",
        desc="Verify that the co-creators signed an overall deal with Warner Bros. Television Group in 2021",
        parent=node,
        critical=True
    )
    creators_str = join_names(data.co_creators)
    claim = f"The co-creators of {series_label(data)} ({creators_str}) signed an overall deal with Warner Bros. Television Group in 2021."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.wbtv_deal_urls,
        additional_instruction="Confirm that the overall deal with Warner Bros. Television Group was signed in 2021 and involves the co-creators.",
        extra_prerequisites=[ref_node]
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
    Evaluate an answer for the comedy series identification task.
    """
    # Initialize evaluator with a parallel root; keep aggregators non-critical to avoid critical-child constraint
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

    # Extract structured series info
    data: SeriesExtraction = await evaluator.extract(
        prompt=prompt_extract_series(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # Build top-level identification node (non-critical aggregator)
    top = evaluator.add_parallel(
        id="series_identification",
        desc="Identify a comedy series that satisfies all specified production, creative, and award criteria",
        parent=root,
        critical=False
    )

    # Build each criterion subtree
    await build_creator_team_nodes(evaluator, top, data)
    await build_production_company_nodes(evaluator, top, data)
    await build_studio_partnership_nodes(evaluator, top, data)
    await build_premiere_details_nodes(evaluator, top, data)
    await build_emmy_comedy_series_nodes(evaluator, top, data)
    await build_lead_actress_emmy_nodes(evaluator, top, data)
    await build_creator_acting_role_nodes(evaluator, top, data)
    await build_broad_city_nodes(evaluator, top, data)
    await build_wbtv_deal_nodes(evaluator, top, data)

    # Return evaluation summary
    return evaluator.get_summary()