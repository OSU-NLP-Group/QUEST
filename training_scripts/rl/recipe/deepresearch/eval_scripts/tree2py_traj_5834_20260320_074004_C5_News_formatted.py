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
TASK_ID = "political_figures_research_march_2026"
TASK_DESCRIPTION = """In recent U.S. political news through March 2026, identify the following 4 political figures and provide the requested information about each:

1. Who became the current host of Meet the Press in September 2023? Provide their educational background (specify the college, degree field, graduation year, and honors), when they became an NBC White House correspondent, and what presidential debate they moderated in 2020.

2. Who was the first Cabinet secretary to leave their post in Trump's second term? Provide the date they were fired or left office, and information about their replacement: the replacement's current political position prior to the Cabinet appointment, the effective date for the replacement as DHS Secretary, and the date of the replacement's confirmation hearing.

3. Which Texas congressman was sworn into office on February 2, 2026, after winning a special election runoff on January 31, 2026? Provide their previous government position (including the years served and any historic firsts associated with that role), their law school, and their current electoral situation as of March 2026.

4. Which Alabama Senator delivered the Republican response to the 2024 State of the Union on March 7, 2024? Provide their birth date and the high school from which they graduated, including the graduation year.
"""


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class Figure1Education(BaseModel):
    college: Optional[str] = None
    college_sources: List[str] = Field(default_factory=list)
    degree_field: Optional[str] = None
    degree_sources: List[str] = Field(default_factory=list)
    graduation_year: Optional[str] = None
    graduation_year_sources: List[str] = Field(default_factory=list)
    honors: Optional[str] = None  # e.g., "cum laude"; if explicitly none, capture "none"
    honors_sources: List[str] = Field(default_factory=list)


class Figure1Details(BaseModel):
    name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)
    education: Figure1Education = Field(default_factory=Figure1Education)
    wh_correspondent_date: Optional[str] = None  # date or month/year
    wh_correspondent_sources: List[str] = Field(default_factory=list)
    debate_2020: Optional[str] = None  # which debate moderated in 2020
    debate_sources: List[str] = Field(default_factory=list)


class ReplacementInfo(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)
    prior_position: Optional[str] = None
    prior_position_sources: List[str] = Field(default_factory=list)
    effective_date_dhs: Optional[str] = None
    effective_date_sources: List[str] = Field(default_factory=list)
    confirmation_hearing_date: Optional[str] = None
    confirmation_sources: List[str] = Field(default_factory=list)


class Figure2Details(BaseModel):
    name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)
    departure_date: Optional[str] = None
    departure_sources: List[str] = Field(default_factory=list)
    replacement: ReplacementInfo = Field(default_factory=ReplacementInfo)


class Figure3Details(BaseModel):
    name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)
    previous_position: Optional[str] = None
    previous_years: Optional[str] = None  # e.g., "2019–2023" or "2019-2023"
    prev_sources: List[str] = Field(default_factory=list)
    historic_firsts: Optional[str] = None  # text; if none, capture "none" when explicitly stated
    historic_sources: List[str] = Field(default_factory=list)
    law_school: Optional[str] = None
    law_school_sources: List[str] = Field(default_factory=list)
    current_electoral_situation: Optional[str] = None  # as of March 2026
    electoral_sources: List[str] = Field(default_factory=list)


class Figure4Details(BaseModel):
    name: Optional[str] = None
    identification_sources: List[str] = Field(default_factory=list)
    birth_date: Optional[str] = None
    birth_date_sources: List[str] = Field(default_factory=list)
    high_school: Optional[str] = None
    hs_graduation_year: Optional[str] = None
    high_school_sources: List[str] = Field(default_factory=list)


class PoliticalFiguresExtraction(BaseModel):
    figure1: Figure1Details = Field(default_factory=Figure1Details)
    figure2: Figure2Details = Field(default_factory=Figure2Details)
    figure3: Figure3Details = Field(default_factory=Figure3Details)
    figure4: Figure4Details = Field(default_factory=Figure4Details)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_political_figures() -> str:
    return """
Extract structured information for the 4 requested political figures exactly as they appear in the answer. Strictly follow these rules:
- Do not invent any information.
- Return null for any value the answer does not explicitly state.
- For every factual field, extract URLs that the answer explicitly provides as evidence for that field. Only include valid URLs actually present in the answer text (including markdown links). If no URL is provided for a field, return an empty list for that field's sources.

Return a JSON object with these fields:

figure1:
  name: the person identified as becoming the host of "Meet the Press" in September 2023
  identification_sources: URLs that support they became host in September 2023
  education:
    college
    college_sources
    degree_field
    degree_sources
    graduation_year
    graduation_year_sources
    honors           # e.g., "cum laude"; use "none" only if the answer explicitly states there were no honors
    honors_sources
  wh_correspondent_date          # date or month/year when they became an NBC White House correspondent
  wh_correspondent_sources
  debate_2020                    # the specific 2020 presidential debate they moderated (e.g., "final presidential debate on Oct 22, 2020, Nashville")
  debate_sources

figure2:
  name: the first Cabinet secretary to leave their post in Trump's second term
  identification_sources: URLs that support that this person was the first Cabinet secretary to leave in Trump's second term
  departure_date: the date they were fired or left office
  departure_sources
  replacement:
    name: the DHS Secretary who replaced them (or was named to replace them)
    name_sources
    prior_position: the replacement's political position immediately prior to the Cabinet appointment
    prior_position_sources
    effective_date_dhs: the effective/start date for the replacement as DHS Secretary
    effective_date_sources
    confirmation_hearing_date: the date of the replacement's confirmation hearing
    confirmation_sources

figure3:
  name: the Texas congressman sworn in on Feb 2, 2026 after winning a Jan 31, 2026 special election runoff
  identification_sources: URLs that confirm the runoff on Jan 31, 2026 and swearing-in on Feb 2, 2026
  previous_position: their previous government position (e.g., "Bexar County DA", "Texas Solicitor General")
  previous_years: the years served for that previous position (as expressed in the answer)
  prev_sources
  historic_firsts: any historic firsts for that role if applicable; use "none" only if the answer explicitly states no such firsts
  historic_sources
  law_school
  law_school_sources
  current_electoral_situation: their electoral situation as of March 2026 (e.g., "running in X primary on Mar 5", "unopposed", "filed for reelection")
  electoral_sources

figure4:
  name: the Alabama Senator who delivered the Republican response to the 2024 State of the Union on March 7, 2024
  identification_sources
  birth_date
  birth_date_sources
  high_school
  hs_graduation_year
  high_school_sources
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool([u for u in (urls or []) if isinstance(u, str) and u.strip()])


def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    return [u.strip() for u in (urls or []) if isinstance(u, str) and u.strip()]


def _nonnull_text(text: Optional[str], placeholder: str = "UNKNOWN") -> str:
    return text.strip() if isinstance(text, str) and text.strip() else placeholder


def _add_sources_presence_node(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent_node,
    urls: Optional[List[str]],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=_urls_present(urls),
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_figure1(evaluator: Evaluator, parent) -> None:
    fig_node = evaluator.add_sequential(
        id="Figure_1_Meet_the_Press_Host",
        desc="Person who became the current host of Meet the Press in September 2023; provide requested background details.",
        parent=parent,
        critical=False,
    )

    # Retrieve extracted info
    extraction: PoliticalFiguresExtraction = evaluator.find_node("root")  # just to hint type
    # The actual extraction object is not accessible via tree; we stored it separately when extracted.
    # We'll re-accept it via closure by reading from evaluator._extraction_results. But better pass it in.
    # Instead, we fetch from the latest extraction record:
    # We'll rely on closure by capturing in outer scope (handled in main evaluate function).
    # This function will be redefined inside evaluate_answer with access to 'data'.
    pass  # placeholder to be overwritten inside evaluate_answer


async def verify_figure2(evaluator: Evaluator, parent) -> None:
    pass  # placeholder to be overwritten inside evaluate_answer


async def verify_figure3(evaluator: Evaluator, parent) -> None:
    pass  # placeholder to be overwritten inside evaluate_answer


async def verify_figure4(evaluator: Evaluator, parent) -> None:
    pass  # placeholder to be overwritten inside evaluate_answer


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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Four independent figures
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

    # Extract structured data for all figures
    data: PoliticalFiguresExtraction = await evaluator.extract(
        prompt=prompt_extract_political_figures(),
        template_class=PoliticalFiguresExtraction,
        extraction_name="political_figures",
    )

    # ---- Define inner verification functions with access to 'data' ---- #
    async def _verify_figure1():
        fig_node = evaluator.add_sequential(
            id="Figure_1_Meet_the_Press_Host",
            desc="Person who became the current host of Meet the Press in September 2023; provide requested background details.",
            parent=root,
            critical=False,
        )

        f1 = data.figure1
        name = _nonnull_text(f1.name)
        id_sources = _clean_urls(f1.identification_sources)

        # Host Identification
        _add_sources_presence_node(
            evaluator,
            "f1_ident_sources_present",
            "Sources are provided to support host identification (September 2023).",
            fig_node,
            id_sources,
            critical=True,
        )
        id_leaf = evaluator.add_leaf(
            id="Host_Identification",
            desc="Answer clearly identifies a specific person as the Meet the Press host who assumed the role in September 2023 (accurate).",
            parent=fig_node,
            critical=True,
        )
        id_claim = f"{name} became the host of 'Meet the Press' in September 2023."
        await evaluator.verify(
            claim=id_claim,
            node=id_leaf,
            sources=id_sources,
            additional_instruction="Verify that the cited sources explicitly state that this person became (or was announced as) the host of 'Meet the Press' in September 2023. Minor date phrasing differences (e.g., announcement vs. first episode) are acceptable if clearly September 2023.",
        )

        # Required details (parallel, critical)
        req_node = evaluator.add_parallel(
            id="Host_Required_Details",
            desc="Provide all required education and career details about the host (accurate).",
            parent=fig_node,
            critical=True,
        )

        # Education - college
        college_sources = _clean_urls(f1.education.college_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_edu_college_sources_present",
            "Sources provided for the host's college.",
            req_node,
            college_sources,
            critical=True,
        )
        leaf_college = evaluator.add_leaf(
            id="Education_College",
            desc="States the college the host attended (accurate).",
            parent=req_node,
            critical=True,
        )
        college_claim = f"According to the cited sources, {name} attended or graduated from {_nonnull_text(f1.education.college)}."
        await evaluator.verify(
            claim=college_claim,
            node=leaf_college,
            sources=college_sources,
            additional_instruction="Confirm the college name from bios or official profiles.",
        )

        # Education - degree field
        degree_sources = _clean_urls(f1.education.degree_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_edu_degree_sources_present",
            "Sources provided for the host's degree field/major.",
            req_node,
            degree_sources,
            critical=True,
        )
        leaf_degree = evaluator.add_leaf(
            id="Education_Degree_Field",
            desc="States the host’s degree field/major (accurate).",
            parent=req_node,
            critical=True,
        )
        degree_claim = f"According to the cited sources, {name}'s degree field or major is '{_nonnull_text(f1.education.degree_field)}'."
        await evaluator.verify(
            claim=degree_claim,
            node=leaf_degree,
            sources=degree_sources,
            additional_instruction="Allow reasonable phrasing variants (e.g., 'history' vs. 'American history').",
        )

        # Education - graduation year
        grad_year_sources = _clean_urls(f1.education.graduation_year_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_edu_gradyear_sources_present",
            "Sources provided for the host's graduation year.",
            req_node,
            grad_year_sources,
            critical=True,
        )
        leaf_grad_year = evaluator.add_leaf(
            id="Education_Graduation_Year",
            desc="States the host’s graduation year (accurate).",
            parent=req_node,
            critical=True,
        )
        grad_year_claim = f"According to the cited sources, {name} graduated in {_nonnull_text(f1.education.graduation_year)}."
        await evaluator.verify(
            claim=grad_year_claim,
            node=leaf_grad_year,
            sources=grad_year_sources,
            additional_instruction="Month/day differences are acceptable as long as the year matches.",
        )

        # Education - honors
        honors_sources = _clean_urls(f1.education.honors_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_edu_honors_sources_present",
            "Sources provided for the host's graduation honors (or lack thereof).",
            req_node,
            honors_sources,
            critical=True,
        )
        leaf_honors = evaluator.add_leaf(
            id="Education_Honors",
            desc="States any graduation honors (accurate, or explicitly states none if applicable).",
            parent=req_node,
            critical=True,
        )
        honors_text = _nonnull_text(f1.education.honors)
        honors_claim = (
            f"According to the cited sources, {name} graduated with honors: {honors_text}."
            if honors_text.lower() not in ["none", "no honors", "not applicable", "n/a", "unknown", "unspecified"]
            else f"According to the cited sources, there are no graduation honors listed for {name}."
        )
        await evaluator.verify(
            claim=honors_claim,
            node=leaf_honors,
            sources=honors_sources,
            additional_instruction="If the answer states 'none' or 'no honors', treat this as correct if authoritative bios do not list any honors. Otherwise, verify the specific honor (e.g., 'cum laude').",
        )

        # NBC White House correspondent date
        whc_sources = _clean_urls(f1.wh_correspondent_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_whc_sources_present",
            "Sources provided for when the host became an NBC White House correspondent.",
            req_node,
            whc_sources,
            critical=True,
        )
        leaf_whc = evaluator.add_leaf(
            id="NBC_White_House_Correspondent_Date",
            desc="States when (date or month/year) the host became an NBC White House correspondent (accurate).",
            parent=req_node,
            critical=True,
        )
        whc_claim = f"According to the cited sources, {name} became an NBC News White House correspondent on or in {_nonnull_text(f1.wh_correspondent_date)}."
        await evaluator.verify(
            claim=whc_claim,
            node=leaf_whc,
            sources=whc_sources,
            additional_instruction="Accept month/year equivalence and minor phrasing differences (e.g., 'named' vs. 'became').",
        )

        # Moderated 2020 presidential debate
        debate_sources = _clean_urls(f1.debate_sources)
        _add_sources_presence_node(
            evaluator,
            "f1_debate_sources_present",
            "Sources provided for the specific 2020 presidential debate they moderated.",
            req_node,
            debate_sources,
            critical=True,
        )
        leaf_debate = evaluator.add_leaf(
            id="Moderated_2020_Presidential_Debate",
            desc="Identifies the specific presidential debate moderated in 2020 (which debate; accurate).",
            parent=req_node,
            critical=True,
        )
        debate_claim = f"According to the cited sources, {name} moderated the following 2020 U.S. presidential debate: {_nonnull_text(f1.debate_2020)}."
        await evaluator.verify(
            claim=debate_claim,
            node=leaf_debate,
            sources=debate_sources,
            additional_instruction="Verify the debate identity (e.g., 'final presidential debate on Oct 22, 2020, in Nashville').",
        )

    async def _verify_figure2():
        fig_node = evaluator.add_sequential(
            id="Figure_2_First_Cabinet_Secretary_To_Leave_Trump_Second_Term",
            desc="First Cabinet secretary to leave their post in Trump's second term; provide departure and replacement details requested.",
            parent=root,
            critical=False,
        )

        f2 = data.figure2
        sec_name = _nonnull_text(f2.name)
        id_sources = _clean_urls(f2.identification_sources)

        # Secretary Identification (first to leave)
        _add_sources_presence_node(
            evaluator,
            "f2_ident_sources_present",
            "Sources are provided to support the identification as the first Cabinet secretary to leave in Trump's second term.",
            fig_node,
            id_sources,
            critical=True,
        )
        leaf_id = evaluator.add_leaf(
            id="Secretary_Identification",
            desc="Answer identifies a specific person who was the first Cabinet secretary to leave their post in Trump's second term (accurate).",
            parent=fig_node,
            critical=True,
        )
        id_claim = f"According to the cited sources, {sec_name} was the first Cabinet secretary to leave their post in Donald Trump's second term."
        await evaluator.verify(
            claim=id_claim,
            node=leaf_id,
            sources=id_sources,
            additional_instruction="Confirm that sources explicitly characterize this person as the 'first' Cabinet secretary to leave or be removed/fired in Trump's second term (synonyms acceptable).",
        )

        # Departure Date
        dep_sources = _clean_urls(f2.departure_sources)
        _add_sources_presence_node(
            evaluator,
            "f2_departure_sources_present",
            "Sources provided for the departure/firing date.",
            fig_node,
            dep_sources,
            critical=True,
        )
        leaf_dep = evaluator.add_leaf(
            id="Departure_Date",
            desc="Provides the date the secretary was fired or left office (accurate).",
            parent=fig_node,
            critical=True,
        )
        dep_claim = f"According to the cited sources, {sec_name} left office or was fired on {_nonnull_text(f2.departure_date)}."
        await evaluator.verify(
            claim=dep_claim,
            node=leaf_dep,
            sources=dep_sources,
            additional_instruction="Treat 'left office', 'removed', 'fired', 'resigned' equivalently if they refer to the same exit event/date.",
        )

        # Replacement Details (parallel, critical)
        rep_node = evaluator.add_parallel(
            id="Replacement_Details",
            desc="Provides the requested information about the replacement DHS Secretary (accurate).",
            parent=fig_node,
            critical=True,
        )
        rep = f2.replacement
        rep_name = _nonnull_text(rep.name)

        # Replacement Identity
        rep_name_sources = _clean_urls(rep.name_sources)
        _add_sources_presence_node(
            evaluator,
            "f2_replacement_name_sources_present",
            "Sources provided for the replacement's identity as DHS Secretary.",
            rep_node,
            rep_name_sources,
            critical=True,
        )
        leaf_rep_name = evaluator.add_leaf(
            id="Replacement_Identity",
            desc="Names the replacement (accurate).",
            parent=rep_node,
            critical=True,
        )
        rep_name_claim = f"According to the cited sources, {rep_name} was selected or served as the replacement DHS Secretary after {sec_name}'s departure."
        await evaluator.verify(
            claim=rep_name_claim,
            node=leaf_rep_name,
            sources=rep_name_sources,
            additional_instruction="Source should clearly tie the named individual to filling the DHS Secretary role after the departure.",
        )

        # Replacement Prior Political Position
        rep_prior_sources = _clean_urls(rep.prior_position_sources)
        _add_sources_presence_node(
            evaluator,
            "f2_replacement_prior_sources_present",
            "Sources provided for the replacement’s prior political position.",
            rep_node,
            rep_prior_sources,
            critical=True,
        )
        leaf_prior = evaluator.add_leaf(
            id="Replacement_Prior_Political_Position",
            desc="States the replacement’s political position immediately prior to the Cabinet appointment (accurate).",
            parent=rep_node,
            critical=True,
        )
        prior_claim = f"According to the cited sources, immediately prior to the Cabinet appointment, {rep_name}'s political position was: {_nonnull_text(rep.prior_position)}."
        await evaluator.verify(
            claim=prior_claim,
            node=leaf_prior,
            sources=rep_prior_sources,
            additional_instruction="Look for biographical context in nomination/announcement articles.",
        )

        # Replacement Effective Date as DHS
        rep_eff_sources = _clean_urls(rep.effective_date_sources)
        _add_sources_presence_node(
            evaluator,
            "f2_replacement_effective_sources_present",
            "Sources provided for the effective/start date as DHS Secretary.",
            rep_node,
            rep_eff_sources,
            critical=True,
        )
        leaf_eff = evaluator.add_leaf(
            id="Replacement_Effective_Date_DHS",
            desc="Provides the effective date for the replacement as DHS Secretary (accurate).",
            parent=rep_node,
            critical=True,
        )
        eff_claim = f"According to the cited sources, the effective date for {rep_name} as DHS Secretary was {_nonnull_text(rep.effective_date_dhs)}."
        await evaluator.verify(
            claim=eff_claim,
            node=leaf_eff,
            sources=rep_eff_sources,
            additional_instruction="Accept the date the person was sworn in/assumed duties. If 'acting', ensure the date corresponds to assuming the DHS role.",
        )

        # Replacement Confirmation Hearing Date
        rep_conf_sources = _clean_urls(rep.confirmation_sources)
        _add_sources_presence_node(
            evaluator,
            "f2_replacement_confirmation_sources_present",
            "Sources provided for the replacement’s confirmation hearing date.",
            rep_node,
            rep_conf_sources,
            critical=True,
        )
        leaf_conf = evaluator.add_leaf(
            id="Replacement_Confirmation_Hearing_Date",
            desc="Provides the date of the replacement’s confirmation hearing (accurate).",
            parent=rep_node,
            critical=True,
        )
        conf_claim = f"According to the cited sources, the confirmation hearing date for {rep_name} was {_nonnull_text(rep.confirmation_hearing_date)}."
        await evaluator.verify(
            claim=conf_claim,
            node=leaf_conf,
            sources=rep_conf_sources,
            additional_instruction="Use official committee calendars, reports, or credible news coverage indicating the hearing date.",
        )

    async def _verify_figure3():
        fig_node = evaluator.add_sequential(
            id="Figure_3_Texas_Congressman_Sworn_In_Feb_2_2026",
            desc="Texas congressman sworn in Feb 2, 2026 after winning the Jan 31, 2026 special election runoff; provide requested background and current electoral situation.",
            parent=root,
            critical=False,
        )

        f3 = data.figure3
        name = _nonnull_text(f3.name)
        id_sources = _clean_urls(f3.identification_sources)

        # Congressman Identification (swearing-in and runoff)
        _add_sources_presence_node(
            evaluator,
            "f3_ident_sources_present",
            "Sources are provided for the Jan 31, 2026 runoff win and Feb 2, 2026 swearing-in.",
            fig_node,
            id_sources,
            critical=True,
        )
        leaf_id = evaluator.add_leaf(
            id="Congressman_Identification",
            desc="Answer identifies the specific Texas congressman matching the described swearing-in (Feb 2, 2026) and runoff win (Jan 31, 2026) events (accurate).",
            parent=fig_node,
            critical=True,
        )
        id_claim = f"According to the cited sources, {name} won a special election runoff on January 31, 2026, and was sworn into the U.S. House on February 2, 2026."
        await evaluator.verify(
            claim=id_claim,
            node=leaf_id,
            sources=id_sources,
            additional_instruction="Both the runoff (Jan 31, 2026) and the swearing-in (Feb 2, 2026) must be supported.",
        )

        # Required Details (parallel, critical)
        req_node = evaluator.add_parallel(
            id="Congressman_Required_Details",
            desc="Provides the requested prior role details, education, and current electoral situation as of March 2026 (accurate).",
            parent=fig_node,
            critical=True,
        )

        # Previous Government Position and Years
        prev_sources = _clean_urls(f3.prev_sources)
        _add_sources_presence_node(
            evaluator,
            "f3_prev_sources_present",
            "Sources provided for the previous government position and years served.",
            req_node,
            prev_sources,
            critical=True,
        )
        leaf_prev = evaluator.add_leaf(
            id="Previous_Government_Position_And_Years",
            desc="States the congressman’s previous government position and the years served (accurate).",
            parent=req_node,
            critical=True,
        )
        prev_claim = f"According to the cited sources, {name} previously served as {_nonnull_text(f3.previous_position)} during {_nonnull_text(f3.previous_years)}."
        await evaluator.verify(
            claim=prev_claim,
            node=leaf_prev,
            sources=prev_sources,
            additional_instruction="Ensure both the position and the service years match the sources.",
        )

        # Historic Firsts (if applicable)
        hist_sources = _clean_urls(f3.historic_sources)
        _add_sources_presence_node(
            evaluator,
            "f3_historic_sources_present",
            "Sources provided for any historic firsts (or lack thereof, if explicitly stated).",
            req_node,
            hist_sources,
            critical=True,
        )
        leaf_hist = evaluator.add_leaf(
            id="Historic_Firsts_In_Previous_Role",
            desc="Describes any historic first(s) associated with that previous role, if applicable (accurate).",
            parent=req_node,
            critical=True,
        )
        hist_text = _nonnull_text(f3.historic_firsts)
        hist_claim = (
            f"According to the cited sources, in that prior role {name} achieved the following historic distinction(s): {hist_text}."
            if hist_text.lower() not in ["none", "not applicable", "n/a", "unknown", "unspecified"]
            else f"According to the cited sources, there are no notable historic firsts associated with {name}'s prior role."
        )
        await evaluator.verify(
            claim=hist_claim,
            node=leaf_hist,
            sources=hist_sources,
            additional_instruction="If the answer says 'none' or similar, treat it as correct if reputable sources do not cite any historic 'first' for that role.",
        )

        # Law School
        law_sources = _clean_urls(f3.law_school_sources)
        _add_sources_presence_node(
            evaluator,
            "f3_law_school_sources_present",
            "Sources provided for the law school.",
            req_node,
            law_sources,
            critical=True,
        )
        leaf_law = evaluator.add_leaf(
            id="Law_School",
            desc="States the law school attended/graduated from (accurate).",
            parent=req_node,
            critical=True,
        )
        law_claim = f"According to the cited sources, {name} attended or graduated from {_nonnull_text(f3.law_school)}."
        await evaluator.verify(
            claim=law_claim,
            node=leaf_law,
            sources=law_sources,
            additional_instruction="Confirm the law school name from bios or official records.",
        )

        # Current Electoral Situation as of March 2026
        elec_sources = _clean_urls(f3.electoral_sources)
        _add_sources_presence_node(
            evaluator,
            "f3_electoral_sources_present",
            "Sources provided for the current electoral situation as of March 2026.",
            req_node,
            elec_sources,
            critical=True,
        )
        leaf_elec = evaluator.add_leaf(
            id="Current_Electoral_Situation_As_Of_March_2026",
            desc="Describes the congressman’s electoral situation as of March 2026 (accurate).",
            parent=req_node,
            critical=True,
        )
        elec_claim = f"According to the cited sources, as of March 2026, {name}'s electoral situation is: {_nonnull_text(f3.current_electoral_situation)}."
        await evaluator.verify(
            claim=elec_claim,
            node=leaf_elec,
            sources=elec_sources,
            additional_instruction="Prefer sources dated in March 2026 or clearly describing the status as of March 2026.",
        )

    async def _verify_figure4():
        fig_node = evaluator.add_sequential(
            id="Figure_4_Alabama_Senator_GOP_SOTU_Response_2024",
            desc="Alabama Senator who delivered the Republican response to the 2024 State of the Union on March 7, 2024; provide birth date and high school details.",
            parent=root,
            critical=False,
        )

        f4 = data.figure4
        name = _nonnull_text(f4.name)
        id_sources = _clean_urls(f4.identification_sources)

        # Senator Identification
        _add_sources_presence_node(
            evaluator,
            "f4_ident_sources_present",
            "Sources provided for delivering the Republican response on March 7, 2024.",
            fig_node,
            id_sources,
            critical=True,
        )
        leaf_id = evaluator.add_leaf(
            id="Senator_Identification",
            desc="Answer identifies the Alabama Senator who delivered the Republican response on March 7, 2024 (accurate).",
            parent=fig_node,
            critical=True,
        )
        id_claim = f"According to the cited sources, {name} delivered the Republican response to the 2024 State of the Union on March 7, 2024."
        await evaluator.verify(
            claim=id_claim,
            node=leaf_id,
            sources=id_sources,
            additional_instruction="Verify the date and the role as the official Republican responder to the 2024 SOTU.",
        )

        # Biographical Details (parallel, critical)
        bio_node = evaluator.add_parallel(
            id="Biographical_Details",
            desc="Provides requested biographical details (accurate).",
            parent=fig_node,
            critical=True,
        )

        # Birth date
        bd_sources = _clean_urls(f4.birth_date_sources)
        _add_sources_presence_node(
            evaluator,
            "f4_birthdate_sources_present",
            "Sources provided for the senator’s birth date.",
            bio_node,
            bd_sources,
            critical=True,
        )
        leaf_bd = evaluator.add_leaf(
            id="Birth_Date",
            desc="Provides the senator’s birth date (accurate).",
            parent=bio_node,
            critical=True,
        )
        bd_claim = f"According to the cited sources, {name}'s birth date is {_nonnull_text(f4.birth_date)}."
        await evaluator.verify(
            claim=bd_claim,
            node=leaf_bd,
            sources=bd_sources,
            additional_instruction="Minor formatting differences (e.g., 'Oct. 7, 1981' vs 'October 7, 1981') are acceptable.",
        )

        # High school and graduation year
        hs_sources = _clean_urls(f4.high_school_sources)
        _add_sources_presence_node(
            evaluator,
            "f4_highschool_sources_present",
            "Sources provided for the senator’s high school and graduation year.",
            bio_node,
            hs_sources,
            critical=True,
        )
        leaf_hs = evaluator.add_leaf(
            id="High_School_And_Graduation_Year",
            desc="Names the high school the senator graduated from and includes the graduation year (accurate).",
            parent=bio_node,
            critical=True,
        )
        hs_claim = f"According to the cited sources, {name} graduated from {_nonnull_text(f4.high_school)} in {_nonnull_text(f4.hs_graduation_year)}."
        await evaluator.verify(
            claim=hs_claim,
            node=leaf_hs,
            sources=hs_sources,
            additional_instruction="Confirm both the high school name and the year of graduation.",
        )

    # ---- Run all four figure verifications ---- #
    await _verify_figure1()
    await _verify_figure2()
    await _verify_figure3()
    await _verify_figure4()

    return evaluator.get_summary()