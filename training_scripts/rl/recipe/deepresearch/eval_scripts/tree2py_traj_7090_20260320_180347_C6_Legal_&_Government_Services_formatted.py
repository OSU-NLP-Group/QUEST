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
TASK_ID = "state_ag_requirements"
TASK_DESCRIPTION = """
In the United States, identify a state that meets ALL of the following criteria: 
(1) Has a bicameral (two-house) state legislature, and 
(2) Has exactly four federal judicial districts. 
For this state, provide comprehensive information about the attorney general position, including: 
• How the attorney general is selected (elected or appointed) and the election timing if elected; 
• All qualification requirements, specifically bar admission requirements (whether required and minimum years if specified), minimum age requirement (if any), state residency requirement (whether required and minimum years if specified), and U.S. citizenship requirement (if any); and 
• Term characteristics including length of term and term limits (if any, and whether consecutive or total). 
Provide the state name and all required information with supporting URL references for verification.
"""


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    state: Optional[str] = None
    bicameral_urls: List[str] = Field(default_factory=list)
    districts4_urls: List[str] = Field(default_factory=list)


class SelectionInfo(BaseModel):
    selection_method: Optional[str] = None  # e.g., "elected", "appointed"
    election_timing: Optional[str] = None   # e.g., "midterm", "presidential", "even-numbered non-presidential years"
    selection_urls: List[str] = Field(default_factory=list)


class BarInfo(BaseModel):
    required: Optional[str] = None                  # "yes"/"no"/text
    years_specified: Optional[str] = None           # "yes"/"no"/text
    years_number: Optional[str] = None              # e.g., "5", "10", "ten"
    bar_urls: List[str] = Field(default_factory=list)


class AgeInfo(BaseModel):
    min_age_exists: Optional[str] = None            # "yes"/"no"/text
    min_age_number: Optional[str] = None            # e.g., "30", "thirty"
    age_urls: List[str] = Field(default_factory=list)


class ResidencyInfo(BaseModel):
    residency_required: Optional[str] = None        # "yes"/"no"/text
    residency_years_specified: Optional[str] = None # "yes"/"no"/text
    residency_years_number: Optional[str] = None    # e.g., "5"
    residency_urls: List[str] = Field(default_factory=list)


class CitizenshipInfo(BaseModel):
    citizenship_required: Optional[str] = None      # "yes"/"no"/text
    citizenship_urls: List[str] = Field(default_factory=list)


class TermLengthInfo(BaseModel):
    term_length: Optional[str] = None               # e.g., "4 years"
    term_length_urls: List[str] = Field(default_factory=list)


class TermLimitsInfo(BaseModel):
    term_limits_exist: Optional[str] = None         # "yes"/"no"/text
    term_limit_type: Optional[str] = None           # e.g., "consecutive", "total (lifetime)", "none stated"
    term_limits_urls: List[str] = Field(default_factory=list)


class AttorneyGeneralInfo(BaseModel):
    selection: Optional[SelectionInfo] = None
    bar: Optional[BarInfo] = None
    age: Optional[AgeInfo] = None
    residency: Optional[ResidencyInfo] = None
    citizenship: Optional[CitizenshipInfo] = None
    term_length: Optional[TermLengthInfo] = None
    term_limits: Optional[TermLimitsInfo] = None


class ExtractionBundle(BaseModel):
    state_info: Optional[StateInfo] = None
    ag_info: Optional[AttorneyGeneralInfo] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured information from the answer. Use null for any missing field. 
When extracting URLs, only include URLs explicitly present in the answer text.

1) State criteria and supporting sources:
- state_info.state: The single U.S. state the answer selects.
- state_info.bicameral_urls: URL(s) cited to support that the state's legislature is bicameral (two chambers).
- state_info.districts4_urls: URL(s) cited to support that the state has exactly four federal judicial districts.

2) Attorney General (AG) information and supporting sources:
- ag_info.selection.selection_method: "elected" or "appointed" (verbatim or normalized from the answer).
- ag_info.selection.election_timing: The timing if elected (e.g., "midterm", "even-numbered non-presidential years", etc.).
- ag_info.selection.selection_urls: URL(s) cited to support selection method and timing.

- ag_info.bar.required: whether bar admission is required (yes/no or a brief phrase).
- ag_info.bar.years_specified: whether a specific minimum number of years is specified (yes/no).
- ag_info.bar.years_number: the minimum number of years if specified (e.g., "5").
- ag_info.bar.bar_urls: URL(s) supporting bar admission requirements.

- ag_info.age.min_age_exists: whether a minimum age requirement exists (yes/no).
- ag_info.age.min_age_number: the minimum age if specified (e.g., "30").
- ag_info.age.age_urls: URL(s) supporting age requirements.

- ag_info.residency.residency_required: whether state residency is required (yes/no).
- ag_info.residency.residency_years_specified: whether a minimum number of residency years is specified (yes/no).
- ag_info.residency.residency_years_number: the required residency years if specified (e.g., "5").
- ag_info.residency.residency_urls: URL(s) supporting residency requirements.

- ag_info.citizenship.citizenship_required: whether U.S. citizenship is required (yes/no).
- ag_info.citizenship.citizenship_urls: URL(s) supporting citizenship requirement (if any).

- ag_info.term_length.term_length: the AG term length as stated (e.g., "4 years").
- ag_info.term_length.term_length_urls: URL(s) supporting the term length.

- ag_info.term_limits.term_limits_exist: whether term limits exist for AG (yes/no).
- ag_info.term_limits.term_limit_type: "consecutive" or "total (lifetime)" or a brief description, if applicable.
- ag_info.term_limits.term_limits_urls: URL(s) supporting term limit info.

Rules:
- Return URLs as full strings. If a URL is given via markdown link, extract the actual URL.
- Do not invent information; only extract what appears in the answer.
- Use null for any field that is not present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_list(v: Optional[List[str]]) -> bool:
    return bool(v) and isinstance(v, list) and len(v) > 0


def _state_name(si: Optional[StateInfo]) -> str:
    return (si.state or "the identified state").strip()


def _clean_years(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip()


def _clean_age(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip()


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_state_identification(evaluator: Evaluator, parent_node, bundle: ExtractionBundle) -> None:
    si = bundle.state_info or StateInfo()
    state_name = _state_name(si)

    # State Identification (critical parallel)
    sid_node = evaluator.add_parallel(
        id="State_Identification",
        desc="Identify the correct state based on legislative structure and federal judicial district criteria",
        parent=parent_node,
        critical=True
    )

    # Bicameral legislature claim (critical, with sources)
    n_bicameral = evaluator.add_leaf(
        id="Bicameral_Legislature",
        desc="The state has a bicameral (two-house) legislature",
        parent=sid_node,
        critical=True
    )
    bicameral_claim = f"The state of {state_name} has a bicameral (two-chamber) state legislature."
    await evaluator.verify(
        claim=bicameral_claim,
        node=n_bicameral,
        sources=si.bicameral_urls,
        additional_instruction="Accept synonymous phrasing like 'two-house' or 'two-chamber'. Verify specifically at the state level."
    )

    # Bicameral reference existence (critical)
    evaluator.add_custom_node(
        result=_non_empty_list(si.bicameral_urls),
        id="Bicameral_Legislature_Reference",
        desc="Provide a URL reference verifying the state's bicameral legislative structure",
        parent=sid_node,
        critical=True
    )

    # Exactly four federal districts claim (critical, with sources)
    n_four = evaluator.add_leaf(
        id="Four_Federal_Districts",
        desc="The state has exactly four federal judicial districts",
        parent=sid_node,
        critical=True
    )
    four_claim = f"The state of {state_name} has exactly four federal judicial districts."
    await evaluator.verify(
        claim=four_claim,
        node=n_four,
        sources=si.districts4_urls,
        additional_instruction="Confirm that the U.S. District Courts in this state are divided into four distinct districts. If the page is about circuits or state courts, this does NOT count."
    )

    # Four districts reference existence (critical)
    evaluator.add_custom_node(
        result=_non_empty_list(si.districts4_urls),
        id="Four_Federal_Districts_Reference",
        desc="Provide a URL reference verifying the state has four federal judicial districts",
        parent=sid_node,
        critical=True
    )


async def build_ag_characteristics_core(evaluator: Evaluator, parent_node, bundle: ExtractionBundle) -> None:
    si = bundle.state_info or StateInfo()
    ag = bundle.ag_info or AttorneyGeneralInfo()
    state_name = _state_name(si)

    # Attorney General Characteristics (critical parallel) - ONLY core (all-critical) checks here
    ag_core = evaluator.add_parallel(
        id="Attorney_General_Characteristics",
        desc="Provide complete information about the state's attorney general position and requirements (core checks)",
        parent=parent_node,
        critical=True
    )

    # Selection Method (critical)
    sel = ag.selection or SelectionInfo()
    sel_node = evaluator.add_parallel(
        id="Selection_Method",
        desc="Provide information about how the attorney general is selected",
        parent=ag_core,
        critical=True
    )

    # Elected position (critical)
    n_elected = evaluator.add_leaf(
        id="Elected_Position",
        desc="The attorney general position is elected (not appointed)",
        parent=sel_node,
        critical=True
    )
    elected_claim = f"In {state_name}, the Attorney General is elected by voters (not appointed)."
    await evaluator.verify(
        claim=elected_claim,
        node=n_elected,
        sources=sel.selection_urls,
        additional_instruction="Verify the officeholder is chosen via statewide election. Synonyms like 'popular vote' or 'statewide election' are acceptable."
    )

    # Midterm election timing (critical)
    n_midterm = evaluator.add_leaf(
        id="Midterm_Election",
        desc="Attorney general elections occur during federal midterm election years",
        parent=sel_node,
        critical=True
    )
    midterm_claim = f"In {state_name}, Attorney General elections are held during U.S. federal midterm election years (even-numbered years without a presidential election)."
    await evaluator.verify(
        claim=midterm_claim,
        node=n_midterm,
        sources=sel.selection_urls,
        additional_instruction="Look for phrasing like 'elected in even-numbered non-presidential years' or ‘midterm years’. If the state holds AG elections during presidential cycles, this should fail."
    )

    # Selection reference provided (critical existence)
    evaluator.add_custom_node(
        result=_non_empty_list(sel.selection_urls),
        id="Selection_Reference",
        desc="Provide a URL reference verifying the attorney general selection method and election timing",
        parent=sel_node,
        critical=True
    )

    # Qualification Requirements CORE (critical)
    qual_core = evaluator.add_parallel(
        id="Qualification_Requirements_CORE",
        desc="Provide all core qualification requirements for the attorney general position (existence and requiredness checks)",
        parent=ag_core,
        critical=True
    )

    # Bar Admission CORE (critical)
    bar = ag.bar or BarInfo()
    bar_core = evaluator.add_parallel(
        id="Bar_Admission_CORE",
        desc="Provide bar admission requirements (core)",
        parent=qual_core,
        critical=True
    )

    n_bar_required = evaluator.add_leaf(
        id="Bar_Required",
        desc="Bar admission is required for the position",
        parent=bar_core,
        critical=True
    )
    bar_req_claim = f"To serve as Attorney General of {state_name}, admission to the state bar (or being a licensed attorney admitted to practice law in the state) is required."
    await evaluator.verify(
        claim=bar_req_claim,
        node=n_bar_required,
        sources=bar.bar_urls,
        additional_instruction="Accept synonymous phrases like 'licensed attorney', 'admitted to practice law', or 'member of the State Bar'."
    )

    n_bar_years_spec = evaluator.add_leaf(
        id="Bar_Years_Specified",
        desc="A specific minimum number of years of bar admission is specified",
        parent=bar_core,
        critical=True
    )
    bar_years_spec_claim = f"{state_name} specifies a minimum number of years of bar admission or legal practice experience as a qualification to serve as Attorney General."
    await evaluator.verify(
        claim=bar_years_spec_claim,
        node=n_bar_years_spec,
        sources=bar.bar_urls,
        additional_instruction="The requirement may be phrased as 'admitted to practice for at least N years' or 'engaged in the practice of law for N years'."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(bar.bar_urls),
        id="Bar_Reference",
        desc="Provide a URL reference verifying the bar admission requirement",
        parent=bar_core,
        critical=True
    )

    # Age Requirement CORE (critical)
    age = ag.age or AgeInfo()
    age_core = evaluator.add_parallel(
        id="Age_Requirement_CORE",
        desc="Provide age requirement information (core)",
        parent=qual_core,
        critical=True
    )

    n_age_exists = evaluator.add_leaf(
        id="Minimum_Age_Exists",
        desc="A minimum age requirement exists for the position",
        parent=age_core,
        critical=True
    )
    age_exists_claim = f"There is a minimum age requirement to serve as Attorney General in {state_name}."
    await evaluator.verify(
        claim=age_exists_claim,
        node=n_age_exists,
        sources=age.age_urls,
        additional_instruction="Confirm existence of an age threshold (e.g., minimum age X). If explicitly 'no minimum age', this should fail."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(age.age_urls),
        id="Age_Reference",
        desc="Provide a URL reference verifying the age requirement",
        parent=age_core,
        critical=True
    )

    # Residency Requirement CORE (critical)
    res = ag.residency or ResidencyInfo()
    res_core = evaluator.add_parallel(
        id="Residency_Requirement_CORE",
        desc="Provide state residency requirement information (core)",
        parent=qual_core,
        critical=True
    )

    n_res_req = evaluator.add_leaf(
        id="Residency_Required",
        desc="State residency is required for the position",
        parent=res_core,
        critical=True
    )
    res_req_claim = f"State residency is required to serve as Attorney General of {state_name}."
    await evaluator.verify(
        claim=res_req_claim,
        node=n_res_req,
        sources=res.residency_urls,
        additional_instruction="Look for 'resident of the state' requirements. If explicitly no residency requirement, this should fail."
    )

    n_res_years_spec = evaluator.add_leaf(
        id="Residency_Years_Specified",
        desc="A specific minimum number of years of state residency is specified",
        parent=res_core,
        critical=True
    )
    res_years_spec_claim = f"{state_name} specifies a minimum number of years of state residency to serve as Attorney General."
    await evaluator.verify(
        claim=res_years_spec_claim,
        node=n_res_years_spec,
        sources=res.residency_urls,
        additional_instruction="Accept phrasing like 'resident for at least N years'."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(res.residency_urls),
        id="Residency_Reference",
        desc="Provide a URL reference verifying the residency requirement",
        parent=res_core,
        critical=True
    )

    # Term Characteristics CORE (critical)
    term_core = evaluator.add_parallel(
        id="Term_Characteristics_CORE",
        desc="Provide information about term length and limits (core checks)",
        parent=ag_core,
        critical=True
    )

    # Term Length (critical)
    tlen = ag.term_length or TermLengthInfo()
    tlen_node = evaluator.add_parallel(
        id="Term_Length",
        desc="Provide term length information",
        parent=term_core,
        critical=True
    )

    n_four_year = evaluator.add_leaf(
        id="Four_Year_Term",
        desc="The attorney general serves a 4-year term",
        parent=tlen_node,
        critical=True
    )
    four_year_claim = f"In {state_name}, the Attorney General serves a 4-year term."
    await evaluator.verify(
        claim=four_year_claim,
        node=n_four_year,
        sources=tlen.term_length_urls,
        additional_instruction="If the page shows a different term length, this should fail."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(tlen.term_length_urls),
        id="Term_Length_Reference",
        desc="Provide a URL reference verifying the term length",
        parent=tlen_node,
        critical=True
    )

    # Term Limits CORE (critical)
    tlim = ag.term_limits or TermLimitsInfo()
    tlim_core = evaluator.add_parallel(
        id="Term_Limits_CORE",
        desc="Provide term limit information (core)",
        parent=term_core,
        critical=True
    )

    n_tl_exist = evaluator.add_leaf(
        id="Term_Limits_Exist",
        desc="Term limits exist for the attorney general position",
        parent=tlim_core,
        critical=True
    )
    tl_exist_claim = f"Term limits apply to the Attorney General in {state_name}."
    await evaluator.verify(
        claim=tl_exist_claim,
        node=n_tl_exist,
        sources=tlim.term_limits_urls,
        additional_instruction="Evidence can be phrased as 'limited to two terms', etc. If explicitly no term limits, this should fail."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(tlim.term_limits_urls),
        id="Term_Limits_Reference",
        desc="Provide a URL reference verifying the term limit information",
        parent=tlim_core,
        critical=True
    )


async def build_optional_details(evaluator: Evaluator, parent_node, bundle: ExtractionBundle) -> None:
    si = bundle.state_info or StateInfo()
    ag = bundle.ag_info or AttorneyGeneralInfo()
    state_name = _state_name(si)

    opt = evaluator.add_parallel(
        id="Optional_Additional_Details",
        desc="Optional details and numeric specifics (non-critical)",
        parent=parent_node,
        critical=False
    )

    # Bar years number (non-critical)
    bar = ag.bar or BarInfo()
    bar_years = _clean_years(bar.years_number)
    n_bar_years_num = evaluator.add_leaf(
        id="Bar_Years_Number",
        desc="Provide the specific number of years required",
        parent=opt,
        critical=False
    )
    byn_claim = f"The minimum number of years of bar admission or legal practice required to serve as Attorney General in {state_name} is {bar_years} years."
    await evaluator.verify(
        claim=byn_claim,
        node=n_bar_years_num,
        sources=bar.bar_urls,
        additional_instruction="Match the numeric minimum if specified. If the sources show a different number or none, this should fail."
    )

    # Age number (non-critical)
    age = ag.age or AgeInfo()
    age_num = _clean_age(age.min_age_number)
    n_age_num = evaluator.add_leaf(
        id="Minimum_Age_Number",
        desc="Provide the specific minimum age required",
        parent=opt,
        critical=False
    )
    age_claim = f"The minimum age required to serve as Attorney General in {state_name} is {age_num} years."
    await evaluator.verify(
        claim=age_claim,
        node=n_age_num,
        sources=age.age_urls,
        additional_instruction="Verify the exact numeric minimum age if available."
    )

    # Residency years number (non-critical)
    res = ag.residency or ResidencyInfo()
    res_years = _clean_years(res.residency_years_number)
    n_res_years_num = evaluator.add_leaf(
        id="Residency_Years_Number",
        desc="Provide the specific number of years of residency required",
        parent=opt,
        critical=False
    )
    ryn_claim = f"The minimum state residency required to serve as Attorney General in {state_name} is {res_years} years."
    await evaluator.verify(
        claim=ryn_claim,
        node=n_res_years_num,
        sources=res.residency_urls,
        additional_instruction="Verify the exact numeric residency years if available."
    )

    # Term limit type (non-critical)
    tlim = ag.term_limits or TermLimitsInfo()
    tl_type = (tlim.term_limit_type or "").strip()
    n_tl_type = evaluator.add_leaf(
        id="Term_Limit_Type",
        desc="Specify whether term limits are consecutive or total",
        parent=opt,
        critical=False
    )
    tl_type_claim = f"In {state_name}, the Attorney General term limits are '{tl_type}' (e.g., consecutive vs total/lifetime)."
    await evaluator.verify(
        claim=tl_type_claim,
        node=n_tl_type,
        sources=tlim.term_limits_urls,
        additional_instruction="Check if the limit is described as consecutive (can serve again after a break) or total (lifetime cap)."
    )

    # Citizenship requirement (non-critical) + reference existence (non-critical)
    cit = ag.citizenship or CitizenshipInfo()
    n_cit_req = evaluator.add_leaf(
        id="US_Citizenship_Required",
        desc="U.S. citizenship is required for the position",
        parent=opt,
        critical=False
    )
    cit_claim = f"U.S. citizenship is required to serve as Attorney General in {state_name}."
    await evaluator.verify(
        claim=cit_claim,
        node=n_cit_req,
        sources=cit.citizenship_urls,
        additional_instruction="If sources explicitly say no citizenship requirement or are silent, this should fail."
    )

    evaluator.add_custom_node(
        result=_non_empty_list(cit.citizenship_urls),
        id="Citizenship_Reference",
        desc="Provide a URL reference verifying the citizenship requirement",
        parent=opt,
        critical=False
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
    # Initialize evaluator with a Sequential root (task order matters: identify state -> AG details -> optional)
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
        default_model=model
    )

    # Extract all information in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=ExtractionBundle,
        extraction_name="extracted_state_and_ag_info"
    )

    # Build verification tree according to rubric (with minor structural adjustments to satisfy framework constraints)
    # 1) State Identification (critical)
    await build_state_identification(evaluator, root, extracted)

    # 2) Attorney General Characteristics CORE (critical set)
    await build_ag_characteristics_core(evaluator, root, extracted)

    # 3) Optional/non-critical numeric specifics and citizenship requirement
    await build_optional_details(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()