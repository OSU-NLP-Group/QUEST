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
TASK_ID = "four_tax_states"
TASK_DESCRIPTION = """
You are a 65-year-old military veteran with a 100% permanent and total (P&T) service-connected disability rating who is retiring and planning to invest in rental real estate. You want to minimize your tax burden by choosing states that offer favorable tax treatment.

Identify four U.S. states that meet ALL of the following criteria:

1. No State Income Tax: The state must impose no state income tax on residents
2. Low Property Tax Rate: The state must have an effective property tax rate at or below 1.00%
3. Senior Citizen Property Tax Exemption: The state must offer a property tax exemption, reduction, or relief program for homeowners aged 65 or older on their primary residence
4. Veteran Disability Property Tax Exemption: The state must offer a property tax exemption or significant reduction for veterans with a 100% permanent and total disability rating on their primary residence

For each of the four states you identify, provide:
- The state name
- Confirmation that it has no state income tax (with a reference URL from an official government source or recognized tax authority)
- The effective property tax rate (with a reference URL from a tax research organization or official source)
- Description of the senior citizen property tax exemption available, including any age requirement (with a reference URL from the state's official property tax or revenue department)
- Description of the veteran disability property tax exemption available for 100% P&T disabled veterans (with a reference URL from the state's official veterans affairs, property tax, or revenue department)
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class ProgramInfo(BaseModel):
    """Description and sources for a tax program."""
    description: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StateTaxEntry(BaseModel):
    """All extracted fields for one state."""
    state_name: Optional[str] = None

    # No income tax sources (official or recognized authority cited in answer)
    no_income_tax_urls: List[str] = Field(default_factory=list)

    # Effective property tax rate (string as written in answer, e.g., "0.86%")
    property_tax_rate_value: Optional[str] = None
    property_tax_rate_urls: List[str] = Field(default_factory=list)

    # Senior and veteran programs
    senior_program: ProgramInfo = Field(default_factory=ProgramInfo)
    veteran_program: ProgramInfo = Field(default_factory=ProgramInfo)


class StatesExtraction(BaseModel):
    """Top-level extraction: up to four states."""
    states: List[StateTaxEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
    Extract up to four U.S. states that the answer claims meet ALL of the specified criteria.
    For each state, create one object with the following fields:

    - state_name: The state's full name (e.g., "Florida"). If missing, return null.
    - no_income_tax_urls: An array of URLs cited in the answer confirming no state personal income tax. 
      Only include URLs explicitly present in the answer, in any format (plain URLs, markdown links, etc.).
      Prefer official government sources (e.g., state revenue/tax department .gov sites) or recognized tax authorities
      (e.g., Federation of Tax Administrators, Tax Foundation). If none are provided, return an empty array.
    - property_tax_rate_value: The effective statewide property tax rate value stated in the answer (e.g., "0.86%").
      Extract exactly as written in the answer; if not given, return null.
    - property_tax_rate_urls: An array of URLs cited in the answer for the effective statewide property tax rate.
      These may include official sources or recognized tax research orgs (e.g., Tax Foundation). If none, return an empty array.
    - senior_program:
        - description: A brief summary of the senior property tax exemption/relief program for age 65+ homeowners on their primary residence, 
          as written in the answer. If missing, return null.
        - urls: Array of URLs cited in the answer from the state’s official property tax or revenue department describing this program.
          If none, return an empty array.
    - veteran_program:
        - description: A brief summary of the property tax exemption or significant reduction for veterans with a 100% permanent and total disability rating 
          on their primary residence, as written in the answer. If missing, return null.
        - urls: Array of URLs cited in the answer from the state’s official veterans affairs, property tax, or revenue department describing this program.
          If none, return an empty array.

    Return a JSON object:
    {
      "states": [ StateTaxEntry, StateTaxEntry, ... up to 4]
    }

    IMPORTANT:
    - Extract only URLs explicitly mentioned in the answer text. Do not invent or infer URLs.
    - If the answer lists more than four states, include only the first four mentioned.
    - If fewer than four are provided, include as many as are present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"State #{n}")


# --------------------------------------------------------------------------- #
# Verification logic per state                                                #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    parent_node,
    state: StateTaxEntry,
    idx: int,
) -> None:
    """
    Build the verification sub-tree for one state and run checks.

    Structure (mirrors rubric; minor ordering adjusted to ensure evidence gating in sequential nodes):
    - Identification (critical)
    - No Income Tax (sequential):
        • Reference (critical, existence of authoritative URL(s))
        • Requirement (critical, verify claim against provided URLs)
    - Property Tax Rate (sequential):
        • Reference (critical, existence of URL(s))
        • Requirement (critical, verify <= 1.00% against provided URLs)
    - Senior Exemption (sequential):
        • Reference (critical, existence of official URL(s))
        • Requirement (critical, verify age 65+ primary residence program)
    - Veteran Exemption (sequential):
        • Reference (critical, existence of official URL(s))
        • Requirement (critical, verify program for 100% P&T disabled veterans on primary residence)
    """
    state_num = idx + 1
    state_id_prefix = f"State_{state_num}"

    # Top-level node per state (parallel, non-critical to allow partial credit across states)
    state_node = evaluator.add_parallel(
        id=state_id_prefix,
        desc=f"{ordinal(state_num)} qualifying state that meets all tax criteria",
        parent=parent_node,
        critical=False,
    )

    # 1) Identification (critical)
    id_exists = bool(state.state_name and state.state_name.strip())
    evaluator.add_custom_node(
        result=id_exists,
        id=f"{state_id_prefix}_Identification",
        desc=f"Provides the name of the {ordinal(state_num).lower()} qualifying state",
        parent=state_node,
        critical=True,
    )

    # 2) No State Income Tax (sequential)
    no_inc_node = evaluator.add_sequential(
        id=f"{state_id_prefix}_No_Income_Tax",
        desc="Verification criteria for no state income tax requirement",
        parent=state_node,
        critical=False,
    )
    # Reference existence first (critical)
    no_inc_ref = evaluator.add_custom_node(
        result=bool(state.no_income_tax_urls),
        id=f"{state_id_prefix}_No_Income_Tax_Reference",
        desc="Provides a reference URL from an official state government source or recognized tax authority confirming the no-income-tax status",
        parent=no_inc_node,
        critical=True,
    )
    # Requirement (critical, verify by URLs; gated on reference existence)
    no_inc_req = evaluator.add_leaf(
        id=f"{state_id_prefix}_No_Income_Tax_Requirement",
        desc="Confirms that the state imposes no state income tax on residents",
        parent=no_inc_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{state.state_name or 'The state'} imposes no state personal income tax on residents.",
        node=no_inc_req,
        sources=state.no_income_tax_urls,
        additional_instruction=(
            "Use only authoritative pages: official state government (.gov, Department of Revenue/Taxation) "
            "or widely recognized tax authorities (e.g., Federation of Tax Administrators, Tax Foundation). "
            "Focus on personal income tax on wages/general income; ignore niche taxes like interest/dividends-only taxes. "
            "If sources are not authoritative or the page contradicts the claim, mark as not supported."
        ),
        extra_prerequisites=[no_inc_ref],
    )

    # 3) Property Tax Rate (sequential)
    prop_rate_node = evaluator.add_sequential(
        id=f"{state_id_prefix}_Property_Tax_Rate",
        desc="Verification criteria for low property tax rate requirement",
        parent=state_node,
        critical=False,
    )
    # Reference existence first (critical)
    prop_rate_ref = evaluator.add_custom_node(
        result=bool(state.property_tax_rate_urls),
        id=f"{state_id_prefix}_Property_Tax_Rate_Reference",
        desc="Provides a reference URL from a tax research organization or official source documenting the effective property tax rate",
        parent=prop_rate_node,
        critical=True,
    )
    # Requirement (critical, verify <= 1.00% by URLs)
    prop_rate_req = evaluator.add_leaf(
        id=f"{state_id_prefix}_Property_Tax_Rate_Requirement",
        desc="Confirms that the state has an effective property tax rate at or below 1.00%",
        parent=prop_rate_node,
        critical=True,
    )
    rate_fragment = (
        f"The extracted rate value is {state.property_tax_rate_value}."
        if state.property_tax_rate_value else
        "Use the figure shown on the source page."
    )
    await evaluator.verify(
        claim=f"The statewide effective property tax rate in {state.state_name or 'the state'} is at or below 1.00%. {rate_fragment}",
        node=prop_rate_req,
        sources=state.property_tax_rate_urls,
        additional_instruction=(
            "Confirm a statewide effective (average effective) property tax rate from the cited page(s). "
            "If the page only shows county-level or non-statewide figures, or the statewide rate exceeds 1.00%, mark as not supported. "
            "Treat 1.00% equivalently to 1.0%; allow minor rounding. Prefer recognized tax research orgs (e.g., Tax Foundation) or official sources."
        ),
        extra_prerequisites=[prop_rate_ref],
    )

    # 4) Senior Exemption (sequential)
    senior_node = evaluator.add_sequential(
        id=f"{state_id_prefix}_Senior_Exemption",
        desc="Verification criteria for senior citizen property tax exemption",
        parent=state_node,
        critical=False,
    )
    # Reference existence first (critical)
    senior_ref = evaluator.add_custom_node(
        result=bool(state.senior_program.urls),
        id=f"{state_id_prefix}_Senior_Exemption_Reference",
        desc="Provides a reference URL from the state's official property tax or revenue department website describing the senior exemption program",
        parent=senior_node,
        critical=True,
    )
    # Requirement (critical, verify by URLs)
    senior_req = evaluator.add_leaf(
        id=f"{state_id_prefix}_Senior_Exemption_Requirement",
        desc="Confirms that the state offers a property tax exemption, reduction, or relief program for homeowners aged 65 or older on their primary residence",
        parent=senior_node,
        critical=True,
    )
    senior_claim_detail = (
        f"Program detail: {state.senior_program.description}."
        if state.senior_program.description else "Verify program existence and age threshold."
    )
    await evaluator.verify(
        claim=(
            f"{state.state_name or 'The state'} offers a property tax exemption, reduction, or relief program "
            f"for homeowners aged 65 or older on their primary residence. {senior_claim_detail}"
        ),
        node=senior_req,
        sources=state.senior_program.urls,
        additional_instruction=(
            "Use the state's official property tax or revenue department page. "
            "Confirm that homeowners aged 65+ (or higher) on their primary residence are eligible. "
            "Programs may be called 'homestead exemption', 'senior exemption', 'senior freeze', etc. "
            "If the minimum age is below 65 or the benefit is not tied to primary residence, mark as not supported."
        ),
        extra_prerequisites=[senior_ref],
    )

    # 5) Veteran Exemption (sequential)
    veteran_node = evaluator.add_sequential(
        id=f"{state_id_prefix}_Veteran_Exemption",
        desc="Verification criteria for veteran property tax exemption",
        parent=state_node,
        critical=False,
    )
    # Reference existence first (critical)
    veteran_ref = evaluator.add_custom_node(
        result=bool(state.veteran_program.urls),
        id=f"{state_id_prefix}_Veteran_Exemption_Reference",
        desc="Provides a reference URL from the state's official veterans affairs, property tax, or revenue department website describing the veteran disability exemption",
        parent=veteran_node,
        critical=True,
    )
    # Requirement (critical, verify by URLs)
    veteran_req = evaluator.add_leaf(
        id=f"{state_id_prefix}_Veteran_Exemption_Requirement",
        desc="Confirms that the state offers property tax exemption or significant reduction for veterans with 100% permanent and total disability rating on their primary residence",
        parent=veteran_node,
        critical=True,
    )
    vet_claim_detail = (
        f"Program detail: {state.veteran_program.description}."
        if state.veteran_program.description else "Verify program scope and eligibility."
    )
    await evaluator.verify(
        claim=(
            f"{state.state_name or 'The state'} offers a property tax exemption or significant reduction "
            f"for veterans with a 100% permanent and total (P&T) service-connected disability on their primary residence. "
            f"{vet_claim_detail}"
        ),
        node=veteran_req,
        sources=state.veteran_program.urls,
        additional_instruction=(
            "Use official state veterans affairs, property tax, or revenue department pages. "
            "Accept equivalent phrasing such as '100% disabled', 'totally and permanently disabled', or '100% service-connected permanent and total'. "
            "The benefit must apply to the primary residence. If the provided page only describes lesser ratings or non-primary residence benefits, mark as not supported."
        ),
        extra_prerequisites=[veteran_ref],
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
    Evaluate an agent's answer for selecting four qualifying states with favorable tax treatment
    for a 65+ veteran with 100% P&T disability, per the specified rubric.
    """
    # Initialize evaluator with PARALLEL root (matches rubric root aggregation)
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

    # Optional: create an explicit main node to mirror rubric "Four_Qualifying_States"
    main_node = evaluator.add_parallel(
        id="Four_Qualifying_States",
        desc="Identifies four U.S. states that each meet all specified tax-related criteria beneficial for a retiring real estate investor who is a 65+ year-old veteran with 100% permanent and total disability rating",
        parent=root,
        critical=False,
    )

    # Extract structured state information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Prepare exactly four states: first four, padding with empty entries if needed
    states: List[StateTaxEntry] = list(extracted.states[:4])
    while len(states) < 4:
        states.append(StateTaxEntry())

    # Add custom info for transparency
    evaluator.add_custom_info(
        info={
            "states_count_in_answer": len(extracted.states),
            "states_used_for_verification": [s.state_name for s in states],
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    # Build verification subtrees for each of the four states
    for i, state in enumerate(states):
        await verify_state(evaluator, main_node, state, i)

    # Return standard summary
    return evaluator.get_summary()