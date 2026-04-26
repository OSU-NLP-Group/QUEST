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
TASK_ID = "state_llc_suitability"
TASK_DESCRIPTION = (
    "An entrepreneur is planning to start a small online consulting business and wants to form a Limited Liability "
    "Company (LLC) in a U.S. state that minimizes administrative burden and costs while maintaining legal compliance. "
    "The entrepreneur is a single-member LLC with no employees initially.\n\n"
    "Identify ONE U.S. state where the entrepreneur should form their LLC that meets the maximum number of the following criteria:\n\n"
    "1. Initial LLC filing fee is $100 or less\n"
    "2. Annual report filing fee is $50 or less (or no annual report requirement)\n"
    "3. No annual franchise tax exceeding $100\n"
    "4. Online filing available through the Secretary of State website\n"
    "5. No newspaper publication requirement for LLC formation\n"
    "6. Allows commercial registered agent services\n"
    "7. Does not require filing a written operating agreement with the state\n"
    "8. Does not require a separate general statewide business license\n"
    "9. Workers' compensation insurance required only after hiring first employee\n"
    "10. Standard processing time is 15 business days or less\n"
    "11. Name reservation fee is $50 or less (if offered)\n"
    "12. Reports required no more frequently than annually\n"
    "13. Allows foreign LLCs to register\n"
    "14. Total first-year cost (filing + year-one fees/taxes) is $300 or less\n\n"
    "Provide the state name and verify which of these criteria it meets. Include reference URLs from official state "
    "government websites (Secretary of State or similar agencies) to support your answer."
)


# --------------------------------------------------------------------------- #
# Extraction data models                                                      #
# --------------------------------------------------------------------------- #
class CriterionEvidence(BaseModel):
    sources: List[str] = Field(default_factory=list)


class LLCStateExtraction(BaseModel):
    state: Optional[str] = None

    # General source list if the answer doesn't map URLs per-criterion
    general_sources: List[str] = Field(default_factory=list)

    # Per-criterion evidence URLs explicitly mentioned in the answer
    initial_filing_fee: CriterionEvidence = Field(default_factory=CriterionEvidence)
    annual_report_fee: CriterionEvidence = Field(default_factory=CriterionEvidence)
    annual_franchise_tax: CriterionEvidence = Field(default_factory=CriterionEvidence)
    online_filing: CriterionEvidence = Field(default_factory=CriterionEvidence)
    no_publication: CriterionEvidence = Field(default_factory=CriterionEvidence)
    registered_agent_flexibility: CriterionEvidence = Field(default_factory=CriterionEvidence)
    operating_agreement_not_required: CriterionEvidence = Field(default_factory=CriterionEvidence)
    no_general_business_license: CriterionEvidence = Field(default_factory=CriterionEvidence)
    workers_comp_requirement: CriterionEvidence = Field(default_factory=CriterionEvidence)
    processing_time: CriterionEvidence = Field(default_factory=CriterionEvidence)
    name_reservation_fee: CriterionEvidence = Field(default_factory=CriterionEvidence)
    annual_report_frequency: CriterionEvidence = Field(default_factory=CriterionEvidence)
    foreign_llc_registration: CriterionEvidence = Field(default_factory=CriterionEvidence)
    total_first_year_cost: CriterionEvidence = Field(default_factory=CriterionEvidence)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_llc_state_info() -> str:
    return """
Extract the single U.S. state recommended by the answer for forming the LLC, and collect all official source URLs cited in the answer that support each specific criterion below.

Rules for extraction:
- Do not invent any state or URLs. Extract only what appears in the answer text.
- Capture URLs in any reasonable format (plain URL, markdown link, etc.), but return the actual URL strings.
- Prefer URLs from official state government websites (Secretary of State or other official state agencies). Still extract all URLs that are present in the answer (even if non-official); the verification will handle domain evaluation.
- If the answer provides a general sources list not tied to any single criterion, put them in 'general_sources'.
- If a criterion is mentioned but has no URLs in the answer, return an empty list for that criterion’s 'sources'.

Return a JSON object with fields:
- state: string, the single U.S. state the answer recommends forming the LLC in. If multiple states are listed, pick the one explicitly recommended; if still ambiguous, pick the first one.
- general_sources: array of strings, any overall sources the answer cites (optional).

- initial_filing_fee: { "sources": [ ... ] }
- annual_report_fee: { "sources": [ ... ] }
- annual_franchise_tax: { "sources": [ ... ] }
- online_filing: { "sources": [ ... ] }
- no_publication: { "sources": [ ... ] }
- registered_agent_flexibility: { "sources": [ ... ] }
- operating_agreement_not_required: { "sources": [ ... ] }
- no_general_business_license: { "sources": [ ... ] }
- workers_comp_requirement: { "sources": [ ... ] }
- processing_time: { "sources": [ ... ] }
- name_reservation_fee: { "sources": [ ... ] }
- annual_report_frequency: { "sources": [ ... ] }
- foreign_llc_registration: { "sources": [ ... ] }
- total_first_year_cost: { "sources": [ ... ] }

If any field is missing in the answer, set its value to null (for 'state') or an empty array (for the 'sources' lists).
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions to build verification nodes                                #
# --------------------------------------------------------------------------- #
def _gather_sources(extracted: LLCStateExtraction, field_name: str) -> List[str]:
    """Return per-criterion sources; if empty, fall back to general_sources."""
    criterion = getattr(extracted, field_name, None)
    sources = []
    if criterion and isinstance(criterion, CriterionEvidence) and criterion.sources:
        sources = list(criterion.sources)
    if not sources and extracted.general_sources:
        sources = list(extracted.general_sources)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in sources:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


OFFICIAL_ONLY_INSTRUCTION = (
    "Only treat the claim as SUPPORTED if it is explicitly supported by the content of the provided page(s) AND the URL "
    "is an official U.S. state government website (e.g., domains ending in .gov, *.state.xx.us, *.xx.gov, or a clearly "
    "official state agency domain such as Florida's sunbiz.org). Ignore law firms, blogs, private SaaS, or non-government sites.\n"
    "If multiple URLs are provided, you may use any one official page that clearly supports the claim.\n"
    "For fees, evaluate the base statutory/agency fee; exclude optional expedite, credit card, portal, or third‑party service charges.\n"
    "For timeframes, evaluate the standard non‑expedited processing published by the agency. If the page lacks enough "
    "information to establish the claim, respond 'Not supported'."
)


async def _add_criterion_group(
    evaluator: Evaluator,
    parent_node,
    state_name: Optional[str],
    group_id: str,
    group_desc: str,
    claim_text: str,
    sources: List[str],
    specific_instruction: str = "",
    critical_group: bool = False
) -> None:
    """
    Build a sequential group for a single criterion:
    1) sources_exist (critical custom leaf)
    2) supported_by_official_url (critical verification leaf using the provided URLs)
    The group itself is non-critical to allow partial credit across criteria unless overridden by critical_group.
    """
    # Create the criterion group node
    group_node = evaluator.add_sequential(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=critical_group  # usually False per rubric (non-critical criteria)
    )

    # Step 1: Ensure the answer provided at least one URL so we do not degrade to source-free verification
    sources_exist = evaluator.add_custom_node(
        result=bool(sources),
        id=f"{group_id}_sources_provided",
        desc=f"Source URL(s) provided in the answer for: {group_desc}",
        parent=group_node,
        critical=True
    )

    # Step 2: Actual evidence-based verification against the provided URLs
    support_leaf = evaluator.add_leaf(
        id=f"{group_id}_supported",
        desc=f"{group_desc} — supported by official state source(s)",
        parent=group_node,
        critical=True
    )

    # Compose instruction
    add_ins = OFFICIAL_ONLY_INSTRUCTION
    if specific_instruction:
        add_ins = f"{OFFICIAL_ONLY_INSTRUCTION}\n{specific_instruction}"

    # Build a robust claim; fall back wording if state unknown (should be gated by state precondition at higher level)
    state_phrase = state_name if state_name else "the chosen state"
    claim = claim_text.format(state=state_phrase)

    # Perform verification using provided URLs
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Criteria-specific builders                                                   #
# --------------------------------------------------------------------------- #
async def build_all_criteria(
    evaluator: Evaluator,
    parent_node,
    extracted: LLCStateExtraction
) -> None:
    state = extracted.state

    # 1. Initial LLC filing fee ≤ $100
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Initial_Filing_Fee",
        group_desc="The state charges $100 or less for initial LLC Articles of Organization filing fee",
        claim_text="In {state}, the standard base filing fee for forming an LLC (Articles of Organization) is $100 or less.",
        sources=_gather_sources(extracted, "initial_filing_fee"),
        specific_instruction="Look for the base filing fee charged by the Secretary of State (or equivalent). Exclude expedite or card fees."
    )

    # 2. Annual report fee ≤ $50 or no report required
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Annual_Report_Fee",
        group_desc="The state charges $50 or less for annual report filing, or has no annual report requirement",
        claim_text="In {state}, the LLC periodic report fee (annual or similar) is $50 or less per filing, or no periodic report is required for LLCs.",
        sources=_gather_sources(extracted, "annual_report_fee"),
        specific_instruction="If the state uses biennial or other periodicity, evaluate the per‑filing fee; it must be $50 or less to satisfy the criterion."
    )

    # 3. No annual franchise tax exceeding $100
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Annual_Franchise_Tax",
        group_desc="The state does not impose an annual franchise tax exceeding $100 for standard LLCs",
        claim_text="In {state}, LLCs do not owe an annual franchise/privilege/minimum tax greater than $100 (either none, or $100 or less).",
        sources=_gather_sources(extracted, "annual_franchise_tax"),
        specific_instruction="Use the state's Department of Revenue/Taxation or agency guidance. Focus on state-level annual taxes owed by standard LLCs."
    )

    # 4. Online filing available
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Online_Filing_Available",
        group_desc="The state allows online filing of LLC formation documents through its Secretary of State website",
        claim_text="In {state}, the Secretary of State (or equivalent) provides an online filing option for LLC formation.",
        sources=_gather_sources(extracted, "online_filing"),
        specific_instruction="Confirm the state’s official portal allows forming an LLC online."
    )

    # 5. No newspaper publication requirement
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="No_Publication_Requirement",
        group_desc="The state does not require newspaper publication of LLC formation",
        claim_text="In {state}, there is no newspaper publication requirement to form an LLC.",
        sources=_gather_sources(extracted, "no_publication"),
        specific_instruction="The page should indicate that publication is not required for LLC formation. If unclear, treat as not supported."
    )

    # 6. Registered agent flexibility (allows commercial service)
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Registered_Agent_Flexibility",
        group_desc="The state allows either an in-state resident OR a commercial registered agent service to serve as registered agent",
        claim_text="In {state}, an LLC may appoint a commercial registered agent (a business entity or service) as its registered agent.",
        sources=_gather_sources(extracted, "registered_agent_flexibility"),
        specific_instruction="Look for text indicating a registered agent can be an individual or business entity/commercial registered agent."
    )

    # 7. Operating agreement not required to be filed with the state
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Operating_Agreement_Not_Required",
        group_desc="The state does not require a written operating agreement to be filed with the state",
        claim_text="In {state}, the LLC operating agreement is internal and is not filed with the state.",
        sources=_gather_sources(extracted, "operating_agreement_not_required"),
        specific_instruction="Seek explicit language such as 'do not file the operating agreement' or 'the operating agreement is kept internally'."
    )

    # 8. No general statewide business license
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="No_General_Business_License",
        group_desc="The state does not require a separate general statewide business license for all LLCs beyond entity registration",
        claim_text="In {state}, there is no statewide general business license required for all LLCs (separate from entity registration).",
        sources=_gather_sources(extracted, "no_general_business_license"),
        specific_instruction="The page may note local licenses/permits may be required; that's acceptable. Confirm no blanket state general business license."
    )

    # 9. Workers' comp required only after first employee
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Workers_Comp_Requirement",
        group_desc="The state requires workers' compensation insurance only after hiring the first employee, not for single-member LLCs with no employees",
        claim_text="In {state}, workers’ compensation insurance is required only when the business has employees; a single‑member LLC with no employees is not required.",
        sources=_gather_sources(extracted, "workers_comp_requirement"),
        specific_instruction="Use the state labor/industrial insurance agency guidance. Confirm coverage is not required with zero employees."
    )

    # 10. Standard processing time ≤ 15 business days
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Processing_Time",
        group_desc="The state has standard LLC formation processing time of 15 business days or less for non-expedited filing",
        claim_text="In {state}, the standard non‑expedited processing time for LLC formation is 15 business days or less.",
        sources=_gather_sources(extracted, "processing_time"),
        specific_instruction="Use the agency's standard processing time posting for LLC formation. If only 'days' are shown, interpret reasonably as business days unless stated otherwise."
    )

    # 11. Name reservation fee ≤ $50 (if offered)
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Name_Reservation_Fee",
        group_desc="The state charges $50 or less for LLC name reservation if offered as a separate service",
        claim_text="In {state}, the LLC name reservation fee, if the state offers name reservation, is $50 or less.",
        sources=_gather_sources(extracted, "name_reservation_fee"),
        specific_instruction="If name reservation is not offered, treat the criterion as satisfied; otherwise confirm the fee is $50 or less."
    )

    # 12. Reports required no more frequently than annually
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Annual_Report_Frequency",
        group_desc="The state requires reports to be filed no more frequently than annually (allows annual, biennial, or no requirement)",
        claim_text="In {state}, required business entity reports for LLCs are filed no more frequently than once per year (annual or less frequent).",
        sources=_gather_sources(extracted, "annual_report_frequency"),
        specific_instruction="Confirm frequency such as annual, biennial, or none. If quarterly or more frequent entity reports exist, do not support."
    )

    # 13. Allows foreign LLCs to register
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Foreign_LLC_Registration",
        group_desc="The state allows foreign LLCs (formed in other states) to register and operate within the state",
        claim_text="In {state}, foreign LLCs can register (obtain authority) to transact business in the state.",
        sources=_gather_sources(extracted, "foreign_llc_registration"),
        specific_instruction="Look for 'Foreign LLC', 'Certificate of Authority', or similar official instructions for out‑of‑state LLCs."
    )

    # 14. Total first-year cost ≤ $300
    await _add_criterion_group(
        evaluator,
        parent_node,
        state,
        group_id="Total_First_Year_Cost",
        group_desc="The total first-year cost (filing fee + any annual fees/taxes due in year one) is $300 or less",
        claim_text="In {state}, the total first‑year cost for an LLC (formation filing fee plus any required fees/taxes due in the first year) is $300 or less.",
        sources=_gather_sources(extracted, "total_first_year_cost"),
        specific_instruction="Only treat as supported if the official pages provided allow you to deduce or compute that the total of first‑year mandatory state charges is ≤ $300. If unclear, mark as not supported."
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
    Evaluate an answer for the 'State Suitability for LLC Formation' task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # per rubric: parallel aggregation at top level
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_llc_state_info(),
        template_class=LLCStateExtraction,
        extraction_name="llc_state_info"
    )

    # Record the extracted state as custom info for traceability
    evaluator.add_custom_info(
        info={"state": extracted.state or None},
        info_type="extracted",
        info_name="chosen_state"
    )

    # Build main rubric node (matches rubric JSON)
    state_node = evaluator.add_parallel(
        id="State_Suitability_for_LLC_Formation",
        desc="Evaluates whether the identified state meets the specified criteria for small business LLC formation",
        parent=root,
        critical=False
    )

    # Preconditions (critical siblings under the same parent to gate the rest)
    evaluator.add_custom_node(
        result=bool(extracted.state and extracted.state.strip()),
        id="State_Identified",
        desc="The answer identifies a specific U.S. state",
        parent=state_node,
        critical=True
    )

    # Build all criteria groups and verifications
    await build_all_criteria(evaluator, state_node, extracted)

    # Return final structured evaluation summary
    return evaluator.get_summary()