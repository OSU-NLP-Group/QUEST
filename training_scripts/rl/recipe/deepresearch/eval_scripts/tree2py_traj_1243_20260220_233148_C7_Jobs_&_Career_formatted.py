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
TASK_ID = "hs_ad_requirements_three_states"
TASK_DESCRIPTION = (
    "Research the professional requirements for high school athletic director positions across three states: "
    "Texas, Ohio, and Alabama. For each state, provide the following information: "
    "(1) The official state high school athletic association that governs interscholastic athletics, "
    "(2) Whether a valid teaching certificate is typically required or preferred for athletic director positions, "
    "(3) The typical minimum years of experience required in coaching or athletic administration, "
    "(4) The average annual salary range for high school athletic directors (as of 2026), and "
    "(5) Whether the state offers a specific athletic administrator certification program beyond NIAAA certifications. "
    "Provide comprehensive information for all three states with appropriate source documentation."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateRequirement(BaseModel):
    # (1) Association
    association_name: Optional[str] = None
    association_sources: List[str] = Field(default_factory=list)

    # (2) Teaching certificate typical requirement
    teaching_certificate: Optional[str] = None  # e.g., "required", "preferred", "not required", or a sentence
    teaching_certificate_sources: List[str] = Field(default_factory=list)

    # (3) Typical minimum years of experience
    experience_min_years: Optional[str] = None  # e.g., "3 years", "3–5 years"
    experience_sources: List[str] = Field(default_factory=list)

    # (4) Salary range (as of 2026)
    salary_range_2026: Optional[str] = None  # e.g., "$60,000–$85,000"
    salary_sources: List[str] = Field(default_factory=list)

    # (5) State-specific certification program beyond NIAAA
    state_certification_program: Optional[str] = None  # e.g., "Yes: [Program Name]" or "No state-specific program"
    state_certification_sources: List[str] = Field(default_factory=list)


class ThreeStatesExtraction(BaseModel):
    texas: Optional[StateRequirement] = None
    ohio: Optional[StateRequirement] = None
    alabama: Optional[StateRequirement] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_three_states() -> str:
    return """
Extract structured information about high school athletic director requirements for three states (Texas, Ohio, Alabama) exactly as presented in the answer. For each state, extract the following fields and the specific URLs that the answer cites to support them. Do not invent or infer any information not explicitly present in the answer text. If a field is missing, return null for the value and an empty array for sources.

For each of the three states, extract:
- association_name: The official state high school athletic association that governs interscholastic athletics (e.g., OHSAA, AHSAA, UIL), as named in the answer.
- association_sources: URL(s) in the answer that support that association (prefer the official association site or authoritative pages).

- teaching_certificate: A short phrase or sentence summarizing whether a valid teaching certificate is typically required or preferred for high school athletic director positions (e.g., "required", "preferred", "not required", or a brief sentence).
- teaching_certificate_sources: URL(s) in the answer that support this typical requirement (job postings, district HR pages, association guidance).

- experience_min_years: The typical minimum years of experience required in coaching or athletic administration (e.g., "3 years", "3–5 years", "5+ years").
- experience_sources: URL(s) in the answer that support this typical minimum requirement.

- salary_range_2026: The average annual salary range (as of 2026) for high school athletic directors, as reported in the answer (e.g., "$60,000–$85,000").
- salary_sources: URL(s) in the answer that support the stated salary range (state/district salary schedules, reputable salary aggregators, recent reports).

- state_certification_program: A concise statement of whether the state offers a specific state-run athletic administrator certification program beyond the NIAAA (e.g., "Yes: [Program Name]" or "No state-specific program beyond NIAAA").
- state_certification_sources: URL(s) in the answer that support the presence or absence of a state-specific athletic administrator certification program.

Return a JSON object with keys: "texas", "ohio", and "alabama", each containing the fields above. If the answer does not contain any info for a state, set that state to null.
"""


# --------------------------------------------------------------------------- #
# Helper for adding criterion checks                                          #
# --------------------------------------------------------------------------- #
async def add_criterion_with_existence_and_verify(
    evaluator: Evaluator,
    parent_node,
    *,
    criterion_id: str,
    criterion_desc: str,
    value: Optional[str],
    sources: List[str],
    claim: str,
    additional_instruction: str,
) -> None:
    """
    For a single criterion, create a critical sequential group:
      1) Existence check (value present + at least one source URL)
      2) Source-supported verification of the claim via the provided URLs
    """
    group_node = evaluator.add_sequential(
        id=criterion_id,
        desc=criterion_desc,
        parent=parent_node,
        critical=True
    )

    exists = bool(value and str(value).strip()) and bool(sources and len(sources) > 0)
    evaluator.add_custom_node(
        result=exists,
        id=f"{criterion_id}_exists",
        desc=f"Evidence provided: non-empty value and at least one supporting URL for '{criterion_id}'",
        parent=group_node,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{criterion_id}_supported",
        desc=f"Claim for '{criterion_id}' is supported by cited sources",
        parent=group_node,
        critical=True
    )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# State verification routine                                                  #
# --------------------------------------------------------------------------- #
async def verify_state_requirements(
    evaluator: Evaluator,
    parent_node,
    state_key: str,
    state_display_name: str,
    info: Optional[StateRequirement]
) -> None:
    """
    Build the verification subtree for a single state. Each of the five required
    items is modeled as a critical sequential node (existence + source-backed check).
    """
    state_node = evaluator.add_parallel(
        id=f"{state_display_name}_Requirements",
        desc=f"Information about {state_display_name} athletic director requirements is provided",
        parent=parent_node,
        critical=False
    )

    # If the entire state info is missing, add five failing existence checks to reflect absence.
    if info is None:
        # Create empty placeholders for criteria to transparently fail
        empty_sources: List[str] = []
        await add_criterion_with_existence_and_verify(
            evaluator,
            state_node,
            criterion_id=f"{state_display_name}_Athletic_Association",
            criterion_desc=f"Provides the name of {state_display_name}'s official state high school athletic association that governs interscholastic athletics",
            value=None,
            sources=empty_sources,
            claim=f"The official state high school athletic association that governs interscholastic athletics in {state_display_name} is ''.",
            additional_instruction="No information provided; this should fail unless supported by URLs."
        )
        await add_criterion_with_existence_and_verify(
            evaluator,
            state_node,
            criterion_id=f"{state_display_name}_Teaching_Certificate",
            criterion_desc=f"States whether a valid teaching certificate is typically required or preferred for athletic director positions in {state_display_name}",
            value=None,
            sources=empty_sources,
            claim=f"In {state_display_name}, a valid teaching certificate is typically ''.",
            additional_instruction="No information provided; this should fail unless supported by URLs."
        )
        await add_criterion_with_existence_and_verify(
            evaluator,
            state_node,
            criterion_id=f"{state_display_name}_Experience_Requirement",
            criterion_desc=f"Specifies the typical minimum years of experience required in coaching or athletic administration for {state_display_name} athletic director positions",
            value=None,
            sources=empty_sources,
            claim=f"In {state_display_name}, the typical minimum years of experience required is ''.",
            additional_instruction="No information provided; this should fail unless supported by URLs."
        )
        await add_criterion_with_existence_and_verify(
            evaluator,
            state_node,
            criterion_id=f"{state_display_name}_Salary_Range",
            criterion_desc=f"Provides an annual salary range for high school athletic directors in {state_display_name} (as of 2026)",
            value=None,
            sources=empty_sources,
            claim=f"As of 2026, the average annual salary range for high school athletic directors in {state_display_name} is ''.",
            additional_instruction="No information provided; this should fail unless supported by URLs."
        )
        await add_criterion_with_existence_and_verify(
            evaluator,
            state_node,
            criterion_id=f"{state_display_name}_State_Certification",
            criterion_desc=f"Indicates whether {state_display_name} offers a state-specific athletic administrator certification program beyond NIAAA certifications",
            value=None,
            sources=empty_sources,
            claim=f"{state_display_name} has the following state-specific athletic administrator certification status (beyond NIAAA): ''.",
            additional_instruction="No information provided; this should fail unless supported by URLs."
        )
        return

    # (1) Association
    assoc_value = info.association_name
    assoc_sources = info.association_sources
    assoc_claim = (
        f"The official state high school athletic association that governs interscholastic athletics "
        f"in {state_display_name} is '{assoc_value}'."
    )
    assoc_instruction = (
        f"Verify that the provided page(s) explicitly identify the official state high school athletic association "
        f"for {state_display_name}. Prefer the official association webpage or authoritative state-level sources."
    )
    await add_criterion_with_existence_and_verify(
        evaluator,
        state_node,
        criterion_id=f"{state_display_name}_Athletic_Association",
        criterion_desc=f"Provides the name of {state_display_name}'s official state high school athletic association that governs interscholastic athletics",
        value=assoc_value,
        sources=assoc_sources,
        claim=assoc_claim,
        additional_instruction=assoc_instruction
    )

    # (2) Teaching certificate typical requirement
    tc_value = info.teaching_certificate
    tc_sources = info.teaching_certificate_sources
    tc_claim = (
        f"In {state_display_name}, a valid teaching certificate is typically {tc_value} for high school athletic director positions."
    )
    tc_instruction = (
        "Judge whether the sources substantiate the typical requirement or preference (e.g., 'required', "
        "'preferred', 'not required'), using representative job postings, district HR pages, or association guidance. "
        "Accept reasonable phrasing variations (e.g., 'valid educator license'). The judgment is about typical practice, "
        "not universal rules."
    )
    await add_criterion_with_existence_and_verify(
        evaluator,
        state_node,
        criterion_id=f"{state_display_name}_Teaching_Certificate",
        criterion_desc=f"States whether a valid teaching certificate is typically required or preferred for athletic director positions in {state_display_name}",
        value=tc_value,
        sources=tc_sources,
        claim=tc_claim,
        additional_instruction=tc_instruction
    )

    # (3) Typical minimum years of experience
    exp_value = info.experience_min_years
    exp_sources = info.experience_sources
    exp_claim = (
        f"In {state_display_name}, the typical minimum years of experience required in coaching or athletic administration "
        f"for high school athletic director positions is {exp_value}."
    )
    exp_instruction = (
        "Verify that the sources indicate a minimum experience requirement consistent with the claimed value. "
        "Accept ranges (e.g., '3–5 years') and minor variations if they imply the same minimum threshold."
    )
    await add_criterion_with_existence_and_verify(
        evaluator,
        state_node,
        criterion_id=f"{state_display_name}_Experience_Requirement",
        criterion_desc=f"Specifies the typical minimum years of experience required in coaching or athletic administration for {state_display_name} athletic director positions",
        value=exp_value,
        sources=exp_sources,
        claim=exp_claim,
        additional_instruction=exp_instruction
    )

    # (4) Salary range (as of 2026)
    sal_value = info.salary_range_2026
    sal_sources = info.salary_sources
    sal_claim = (
        f"As of 2026, the average annual salary range for high school athletic directors in {state_display_name} is {sal_value}."
    )
    sal_instruction = (
        "Check whether the URLs support the stated salary range for high school athletic directors in the specified state. "
        "Accept recent, reputable sources (e.g., district salary schedules, state reports, or credible aggregators). "
        "Allow reasonable rounding and phrasing differences. If the source is dated 2024–2026 and plausibly reflects 2026 levels, treat as acceptable."
    )
    await add_criterion_with_existence_and_verify(
        evaluator,
        state_node,
        criterion_id=f"{state_display_name}_Salary_Range",
        criterion_desc=f"Provides an annual salary range for high school athletic directors in {state_display_name} (as of 2026)",
        value=sal_value,
        sources=sal_sources,
        claim=sal_claim,
        additional_instruction=sal_instruction
    )

    # (5) State-specific certification program beyond NIAAA
    cert_value = info.state_certification_program
    cert_sources = info.state_certification_sources
    cert_claim = (
        f"{state_display_name} has the following state-specific athletic administrator certification status (beyond NIAAA): {cert_value}."
    )
    cert_instruction = (
        "Assess whether the provided sources explicitly confirm the presence (program name, official state-run certification) "
        "or the absence (no state-run program beyond NIAAA) of a state-specific athletic administrator certification. "
        "If the claim is 'no state-specific program', require strong evidence from official state/association pages or authoritative references."
    )
    await add_criterion_with_existence_and_verify(
        evaluator,
        state_node,
        criterion_id=f"{state_display_name}_State_Certification",
        criterion_desc=f"Indicates whether {state_display_name} offers a state-specific athletic administrator certification program beyond NIAAA certifications",
        value=cert_value,
        sources=cert_sources,
        claim=cert_claim,
        additional_instruction=cert_instruction
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
    Evaluate an answer for the multi-state high school athletic director requirements task.
    """
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

    # Extract structured information for the three states
    extraction = await evaluator.extract(
        prompt=prompt_extract_three_states(),
        template_class=ThreeStatesExtraction,
        extraction_name="three_states_requirements"
    )

    # Top-level (non-critical) aggregator to allow partial credit across states
    main_node = evaluator.add_parallel(
        id="Athletic_Director_Requirements_Three_States",
        desc="Evaluate comprehensive research on high school athletic director requirements across Texas, Ohio, and Alabama",
        parent=root,
        critical=False
    )

    # Verify each state
    await verify_state_requirements(
        evaluator,
        main_node,
        state_key="texas",
        state_display_name="Texas",
        info=extraction.texas
    )

    await verify_state_requirements(
        evaluator,
        main_node,
        state_key="ohio",
        state_display_name="Ohio",
        info=extraction.ohio
    )

    await verify_state_requirements(
        evaluator,
        main_node,
        state_key="Alabama",  # display uses proper capitalization
        state_display_name="Alabama",
        info=extraction.alabama
    )

    return evaluator.get_summary()