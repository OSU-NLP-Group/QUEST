import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "state_constitution_requirements_2023"
TASK_DESCRIPTION = """Which U.S. state meets all of the following constitutional and governmental structure requirements as of December 31, 2023:

1. The state legislature requires a three-fifths (3/5) vote in each chamber to override a gubernatorial veto.

2. The Attorney General is elected by voters (not appointed by the governor, legislature, or courts).

3. The Attorney General serves a 4-year term.

4. There are no term limits for the Attorney General position.

5. The Attorney General must be at least 25 years old to hold office.

6. The Attorney General must have been a state resident for exactly 3 years prior to election.

7. The state Supreme Court Chief Justice is selected by peer vote (chosen by other justices on the court).

8. The Chief Justice serves a 3-year term.

9. The state legislature convenes annually on the second Wednesday of January.

Provide the name of the state and supporting URL references for each requirement.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateRequirementsEvidence(BaseModel):
    """
    Extracted evidence from the agent's answer for the specified requirements.
    URLs must be explicitly present in the answer text. If not provided, leave the list empty.
    """
    state_name: Optional[str] = None

    veto_override_urls: List[str] = Field(default_factory=list)
    ag_elected_urls: List[str] = Field(default_factory=list)
    ag_term_4_years_urls: List[str] = Field(default_factory=list)
    ag_no_term_limits_urls: List[str] = Field(default_factory=list)
    ag_min_age_25_urls: List[str] = Field(default_factory=list)
    ag_residency_exact_3_years_urls: List[str] = Field(default_factory=list)
    chief_selected_peer_vote_urls: List[str] = Field(default_factory=list)
    chief_term_3_years_urls: List[str] = Field(default_factory=list)
    legislature_convenes_second_wed_jan_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_state_requirements() -> str:
    return """
    Extract the single U.S. state named in the answer that is claimed to satisfy all the listed requirements, and collect the supporting URLs that the answer associates with each requirement.

    Return JSON with the following fields:
    - state_name: The exact state name (e.g., "Pennsylvania"). If multiple states are mentioned, choose the single one presented as the final answer. If no state is presented, set to null.

    For each requirement below, extract all supporting URLs explicitly mentioned in the answer for that specific requirement (do NOT invent or infer URLs). If none are given for a requirement, return an empty list.

    1) veto_override_urls: URLs that support "a three-fifths (3/5) vote in each chamber is required to override a gubernatorial veto".
    2) ag_elected_urls: URLs that support "the Attorney General is elected by voters".
    3) ag_term_4_years_urls: URLs that support "the Attorney General serves a 4-year term".
    4) ag_no_term_limits_urls: URLs that support "there are no term limits for the Attorney General".
    5) ag_min_age_25_urls: URLs that support "the Attorney General must be at least 25 years old".
    6) ag_residency_exact_3_years_urls: URLs that support "the Attorney General must have been a state resident for exactly 3 years prior to election".
    7) chief_selected_peer_vote_urls: URLs that support "the state Supreme Court Chief Justice is selected by peer vote (chosen by other justices)".
    8) chief_term_3_years_urls: URLs that support "the Chief Justice serves a 3-year term".
    9) legislature_convenes_second_wed_jan_urls: URLs that support "the legislature convenes annually on the second Wednesday of January".

    Notes:
    - Extract only URLs explicitly present in the answer (plain URLs or markdown links). If a requirement mentions a source by name without a URL, do not include it.
    - Do not include duplicate URLs within a list for the same requirement.
    - Preserve the order as they appear in the answer when possible.
    """


# --------------------------------------------------------------------------- #
# Helper functions for building verification nodes                            #
# --------------------------------------------------------------------------- #
def _has_at_least_one_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


async def add_requirement_check(
    evaluator: Evaluator,
    parent_node,
    requirement_id: str,
    requirement_desc: str,
    claim_text: str,
    sources: List[str],
    add_ins: str,
) -> None:
    """
    Build a sequential verification sub-tree for a specific requirement:
    - Step 1: Existence check (at least one supporting URL provided)
    - Step 2: Verify the claim against the provided URLs
    All nodes are critical, because the overall "State_Identification" node is critical.
    """
    req_node = evaluator.add_sequential(
        id=requirement_id,
        desc=requirement_desc,
        parent=parent_node,
        critical=True
    )

    # Step 1: Existence of at least one supporting URL
    evaluator.add_custom_node(
        result=_has_at_least_one_url(sources),
        id=f"{requirement_id}_sources_provided",
        desc=f"At least one supporting URL is provided for: {requirement_desc}",
        parent=req_node,
        critical=True
    )

    # Step 2: Verify the claim using the provided URLs
    verify_leaf = evaluator.add_leaf(
        id=f"{requirement_id}_supported_by_sources",
        desc=requirement_desc,
        parent=req_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=sources,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the agent's answer for the 'state_constitution_requirements_2023' task
    using the Mind2Web2 evaluation framework.
    """
    # Initialize evaluator (framework root is a non-critical wrapper)
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

    # Extract state name and per-requirement URLs from the answer
    extracted: StateRequirementsEvidence = await evaluator.extract(
        prompt=prompt_extract_state_requirements(),
        template_class=StateRequirementsEvidence,
        extraction_name="state_requirements_evidence"
    )

    # Add a critical node representing the JSON "State_Identification" root
    state_main = evaluator.add_parallel(
        id="State_Identification",
        desc="Answer identifies a single U.S. state that satisfies all listed requirements as of Dec 31, 2023, and provides supporting URL references for each requirement.",
        parent=root,
        critical=True
    )

    # Record the extracted state name as custom info
    evaluator.add_custom_info(
        info={"extracted_state_name": extracted.state_name or None},
        info_type="extraction_summary",
        info_name="state_name"
    )

    # Leaf: State name must be provided (critical)
    evaluator.add_custom_node(
        result=(extracted.state_name is not None and isinstance(extracted.state_name, str) and extracted.state_name.strip() != ""),
        id="State_Name_Provided",
        desc="Provide the name of the U.S. state (one specific state).",
        parent=state_main,
        critical=True
    )

    # Build claims for each requirement using the extracted state name
    state = (extracted.state_name or "").strip()

    # Requirement 1: Veto override 3/5 in each chamber
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="Veto_Override_3_5_Each_Chamber",
        requirement_desc="State requires a three-fifths (3/5) vote in each chamber to override a gubernatorial veto, with at least one supporting URL reference.",
        claim_text=f"In {state}, overriding a gubernatorial veto requires a three-fifths (3/5) vote in each chamber of the state legislature.",
        sources=extracted.veto_override_urls,
        add_ins="Confirm the override threshold is 3/5 in each chamber (House and Senate/Assembly). Prefer constitutional/statutory sources. As of Dec 31, 2023."
    )

    # Requirement 2: AG elected by voters
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="AG_Elected_By_Voters",
        requirement_desc="Attorney General is elected by voters (not appointed), with at least one supporting URL reference.",
        claim_text=f"In {state}, the Attorney General is elected by the state's voters (not appointed by the governor, legislature, or courts).",
        sources=extracted.ag_elected_urls,
        add_ins="Verify the Attorney General is elected in statewide elections. Reject sources indicating appointment. As of Dec 31, 2023."
    )

    # Requirement 3: AG serves 4-year term
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="AG_Term_4_Years",
        requirement_desc="Attorney General serves a 4-year term, with at least one supporting URL reference.",
        claim_text=f"In {state}, the Attorney General serves a 4-year term.",
        sources=extracted.ag_term_4_years_urls,
        add_ins="Confirm the specified term length is four years for the Attorney General. Prefer constitution/statute pages. As of Dec 31, 2023."
    )

    # Requirement 4: No AG term limits
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="AG_No_Term_Limits",
        requirement_desc="There are no term limits for the Attorney General position, with at least one supporting URL reference.",
        claim_text=f"In {state}, there are no term limits for the office of Attorney General.",
        sources=extracted.ag_no_term_limits_urls,
        add_ins="Check whether statutes or constitution impose any maximum number of terms on the Attorney General. As of Dec 31, 2023."
    )

    # Requirement 5: AG minimum age 25
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="AG_Minimum_Age_25",
        requirement_desc="Attorney General must be at least 25 years old to hold office, with at least one supporting URL reference.",
        claim_text=f"In {state}, a person must be at least 25 years old to be eligible to serve as Attorney General.",
        sources=extracted.ag_min_age_25_urls,
        add_ins="Confirm the minimum age requirement is 25 for the Attorney General eligibility. As of Dec 31, 2023."
    )

    # Requirement 6: AG residency exactly 3 years prior to election
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="AG_Residency_Exactly_3_Years",
        requirement_desc="Attorney General must have been a state resident for exactly 3 years prior to election, with at least one supporting URL reference.",
        claim_text=f"In {state}, a candidate for Attorney General must have been a resident of the state for exactly 3 years prior to election.",
        sources=extracted.ag_residency_exact_3_years_urls,
        add_ins="Confirm the residency requirement is exactly 3 years (not 'at least' or any other duration). As of Dec 31, 2023."
    )

    # Requirement 7: Chief Justice selected by peer vote
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="Chief_Justice_Selected_By_Peer_Vote",
        requirement_desc="State Supreme Court Chief Justice is selected by peer vote (chosen by other justices), with at least one supporting URL reference.",
        claim_text=f"In {state}, the Chief Justice of the state Supreme Court is selected by a vote of the other justices on the court.",
        sources=extracted.chief_selected_peer_vote_urls,
        add_ins="Confirm the selection mechanism is internal peer vote by the justices (not appointment by governor or seniority rule). As of Dec 31, 2023."
    )

    # Requirement 8: Chief Justice serves a 3-year term
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="Chief_Justice_Term_3_Years",
        requirement_desc="Chief Justice serves a 3-year term, with at least one supporting URL reference.",
        claim_text=f"In {state}, the Chief Justice serves a 3-year term.",
        sources=extracted.chief_term_3_years_urls,
        add_ins="Confirm the Chief Justice's term length is three years. As of Dec 31, 2023."
    )

    # Requirement 9: Legislature convenes annually on the second Wednesday of January
    await add_requirement_check(
        evaluator=evaluator,
        parent_node=state_main,
        requirement_id="Legislature_Convenes_Second_Wednesday_January_Annually",
        requirement_desc="State legislature convenes annually on the second Wednesday of January, with at least one supporting URL reference.",
        claim_text=f"In {state}, the state legislature convenes annually on the second Wednesday of January.",
        sources=extracted.legislature_convenes_second_wed_jan_urls,
        add_ins="Verify the regular session convening day is the second Wednesday of January each year. Prefer official legislative calendar or constitution/statutes. As of Dec 31, 2023."
    )

    # Return evaluation summary
    return evaluator.get_summary()