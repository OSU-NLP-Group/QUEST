import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "legislative_procedures_states"
TASK_DESCRIPTION = (
    "Identify four different U.S. states that demonstrate distinct legislative procedural requirements, "
    "with each state satisfying its specific criteria and provide documentation URLs from official government sources, "
    "Ballotpedia, or NCSL."
)

# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class State1Info(BaseModel):
    name: Optional[str] = None
    explanation: Optional[str] = None
    quorum_value: Optional[str] = None
    tax_threshold_value: Optional[str] = None
    quorum_urls: List[str] = Field(default_factory=list)
    tax_urls: List[str] = Field(default_factory=list)


class State2Info(BaseModel):
    name: Optional[str] = None
    explanation: Optional[str] = None
    veto_override_value: Optional[str] = None
    veto_urls: List[str] = Field(default_factory=list)
    uniformity_urls: List[str] = Field(default_factory=list)


class State3Info(BaseModel):
    name: Optional[str] = None
    explanation: Optional[str] = None
    variable_thresholds_note: Optional[str] = None
    regular_veto_value: Optional[str] = None
    variable_veto_urls: List[str] = Field(default_factory=list)
    regular_veto_urls: List[str] = Field(default_factory=list)


class State4Info(BaseModel):
    name: Optional[str] = None
    explanation: Optional[str] = None
    amendment_approval_value: Optional[str] = None
    amendment_urls: List[str] = Field(default_factory=list)


class StatesExtraction(BaseModel):
    state1: Optional[State1Info] = None
    state2: Optional[State2Info] = None
    state3: Optional[State3Info] = None
    state4: Optional[State4Info] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states_info() -> str:
    return """
    Extract structured information for four different U.S. states that the answer claims meet the specified legislative procedural requirements.
    For each state, return the following JSON fields. Only include URLs explicitly present in the answer text. If a field is missing, return null or an empty array as appropriate.

    state1:
      - name: State name
      - explanation: Brief description of how the state satisfies BOTH (a) a two-thirds quorum requirement to conduct official legislative business AND (b) a supermajority requirement to increase or impose taxes.
      - quorum_value: The quorum threshold value mentioned (e.g., "two-thirds", "2/3", "66.67%"), if present.
      - tax_threshold_value: The tax vote threshold mentioned (e.g., "two-thirds", "three-fifths", "60%"), if present.
      - quorum_urls: List of URLs cited for the quorum requirement. Include all URLs mentioned for this item.
      - tax_urls: List of URLs cited for the tax supermajority requirement. Include all URLs mentioned for this item.

    state2:
      - name: State name
      - explanation: Brief description of how the state satisfies BOTH (a) a three-fifths (3/5) veto override threshold in both chambers AND (b) a uniform/non-variable override threshold that does NOT vary by bill type.
      - veto_override_value: The override threshold value mentioned (e.g., "three-fifths", "3/5", "60%"), if present.
      - veto_urls: List of URLs cited for the 3/5 override requirement (any chamber or both).
      - uniformity_urls: List of URLs cited to support that the override threshold does NOT vary by bill type.

    state3:
      - name: State name
      - explanation: Brief description of how the state satisfies BOTH (a) variable veto override thresholds depending on bill type AND (b) a three-fifths threshold for regular bills.
      - variable_thresholds_note: Any text noting categories (e.g., emergency, appropriations, taxes) and their thresholds, if present.
      - regular_veto_value: The threshold for regular bills (e.g., "three-fifths", "3/5", "60%"), if present.
      - variable_veto_urls: List of URLs documenting variable veto override thresholds by bill type.
      - regular_veto_urls: List of URLs documenting the 3/5 threshold for regular bills.

    state4:
      - name: State name
      - explanation: Brief description explaining the voter approval threshold for constitutional amendments (>50%).
      - amendment_approval_value: The threshold value mentioned (e.g., "55%", "60%", "two-thirds"), if present.
      - amendment_urls: List of URLs documenting the voter approval requirement for constitutional amendments.

    Notes:
    - Include only valid, complete URLs as they appear in the answer (plain or markdown links).
    - Do not invent URLs.
    - If a URL is missing a protocol, prepend http://.
    - Keep arrays empty if no URLs are cited.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_allowed_source(url: str) -> bool:
    """
    Basic heuristic to determine if a URL belongs to an allowed source:
    - Official government sources (commonly '.gov' domains or state legislature/state government domains)
    - Ballotpedia (ballotpedia.org)
    - NCSL (ncsl.org)

    This is a lightweight check; it is not exhaustive.
    """
    if not url:
        return False
    u = url.lower()
    allowed_keywords = [
        ".gov",
        "ballotpedia.org",
        "ncsl.org",
        ".state.us",  # some legacy state domains
        "legislature.",  # state legislature subdomains
        "/legislature/",  # path indicating legislature resource
        "senate.", "house.",  # chambers subdomains (commonly on .gov domains)
    ]
    return any(k in u for k in allowed_keywords)


def any_allowed_url(urls: List[str]) -> bool:
    return any(is_allowed_source(u) for u in urls)


def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def non_empty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_state1_tree(evaluator: Evaluator, parent_node, s1: Optional[State1Info]) -> None:
    node = evaluator.add_parallel(
        id="State_1_Requirements",
        desc="State 1: 2/3 quorum requirement + tax-increase supermajority requirement, with explanation and sources",
        parent=parent_node,
        critical=False,
    )

    name_val = s1.name if s1 else None
    # Identification (critical)
    evaluator.add_custom_node(
        result=non_empty_text(name_val),
        id="State_1_Identification",
        desc="Provide the State 1 name",
        parent=node,
        critical=True,
    )

    # Quorum Requirement group (critical)
    quorum_group = evaluator.add_parallel(
        id="State_1_Quorum_Requirement",
        desc="State 1 has a two-thirds (2/3) quorum requirement to conduct legislative business",
        parent=node,
        critical=True,
    )

    # Quorum Value (critical leaf verified by URLs)
    quorum_value_leaf = evaluator.add_leaf(
        id="State_1_Quorum_Value",
        desc="Quorum threshold equals 2/3 of members (not a simple majority)",
        parent=quorum_group,
        critical=True,
    )
    quorum_claim = (
        f"{name_val} requires a two-thirds (2/3) quorum of members to conduct official legislative business, "
        f"which is greater than a simple majority."
        if name_val else
        "The state requires a two-thirds (2/3) quorum of members to conduct official legislative business, which is greater than a simple majority."
    )
    await evaluator.verify(
        claim=quorum_claim,
        node=quorum_value_leaf,
        sources=safe_urls(s1.quorum_urls if s1 else []),
        additional_instruction="Confirm explicitly in the cited source that the quorum to conduct business is two-thirds (≈66.7%). "
                              "Allow equivalent phrasing (e.g., 2/3, two-thirds).",
    )

    # Quorum Documentation (critical existence of allowed source)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s1.quorum_urls if s1 else [])),
        id="State_1_Quorum_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting the 2/3 quorum requirement",
        parent=quorum_group,
        critical=True,
    )

    # Tax Supermajority Requirement group (critical)
    tax_group = evaluator.add_parallel(
        id="State_1_Tax_Supermajority_Requirement",
        desc="State 1 has a supermajority (>50%) voting requirement to increase or impose taxes",
        parent=node,
        critical=True,
    )

    # Tax supermajority Value (critical leaf verified by URLs)
    tax_value_leaf = evaluator.add_leaf(
        id="State_1_Tax_Supermajority_Value",
        desc="Tax increase/imposition requires a threshold greater than 50%",
        parent=tax_group,
        critical=True,
    )
    tax_claim = (
        f"In {name_val}, increasing or imposing taxes requires a supermajority vote greater than a simple majority (>50%)."
        if name_val else
        "Increasing or imposing taxes requires a supermajority vote greater than a simple majority (>50%)."
    )
    await evaluator.verify(
        claim=tax_claim,
        node=tax_value_leaf,
        sources=safe_urls(s1.tax_urls if s1 else []),
        additional_instruction="Verify from the cited source that the threshold to raise or impose taxes exceeds 50% (e.g., 3/5, 2/3, 60%). "
                              "Equivalents (three-fifths=60%, two-thirds≈66.7%) are acceptable.",
    )

    # Tax Documentation (critical existence of allowed source)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s1.tax_urls if s1 else [])),
        id="State_1_Tax_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting the tax supermajority requirement",
        parent=tax_group,
        critical=True,
    )

    # Explanation (critical presence)
    evaluator.add_custom_node(
        result=non_empty_text(s1.explanation if s1 else None),
        id="State_1_Explanation",
        desc="Provide a brief description explaining how State 1 satisfies both the quorum and tax supermajority criteria",
        parent=node,
        critical=True,
    )


async def build_state2_tree(evaluator: Evaluator, parent_node, s1: Optional[State1Info], s2: Optional[State2Info]) -> None:
    node = evaluator.add_parallel(
        id="State_2_Requirements",
        desc="State 2: 3/5 veto override in both chambers + no bill-type-specific override thresholds, with explanation and sources",
        parent=parent_node,
        critical=False,
    )

    name2 = s2.name if s2 else None
    name1 = s1.name if s1 else None

    # Identification (critical)
    evaluator.add_custom_node(
        result=non_empty_text(name2),
        id="State_2_Identification",
        desc="Provide the State 2 name",
        parent=node,
        critical=True,
    )

    # Distinctness from State 1 (critical)
    evaluator.add_custom_node(
        result=(non_empty_text(name2) and non_empty_text(name1) and name2.strip() != name1.strip()),
        id="State_2_Distinctness",
        desc="State 2 must be different from State 1",
        parent=node,
        critical=True,
    )

    # Veto Override Threshold group (critical)
    veto_group = evaluator.add_parallel(
        id="State_2_Veto_Override_Threshold",
        desc="State 2 requires a three-fifths (3/5) vote in both legislative chambers to override a gubernatorial veto",
        parent=node,
        critical=True,
    )

    # Veto value (critical leaf verified by URLs)
    veto_value_leaf = evaluator.add_leaf(
        id="State_2_Veto_Value",
        desc="Override threshold equals 3/5 in both chambers",
        parent=veto_group,
        critical=True,
    )
    veto_claim = (
        f"{name2} requires a three-fifths (3/5) vote in both the House and the Senate to override a gubernatorial veto."
        if name2 else
        "The state requires a three-fifths (3/5) vote in both the House and the Senate to override a gubernatorial veto."
    )
    await evaluator.verify(
        claim=veto_claim,
        node=veto_value_leaf,
        sources=safe_urls(s2.veto_urls if s2 else []),
        additional_instruction="Confirm explicitly that both chambers require three-fifths (≈60%) to override a veto, allowing equivalent phrasing (e.g., 60%, 3/5).",
    )

    # Veto documentation from allowed sources (critical)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s2.veto_urls if s2 else [])),
        id="State_2_Veto_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting the 3/5 veto override rule",
        parent=veto_group,
        critical=True,
    )

    # No variable veto thresholds group (critical)
    uniform_group = evaluator.add_parallel(
        id="State_2_No_Variable_Veto_Thresholds",
        desc="State 2 does NOT have bill-type-specific veto override thresholds (override threshold does not vary by bill type)",
        parent=node,
        critical=True,
    )

    # Uniformity check (critical leaf verified by URLs)
    uniform_leaf = evaluator.add_leaf(
        id="State_2_No_Variation_Check",
        desc="Override threshold is uniform across bill types (no variable thresholds based on bill category)",
        parent=uniform_group,
        critical=True,
    )
    uniform_claim = (
        f"In {name2}, the veto override threshold is uniform and does not vary by bill type/category."
        if name2 else
        "The veto override threshold is uniform and does not vary by bill type/category."
    )
    await evaluator.verify(
        claim=uniform_claim,
        node=uniform_leaf,
        sources=safe_urls(s2.uniformity_urls if s2 else []),
        additional_instruction="Check the cited legal text or summary to ensure the override threshold is a single value applied to all bill types (no separate thresholds for emergency, appropriations, taxes, etc.).",
    )

    # Uniformity documentation from allowed sources (critical)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s2.uniformity_urls if s2 else [])),
        id="State_2_Uniformity_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) supporting that the override threshold does not vary by bill type",
        parent=uniform_group,
        critical=True,
    )

    # Explanation (critical presence)
    evaluator.add_custom_node(
        result=non_empty_text(s2.explanation if s2 else None),
        id="State_2_Explanation",
        desc="Provide a brief description explaining how State 2 satisfies both the 3/5 override and the non-variability requirement",
        parent=node,
        critical=True,
    )


async def build_state3_tree(evaluator: Evaluator, parent_node, s1: Optional[State1Info], s2: Optional[State2Info], s3: Optional[State3Info]) -> None:
    node = evaluator.add_parallel(
        id="State_3_Requirements",
        desc="State 3: variable veto override thresholds by bill type + 3/5 for regular bills, with explanation and sources",
        parent=parent_node,
        critical=False,
    )

    name3 = s3.name if s3 else None
    name1 = s1.name if s1 else None
    name2 = s2.name if s2 else None

    # Identification (critical)
    evaluator.add_custom_node(
        result=non_empty_text(name3),
        id="State_3_Identification",
        desc="Provide the State 3 name",
        parent=node,
        critical=True,
    )

    # Distinctness from States 1 and 2 (critical)
    evaluator.add_custom_node(
        result=(non_empty_text(name3) and non_empty_text(name1) and non_empty_text(name2)
                and name3.strip() != name1.strip() and name3.strip() != name2.strip()),
        id="State_3_Distinctness",
        desc="State 3 must be different from States 1 and 2",
        parent=node,
        critical=True,
    )

    # Variable veto thresholds group (critical)
    variable_group = evaluator.add_parallel(
        id="State_3_Variable_Veto_Thresholds",
        desc="State 3 has different veto override voting thresholds depending on bill type/category",
        parent=node,
        critical=True,
    )

    # Variation present (critical leaf verified by URLs)
    variation_leaf = evaluator.add_leaf(
        id="State_3_Variation_Present",
        desc="There exist at least two different veto-override thresholds that apply to different bill categories",
        parent=variable_group,
        critical=True,
    )
    variation_claim = (
        f"In {name3}, veto override thresholds vary by bill type, with different thresholds for certain categories (e.g., emergency, appropriations, taxes) compared to regular legislation."
        if name3 else
        "Veto override thresholds vary by bill type, with different thresholds for certain categories compared to regular legislation."
    )
    await evaluator.verify(
        claim=variation_claim,
        node=variation_leaf,
        sources=safe_urls(s3.variable_veto_urls if s3 else []),
        additional_instruction="Look for constitutional or statutory text or authoritative summaries indicating multiple override thresholds depending on bill category.",
    )

    # Variation documentation from allowed sources (critical)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s3.variable_veto_urls if s3 else [])),
        id="State_3_Variation_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting variable veto override thresholds by bill type",
        parent=variable_group,
        critical=True,
    )

    # Regular bills 3/5 group (critical)
    regular_group = evaluator.add_parallel(
        id="State_3_Regular_Bills_Three_Fifths",
        desc="State 3 requires a 3/5 vote to override vetoes of regular (non-special-category) bills",
        parent=node,
        critical=True,
    )

    # Regular bill value (critical leaf verified by URLs)
    regular_value_leaf = evaluator.add_leaf(
        id="State_3_Regular_Bill_Value",
        desc="Regular-bill veto override threshold equals 3/5",
        parent=regular_group,
        critical=True,
    )
    regular_claim = (
        f"For regular bills in {name3}, a three-fifths (3/5) vote is required to override a gubernatorial veto."
        if name3 else
        "For regular bills, a three-fifths (3/5) vote is required to override a gubernatorial veto."
    )
    await evaluator.verify(
        claim=regular_claim,
        node=regular_value_leaf,
        sources=safe_urls(s3.regular_veto_urls if s3 else []),
        additional_instruction="Confirm the threshold for regular (non-special-category) bills is three-fifths (≈60%).",
    )

    # Regular bill documentation from allowed sources (critical)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s3.regular_veto_urls if s3 else [])),
        id="State_3_Regular_Bill_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting the 3/5 threshold for regular bills",
        parent=regular_group,
        critical=True,
    )

    # Explanation (critical presence)
    evaluator.add_custom_node(
        result=non_empty_text(s3.explanation if s3 else None),
        id="State_3_Explanation",
        desc="Provide a brief description explaining how State 3 satisfies both the variable-threshold requirement and the 3/5 regular-bill requirement",
        parent=node,
        critical=True,
    )


async def build_state4_tree(evaluator: Evaluator, parent_node, s1: Optional[State1Info], s2: Optional[State2Info], s3: Optional[State3Info], s4: Optional[State4Info]) -> None:
    node = evaluator.add_parallel(
        id="State_4_Requirements",
        desc="State 4: >50% voter approval required to ratify constitutional amendments, with explanation and sources",
        parent=parent_node,
        critical=False,
    )

    name4 = s4.name if s4 else None
    names_prev = [s1.name if s1 else None, s2.name if s2 else None, s3.name if s3 else None]
    # Identification (critical)
    evaluator.add_custom_node(
        result=non_empty_text(name4),
        id="State_4_Identification",
        desc="Provide the State 4 name",
        parent=node,
        critical=True,
    )

    # Distinctness from States 1, 2, 3 (critical)
    distinct_ok = non_empty_text(name4) and all(non_empty_text(n) for n in names_prev) and all(
        name4.strip() != n.strip() for n in names_prev  # type: ignore
    )
    evaluator.add_custom_node(
        result=bool(distinct_ok),
        id="State_4_Distinctness",
        desc="State 4 must be different from States 1, 2, and 3",
        parent=node,
        critical=True,
    )

    # Amendment supermajority group (critical)
    amend_group = evaluator.add_parallel(
        id="State_4_Amendment_Supermajority",
        desc="State 4 requires more than a simple majority (>50%) of voters to approve constitutional amendments",
        parent=node,
        critical=True,
    )

    # Supermajority value (critical leaf verified by URLs)
    amend_value_leaf = evaluator.add_leaf(
        id="State_4_Supermajority_Value",
        desc="Voter approval threshold is strictly greater than 50% (e.g., 55%, 60%, or 2/3)",
        parent=amend_group,
        critical=True,
    )
    amend_claim = (
        f"In {name4}, approving constitutional amendments requires more than a simple majority (>50%) of voters (e.g., 55%, 60%, two-thirds)."
        if name4 else
        "Approving constitutional amendments requires more than a simple majority (>50%) of voters (e.g., 55%, 60%, two-thirds)."
    )
    await evaluator.verify(
        claim=amend_claim,
        node=amend_value_leaf,
        sources=safe_urls(s4.amendment_urls if s4 else []),
        additional_instruction="Confirm the voter approval threshold for constitutional amendments exceeds 50% (e.g., 55%, 60%, two-thirds).",
    )

    # Amendment documentation from allowed sources (critical)
    evaluator.add_custom_node(
        result=any_allowed_url(safe_urls(s4.amendment_urls if s4 else [])),
        id="State_4_Amendment_Documentation",
        desc="Provide at least one URL from an allowed source (official government, Ballotpedia, or NCSL) documenting the amendment approval threshold",
        parent=amend_group,
        critical=True,
    )

    # Explanation (critical presence)
    evaluator.add_custom_node(
        result=non_empty_text(s4.explanation if s4 else None),
        id="State_4_Explanation",
        desc="Provide a brief description explaining how State 4 satisfies the >50% constitutional amendment voter-approval requirement",
        parent=node,
        critical=True,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the legislative procedures states task and return a structured summary.
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
        default_model=model,
    )

    # Extraction
    extracted_states = await evaluator.extract(
        prompt=prompt_extract_states_info(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Build tree for each state
    await build_state1_tree(evaluator, root, extracted_states.state1)
    await build_state2_tree(evaluator, root, extracted_states.state1, extracted_states.state2)
    await build_state3_tree(evaluator, root, extracted_states.state1, extracted_states.state2, extracted_states.state3)
    await build_state4_tree(evaluator, root, extracted_states.state1, extracted_states.state2, extracted_states.state3, extracted_states.state4)

    # Optional: add custom info counts of allowed sources
    def count_allowed(urls: List[str]) -> int:
        return sum(1 for u in urls if is_allowed_source(u))

    custom_info = {
        "state1_allowed_quorum_sources": count_allowed(safe_urls(extracted_states.state1.quorum_urls if extracted_states.state1 else [])),
        "state1_allowed_tax_sources": count_allowed(safe_urls(extracted_states.state1.tax_urls if extracted_states.state1 else [])),
        "state2_allowed_veto_sources": count_allowed(safe_urls(extracted_states.state2.veto_urls if extracted_states.state2 else [])),
        "state2_allowed_uniformity_sources": count_allowed(safe_urls(extracted_states.state2.uniformity_urls if extracted_states.state2 else [])),
        "state3_allowed_variable_sources": count_allowed(safe_urls(extracted_states.state3.variable_veto_urls if extracted_states.state3 else [])),
        "state3_allowed_regular_sources": count_allowed(safe_urls(extracted_states.state3.regular_veto_urls if extracted_states.state3 else [])),
        "state4_allowed_amendment_sources": count_allowed(safe_urls(extracted_states.state4.amendment_urls if extracted_states.state4 else [])),
    }
    evaluator.add_custom_info(custom_info, info_type="allowed_source_counts", info_name="allowed_source_counts_summary")

    return evaluator.get_summary()