import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_state_rental_regulations"
TASK_DESCRIPTION = """
I am starting a property management company that will operate across multiple states. To ensure compliance with state-specific regulations, I need to understand the key differences in rental property laws.

Compare rental property regulations for the following four states: Texas, Florida, Maryland, and California.

For each state, provide the following information with supporting reference URLs:

1. Security Deposit Regulations: What is the maximum security deposit amount allowed (if any limit exists) and what is the deadline for returning security deposits to tenants after they vacate?

2. Property Management Licensing Requirements: What type of license is required to operate a property management business in the state, and are there any specific educational or experience requirements?

3. Eviction Notice Requirements: What is the minimum notice period required before filing eviction proceedings for non-payment of rent?

Each piece of information must include a direct reference URL to an official state government website, established legal resource, or recognized property management organization that confirms the requirement.
"""

STATE_PRETTY_NAME = {
    "texas": "Texas",
    "florida": "Florida",
    "maryland": "Maryland",
    "california": "California",
}

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class StateSecurityDeposit(BaseModel):
    max_amount: Optional[str] = None  # e.g., "no statutory limit", "2 months' rent", "1.5x monthly rent"
    return_deadline: Optional[str] = None  # e.g., "30 days", "21 days", "45 days"
    sources: List[str] = Field(default_factory=list)


class StateLicensing(BaseModel):
    license_type: Optional[str] = None  # e.g., "real estate broker license", "real estate license"
    education_experience: Optional[str] = None  # e.g., "63-hour pre-licensing course", "2 years experience", "none"
    sources: List[str] = Field(default_factory=list)


class StateEviction(BaseModel):
    notice_nonpayment: Optional[str] = None  # e.g., "3-day notice", "10 days", "no notice required"
    sources: List[str] = Field(default_factory=list)


class StateRegulation(BaseModel):
    security_deposit: Optional[StateSecurityDeposit] = None
    licensing: Optional[StateLicensing] = None
    eviction: Optional[StateEviction] = None


class FourStatesExtraction(BaseModel):
    texas: Optional[StateRegulation] = None
    florida: Optional[StateRegulation] = None
    maryland: Optional[StateRegulation] = None
    california: Optional[StateRegulation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_four_states() -> str:
    return """
Extract the requested regulatory information for the following four U.S. states as presented in the answer text: Texas, Florida, Maryland, and California.

For each state, extract these three sections with the exact field names and structure below. Do not infer or invent; use only what the answer explicitly states. If the answer states that there is "no limit" or "no statutory cap", record that exact language in the 'max_amount' field. If any field is missing, set it to null; for sources, return an empty array if none are explicitly provided as URLs.

Return a single JSON object with the following structure:

{
  "texas": {
    "security_deposit": {
      "max_amount": string | null,
      "return_deadline": string | null,
      "sources": string[]   // URLs explicitly mentioned in the answer for security-deposit info
    },
    "licensing": {
      "license_type": string | null,
      "education_experience": string | null,
      "sources": string[]   // URLs explicitly mentioned in the answer for licensing info
    },
    "eviction": {
      "notice_nonpayment": string | null,
      "sources": string[]   // URLs explicitly mentioned in the answer for eviction (non-payment) notice info
    }
  },
  "florida": { ...same structure... },
  "maryland": { ...same structure... },
  "california": { ...same structure... }
}

Special instructions for URL extraction:
- Extract only URLs that are explicitly present in the answer; they may be in plain form or markdown links. Do not create or infer any URL.
- Include only valid URLs. If a URL is missing a protocol, prepend "http://".
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_text(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _nonempty_urls(urls: Optional[List[str]]) -> bool:
    return isinstance(urls, list) and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


# --------------------------------------------------------------------------- #
# Verification builders (per-category)                                        #
# --------------------------------------------------------------------------- #
async def verify_security_deposit(
    evaluator: Evaluator,
    state_parent,
    state_key: str,
    state_name: str,
    data: Optional[StateSecurityDeposit],
) -> Tuple[Optional[StateSecurityDeposit], Optional[Any]]:
    """
    Build and verify the Security Deposit subtree for a state.
    Returns (data_obj, existence_node) for potential cross-prerequisites.
    """
    # Sequential subtree to enforce existence gating before detailed checks
    secdep_node = evaluator.add_sequential(
        id=f"{state_key}_security_deposit",
        desc=f"{state_name} security deposit regulations are correctly stated and supported",
        parent=state_parent,
        critical=True  # Critical under the state's verification
    )

    # Existence check: both return_deadline and at least one URL must be provided; max_amount can be a textual "no limit"
    exists = (data is not None) and _nonempty_text(data.return_deadline) and _nonempty_urls(data.sources) and _nonempty_text(data.max_amount)
    secdep_exist = evaluator.add_custom_node(
        result=bool(exists),
        id=f"{state_key}_secdep_exists",
        desc=f"{state_name} security deposit info is provided with sources",
        parent=secdep_node,
        critical=True
    )

    # Max amount verification (supports "no limit"/"no statutory cap" phrasing)
    max_leaf = evaluator.add_leaf(
        id=f"{state_key}_secdep_max_supported",
        desc=f"{state_name} maximum security deposit amount is accurately stated and supported",
        parent=secdep_node,
        critical=True
    )
    max_amount = (data.max_amount if data else None) or ""
    await evaluator.verify(
        claim=f"In {state_name}, the maximum residential security deposit amount allowed by law is {max_amount}. "
              f"If the statement says 'no limit' or 'no statutory cap', interpret it as no statutory maximum cap.",
        node=max_leaf,
        sources=_safe_urls(data.sources if data else []),
        additional_instruction="Check the statute or official guidance. Accept equivalent phrasings (e.g., 'no statutory cap', 'no maximum limit')."
    )

    # Return deadline verification
    deadline_leaf = evaluator.add_leaf(
        id=f"{state_key}_secdep_deadline_supported",
        desc=f"{state_name} security deposit return deadline is accurately stated and supported",
        parent=secdep_node,
        critical=True
    )
    deadline = (data.return_deadline if data else None) or ""
    await evaluator.verify(
        claim=f"In {state_name}, landlords must return the residential security deposit within {deadline} after the tenant vacates or lease termination, as required by law.",
        node=deadline_leaf,
        sources=_safe_urls(data.sources if data else []),
        additional_instruction="Allow reasonable wording differences (e.g., '21 days' vs 'within 21 days'). Focus on the official deadline for returning deposits."
    )

    return data, secdep_exist


async def verify_licensing(
    evaluator: Evaluator,
    state_parent,
    state_key: str,
    state_name: str,
    data: Optional[StateLicensing],
) -> Tuple[Optional[StateLicensing], Optional[Any]]:
    """
    Build and verify the Licensing subtree for a state.
    Returns (data_obj, existence_node) for potential cross-prerequisites.
    """
    licensing_node = evaluator.add_sequential(
        id=f"{state_key}_licensing",
        desc=f"{state_name} property management licensing requirements are correctly stated and supported",
        parent=state_parent,
        critical=True
    )

    exists = (data is not None) and _nonempty_text(data.license_type) and _nonempty_text(data.education_experience) and _nonempty_urls(data.sources)
    licensing_exist = evaluator.add_custom_node(
        result=bool(exists),
        id=f"{state_key}_licensing_exists",
        desc=f"{state_name} licensing info (license type and education/experience) is provided with sources",
        parent=licensing_node,
        critical=True
    )

    # License type verification
    type_leaf = evaluator.add_leaf(
        id=f"{state_key}_license_type_supported",
        desc=f"{state_name} required license type for property management is accurately stated and supported",
        parent=licensing_node,
        critical=True
    )
    license_type = (data.license_type if data else None) or ""
    await evaluator.verify(
        claim=f"In {state_name}, to perform property management for others for compensation, the required license is: {license_type}.",
        node=type_leaf,
        sources=_safe_urls(data.sources if data else []),
        additional_instruction="Verify using state licensing authority or statute. Consider exemptions as context, but the core requirement should match the source."
    )

    # Education/experience verification
    edu_leaf = evaluator.add_leaf(
        id=f"{state_key}_license_edu_supported",
        desc=f"{state_name} specific educational or experience requirements are accurately stated and supported",
        parent=licensing_node,
        critical=True
    )
    edu = (data.education_experience if data else None) or ""
    await evaluator.verify(
        claim=f"In {state_name}, the specific educational and/or experience requirements for the required license are: {edu}.",
        node=edu_leaf,
        sources=_safe_urls(data.sources if data else []),
        additional_instruction="Confirm exact hours, coursework, or experience if stated. If 'none' is stated, confirm that no specific requirement exists."
    )

    return data, licensing_exist


async def verify_eviction_nonpayment(
    evaluator: Evaluator,
    state_parent,
    state_key: str,
    state_name: str,
    data: Optional[StateEviction],
) -> Tuple[Optional[StateEviction], Optional[Any]]:
    """
    Build and verify the Eviction (non-payment) subtree for a state.
    Returns (data_obj, existence_node) for potential cross-prerequisites.
    """
    eviction_node = evaluator.add_sequential(
        id=f"{state_key}_eviction",
        desc=f"{state_name} eviction notice period for non-payment is correctly stated and supported",
        parent=state_parent,
        critical=True
    )

    exists = (data is not None) and _nonempty_text(data.notice_nonpayment) and _nonempty_urls(data.sources)
    eviction_exist = evaluator.add_custom_node(
        result=bool(exists),
        id=f"{state_key}_eviction_exists",
        desc=f"{state_name} non-payment eviction notice info is provided with sources",
        parent=eviction_node,
        critical=True
    )

    notice_leaf = evaluator.add_leaf(
        id=f"{state_key}_eviction_notice_supported",
        desc=f"{state_name} minimum notice before filing eviction for non-payment is accurately stated and supported",
        parent=eviction_node,
        critical=True
    )
    notice = (data.notice_nonpayment if data else None) or ""
    await evaluator.verify(
        claim=f"In {state_name}, before filing an eviction case for non-payment of rent, the minimum notice period required is {notice}.",
        node=notice_leaf,
        sources=_safe_urls(data.sources if data else []),
        additional_instruction="Interpret the requirement as the pre-filing statutory notice (e.g., 3-day notice to pay or quit). Allow reasonable wording variants; confirm via official or recognized sources."
    )

    return data, eviction_exist


async def verify_sources_authority(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    urls: List[str],
    *,
    extra_prereq: Optional[Any] = None,
    critical: bool = True
) -> None:
    """
    Add a leaf to ensure that at least one of the provided URLs is an official state government site
    or a recognized, high-authority legal resource/recognized property management organization.
    This is a separate check to enforce the source-authority requirement.
    """
    authority_leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )

    # If an existence node is provided (from a different subtree), make this verification depend on it
    extra_nodes = [extra_prereq] if extra_prereq is not None else None

    await evaluator.verify(
        claim="This page is an official state government or judiciary/legislature website (e.g., .gov domains or state code/judicial sites) "
              "OR a well-established, recognized legal resource or property management association that authoritatively discusses the topic.",
        node=authority_leaf,
        sources=_safe_urls(urls),
        extra_prerequisites=extra_nodes,
        additional_instruction="Use the domain, publisher, page header, and branding to assess authority. "
                               "Accept: state code/legislature/court/agency sites; recognized legal resources (e.g., state bar, official code hosts); "
                               "recognized property management organizations. Do not count generic blogs or marketing pages."
    )


# --------------------------------------------------------------------------- #
# State-level verification builder                                            #
# --------------------------------------------------------------------------- #
async def verify_state(
    evaluator: Evaluator,
    root_parent,
    state_key: str,
    state_data: Optional[StateRegulation],
) -> None:
    """
    Build the verification subtree for one state.
    """
    state_name = STATE_PRETTY_NAME[state_key]
    state_node = evaluator.add_parallel(
        id=f"{state_key}_regulations",
        desc=f"Document {state_name} rental property regulations including security deposit laws, licensing, and eviction notice procedures",
        parent=root_parent,
        critical=False  # allow partial credit across categories within a state
    )

    # Security Deposit
    secdep_data = state_data.security_deposit if state_data else None
    secdep_data, secdep_exist_node = await verify_security_deposit(
        evaluator, state_node, state_key, state_name, secdep_data
    )
    await verify_sources_authority(
        evaluator,
        parent_node=state_node,
        node_id=f"{state_key}_secdep_sources_authority",
        desc=f"{state_name} security deposit sources are authoritative (official or recognized)",
        urls=_safe_urls(secdep_data.sources if secdep_data else []),
        extra_prereq=secdep_exist_node,
        critical=True
    )

    # Licensing
    licensing_data = state_data.licensing if state_data else None
    licensing_data, licensing_exist_node = await verify_licensing(
        evaluator, state_node, state_key, state_name, licensing_data
    )
    await verify_sources_authority(
        evaluator,
        parent_node=state_node,
        node_id=f"{state_key}_licensing_sources_authority",
        desc=f"{state_name} licensing sources are authoritative (official or recognized)",
        urls=_safe_urls(licensing_data.sources if licensing_data else []),
        extra_prereq=licensing_exist_node,
        critical=True
    )

    # Eviction (non-payment)
    eviction_data = state_data.eviction if state_data else None
    eviction_data, eviction_exist_node = await verify_eviction_nonpayment(
        evaluator, state_node, state_key, state_name, eviction_data
    )
    await verify_sources_authority(
        evaluator,
        parent_node=state_node,
        node_id=f"{state_key}_eviction_sources_authority",
        desc=f"{state_name} non-payment eviction sources are authoritative (official or recognized)",
        urls=_safe_urls(eviction_data.sources if eviction_data else []),
        extra_prereq=eviction_exist_node,
        critical=True
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
    Evaluate an answer for the Four-State Rental Regulation Comparison task.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator (root as non-critical to allow partial credit across states)
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_four_states(),
        template_class=FourStatesExtraction,
        extraction_name="four_states_extraction",
        additional_instruction="Ensure each 'sources' array only contains explicit URLs present in the answer, and assign 'no limit' if the answer explicitly states there is no statutory cap."
    )

    # Build and verify per-state subtrees
    state_tasks = []
    for sk in ["texas", "florida", "maryland", "california"]:
        state_reg: Optional[StateRegulation] = getattr(extraction, sk)
        task = verify_state(evaluator, root, sk, state_reg)
        state_tasks.append(task)

    # Run verifications
    await asyncio.gather(*state_tasks)

    # Return structured evaluation summary
    return evaluator.get_summary()