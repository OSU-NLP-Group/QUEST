import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "state_absolute_vaccine_mandates_feb2026"
TASK_DESCRIPTION = """
As of February 2026, following the federal government's significant revision of childhood vaccination recommendations in January 2026, identify three U.S. states that maintain absolute vaccine mandates for school entry (meaning they allow only medical exemptions and do not permit religious or philosophical exemptions). For each state, provide the following information: (1) The state name, (2) Confirmation that the state has an absolute vaccine mandate with no religious or philosophical exemptions allowed, (3) Specific details on whether the state requires each of the following vaccinations for school entry: chickenpox (varicella), hepatitis B, measles, mumps, rubella, polio, pertussis, tetanus, diphtheria, and meningitis, (4) Whether the state law incorporates ACIP (Advisory Committee on Immunization Practices) recommendations, and (5) A reference URL that confirms the state's absolute vaccine mandate status. Note: The federal guidance released on January 5, 2026, reduced universal childhood vaccine recommendations from 18 to 11 diseases, but states may maintain their own more comprehensive requirements.
"""


# -----------------------------------------------------------------------------
# Extraction models
# -----------------------------------------------------------------------------
class VaccineStatus(BaseModel):
    varicella: Optional[str] = None
    hepatitis_b: Optional[str] = None
    measles: Optional[str] = None
    mumps: Optional[str] = None
    rubella: Optional[str] = None
    polio: Optional[str] = None
    pertussis: Optional[str] = None
    tetanus: Optional[str] = None
    diphtheria: Optional[str] = None
    meningitis: Optional[str] = None  # meningococcal (MenACWY) typically


class StateItem(BaseModel):
    name: Optional[str] = None
    mandate_type: Optional[str] = None  # e.g., "only medical exemptions; no religious or philosophical exemptions"
    vaccines: Optional[VaccineStatus] = None
    acip_incorporation: Optional[str] = None  # e.g., "incorporates ACIP by reference" or "does not incorporate"
    reference_url: Optional[str] = None  # a canonical URL confirming absolute mandate status
    source_urls: List[str] = Field(default_factory=list)  # any additional URLs cited in the answer for this state


class StatesExtraction(BaseModel):
    states: List[StateItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_states() -> str:
    return """
Extract up to three U.S. states mentioned in the answer that are claimed to maintain absolute school-entry vaccine mandates as of February 2026 (i.e., only medical exemptions allowed; no religious or philosophical exemptions). For each such state, extract:

- name: The state's name (e.g., "California").
- mandate_type: The exact wording the answer uses to describe the exemption regime for school-entry immunizations (e.g., "only medical exemptions; no religious/philosophical exemptions").
- vaccines: For each of the following, copy exactly what the answer states about whether it is required for school entry: 
  - varicella (a.k.a. chickenpox)
  - hepatitis_b
  - measles
  - mumps
  - rubella
  - polio
  - pertussis
  - tetanus
  - diphtheria
  - meningitis (i.e., meningococcal vaccine such as MenACWY)
  If the answer does not clearly indicate whether a given vaccine is required, set that field to null.
- acip_incorporation: What the answer claims about whether the state law or regulation incorporates ACIP (Advisory Committee on Immunization Practices) recommendations (copy the phrasing; if not mentioned, set to null).
- reference_url: A single primary URL explicitly cited in the answer that confirms the state's "absolute mandate" (medical-only; no religious/philosophical exemptions). If multiple URLs are shown, pick the one most directly confirming the absolute status. If none, set to null.
- source_urls: All URLs cited in the answer for this state (including official state health department pages, statutes, regulations, immunization program sites, etc.). If none, return an empty list.

Return a JSON object with a top-level field "states" which is a list of at most three StateItem objects following the schema.
Important rules:
- Extract only what the answer explicitly provides; do not invent or infer new URLs or statuses.
- For vaccines, preserve whatever wording is present (e.g., "Required", "Not required", "Yes", "No", "Required for grades 7–12", etc.). Do not normalize to yes/no yourself.
- Include full URLs (with http/https). Ignore malformed URLs.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
DISEASE_LABELS = {
    "varicella": "chickenpox (varicella)",
    "hepatitis_b": "hepatitis B",
    "measles": "measles",
    "mumps": "mumps",
    "rubella": "rubella",
    "polio": "polio",
    "pertussis": "pertussis",
    "tetanus": "tetanus",
    "diphtheria": "diphtheria",
    "meningitis": "meningitis (meningococcal vaccine, e.g., MenACWY)",
}


def _is_valid_url(u: Optional[str]) -> bool:
    if not isinstance(u, str):
        return False
    return u.strip().lower().startswith("http")


def gather_state_urls(state: StateItem) -> List[str]:
    all_urls: List[str] = []
    if _is_valid_url(state.reference_url):
        all_urls.append(state.reference_url.strip())
    for u in state.source_urls or []:
        if _is_valid_url(u) and u.strip() not in all_urls:
            all_urls.append(u.strip())
    return all_urls


def normalize_requirement_status(status: Optional[str]) -> Optional[str]:
    """
    Convert free-text requirement status extracted from the answer into one of:
      - 'required'
      - 'not required'
      - 'unknown' (treated as None)
    Heuristic-based, tolerant to varied phrasing. Prefer explicit negation if present.
    """
    if not status or not isinstance(status, str):
        return None
    s = status.strip().lower()

    # Remove punctuation for simpler matching
    s_simple = re.sub(r"[^a-z0-9\s]", " ", s)

    # Negations or explicit "not required"
    neg_markers = [
        "not required", "no requirement", "no required", "not mandated", "no mandate",
        "optional", "exempt", "exemption available", "waiver available", "does not require",
        "not needed", "no need", "personal belief exemption", "religious exemption allowed",
        "philosophical exemption allowed"
    ]
    for m in neg_markers:
        if m in s_simple:
            return "not required"

    # Affirmatives
    pos_markers = [
        "required", "requires", "mandated", "must", "mandatory", "is required", "are required"
    ]
    for m in pos_markers:
        if m in s_simple:
            return "required"

    # Short answers
    if s_simple in {"yes", "y"}:
        return "required"
    if s_simple in {"no", "n"}:
        return "not required"

    return None


def build_vaccine_claim(state_name: str, disease_key: str, norm_status: str) -> str:
    disease_text = DISEASE_LABELS[disease_key]
    if norm_status == "required":
        return f"As of February 2026, {state_name} requires {disease_text} vaccination for school entry (K–12)."
    else:
        return f"As of February 2026, {state_name} does not require {disease_text} vaccination for school entry (K–12)."


VACCINE_ADDITIONAL_INSTRUCTION = (
    "Verify the state's current school-entry immunization requirements as of February 2026 on the cited page(s). "
    "Accept combined vaccines that satisfy the component diseases: e.g., MMR satisfies measles, mumps, and rubella; "
    "DTaP/Tdap/Td cover diphtheria, tetanus, and pertussis depending on grade levels. "
    "If the requirement is grade-specific (e.g., kindergarten entry or 7th grade), you should still treat this "
    "as 'required for school entry' if any K–12 entry point requires it. Focus on state law, regulation, or "
    "official health department requirements; do not rely on federal ACIP guidance alone."
)

MANDATE_TYPE_ADDITIONAL_INSTRUCTION = (
    "Determine whether the state allows only medical exemptions and does not permit religious or philosophical/personal "
    "belief exemptions for school-entry immunizations as of February 2026. If the cited source shows that religious or "
    "philosophical (or personal belief) exemptions are permitted, the claim is not supported. If the source explicitly "
    "states that non-medical exemptions have been removed or are not available, the claim is supported. Give priority "
    "to official state statutes, regulations, or health department pages."
)

ACIP_ADDITIONAL_INSTRUCTION = (
    "Check whether the state's statute or regulation explicitly incorporates or references ACIP (Advisory Committee on "
    "Immunization Practices) recommendations for determining the list/schedule of required school immunizations. "
    "Phrases such as 'as recommended by ACIP' or 'incorporated by reference to ACIP' count as incorporation."
)


# -----------------------------------------------------------------------------
# Verification helpers
# -----------------------------------------------------------------------------
async def add_vaccine_requirement_check(
    evaluator: Evaluator,
    parent_node,
    state_idx: int,
    state_name: Optional[str],
    disease_key: str,
    reported_status: Optional[str],
    urls: List[str],
) -> None:
    """
    Add a leaf node that verifies the correctness of the reported vaccine requirement status.
    If the answer didn't provide a status or there are no URLs, we mark this leaf as failed (non-critical).
    """
    node_id = f"state_{state_idx}_{disease_key}_required"
    disease_text = DISEASE_LABELS[disease_key]
    desc = f"{disease_text.capitalize()} requirement status for the state is correctly reported (as of Feb 2026)"

    norm = normalize_requirement_status(reported_status)

    if not state_name or not norm or not urls:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{disease_text.capitalize()} requirement verification failed due to missing info or sources",
            parent=parent_node,
            critical=False,  # Non-critical per rubric
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=False,  # Non-critical per rubric
    )

    claim = build_vaccine_claim(state_name, disease_key, norm)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=VACCINE_ADDITIONAL_INSTRUCTION,
    )


async def verify_one_state(
    evaluator: Evaluator,
    parent_node,
    state: StateItem,
    state_idx: int,
) -> None:
    """
    Build the subtree for a single state and run all checks according to the rubric.
    """
    # Parent node for the state (non-critical; allows partial credit per state)
    state_node = evaluator.add_parallel(
        id=f"state_{state_idx}",
        desc=f"State #{state_idx} with absolute vaccine mandate identified and verified",
        parent=parent_node,
        critical=False,
    )

    # --- Critical: state name provided (treat as existence check; we cannot 'verify' correctness without GT) ---
    evaluator.add_custom_node(
        result=bool(state and state.name and state.name.strip()),
        id=f"state_{state_idx}_state_name",
        desc="The state is named (non-empty)",
        parent=state_node,
        critical=True,
    )

    # Gather URLs for this state
    urls = gather_state_urls(state)

    # --- Critical: reference URL confirms absolute mandate ---
    # If missing reference_url, fail this critical node immediately
    if not _is_valid_url(state.reference_url):
        evaluator.add_custom_node(
            result=False,
            id=f"state_{state_idx}_reference_url",
            desc="A valid reference URL confirming absolute mandate status is provided",
            parent=state_node,
            critical=True,
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id=f"state_{state_idx}_reference_url",
            desc="Reference URL confirms absolute (medical-only) mandate status",
            parent=state_node,
            critical=True,
        )
        ref_claim = (
            f"As of February 2026, {state.name} allows only medical exemptions and does not permit "
            f"religious or philosophical/personal belief exemptions to K–12 school-entry immunization requirements."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=state.reference_url,
            additional_instruction=MANDATE_TYPE_ADDITIONAL_INSTRUCTION,
        )

    # --- Critical: mandate_type (substantive confirmation using all URLs if available) ---
    if not urls:
        evaluator.add_custom_node(
            result=False,
            id=f"state_{state_idx}_mandate_type",
            desc="The state has an absolute vaccine mandate (no religious or philosophical exemptions) for school entry",
            parent=state_node,
            critical=True,
        )
    else:
        mandate_leaf = evaluator.add_leaf(
            id=f"state_{state_idx}_mandate_type",
            desc="Absolute mandate (medical-only; no religious/philosophical exemptions) is correctly claimed",
            parent=state_node,
            critical=True,
        )
        mandate_claim = (
            f"As of February 2026, {state.name} allows only medical exemptions and does not permit "
            f"religious or philosophical/personal belief exemptions to K–12 school-entry immunization requirements."
        )
        await evaluator.verify(
            claim=mandate_claim,
            node=mandate_leaf,
            sources=urls,
            additional_instruction=MANDATE_TYPE_ADDITIONAL_INSTRUCTION,
        )

    # --- Non-critical: ACIP incorporation ---
    acip_text = (state.acip_incorporation or "").strip()
    if not acip_text or not urls:
        evaluator.add_custom_node(
            result=False,
            id=f"state_{state_idx}_acip_incorporation",
            desc="The state's ACIP incorporation status is correctly reported (as of Feb 2026)",
            parent=state_node,
            critical=False,
        )
    else:
        acip_leaf = evaluator.add_leaf(
            id=f"state_{state_idx}_acip_incorporation",
            desc="The state's ACIP incorporation status is correctly reported (as of Feb 2026)",
            parent=state_node,
            critical=False,
        )
        # Normalize the direction of the claim based on the text (very simple heuristic)
        acip_simple = acip_text.lower()
        if any(k in acip_simple for k in ["incorporate", "acip", "advisory committee on immunization practices", "by reference"]):
            acip_claim = f"As of February 2026, {state.name}'s school immunization law/regulations incorporate ACIP recommendations."
        else:
            acip_claim = f"As of February 2026, {state.name}'s school immunization law/regulations do not incorporate ACIP recommendations."
        await evaluator.verify(
            claim=acip_claim,
            node=acip_leaf,
            sources=urls,
            additional_instruction=ACIP_ADDITIONAL_INSTRUCTION,
        )

    # --- Non-critical: vaccine-specific requirement verifications ---
    vaccines = state.vaccines or VaccineStatus()

    # Build per-disease checks
    for disease_key in [
        "varicella",
        "hepatitis_b",
        "measles",
        "mumps",
        "rubella",
        "polio",
        "pertussis",
        "tetanus",
        "diphtheria",
        "meningitis",
    ]:
        reported = getattr(vaccines, disease_key, None)
        await add_vaccine_requirement_check(
            evaluator=evaluator,
            parent_node=state_node,
            state_idx=state_idx,
            state_name=state.name or "",
            disease_key=disease_key,
            reported_status=reported,
            urls=urls,
        )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the 2026 absolute school-entry vaccine mandates task.
    """
    evaluator = Evaluator()
    # NOTE: The framework enforces that a critical parent node cannot have non-critical children.
    # The rubric's root was marked as critical, but its children (per-state subtrees) are non-critical.
    # To satisfy the framework constraint while preserving intended scoring (partial credit across states),
    # we set the root to non-critical parallel aggregation.
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=StatesExtraction,
        extraction_name="states_extraction",
    )

    # Normalize state list to exactly 3 entries (truncate or pad with empty)
    states: List[StateItem] = list(extracted.states or [])
    states = states[:3]
    while len(states) < 3:
        states.append(StateItem())

    # Add custom info about timeline context
    evaluator.add_custom_info(
        info={
            "as_of": "February 2026",
            "federal_update_context": "January 5, 2026 federal guidance reduced universal childhood recommendations from 18 to 11 diseases; states may maintain more comprehensive requirements."
        },
        info_type="context",
        info_name="timeline_context"
    )

    # Build verification subtrees for each of the three states (parallel under root)
    # Using 1-based indexing for readability and alignment with rubric naming
    for idx in range(1, 4):
        await verify_one_state(
            evaluator=evaluator,
            parent_node=root,
            state=states[idx - 1],
            state_idx=idx,
        )

    return evaluator.get_summary()