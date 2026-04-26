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
TASK_ID = "us_state_vax_policies_4cats"
TASK_DESCRIPTION = """Identify four distinct U.S. states, one from each of the following categories, that meet the specified vaccination policy criteria. For each state, provide comprehensive documentation of their requirements:

Category A (Strictest Exemption Policy): A state that allows only medical exemptions from school vaccination requirements (no religious or personal belief exemptions allowed).

Category B (Recent Policy Change): A state that changed its exemption policy in 2025 by signing an executive order or enacting legislation to allow religious and/or personal belief exemptions, having previously allowed only medical exemptions.

Category C (High Exemption Rate): A state where the kindergarten vaccination exemption rate for the 2024-2025 school year was 9.0% or higher.

Category D (Enhanced Requirements): A state that requires meningococcal (MenACWY) vaccine for middle or high school entry in addition to the four universally required vaccines (MMR, DTaP, polio, varicella).

For each identified state, document:
1. The state name
2. Which category it fulfills
3. All vaccines required for kindergarten entry
4. The number of doses required for DTaP and MMR vaccines
5. The types of exemptions currently allowed (medical, religious, personal belief)
6. If applicable (Category B), details of any 2025 policy changes
7. If applicable (Category C), the specific exemption rate for 2024-2025
8. If applicable (Category D), the grade level(s) for which meningococcal vaccine is required
9. Authoritative source URLs supporting each piece of information
"""

ALLOWED_CATEGORY_A_STATES = ["California", "Connecticut", "Maine", "New York"]
CATEGORY_C_THRESHOLD = 9.0


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateDoc(BaseModel):
    state_name: Optional[str] = None
    category: Optional[str] = None  # Expect exact strings: "Category A"/"Category B"/"Category C"/"Category D"

    kindergarten_vaccines: List[str] = Field(default_factory=list)
    dtap_doses: Optional[str] = None
    mmr_doses: Optional[str] = None

    exemption_types: List[str] = Field(default_factory=list)  # e.g., ["medical"] or ["medical","religious"]
    policy_change_2025: Optional[str] = None  # narrative/details if applicable (Category B)
    exemption_rate_2024_2025: Optional[str] = None  # e.g., "9.5%" (Category C)
    meningococcal_grade_levels: Optional[str] = None  # e.g., "7th and 12th grade" (Category D)

    # Source URL fields (authoritative references only)
    state_identity_urls: List[str] = Field(default_factory=list)
    vaccine_requirements_urls: List[str] = Field(default_factory=list)
    dtap_doses_urls: List[str] = Field(default_factory=list)
    mmr_doses_urls: List[str] = Field(default_factory=list)
    exemption_policy_urls: List[str] = Field(default_factory=list)
    policy_change_urls: List[str] = Field(default_factory=list)
    exemption_rate_urls: List[str] = Field(default_factory=list)
    meningococcal_urls: List[str] = Field(default_factory=list)


class AllCategoriesExtraction(BaseModel):
    category_a: Optional[StateDoc] = None
    category_b: Optional[StateDoc] = None
    category_c: Optional[StateDoc] = None
    category_d: Optional[StateDoc] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_states() -> str:
    return """
Extract exactly one state (and its documentation) for each category (A, B, C, D) from the answer. Ensure the four states are distinct if multiple are given; select the first valid one mentioned per category.

For EACH category's chosen state, extract a JSON object with the following fields:

Common fields across all categories:
- state_name: The U.S. state name as written in the answer (string).
- category: The category label string. MUST be exactly one of: "Category A", "Category B", "Category C", "Category D".
- kindergarten_vaccines: Array of vaccine names explicitly listed for kindergarten entry in the answer (e.g., ["MMR","DTaP","polio","varicella", ...]).
- dtap_doses: String for the number of DTaP doses required for kindergarten entry (e.g., "5 doses" or "4-5 doses").
- mmr_doses: String for the number of MMR doses required for kindergarten entry (e.g., "2 doses").
- exemption_types: Array of currently allowed exemption types stated in the answer. Possible values: "medical", "religious", "personal belief". Use exactly these lowercase terms if present; do not invent.
- state_identity_urls: Array of URLs that clearly identify the state's official health/education site or state code/statute page relevant to school immunizations.
- vaccine_requirements_urls: Array of URLs that document kindergarten entry vaccine requirements for that state.
- dtap_doses_urls: Array of URLs specifically supporting the stated DTaP dose requirement (can be the same as vaccine requirements URL if it shows doses).
- mmr_doses_urls: Array of URLs specifically supporting the stated MMR dose requirement (can be the same as vaccine requirements URL if it shows doses).
- exemption_policy_urls: Array of URLs that document the CURRENT allowed exemption types for that state.

Category-specific fields:
- For Category B:
  - policy_change_2025: A short text describing the 2025 policy change (e.g., "Executive order in May 2025 allowing religious exemptions; previously medical-only").
  - policy_change_urls: Array of URLs that explicitly document the 2025 policy change.
- For Category C:
  - exemption_rate_2024_2025: The kindergarten vaccination exemption rate for 2024–2025 as a percentage string (e.g., "9.4%").
  - exemption_rate_urls: Array of URLs that explicitly show the 2024–2025 kindergarten exemption rate for that state.
- For Category D:
  - meningococcal_grade_levels: Text for the grade levels for which MenACWY is required (e.g., "7th and 12th grades").
  - meningococcal_urls: Array of URLs that document the meningococcal (MenACWY) requirement and grade levels.

Return a single JSON object with four top-level objects:
{
  "category_a": { ... StateDoc ... } | null,
  "category_b": { ... StateDoc ... } | null,
  "category_c": { ... StateDoc ... } | null,
  "category_d": { ... StateDoc ... } | null
}

Important extraction rules:
- Extract ONLY information explicitly present in the answer. Do not infer.
- For any field not found, set it to null or [] accordingly.
- For URL fields, extract only valid, complete URLs that appear in the answer (including within markdown links). If a field requires a URL but none are provided in the answer, return an empty array for that URL list.
- Prefer authoritative sources for URLs: state health/education department sites, state statutes/regulations, CDC, or official governor/legislature releases. News articles are acceptable only if explicitly cited in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _ensure_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _fallback_urls(primary: Optional[List[str]], fallback: Optional[List[str]]) -> List[str]:
    p = _ensure_list(primary)
    if p:
        return p
    return _ensure_list(fallback)


def _join_list(values: Optional[List[str]]) -> str:
    if not values:
        return ""
    return ", ".join(v for v in values if isinstance(v, str))


async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    *,
    claim: str,
    node,
    urls: Optional[List[str]],
    additional_instruction: str = "",
) -> bool:
    clean_urls = _ensure_list(urls)
    if len(clean_urls) == 0:
        # Force failure if no URLs are provided (source-grounding policy)
        forced_instruction = (
            additional_instruction.strip() + "\n"
            "IMPORTANT: No source URLs were provided for this check; you must mark this claim as Incorrect (not supported)."
        ).strip()
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction=forced_instruction,
        )
    elif len(clean_urls) == 1:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=clean_urls[0],
            additional_instruction=additional_instruction,
        )
    else:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=clean_urls,
            additional_instruction=additional_instruction,
        )


# --------------------------------------------------------------------------- #
# Verification builders per category                                          #
# --------------------------------------------------------------------------- #
async def _build_category_A(
    evaluator: Evaluator,
    parent_node,
    doc: Optional[StateDoc],
) -> None:
    cat_node = evaluator.add_sequential(
        id="category_A_state",
        desc="Identification and verification of a state in Category A (strictest exemption policy - medical only)",
        parent=parent_node,
        critical=False,
    )

    state_name = (doc.state_name if doc else None) or "UNSPECIFIED"

    # Identification leaf (critical): must be one of the known Category A states
    id_leaf = evaluator.add_leaf(
        id="category_A_identification",
        desc="State is correctly identified as one that allows only medical exemptions (California, Connecticut, Maine, or New York)",
        parent=cat_node,
        critical=True,
    )
    id_claim = (
        f"The selected state for Category A is one of: {', '.join(ALLOWED_CATEGORY_A_STATES)}. "
        f"The provided state is '{state_name}'."
    )
    await evaluator.verify(
        claim=id_claim,
        node=id_leaf,
        additional_instruction="This is a simple logical membership check. Treat minor spelling/casing variants as the same state.",
    )

    # Documentation (parallel)
    doc_node = evaluator.add_parallel(
        id="category_A_documentation",
        desc="Complete documentation of Category A state's vaccination requirements and policies",
        parent=cat_node,
        critical=False,
    )

    # Basic info (critical)
    basic_node = evaluator.add_parallel(
        id="category_A_basic_info",
        desc="Basic identification information for Category A state",
        parent=doc_node,
        critical=True,
    )

    # State identity leaf with URLs
    state_name_leaf = evaluator.add_leaf(
        id="category_A_state_name",
        desc="State name is provided with URL reference confirming state identity",
        parent=basic_node,
        critical=True,
    )
    state_identity_claim = (
        f"The cited source page(s) clearly pertain to the State of {state_name} (e.g., official state health/education site or state statutes)."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=state_identity_claim,
        node=state_name_leaf,
        urls=doc.state_identity_urls if doc else [],
        additional_instruction="Confirm the page is an authoritative resource for the named state.",
    )

    # Category label leaf (simple)
    category_label_leaf = evaluator.add_leaf(
        id="category_A_category_label",
        desc="Category A designation is stated",
        parent=basic_node,
        critical=True,
    )
    cat_label = doc.category if doc and doc.category else ""
    label_claim = f"The category designation provided for this state is '{cat_label}', and it should be 'Category A'."
    await evaluator.verify(
        claim=label_claim,
        node=category_label_leaf,
        additional_instruction="Accept if the answer explicitly labels this state as Category A (case-insensitive).",
    )

    # Vaccine requirements (critical)
    vax_node = evaluator.add_parallel(
        id="category_A_vaccine_requirements",
        desc="Vaccine requirements documentation for Category A state",
        parent=doc_node,
        critical=True,
    )

    # Kindergarten vaccine list leaf
    kg_vax_leaf = evaluator.add_leaf(
        id="category_A_kindergarten_vaccines",
        desc="All vaccines required for kindergarten entry are listed (including at minimum MMR, DTaP, polio, and varicella) with URL reference documenting requirements",
        parent=vax_node,
        critical=True,
    )
    kg_list = _join_list(doc.kindergarten_vaccines if doc else [])
    kg_claim = (
        f"The source page lists the required vaccines for kindergarten entry in {state_name}. "
        f"The extracted list is: [{kg_list}]. At minimum, the page must confirm MMR, DTaP, polio (IPV), and varicella are required."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=kg_claim,
        node=kg_vax_leaf,
        urls=doc.vaccine_requirements_urls if doc else [],
        additional_instruction="Focus on kindergarten entry requirements; allow common synonyms (e.g., IPV for polio).",
    )

    # DTaP doses leaf
    dtap_leaf = evaluator.add_leaf(
        id="category_A_dtap_doses",
        desc="Number of DTaP doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    dtap_claim = f"The number of DTaP doses required for kindergarten entry in {state_name} is '{(doc.dtap_doses if doc and doc.dtap_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=dtap_claim,
        node=dtap_leaf,
        urls=_fallback_urls(doc.dtap_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    # MMR doses leaf
    mmr_leaf = evaluator.add_leaf(
        id="category_A_mmr_doses",
        desc="Number of MMR doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    mmr_claim = f"The number of MMR doses required for kindergarten entry in {state_name} is '{(doc.mmr_doses if doc and doc.mmr_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=mmr_claim,
        node=mmr_leaf,
        urls=_fallback_urls(doc.mmr_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    # Exemption policy (critical)
    ex_node = evaluator.add_parallel(
        id="category_A_exemption_policy",
        desc="Exemption policy documentation for Category A state",
        parent=doc_node,
        critical=True,
    )

    # Exemption types leaf: medical only
    ex_types_leaf = evaluator.add_leaf(
        id="category_A_exemption_types",
        desc="Types of exemptions allowed are stated as medical only with URL reference confirming exemption policy",
        parent=ex_node,
        critical=True,
    )
    ex_list = _join_list(doc.exemption_types if doc else [])
    ex_claim = (
        f"In {state_name}, only medical exemptions are allowed for school vaccination requirements; religious and personal belief exemptions are NOT allowed. "
        f"The extracted current exemption types are: [{ex_list}]."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=ex_claim,
        node=ex_types_leaf,
        urls=doc.exemption_policy_urls if doc else [],
        additional_instruction="The page must explicitly indicate that only medical exemptions are permitted (no religious or personal belief). Prefer state statutes/regulations or official guidance.",
    )


async def _build_category_B(
    evaluator: Evaluator,
    parent_node,
    doc: Optional[StateDoc],
) -> None:
    cat_node = evaluator.add_sequential(
        id="category_B_state",
        desc="Identification and verification of a state in Category B (recent policy change in 2025)",
        parent=parent_node,
        critical=False,
    )

    state_name = (doc.state_name if doc else None) or "UNSPECIFIED"

    # Identification (critical): explicit 2025 policy change allowing religious/personal belief exemptions,
    # previously medical-only
    id_leaf = evaluator.add_leaf(
        id="category_B_identification",
        desc="State is correctly identified as one that changed exemption policy in 2025",
        parent=cat_node,
        critical=True,
    )
    change_txt = (doc.policy_change_2025 if doc and doc.policy_change_2025 else "")
    id_claim = (
        f"In 2025, the state of {state_name} changed its school vaccination exemption policy (e.g., via executive order or legislation) "
        f"to allow religious and/or personal belief exemptions; prior to 2025 it allowed only medical exemptions. "
        f"Details given: '{change_txt}'."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=id_claim,
        node=id_leaf,
        urls=doc.policy_change_urls if doc else [],
        additional_instruction="Only accept if the provided source(s) explicitly indicate the change occurred in calendar year 2025 and that it allowed religious/personal belief exemptions, having previously permitted only medical exemptions.",
    )

    # Documentation (parallel)
    doc_node = evaluator.add_parallel(
        id="category_B_documentation",
        desc="Complete documentation of Category B state's vaccination requirements and policy changes",
        parent=cat_node,
        critical=False,
    )

    # Basic info (critical)
    basic_node = evaluator.add_parallel(
        id="category_B_basic_info",
        desc="Basic identification information for Category B state",
        parent=doc_node,
        critical=True,
    )

    state_name_leaf = evaluator.add_leaf(
        id="category_B_state_name",
        desc="State name is provided with URL reference confirming state identity",
        parent=basic_node,
        critical=True,
    )
    identity_claim = f"The cited source page(s) clearly pertain to the State of {state_name}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=identity_claim,
        node=state_name_leaf,
        urls=doc.state_identity_urls if doc else [],
        additional_instruction="Confirm this is an authoritative state resource.",
    )

    category_label_leaf = evaluator.add_leaf(
        id="category_B_category_label",
        desc="Category B designation is stated",
        parent=basic_node,
        critical=True,
    )
    cat_label = doc.category if doc and doc.category else ""
    label_claim = f"The category designation provided for this state is '{cat_label}', and it should be 'Category B'."
    await evaluator.verify(
        claim=label_claim,
        node=category_label_leaf,
        additional_instruction="Accept if the answer explicitly labels this state as Category B (case-insensitive).",
    )

    # Vaccine requirements (critical)
    vax_node = evaluator.add_parallel(
        id="category_B_vaccine_requirements",
        desc="Vaccine requirements documentation for Category B state",
        parent=doc_node,
        critical=True,
    )

    kg_vax_leaf = evaluator.add_leaf(
        id="category_B_kindergarten_vaccines",
        desc="All vaccines required for kindergarten entry are listed (including at minimum MMR, DTaP, polio, and varicella) with URL reference documenting requirements",
        parent=vax_node,
        critical=True,
    )
    kg_list = _join_list(doc.kindergarten_vaccines if doc else [])
    kg_claim = (
        f"The source page lists the required vaccines for kindergarten entry in {state_name}. "
        f"The extracted list is: [{kg_list}]. At minimum, the page must confirm MMR, DTaP, polio (IPV), and varicella are required."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=kg_claim,
        node=kg_vax_leaf,
        urls=doc.vaccine_requirements_urls if doc else [],
        additional_instruction="Focus on kindergarten; allow common synonyms (e.g., IPV for polio).",
    )

    dtap_leaf = evaluator.add_leaf(
        id="category_B_dtap_doses",
        desc="Number of DTaP doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    dtap_claim = f"The number of DTaP doses required for kindergarten entry in {state_name} is '{(doc.dtap_doses if doc and doc.dtap_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=dtap_claim,
        node=dtap_leaf,
        urls=_fallback_urls(doc.dtap_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    mmr_leaf = evaluator.add_leaf(
        id="category_B_mmr_doses",
        desc="Number of MMR doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    mmr_claim = f"The number of MMR doses required for kindergarten entry in {state_name} is '{(doc.mmr_doses if doc and doc.mmr_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=mmr_claim,
        node=mmr_leaf,
        urls=_fallback_urls(doc.mmr_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    # Exemption policy and 2025 change (critical)
    ex_node = evaluator.add_parallel(
        id="category_B_exemption_policy",
        desc="Exemption policy documentation for Category B state",
        parent=doc_node,
        critical=True,
    )

    ex_types_leaf = evaluator.add_leaf(
        id="category_B_exemption_types",
        desc="Current types of exemptions allowed are stated with URL reference confirming current exemption policy",
        parent=ex_node,
        critical=True,
    )
    ex_list = _join_list(doc.exemption_types if doc else [])
    ex_claim = f"The currently allowed exemption types in {state_name} are: [{ex_list}]."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=ex_claim,
        node=ex_types_leaf,
        urls=doc.exemption_policy_urls if doc else [],
        additional_instruction="Confirm the present policy from an authoritative page (state statutes/regs or official guidance).",
    )

    policy_change_leaf = evaluator.add_leaf(
        id="category_B_policy_change",
        desc="Details of 2025 policy change are provided with URL reference documenting the policy change",
        parent=ex_node,
        critical=True,
    )
    pc_txt = (doc.policy_change_2025 if doc and doc.policy_change_2025 else "")
    pc_claim = f"In 2025, {state_name} implemented this policy change: '{pc_txt}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=pc_claim,
        node=policy_change_leaf,
        urls=doc.policy_change_urls if doc else [],
        additional_instruction="The page must clearly show the 2025 effective date and describe the change allowing religious and/or personal belief exemptions.",
    )


async def _build_category_C(
    evaluator: Evaluator,
    parent_node,
    doc: Optional[StateDoc],
) -> None:
    cat_node = evaluator.add_sequential(
        id="category_C_state",
        desc="Identification and verification of a state in Category C (high exemption rate ≥9.0%)",
        parent=parent_node,
        critical=False,
    )

    state_name = (doc.state_name if doc else None) or "UNSPECIFIED"
    rate_txt = (doc.exemption_rate_2024_2025 if doc and doc.exemption_rate_2024_2025 else "")

    # Identification (critical): rate >= 9.0%
    id_leaf = evaluator.add_leaf(
        id="category_C_identification",
        desc="State is correctly identified as one with exemption rate ≥9.0% for 2024-2025",
        parent=cat_node,
        critical=True,
    )
    id_claim = (
        f"The kindergarten vaccination exemption rate for the 2024–2025 school year in {state_name} was '{rate_txt}', "
        f"which should be at least {CATEGORY_C_THRESHOLD:.1f}%."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=id_claim,
        node=id_leaf,
        urls=doc.exemption_rate_urls if doc else [],
        additional_instruction="Only accept if the page explicitly shows a 2024–2025 kindergarten exemption rate ≥ 9.0%.",
    )

    # Documentation (parallel)
    doc_node = evaluator.add_parallel(
        id="category_C_documentation",
        desc="Complete documentation of Category C state's vaccination requirements and exemption data",
        parent=cat_node,
        critical=False,
    )

    # Basic info (critical)
    basic_node = evaluator.add_parallel(
        id="category_C_basic_info",
        desc="Basic identification information for Category C state",
        parent=doc_node,
        critical=True,
    )

    state_name_leaf = evaluator.add_leaf(
        id="category_C_state_name",
        desc="State name is provided with URL reference confirming state identity",
        parent=basic_node,
        critical=True,
    )
    identity_claim = f"The cited source page(s) clearly pertain to the State of {state_name}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=identity_claim,
        node=state_name_leaf,
        urls=doc.state_identity_urls if doc else [],
        additional_instruction="Confirm this is an authoritative state resource.",
    )

    category_label_leaf = evaluator.add_leaf(
        id="category_C_category_label",
        desc="Category C designation is stated",
        parent=basic_node,
        critical=True,
    )
    cat_label = doc.category if doc and doc.category else ""
    label_claim = f"The category designation provided for this state is '{cat_label}', and it should be 'Category C'."
    await evaluator.verify(
        claim=label_claim,
        node=category_label_leaf,
        additional_instruction="Accept if the answer explicitly labels this state as Category C (case-insensitive).",
    )

    # Vaccine requirements (critical)
    vax_node = evaluator.add_parallel(
        id="category_C_vaccine_requirements",
        desc="Vaccine requirements documentation for Category C state",
        parent=doc_node,
        critical=True,
    )

    kg_vax_leaf = evaluator.add_leaf(
        id="category_C_kindergarten_vaccines",
        desc="All vaccines required for kindergarten entry are listed (including at minimum MMR, DTaP, polio, and varicella) with URL reference documenting requirements",
        parent=vax_node,
        critical=True,
    )
    kg_list = _join_list(doc.kindergarten_vaccines if doc else [])
    kg_claim = (
        f"The source page lists the required vaccines for kindergarten entry in {state_name}. "
        f"The extracted list is: [{kg_list}]. At minimum, the page must confirm MMR, DTaP, polio (IPV), and varicella are required."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=kg_claim,
        node=kg_vax_leaf,
        urls=doc.vaccine_requirements_urls if doc else [],
        additional_instruction="Focus on kindergarten; allow common synonyms (e.g., IPV for polio).",
    )

    dtap_leaf = evaluator.add_leaf(
        id="category_C_dtap_doses",
        desc="Number of DTaP doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    dtap_claim = f"The number of DTaP doses required for kindergarten entry in {state_name} is '{(doc.dtap_doses if doc and doc.dtap_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=dtap_claim,
        node=dtap_leaf,
        urls=_fallback_urls(doc.dtap_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    mmr_leaf = evaluator.add_leaf(
        id="category_C_mmr_doses",
        desc="Number of MMR doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    mmr_claim = f"The number of MMR doses required for kindergarten entry in {state_name} is '{(doc.mmr_doses if doc and doc.mmr_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=mmr_claim,
        node=mmr_leaf,
        urls=_fallback_urls(doc.mmr_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    # Exemption policy and rate (critical)
    ex_node = evaluator.add_parallel(
        id="category_C_exemption_policy",
        desc="Exemption policy and rate documentation for Category C state",
        parent=doc_node,
        critical=True,
    )

    ex_types_leaf = evaluator.add_leaf(
        id="category_C_exemption_types",
        desc="Types of exemptions allowed are stated with URL reference confirming exemption policy",
        parent=ex_node,
        critical=True,
    )
    ex_list = _join_list(doc.exemption_types if doc else [])
    ex_claim = f"The currently allowed exemption types in {state_name} are: [{ex_list}]."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=ex_claim,
        node=ex_types_leaf,
        urls=doc.exemption_policy_urls if doc else [],
        additional_instruction="Confirm the present policy from an authoritative page (state statutes/regs or official guidance).",
    )

    ex_rate_leaf = evaluator.add_leaf(
        id="category_C_exemption_rate",
        desc="Specific exemption rate for 2024-2025 is provided and is ≥9.0% with URL reference documenting exemption rate data",
        parent=ex_node,
        critical=True,
    )
    rate_claim = (
        f"The kindergarten vaccination exemption rate for 2024–2025 in {state_name} was '{rate_txt}', "
        f"and this value is ≥ {CATEGORY_C_THRESHOLD:.1f}%."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=rate_claim,
        node=ex_rate_leaf,
        urls=doc.exemption_rate_urls if doc else [],
        additional_instruction="Only accept if the page explicitly shows the 2024–2025 kindergarten exemption rate and it meets/exceeds 9.0%.",
    )


async def _build_category_D(
    evaluator: Evaluator,
    parent_node,
    doc: Optional[StateDoc],
) -> None:
    cat_node = evaluator.add_sequential(
        id="category_D_state",
        desc="Identification and verification of a state in Category D (requires meningococcal vaccine)",
        parent=parent_node,
        critical=False,
    )

    state_name = (doc.state_name if doc else None) or "UNSPECIFIED"
    grades_txt = (doc.meningococcal_grade_levels if doc and doc.meningococcal_grade_levels else "")

    # Identification (critical): requires MenACWY for middle or high school entry
    id_leaf = evaluator.add_leaf(
        id="category_D_identification",
        desc="State is correctly identified as one requiring meningococcal (MenACWY) vaccine",
        parent=cat_node,
        critical=True,
    )
    id_claim = (
        f"The state of {state_name} requires meningococcal (MenACWY) vaccine for middle and/or high school entry."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=id_claim,
        node=id_leaf,
        urls=doc.meningococcal_urls if doc else [],
        additional_instruction="The page must explicitly document MenACWY as a requirement for specified grades (e.g., 7th and/or 12th).",
    )

    # Documentation (parallel)
    doc_node = evaluator.add_parallel(
        id="category_D_documentation",
        desc="Complete documentation of Category D state's vaccination requirements including meningococcal",
        parent=cat_node,
        critical=False,
    )

    # Basic info (critical)
    basic_node = evaluator.add_parallel(
        id="category_D_basic_info",
        desc="Basic identification information for Category D state",
        parent=doc_node,
        critical=True,
    )

    state_name_leaf = evaluator.add_leaf(
        id="category_D_state_name",
        desc="State name is provided with URL reference confirming state identity",
        parent=basic_node,
        critical=True,
    )
    identity_claim = f"The cited source page(s) clearly pertain to the State of {state_name}."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=identity_claim,
        node=state_name_leaf,
        urls=doc.state_identity_urls if doc else [],
        additional_instruction="Confirm this is an authoritative state resource.",
    )

    category_label_leaf = evaluator.add_leaf(
        id="category_D_category_label",
        desc="Category D designation is stated",
        parent=basic_node,
        critical=True,
    )
    cat_label = doc.category if doc and doc.category else ""
    label_claim = f"The category designation provided for this state is '{cat_label}', and it should be 'Category D'."
    await evaluator.verify(
        claim=label_claim,
        node=category_label_leaf,
        additional_instruction="Accept if the answer explicitly labels this state as Category D (case-insensitive).",
    )

    # Vaccine requirements (critical)
    vax_node = evaluator.add_parallel(
        id="category_D_vaccine_requirements",
        desc="Vaccine requirements documentation for Category D state",
        parent=doc_node,
        critical=True,
    )

    kg_vax_leaf = evaluator.add_leaf(
        id="category_D_kindergarten_vaccines",
        desc="All vaccines required for kindergarten entry are listed (including at minimum MMR, DTaP, polio, and varicella) with URL reference documenting requirements",
        parent=vax_node,
        critical=True,
    )
    kg_list = _join_list(doc.kindergarten_vaccines if doc else [])
    kg_claim = (
        f"The source page lists the required vaccines for kindergarten entry in {state_name}. "
        f"The extracted list is: [{kg_list}]. At minimum, the page must confirm MMR, DTaP, polio (IPV), and varicella are required."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=kg_claim,
        node=kg_vax_leaf,
        urls=doc.vaccine_requirements_urls if doc else [],
        additional_instruction="Focus on kindergarten; allow common synonyms (e.g., IPV for polio).",
    )

    dtap_leaf = evaluator.add_leaf(
        id="category_D_dtap_doses",
        desc="Number of DTaP doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    dtap_claim = f"The number of DTaP doses required for kindergarten entry in {state_name} is '{(doc.dtap_doses if doc and doc.dtap_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=dtap_claim,
        node=dtap_leaf,
        urls=_fallback_urls(doc.dtap_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    mmr_leaf = evaluator.add_leaf(
        id="category_D_mmr_doses",
        desc="Number of MMR doses required is specified with URL reference confirming dose requirements",
        parent=vax_node,
        critical=True,
    )
    mmr_claim = f"The number of MMR doses required for kindergarten entry in {state_name} is '{(doc.mmr_doses if doc and doc.mmr_doses else '')}'."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=mmr_claim,
        node=mmr_leaf,
        urls=_fallback_urls(doc.mmr_doses_urls if doc else [], doc.vaccine_requirements_urls if doc else []),
        additional_instruction="Verify the dose count exactly as written on the authoritative page.",
    )

    # Meningococcal requirement (critical)
    men_leaf = evaluator.add_leaf(
        id="category_D_meningococcal",
        desc="Meningococcal (MenACWY) vaccine requirement is documented with grade level(s) and URL reference confirming requirement",
        parent=vax_node,
        critical=True,
    )
    men_claim = (
        f"{state_name} requires the meningococcal (MenACWY) vaccine for grade level(s): '{grades_txt}'. "
        f"The cited source must explicitly show MenACWY is required and indicate the grade level(s)."
    )
    await _verify_with_urls_or_fail(
        evaluator,
        claim=men_claim,
        node=men_leaf,
        urls=doc.meningococcal_urls if doc else [],
        additional_instruction="Confirm that MenACWY is required and identify the specific grade levels (e.g., 7th and/or 12th).",
    )

    # Exemption policy (critical)
    ex_node = evaluator.add_parallel(
        id="category_D_exemption_policy",
        desc="Exemption policy documentation for Category D state",
        parent=doc_node,
        critical=True,
    )

    ex_types_leaf = evaluator.add_leaf(
        id="category_D_exemption_types",
        desc="Types of exemptions allowed are stated with URL reference confirming exemption policy",
        parent=ex_node,
        critical=True,
    )
    ex_list = _join_list(doc.exemption_types if doc else [])
    ex_claim = f"The currently allowed exemption types in {state_name} are: [{ex_list}]."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=ex_claim,
        node=ex_types_leaf,
        urls=doc.exemption_policy_urls if doc else [],
        additional_instruction="Confirm the present policy from an authoritative page (state statutes/regs or official guidance).",
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
    # Initialize evaluator (root kept non-critical to allow partial credit; rubric's root critical would force all children critical)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across four categories
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

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_states(),
        template_class=AllCategoriesExtraction,
        extraction_name="states_by_category",
    )

    # Add ground truth/meta context
    evaluator.add_ground_truth({
        "category_A_allowed_states": ALLOWED_CATEGORY_A_STATES,
        "category_C_threshold_percent": CATEGORY_C_THRESHOLD,
        "categories_expected": ["Category A", "Category B", "Category C", "Category D"],
    }, gt_type="task_expectations")

    # Build verification subtrees per category
    await _build_category_A(evaluator, root, extracted.category_a if extracted else None)
    await _build_category_B(evaluator, root, extracted.category_b if extracted else None)
    await _build_category_C(evaluator, root, extracted.category_c if extracted else None)
    await _build_category_D(evaluator, root, extracted.category_d if extracted else None)

    # Return standardized evaluation summary
    return evaluator.get_summary()