import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cte_large_districts_4_states"
TASK_DESCRIPTION = """
Identify 4 large school districts in the United States, with each district located in a different state. Each district must be among the 100 largest school districts in the nation by student enrollment (minimum 47,000 students). For each of the 4 districts, provide the following information: (1) District name and state location, (2) Total student enrollment figure, (3) At least one specific Career Technical Education (CTE) pathway program offered by the district (identify the program by name), (4) The certification or qualification requirements for teachers who teach in CTE programs in that district or state, and (5) Whether the district offers dual enrollment programs that allow high school students to earn college credit. Provide reference URLs for all information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    basic_info_urls: List[str] = Field(default_factory=list)

    enrollment: Optional[str] = None  # Keep as string to handle ranges/text like "≈ 50,000 (2023-24)"
    enrollment_urls: List[str] = Field(default_factory=list)
    top100_urls: List[str] = Field(default_factory=list)

    cte_program_name: Optional[str] = None
    cte_program_urls: List[str] = Field(default_factory=list)

    certification_requirements: Optional[str] = None
    certification_urls: List[str] = Field(default_factory=list)

    dual_enrollment: Optional[str] = None  # e.g., "Yes, offers dual enrollment with XYZ College"
    dual_enrollment_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
Extract up to 6 U.S. school districts described in the answer with the following fields for each district.

For each district, extract:
- district_name: The full district name (e.g., "Miami-Dade County Public Schools")
- state: The U.S. state where the district is located (accept full name or USPS abbreviation)
- basic_info_urls: An array of URL(s) that reference the district generally (e.g., district homepage, "About" page, Wikipedia page) if present in the answer
- enrollment: The total student enrollment figure text exactly as mentioned in the answer
- enrollment_urls: URL(s) that support the enrollment figure or reference student counts
- top100_urls: URL(s) that support the "among the 100 largest districts" claim (if present; otherwise leave empty)
- cte_program_name: The name of at least one specific CTE pathway program offered by the district (e.g., "Information Technology pathway", "Health Science", "Automotive Technology")
- cte_program_urls: URL(s) that support the existence of that specific named CTE pathway/program
- certification_requirements: The described certification or qualification requirements for CTE teachers (district or state requirements) as text from the answer
- certification_urls: URL(s) that support the CTE teacher certification/qualification requirements
- dual_enrollment: The statement about dual enrollment availability (e.g., "Yes, dual enrollment offered with ABC College") exactly as written in the answer
- dual_enrollment_urls: URL(s) that support dual enrollment availability

General URL rules:
- Only include URLs explicitly present in the answer text; do not invent URLs.
- Extract full URLs. If a URL is missing protocol, prepend "http://".
- If a given piece of information has multiple URLs cited, include all of them.

Return a JSON with a "districts" array of objects as described. If some fields are missing for a district, set them to null (for strings) or [] (for arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: Optional[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


# --------------------------------------------------------------------------- #
# Verification builder for one district                                       #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    root: VerificationNode,
    district: DistrictItem,
    index_1based: int,
    prior_state_values: List[str],
    prior_state_nodes: List[VerificationNode],
) -> Dict[str, Any]:
    """
    Build and verify the subtree for a single district.
    Returns a dict with references to some nodes/values for cross-district checks.
    """
    idx = index_1based
    district_node = evaluator.add_parallel(
        id=f"district_{idx}",
        desc=(
            f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx-1] if idx<=6 else f'#{idx}th'} "
            f"large school district meeting all requirements"
        ),
        parent=root,
        critical=False  # Each district contributes partial credit independently
    )

    # ---------------- Basic Info ----------------
    basic_main = evaluator.add_parallel(
        id=f"district_{idx}_basic_info",
        desc=("District name, state"
              + ("" if idx == 1 else " (different from previous districts)")
              + ", and reference URL provided"),
        parent=district_node,
        critical=True
    )

    # name provided
    name_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.district_name),
        id=f"district_{idx}_name_provided",
        desc=f"District {idx}: name is provided",
        parent=basic_main,
        critical=True
    )

    # state provided
    state_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.state),
        id=f"district_{idx}_state_provided",
        desc=f"District {idx}: state is provided",
        parent=basic_main,
        critical=True
    )

    # basic URLs provided
    basic_urls_provided = evaluator.add_custom_node(
        result=len(district.basic_info_urls) > 0,
        id=f"district_{idx}_basic_urls_provided",
        desc=f"District {idx}: basic reference URL(s) provided",
        parent=basic_main,
        critical=True
    )

    # state must be different from previously selected districts (for idx > 1)
    if idx > 1:
        state_distinct_leaf = evaluator.add_leaf(
            id=f"district_{idx}_state_distinct",
            desc=f"District {idx}: state is different from previously selected district states",
            parent=basic_main,
            critical=True
        )
        curr_state = district.state or ""
        prior_states_str = ", ".join([s for s in prior_state_values if s]) if prior_state_values else "(none)"
        claim = (
            f"The state '{curr_state}' is different from each of the following previously chosen states: {prior_states_str}."
        )
        await evaluator.verify(
            claim=claim,
            node=state_distinct_leaf,
            additional_instruction=(
                "Treat state names and USPS abbreviations as equivalent (e.g., 'CA' equals 'California'). "
                "If any prior state matches the current state by full name or standard abbreviation, judge this claim as incorrect."
            ),
            extra_prerequisites=[state_provided] + prior_state_nodes  # gate by required state presence
        )

    # ---------------- Enrollment Info ----------------
    enroll_main = evaluator.add_parallel(
        id=f"district_{idx}_enrollment_info",
        desc="Total student enrollment is provided and supported by references; also satisfies 'top 100' or '≥ 47,000' condition",
        parent=district_node,
        critical=True
    )

    enrollment_text_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.enrollment),
        id=f"district_{idx}_enrollment_text_provided",
        desc=f"District {idx}: enrollment figure text is provided",
        parent=enroll_main,
        critical=True
    )

    enrollment_urls_provided = evaluator.add_custom_node(
        result=len(district.enrollment_urls) > 0 or len(district.top100_urls) > 0,
        id=f"district_{idx}_enrollment_urls_provided",
        desc=f"District {idx}: enrollment or top-100 reference URL(s) provided",
        parent=enroll_main,
        critical=True
    )

    # Verify enrollment value is supported by provided URLs
    enrollment_supported = evaluator.add_leaf(
        id=f"district_{idx}_enrollment_supported",
        desc=f"District {idx}: enrollment value is supported by cited URL(s)",
        parent=enroll_main,
        critical=True
    )

    enrollment_sources = _unique_urls(district.enrollment_urls, district.basic_info_urls)
    claim_enrollment = (
        f"According to the provided sources, the district '{district.district_name or ''}' has a total student enrollment of '{district.enrollment or ''}'. "
        "Minor differences due to rounding or school year phrasing are acceptable."
    )
    await evaluator.verify(
        claim=claim_enrollment,
        node=enrollment_supported,
        sources=enrollment_sources,
        additional_instruction=(
            "Check that the cited page(s) mention a total enrollment consistent with the stated figure (allow small rounding or year notation). "
            "If multiple numbers appear, prefer district-wide total K-12 enrollment figures. "
            "If the claim cannot be supported from the provided URL(s), judge as not supported."
        ),
    )

    # Verify 'among 100 largest' or '>= 47,000' condition
    top100_supported = evaluator.add_leaf(
        id=f"district_{idx}_top100_or_min_supported",
        desc=f"District {idx}: district is among top 100 by enrollment or has at least 47,000 students",
        parent=enroll_main,
        critical=True
    )

    top100_sources = _unique_urls(district.top100_urls, district.enrollment_urls, district.basic_info_urls)
    claim_top100 = (
        "The district is either explicitly listed among the 100 largest U.S. school districts by student enrollment, "
        "or its total enrollment shown on the cited pages is at least 47,000 students."
    )
    await evaluator.verify(
        claim=claim_top100,
        node=top100_supported,
        sources=top100_sources,
        additional_instruction=(
            "This claim is satisfied in EITHER of these ways: "
            "(1) The page clearly shows a ranking/list where the district appears in the top 100 largest by enrollment; OR "
            "(2) The page shows a total enrollment number that is >= 47,000. "
            "If neither is supported on the provided pages, judge as not supported."
        ),
    )

    # ---------------- CTE Program ----------------
    cte_main = evaluator.add_parallel(
        id=f"district_{idx}_cte_program",
        desc="At least one specific CTE pathway program identified and supported by a reference URL",
        parent=district_node,
        critical=True
    )

    cte_text_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.cte_program_name),
        id=f"district_{idx}_cte_program_provided",
        desc=f"District {idx}: a specific CTE pathway/program name is provided",
        parent=cte_main,
        critical=True
    )

    cte_urls_provided = evaluator.add_custom_node(
        result=len(district.cte_program_urls) > 0,
        id=f"district_{idx}_cte_urls_provided",
        desc=f"District {idx}: CTE program reference URL(s) provided",
        parent=cte_main,
        critical=True
    )

    cte_supported = evaluator.add_leaf(
        id=f"district_{idx}_cte_program_supported",
        desc=f"District {idx}: the named CTE pathway/program is supported by cited URL(s)",
        parent=cte_main,
        critical=True
    )

    cte_sources = _unique_urls(district.cte_program_urls, district.basic_info_urls)
    claim_cte = (
        f"The district '{district.district_name or ''}' offers a CTE pathway or program named '{district.cte_program_name or ''}'."
    )
    await evaluator.verify(
        claim=claim_cte,
        node=cte_supported,
        sources=cte_sources,
        additional_instruction=(
            "Accept synonyms like 'CTE pathway', 'career pathway', 'academy', 'program of study', or 'career academy'. "
            "The cited page must clearly indicate the existence of the named program/pathway (or a very close name variant)."
        ),
    )

    # ---------------- Certification Requirements ----------------
    cert_main = evaluator.add_parallel(
        id=f"district_{idx}_certification",
        desc="CTE teacher certification or qualification requirements are described and supported by a reference URL",
        parent=district_node,
        critical=True
    )

    cert_text_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.certification_requirements),
        id=f"district_{idx}_cert_text_provided",
        desc=f"District {idx}: certification/qualification requirements text is provided",
        parent=cert_main,
        critical=True
    )

    cert_urls_provided = evaluator.add_custom_node(
        result=len(district.certification_urls) > 0,
        id=f"district_{idx}_cert_urls_provided",
        desc=f"District {idx}: certification/qualification reference URL(s) provided",
        parent=cert_main,
        critical=True
    )

    cert_supported = evaluator.add_leaf(
        id=f"district_{idx}_cert_supported",
        desc=f"District {idx}: certification/qualification requirements for CTE teachers are supported by cited URL(s)",
        parent=cert_main,
        critical=True
    )

    cert_sources = _unique_urls(district.certification_urls, district.basic_info_urls)
    claim_cert = (
        f"The cited page(s) describe certification or qualification requirements for CTE teachers relevant to {district.state or 'the state'}. "
        "It is acceptable if the requirements are at the state level rather than district-specific."
    )
    await evaluator.verify(
        claim=claim_cert,
        node=cert_supported,
        sources=cert_sources,
        additional_instruction=(
            "Look for wording like 'CTE teacher certification', 'occupational license', 'endorsement', 'industry credential', "
            "'state licensure', or similar. The page should clearly discuss requirements for CTE teachers in the relevant state/district."
        ),
    )

    # ---------------- Dual Enrollment ----------------
    dual_main = evaluator.add_parallel(
        id=f"district_{idx}_dual_enrollment",
        desc="Dual enrollment availability is stated and supported by a reference URL",
        parent=district_node,
        critical=True
    )

    dual_text_provided = evaluator.add_custom_node(
        result=_nonempty_text(district.dual_enrollment),
        id=f"district_{idx}_dual_text_provided",
        desc=f"District {idx}: dual enrollment availability statement is provided",
        parent=dual_main,
        critical=True
    )

    dual_urls_provided = evaluator.add_custom_node(
        result=len(district.dual_enrollment_urls) > 0,
        id=f"district_{idx}_dual_urls_provided",
        desc=f"District {idx}: dual enrollment reference URL(s) provided",
        parent=dual_main,
        critical=True
    )

    dual_supported = evaluator.add_leaf(
        id=f"district_{idx}_dual_supported",
        desc=f"District {idx}: dual enrollment availability is supported by cited URL(s)",
        parent=dual_main,
        critical=True
    )

    dual_sources = _unique_urls(district.dual_enrollment_urls, district.basic_info_urls)
    claim_dual = (
        "The district offers dual enrollment programs that allow high school students to earn college credit."
    )
    await evaluator.verify(
        claim=claim_dual,
        node=dual_supported,
        sources=dual_sources,
        additional_instruction=(
            "Accept synonymous terms like 'dual credit', 'concurrent enrollment', 'early college', or 'dual enrollment (DE)'. "
            "The cited page should indicate that high school students can take college-level courses for credit."
        ),
    )

    return {
        "state_value": district.state or "",
        "state_provided_node": state_provided
    }


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer to the '4 large districts with CTE and dual enrollment' task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root node is non-critical by framework design)
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

    # Extract structured districts info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction"
    )

    # Choose the first 4 districts; pad with empty items if fewer are provided
    items: List[DistrictItem] = list(extracted.districts[:4])
    while len(items) < 4:
        items.append(DistrictItem())

    # Keep track of prior states and their "provided" nodes to enforce distinctness
    prior_states: List[str] = []
    prior_state_nodes: List[VerificationNode] = []

    # Build verification subtrees for each of the 4 districts
    for i in range(4):
        result = await verify_one_district(
            evaluator=evaluator,
            root=root,
            district=items[i],
            index_1based=i + 1,
            prior_state_values=prior_states.copy(),
            prior_state_nodes=prior_state_nodes.copy()
        )
        # Update cross-district constraints tracking
        if result.get("state_value"):
            prior_states.append(result["state_value"])
        if result.get("state_provided_node"):
            prior_state_nodes.append(result["state_provided_node"])

    # Add a compact custom info block summarizing chosen districts and states
    summary_rows = []
    for i, d in enumerate(items, start=1):
        summary_rows.append({
            "idx": i,
            "district_name": d.district_name,
            "state": d.state,
            "cte_program_name": d.cte_program_name,
            "enrollment": d.enrollment
        })
    evaluator.add_custom_info({"selected_districts": summary_rows}, info_type="selection_summary")

    # Return evaluation summary
    return evaluator.get_summary()