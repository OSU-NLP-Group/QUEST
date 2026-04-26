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
TASK_ID = "verizon_outage_2026_report"
TASK_DESCRIPTION = (
    "In January 2026, Verizon experienced a major network outage that affected customers across multiple US states "
    "and prompted an FCC investigation. As a technology analyst preparing a comprehensive incident report, compile the following information:\n\n"
    "1. The exact date when the major Verizon network outage occurred in January 2026\n"
    "2. The technical cause that Verizon identified for this outage\n"
    "3. All four US states that were explicitly documented in news reports as being affected by this outage\n"
    "4. The official FCC email address where customers can submit their outage experiences, and the deadline for these submissions\n"
    "5. Based on 2025-2026 industry reports and coverage data, which mobile carrier (among the three major US carriers: Verizon, T-Mobile, AT&T) has the most extensive 5G network coverage in the United States\n\n"
    "For each piece of information, provide supporting reference URLs from reliable sources that verify your findings."
)

# Expected values used for consistency checks against the answer text
EXPECTED_OUTAGE_DATE = "January 14, 2026"
EXPECTED_CAUSE_KEYWORD = "software"  # allow broader matching via instructions
EXPECTED_STATES = ["Texas", "Georgia", "New York", "California"]
EXPECTED_FCC_EMAIL = "VerizonOutage2026@fcc.gov"
EXPECTED_DEADLINE = "March 16, 2026"
EXPECTED_COVERAGE_LEADER = "T-Mobile"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class OutageDateModel(BaseModel):
    date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OutageCauseModel(BaseModel):
    cause: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AffectedStatesModel(BaseModel):
    states: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FCCInfoModel(BaseModel):
    email: Optional[str] = None
    email_sources: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    deadline_sources: List[str] = Field(default_factory=list)


class CoverageLeaderModel(BaseModel):
    carrier: Optional[str] = None  # one of: Verizon, T-Mobile, AT&T (case-insensitive accepted)
    coverage_metrics: Optional[str] = None  # any percentage or population figure text from the answer
    sources: List[str] = Field(default_factory=list)  # sources supporting carrier leadership
    metrics_sources: List[str] = Field(default_factory=list)  # sources supporting coverage metrics (if separate)


class IncidentExtraction(BaseModel):
    outage_date: OutageDateModel = Field(default_factory=OutageDateModel)
    outage_cause: OutageCauseModel = Field(default_factory=OutageCauseModel)
    affected_states: AffectedStatesModel = Field(default_factory=AffectedStatesModel)
    fcc: FCCInfoModel = Field(default_factory=FCCInfoModel)
    coverage: CoverageLeaderModel = Field(default_factory=CoverageLeaderModel)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_incident() -> str:
    return """
    Extract the following fields exactly as they appear in the answer text. Do not infer or add information.

    1) outage_date:
       - date: the exact date of the major Verizon outage as stated in the answer (e.g., "January 14, 2026", "Jan 14, 2026", "1/14/2026")
       - sources: all URLs in the answer that directly support the outage date (e.g., news reports, official posts)

    2) outage_cause:
       - cause: the technical cause Verizon identified for the outage, as stated in the answer (e.g., "software issue", "software bug", "software update problem")
       - sources: all URLs in the answer that directly support the outage cause

    3) affected_states:
       - states: a list of the US states explicitly claimed in the answer to have been affected by the outage (each item should be a state name, properly capitalized; do not include cities)
       - sources: all URLs in the answer that support which states were affected

    4) fcc:
       - email: the FCC email address provided for customers to submit outage experiences (return null if not present)
       - email_sources: all URLs that support/provide the FCC email address
       - deadline: the deadline date for submitting outage experiences to the FCC (return null if not present)
       - deadline_sources: all URLs that support/provide the submission deadline

    5) coverage:
       - carrier: which carrier (Verizon, T-Mobile, or AT&T) the answer identifies as having the most extensive US 5G network coverage based on 2025-2026 reports
       - coverage_metrics: any coverage percentage or population figure mentioned in the answer to justify this (e.g., "covers 98% of Americans", "more than 330 million people", "two times the coverage of X")
       - sources: all URLs that support the identified carrier being the coverage leader (2025-2026 timeframe)
       - metrics_sources: all URLs that support the provided coverage metrics (if distinct from the above; otherwise, repeat if the same URL supports both)

    Rules for URLs:
    - Include only URLs explicitly present in the answer (plain links or markdown links).
    - Ensure URLs are valid and include the protocol. If missing protocol, prepend "http://".
    - Do not fabricate or infer URLs.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and any(isinstance(u, str) and u.strip() for u in urls))


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_outage_date_branch(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'outage_date' branch.
    """
    node = evaluator.add_parallel(
        id="outage_date",
        desc="Correct outage date provided",
        parent=parent,
        critical=True
    )

    # Leaf: date value matches expected
    date_value_leaf = evaluator.add_leaf(
        id="date_value",
        desc="Date is January 14, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the major Verizon network outage occurred on January 14, 2026 (accept 'Jan 14, 2026', 'January 14th, 2026', or '1/14/2026').",
        node=date_value_leaf,
        additional_instruction=(
            "Read only the answer text to determine the date claimed for the outage. "
            "If the answer claims a different date or does not clearly provide the date, mark as incorrect."
        )
    )

    # Leaf: date supported by provided URLs
    date_ref_leaf = evaluator.add_leaf(
        id="date_reference_url",
        desc="Valid URL source for outage date",
        parent=node,
        critical=True
    )
    # Fetch extracted data
    # We'll retrieve from the recorded extraction results later in main flow;
    # For modularity, we instead look up the last IncidentExtraction stored via evaluator._extraction_results.
    # However, better to pass data into this function. We'll search from evaluator._extraction_results safely.
    # To avoid hidden coupling, we'll set a placeholder; the caller will replace with the real verify call.
    # Here, we just leave the node; actual verify call will be done in main after extraction.
    # We'll store the node id to custom info for later reference.
    evaluator.add_custom_info({"node_id": date_ref_leaf.id}, "node_handles", "date_ref_leaf_handle")


async def build_outage_cause_branch(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'outage_cause' branch.
    """
    node = evaluator.add_parallel(
        id="outage_cause",
        desc="Correct technical cause provided",
        parent=parent,
        critical=True
    )

    cause_value_leaf = evaluator.add_leaf(
        id="cause_value",
        desc="Cause identified as software issue",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer attributes the outage to a software issue (e.g., software bug, software update/configuration problem).",
        node=cause_value_leaf,
        additional_instruction=(
            "Check the answer text only. Accept paraphrases that clearly indicate a software-related root cause. "
            "If the cause is different or missing, mark incorrect."
        )
    )

    cause_ref_leaf = evaluator.add_leaf(
        id="cause_reference_url",
        desc="Valid URL source for outage cause",
        parent=node,
        critical=True
    )
    evaluator.add_custom_info({"node_id": cause_ref_leaf.id}, "node_handles", "cause_ref_leaf_handle")


async def build_affected_states_branch(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'affected_states' branch.
    """
    node = evaluator.add_parallel(
        id="affected_states",
        desc="All four affected states correctly identified",
        parent=parent,
        critical=True
    )

    states_list_node = evaluator.add_parallel(
        id="states_list",
        desc="Complete list of affected states provided",
        parent=node,
        critical=True
    )

    # Four per-state leaves (set critical=True to satisfy critical parent constraint)
    state_leaves = {}
    for state_id, state_name in [
        ("texas", "Texas"),
        ("georgia", "Georgia"),
        ("new_york", "New York"),
        ("california", "California"),
    ]:
        leaf = evaluator.add_leaf(
            id=state_id,
            desc=f"{state_name} identified as affected state",
            parent=states_list_node,
            critical=True
        )
        state_leaves[state_name] = leaf
        # We'll verify after extraction when we have sources; record handles
        evaluator.add_custom_info({"node_id": leaf.id, "state": state_name}, "node_handles", f"state_leaf_{state_id}_handle")

    states_ref_leaf = evaluator.add_leaf(
        id="states_reference_url",
        desc="Valid URL source for affected states information",
        parent=node,
        critical=True
    )
    evaluator.add_custom_info({"node_id": states_ref_leaf.id}, "node_handles", "states_ref_leaf_handle")


async def build_fcc_submission_branch(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'fcc_submission_info' branch.
    """
    node = evaluator.add_parallel(
        id="fcc_submission_info",
        desc="Complete FCC submission details provided",
        parent=parent,
        critical=True
    )

    email_node = evaluator.add_parallel(
        id="fcc_email",
        desc="Correct FCC email address provided",
        parent=node,
        critical=True
    )

    email_value_leaf = evaluator.add_leaf(
        id="email_value",
        desc="Email is VerizonOutage2026@fcc.gov",
        parent=email_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer provides the FCC email address for outage submissions as {EXPECTED_FCC_EMAIL}.",
        node=email_value_leaf,
        additional_instruction=(
            "Check the answer text only. Email matching can be case-insensitive and should ignore minor formatting noise. "
            "If a different email is provided or missing, mark incorrect."
        )
    )

    email_ref_leaf = evaluator.add_leaf(
        id="email_reference_url",
        desc="Valid URL source for FCC email",
        parent=email_node,
        critical=True
    )
    evaluator.add_custom_info({"node_id": email_ref_leaf.id}, "node_handles", "email_ref_leaf_handle")

    deadline_node = evaluator.add_parallel(
        id="fcc_deadline",
        desc="Correct submission deadline provided",
        parent=node,
        critical=True
    )

    deadline_value_leaf = evaluator.add_leaf(
        id="deadline_value",
        desc="Deadline is March 16, 2026",
        parent=deadline_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer states that the deadline for submissions is {EXPECTED_DEADLINE} (accept 'March 16th, 2026').",
        node=deadline_value_leaf,
        additional_instruction=(
            "Check the answer text only. Accept minor formatting variations like 'March 16th, 2026'. "
            "If the deadline is missing or different, mark incorrect."
        )
    )

    deadline_ref_leaf = evaluator.add_leaf(
        id="deadline_reference_url",
        desc="Valid URL source for deadline",
        parent=deadline_node,
        critical=True
    )
    evaluator.add_custom_info({"node_id": deadline_ref_leaf.id}, "node_handles", "deadline_ref_leaf_handle")


async def build_alternative_carrier_branch(evaluator: Evaluator, parent) -> None:
    """
    Build and verify the 'alternative_carrier' branch.
    Note: To satisfy critical-child constraints while keeping justification optional,
    we set this parent as non-critical, with a critical identity sub-branch and non-critical justification.
    """
    node = evaluator.add_parallel(
        id="alternative_carrier",
        desc="Carrier with most extensive 5G coverage identified",
        parent=parent,
        critical=False  # allow non-critical justification subtree
    )

    identity_node = evaluator.add_parallel(
        id="carrier_identity",
        desc="Correct carrier identified as coverage leader",
        parent=node,
        critical=True
    )

    carrier_name_leaf = evaluator.add_leaf(
        id="carrier_name",
        desc="Carrier identified as having most extensive US 5G coverage based on multiple 2025-2026 sources",
        parent=identity_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The answer identifies {EXPECTED_COVERAGE_LEADER} as having the most extensive 5G coverage in the United States.",
        node=carrier_name_leaf,
        additional_instruction=(
            "Check the answer text only. Accept minor variants like 'T‑Mobile', 'T Mobile', or 'T-Mobile US'. "
            "If the answer names a different carrier, mark incorrect."
        )
    )

    carrier_ref_leaf = evaluator.add_leaf(
        id="carrier_reference_url",
        desc="Valid URL source for carrier coverage data",
        parent=identity_node,
        critical=True
    )
    evaluator.add_custom_info({"node_id": carrier_ref_leaf.id}, "node_handles", "carrier_ref_leaf_handle")

    justification_node = evaluator.add_parallel(
        id="coverage_justification",
        desc="Supporting coverage data provided",
        parent=node,
        critical=False
    )

    # Existence check for coverage metrics
    coverage_metrics_exists = evaluator.add_custom_node(
        result=False,  # placeholder; will update via actual claim verification below if needed
        id="coverage_metrics",
        desc="Coverage percentage or population figures mentioned",
        parent=justification_node,
        critical=False
    )
    # Record handle to replace later
    evaluator.add_custom_info({"node_id": coverage_metrics_exists.id}, "node_handles", "coverage_metrics_exists_handle")

    metrics_ref_leaf = evaluator.add_leaf(
        id="metrics_reference_url",
        desc="Valid URL source for coverage metrics",
        parent=justification_node,
        critical=False
    )
    evaluator.add_custom_info({"node_id": metrics_ref_leaf.id}, "node_handles", "metrics_ref_leaf_handle")


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
    Entry point for evaluating the outage report answer.
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

    # Extract structured information
    extraction: IncidentExtraction = await evaluator.extract(
        prompt=prompt_extract_incident(),
        template_class=IncidentExtraction,
        extraction_name="incident_extraction"
    )

    # Record expected GT info (for transparency; not used for gating)
    evaluator.add_ground_truth({
        "expected_outage_date": EXPECTED_OUTAGE_DATE,
        "expected_cause_keyword": EXPECTED_CAUSE_KEYWORD,
        "expected_states": EXPECTED_STATES,
        "expected_fcc_email": EXPECTED_FCC_EMAIL,
        "expected_deadline": EXPECTED_DEADLINE,
        "expected_coverage_leader": EXPECTED_COVERAGE_LEADER
    })

    # Build the tree structure (create all nodes)
    await build_outage_date_branch(evaluator, root)
    await build_outage_cause_branch(evaluator, root)
    await build_affected_states_branch(evaluator, root)
    await build_fcc_submission_branch(evaluator, root)
    await build_alternative_carrier_branch(evaluator, root)

    # ----------------- Post-extraction verifications with URLs ---------------- #

    # Outage date reference verification
    date_ref_node = evaluator.find_node("date_reference_url")
    if date_ref_node:
        if _has_urls(extraction.outage_date.sources):
            await evaluator.verify(
                claim=f"The major Verizon network outage occurred on {EXPECTED_OUTAGE_DATE}.",
                node=date_ref_node,
                sources=extraction.outage_date.sources,
                additional_instruction=(
                    "Verify that the source explicitly states the outage date. "
                    "Allow minor format differences (e.g., 'Jan 14, 2026')."
                )
            )
        else:
            # Fallback: check that the answer includes at least one valid URL for the date (should fail if none)
            await evaluator.verify(
                claim="The answer includes at least one valid URL that supports that the outage occurred on January 14, 2026.",
                node=date_ref_node,
                sources=None,
                additional_instruction=(
                    "Check only the answer text. If no such URL is present, mark incorrect."
                )
            )

    # Outage cause reference verification
    cause_ref_node = evaluator.find_node("cause_reference_url")
    if cause_ref_node:
        if _has_urls(extraction.outage_cause.sources):
            await evaluator.verify(
                claim="Verizon attributed the January 2026 outage to a software issue (e.g., software bug/update/configuration).",
                node=cause_ref_node,
                sources=extraction.outage_cause.sources,
                additional_instruction=(
                    "Verify that the source clearly attributes the cause to a software issue (accept synonyms)."
                )
            )
        else:
            await evaluator.verify(
                claim="The answer includes at least one valid URL that supports Verizon attributing the outage to a software issue.",
                node=cause_ref_node,
                sources=None,
                additional_instruction="If the answer lacks such a URL, mark incorrect."
            )

    # Affected states per-state verification and states reference
    states_sources = extraction.affected_states.sources
    for state in EXPECTED_STATES:
        # Map state to node id
        node_id_map = {
            "Texas": "texas",
            "Georgia": "georgia",
            "New York": "new_york",
            "California": "california"
        }
        leaf = evaluator.find_node(node_id_map[state])
        if leaf:
            if _has_urls(states_sources):
                await evaluator.verify(
                    claim=f"{state} was affected by the Verizon network outage in January 2026.",
                    node=leaf,
                    sources=states_sources,
                    additional_instruction="The source should explicitly mention this state as affected."
                )
            else:
                await evaluator.verify(
                    claim=f"The answer lists {state} among the affected states.",
                    node=leaf,
                    sources=None,
                    additional_instruction="Check the answer text only. If not listed, mark incorrect."
                )

    states_ref_node = evaluator.find_node("states_reference_url")
    if states_ref_node:
        if _has_urls(states_sources):
            await evaluator.verify(
                claim="This source discusses which US states were affected by the January 2026 Verizon outage.",
                node=states_ref_node,
                sources=states_sources,
                additional_instruction="Verify the source provides information about affected states."
            )
        else:
            await evaluator.verify(
                claim="The answer includes at least one valid URL that supports which US states were affected by the outage.",
                node=states_ref_node,
                sources=None,
                additional_instruction="If the answer lacks such a URL, mark incorrect."
            )

    # FCC email reference
    email_ref_node = evaluator.find_node("email_reference_url")
    if email_ref_node:
        if _has_urls(extraction.fcc.email_sources):
            await evaluator.verify(
                claim=f"The FCC directed customers to email {EXPECTED_FCC_EMAIL} to submit their outage experiences.",
                node=email_ref_node,
                sources=extraction.fcc.email_sources,
                additional_instruction="Verify the email address appears on the page in this context."
            )
        else:
            await evaluator.verify(
                claim=f"The answer includes at least one valid URL that provides the FCC email address {EXPECTED_FCC_EMAIL} for submissions.",
                node=email_ref_node,
                sources=None,
                additional_instruction="If absent, mark incorrect."
            )

    # FCC deadline reference
    deadline_ref_node = evaluator.find_node("deadline_reference_url")
    if deadline_ref_node:
        if _has_urls(extraction.fcc.deadline_sources):
            await evaluator.verify(
                claim=f"The deadline for submitting outage experiences to the FCC is {EXPECTED_DEADLINE}.",
                node=deadline_ref_node,
                sources=extraction.fcc.deadline_sources,
                additional_instruction="Accept minor formatting differences like 'March 16th, 2026'."
            )
        else:
            await evaluator.verify(
                claim=f"The answer includes at least one valid URL that supports the submission deadline of {EXPECTED_DEADLINE}.",
                node=deadline_ref_node,
                sources=None,
                additional_instruction="If absent, mark incorrect."
            )

    # Carrier reference verification
    carrier_ref_node = evaluator.find_node("carrier_reference_url")
    if carrier_ref_node:
        if _has_urls(extraction.coverage.sources):
            await evaluator.verify(
                claim=f"{EXPECTED_COVERAGE_LEADER} has the most extensive 5G coverage in the United States based on 2025-2026 reports.",
                node=carrier_ref_node,
                sources=extraction.coverage.sources,
                additional_instruction=(
                    "Verify that the source(s) clearly indicate the identified carrier leads in overall 5G coverage. "
                    "Allow synonymous phrasing such as 'largest 5G network', 'covers the most people/area', etc."
                )
            )
        else:
            await evaluator.verify(
                claim=f"The answer includes at least one valid URL supporting that {EXPECTED_COVERAGE_LEADER} has the most extensive US 5G coverage.",
                node=carrier_ref_node,
                sources=None,
                additional_instruction="If absent, mark incorrect."
            )

    # Coverage metrics existence (custom node replacement): update result based on extraction
    metrics_exists_handle = evaluator.find_node("coverage_metrics")
    if metrics_exists_handle:
        # Replace the placeholder custom node by adding an additional custom child marking actual existence
        # Note: We cannot modify existing node fields easily; we add a sibling custom node to ensure a concrete binary leaf exists.
        # However, to adhere to the given ID, we will add a parallel sibling node only if necessary.
        # Here, we'll just add another custom node indicating the same requirement.
        metrics_present = bool(extraction.coverage.coverage_metrics and extraction.coverage.coverage_metrics.strip())
        # Since the original placeholder node already exists with default failed state (0.0),
        # add an additional node with explicit result to reflect actual status.
        evaluator.add_custom_node(
            result=metrics_present,
            id="coverage_metrics_extracted",
            desc="Coverage percentage or population figures mentioned (extracted presence check)",
            parent=metrics_exists_handle and evaluator.find_node("coverage_justification"),
            critical=False
        )

    # Coverage metrics reference verification
    metrics_ref_node = evaluator.find_node("metrics_reference_url")
    if metrics_ref_node:
        # Combine metrics-specific sources and general coverage sources for flexibility
        candidate_urls = []
        if _has_urls(extraction.coverage.metrics_sources):
            candidate_urls.extend(extraction.coverage.metrics_sources)
        if _has_urls(extraction.coverage.sources):
            candidate_urls.extend(extraction.coverage.sources)
        if _has_urls(candidate_urls):
            await evaluator.verify(
                claim=f"The coverage metric(s) '{extraction.coverage.coverage_metrics or ''}' are supported by the cited source(s) (exact phrasing may vary).",
                node=metrics_ref_node,
                sources=candidate_urls,
                additional_instruction=(
                    "Verify that the page provides coverage metrics consistent with those mentioned in the answer. "
                    "Allow minor differences in rounding or wording."
                )
            )
        else:
            await evaluator.verify(
                claim="The answer includes at least one valid URL that supports the coverage metrics cited.",
                node=metrics_ref_node,
                sources=None,
                additional_instruction="If absent, mark incorrect."
            )

    # Return final summary
    return evaluator.get_summary()