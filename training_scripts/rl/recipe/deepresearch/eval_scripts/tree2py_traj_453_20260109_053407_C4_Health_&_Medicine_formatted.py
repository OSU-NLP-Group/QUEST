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
TASK_ID = "fda_2024_cder_multi_designations_oncology"
TASK_DESCRIPTION = (
    "Identify a drug that was approved by the FDA's Center for Drug Evaluation and Research (CDER) "
    "as a novel drug in 2024 and received all three of the following expedited program designations: "
    "Breakthrough Therapy, Orphan Drug, and Priority Review. The drug must treat a cancer-related condition. "
    "Provide the drug's trade name, its specific FDA-approved oncology indication, and a reference to official "
    "FDA documentation confirming these designations (specifically the FDA 2024 New Drug Therapy Approvals Annual Report)."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugInfo(BaseModel):
    """Structured info extracted from the agent's answer."""
    trade_name: Optional[str] = None
    generic_name: Optional[str] = None
    specific_oncology_indication: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return (
        "Extract the following fields from the answer about the FDA CDER 2024 novel oncology drug:\n"
        "1) trade_name: The drug's trade (brand) name exactly as stated in the answer.\n"
        "2) generic_name: The generic/INN name if explicitly stated in the answer (else null).\n"
        "3) specific_oncology_indication: A concise text string stating the FDA-approved oncology indication exactly as the answer provides (e.g., 'for the treatment of unresectable or metastatic X').\n"
        "4) sources: A list of all URLs explicitly cited in the answer that are intended to support the drug's status/designations/indication. "
        "   Only include actual URLs visible in the answer (plain URLs or markdown links). Do not invent or infer.\n"
        "If any field is missing from the answer, return null (or an empty list for sources)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_sources(urls: List[str]) -> List[str]:
    """Basic normalization and de-duplication for URL list."""
    if not urls:
        return []
    seen = set()
    normed = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Prepend protocol if missing (per framework suggestion)
        if not s.lower().startswith(("http://", "https://")):
            s = "http://" + s
        if s not in seen:
            seen.add(s)
            normed.append(s)
    return normed


def _is_annual_report_url(url: str) -> bool:
    """
    Heuristic check for the FDA 2024 New Drug Therapy Approvals annual report URL (webpage or PDF).
    Accepts various official FDA paths that contain NDTA 2024 content.
    """
    if not url or not isinstance(url, str):
        return False
    u = url.lower()
    if "fda.gov" not in u:
        return False
    has_year = "2024" in u
    # Look for common NDTA strings / slugs
    tokens = [
        "new-drug-therapy-approvals",
        "ndta",
        "new_drug_therapy_approvals",
        "newdrugtherapyapprovals",
    ]
    has_ndta_phrase = any(tok in u for tok in tokens)
    # Often NDTA PDFs are under /media/ paths
    is_pdf = u.endswith(".pdf") and "media" in u
    return (has_year and has_ndta_phrase) or is_pdf


def _filter_annual_report_urls(urls: List[str]) -> List[str]:
    """Return only URLs that likely correspond to the FDA 2024 NDTA annual report."""
    return [u for u in urls if _is_annual_report_url(u)]


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    parent_node,
    info: DrugInfo,
) -> None:
    """
    Build verification leaves based on the rubric and run verifications.
    The parent_node is a critical parallel node representing the overall task.
    """
    trade = (info.trade_name or "").strip()
    generic = (info.generic_name or "").strip()
    indication = (info.specific_oncology_indication or "").strip()
    all_sources = _normalize_sources(info.sources)
    annual_report_sources = _filter_annual_report_urls(all_sources)

    # Record some helpful custom info in the summary
    evaluator.add_custom_info(
        {"all_sources": all_sources, "annual_report_sources": annual_report_sources},
        info_type="source_collection",
        info_name="extracted_sources",
    )

    # 1) Trade_Name_Provided (critical existence check)
    evaluator.add_custom_node(
        result=bool(trade),
        id="Trade_Name_Provided",
        desc="The response provides the drug’s trade (brand) name",
        parent=parent_node,
        critical=True,
    )

    # 2) Specific_FDA_Approved_Indication_Stated (critical existence check)
    evaluator.add_custom_node(
        result=bool(indication),
        id="Specific_FDA_Approved_Indication_Stated",
        desc="The response clearly states the specific FDA-approved oncology indication for the drug",
        parent=parent_node,
        critical=True,
    )

    # 3) FDA_Annual_Report_Citation_For_Designations (critical existence of NDTA 2024 citation)
    evaluator.add_custom_node(
        result=len(annual_report_sources) > 0,
        id="FDA_Annual_Report_Citation_For_Designations",
        desc="The response provides a reference to the FDA 2024 NDTA Annual Report confirming designations",
        parent=parent_node,
        critical=True,
    )

    # 4) CDER_Novel_Drug_2024 (verify via URLs)
    cder_novel_node = evaluator.add_leaf(
        id="CDER_Novel_Drug_2024",
        desc="The drug is one of the CDER novel drugs approved in 2024",
        parent=parent_node,
        critical=True,
    )
    novel_claim = (
        f"The drug {trade or generic} is listed among CDER novel drug approvals in 2024 on official FDA documentation "
        f"(e.g., the FDA 2024 New Drug Therapy Approvals annual report)."
    )
    await evaluator.verify(
        claim=novel_claim,
        node=cder_novel_node,
        sources=annual_report_sources if annual_report_sources else all_sources,
        additional_instruction=(
            "Confirm using FDA official pages that the drug appears on the 2024 CDER novel drug approvals list. "
            "Allow minor name variants (brand vs. generic). If the provided URL is not an FDA page or irrelevant, "
            "consider it not supported."
        ),
    )

    # 5) Oncology_Indication (verify that indication is cancer-related)
    oncology_node = evaluator.add_leaf(
        id="Oncology_Indication",
        desc="The drug’s FDA-approved indication is cancer-related (oncology/malignancy)",
        parent=parent_node,
        critical=True,
    )
    oncology_claim = (
        f"The FDA-approved indication for {trade or generic} is oncology-related: '{indication}'. "
        f"This is a malignancy/cancer/tumor/neoplasm indication."
    )
    await evaluator.verify(
        claim=oncology_claim,
        node=oncology_node,
        sources=annual_report_sources if annual_report_sources else all_sources,
        additional_instruction=(
            "Verify from the official FDA source(s) that the indication is for a cancer-related condition. "
            "Treat synonyms like malignancy, tumor, carcinoma, sarcoma, lymphoma, leukemia, myeloma, neoplasm as cancer-related."
        ),
    )

    # 6) Breakthrough_Therapy_Designation
    breakthrough_node = evaluator.add_leaf(
        id="Breakthrough_Therapy_Designation",
        desc="The drug received FDA Breakthrough Therapy designation",
        parent=parent_node,
        critical=True,
    )
    breakthrough_claim = f"The drug {trade or generic} received FDA Breakthrough Therapy designation."
    await evaluator.verify(
        claim=breakthrough_claim,
        node=breakthrough_node,
        sources=annual_report_sources if annual_report_sources else all_sources,
        additional_instruction=(
            "Confirm from the FDA 2024 NDTA annual report (or other official FDA documentation) that this drug "
            "has the Breakthrough Therapy designation."
        ),
    )

    # 7) Orphan_Drug_Designation
    orphan_node = evaluator.add_leaf(
        id="Orphan_Drug_Designation",
        desc="The drug received FDA Orphan Drug designation",
        parent=parent_node,
        critical=True,
    )
    orphan_claim = f"The drug {trade or generic} received FDA Orphan Drug designation."
    await evaluator.verify(
        claim=orphan_claim,
        node=orphan_node,
        sources=annual_report_sources if annual_report_sources else all_sources,
        additional_instruction=(
            "Confirm from the FDA 2024 NDTA annual report (or other official FDA documentation) that this drug "
            "has the Orphan Drug designation."
        ),
    )

    # 8) Priority_Review_Designation
    priority_node = evaluator.add_leaf(
        id="Priority_Review_Designation",
        desc="The drug received FDA Priority Review designation",
        parent=parent_node,
        critical=True,
    )
    priority_claim = f"The drug {trade or generic} received FDA Priority Review designation."
    await evaluator.verify(
        claim=priority_claim,
        node=priority_node,
        sources=annual_report_sources if annual_report_sources else all_sources,
        additional_instruction=(
            "Confirm from the FDA 2024 NDTA annual report (or other official FDA documentation) that this drug "
            "has Priority Review designation."
        ),
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
    Evaluate an agent's answer for the FDA 2024 CDER multi-designations oncology task.

    Returns:
        A structured evaluation summary dict produced by the evaluator.
    """
    # Initialize evaluator with a non-critical root; create a critical task node under it
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

    # Create the task node (critical, as all criteria are mandatory)
    task_node = evaluator.add_parallel(
        id="2024_FDA_Drug_Multiple_Designations",
        desc="Identify a 2024 CDER novel drug (oncology) with Breakthrough Therapy, Orphan Drug, and Priority Review designations, and provide the required fields with FDA documentation",
        parent=root,
        critical=True,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugInfo,
        extraction_name="drug_info",
    )

    # Build the verification leaves and run verifications
    await build_and_verify(evaluator, task_node, extracted)

    # Return evaluation summary
    return evaluator.get_summary()