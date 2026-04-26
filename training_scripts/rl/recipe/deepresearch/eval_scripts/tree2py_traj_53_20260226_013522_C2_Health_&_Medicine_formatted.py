import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gene_therapy_intrathecal_2025_2026"
TASK_DESCRIPTION = (
    "Identify a gene therapy that was approved by the FDA in 2025 or 2026 for treating a pediatric neurological or "
    "neuromuscular condition, specifically one that is administered via the intrathecal (into the spinal canal) route "
    "and is approved for patients aged 2 years or older. For the identified therapy, provide the following "
    "information: (1) the generic (nonproprietary) drug name, (2) the brand (proprietary/trade) name, "
    "(3) the specific medical condition it treats, (4) the FDA approval date, and (5) a direct link to an official "
    "FDA press announcement or approval notice."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TherapyInfo(BaseModel):
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    condition: Optional[str] = None
    approval_date: Optional[str] = None
    age_eligibility_statement: Optional[str] = None
    administration_route_statement: Optional[str] = None
    official_fda_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_therapy_info() -> str:
    return """
    Extract the following information about the identified therapy exactly as stated in the answer:

    Fields to extract (return null if missing):
    - generic_name: The generic (nonproprietary) name of the drug.
    - brand_name: The brand (proprietary/trade) name of the drug.
    - condition: The specific medical condition the therapy treats.
    - approval_date: The FDA approval date (any reasonable date format as stated in the answer).
    - age_eligibility_statement: The exact phrase/sentence in the answer describing age eligibility (e.g., "for patients aged 2 years and older").
    - administration_route_statement: The exact phrase/sentence in the answer describing how the therapy is administered (e.g., "administered intrathecally").
    - official_fda_url: A direct URL provided in the answer that points to an official FDA press announcement or approval notice (must be on an fda.gov domain, for example newsroom press announcements, 'FDA approves ...' pages, Drugs@FDA approval notices, or biologics approval pages). If multiple FDA URLs are present, choose the most directly relevant approval/press page. If the answer provides no such URL, set to null.
    - additional_urls: All other URLs (if any) provided in the answer that are relevant to this therapy (do not re-include the official_fda_url here).

    Rules:
    - Only extract what is explicitly present in the answer text. Do not infer or fabricate.
    - For URLs, include only complete and valid URLs. If a URL lacks protocol, prepend http://.
    - If multiple FDA URLs are present, select the best candidate for official_fda_url and list the rest in additional_urls.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def gather_sources(info: TherapyInfo) -> List[str]:
    """Combine official FDA URL (if any) and any additional urls into a single list for multi-URL verification."""
    urls: List[str] = []
    if info.official_fda_url and info.official_fda_url.strip():
        urls.append(info.official_fda_url.strip())
    for u in info.additional_urls:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: TherapyInfo) -> None:
    """
    Build the verification tree and run all checks based on the rubric:
      - Root: Gene_Therapy_Report_Completeness (parallel, non-critical)
      - Child 1: Clinical_Criteria_Compliance (parallel, critical)
      - Child 2: Documentation_Requirements (parallel, critical)
      - Child 3: Drug_Identification (parallel, critical)
    """
    # Child 1: Clinical_Criteria_Compliance (critical, parallel)
    clinical_node = evaluator.add_parallel(
        id="Clinical_Criteria_Compliance",
        desc="The identified gene therapy must meet all clinical selection criteria from the constraints",
        parent=root_node,
        critical=True
    )

    # Sources to use for clinical verification
    clinical_sources = gather_sources(extracted)

    # Leaf: FDA_Approval_Timing (critical)
    timing_leaf = evaluator.add_leaf(
        id="FDA_Approval_Timing",
        desc="The therapy was approved by the FDA in 2025 or 2026",
        parent=clinical_node,
        critical=True
    )
    await evaluator.verify(
        claim="The FDA approval for this therapy occurred in 2025 or 2026.",
        node=timing_leaf,
        sources=clinical_sources,
        additional_instruction=(
            "Use the provided FDA page(s) to determine the approval year. "
            "Accept initial approval or clearly labeled 'FDA approves' notices in 2025 or 2026. "
            "If the page only refers to earlier years for approval, mark as incorrect."
        )
    )

    # Leaf: Target_Condition (critical)
    target_leaf = evaluator.add_leaf(
        id="Target_Condition",
        desc="The therapy is indicated for treating a pediatric neurological or neuromuscular condition",
        parent=clinical_node,
        critical=True
    )
    cond_hint = extracted.condition or "the stated condition"
    await evaluator.verify(
        claim="This therapy is indicated for a pediatric neurological or neuromuscular condition.",
        node=target_leaf,
        sources=clinical_sources,
        additional_instruction=(
            f"Rely on the FDA page(s). Determine whether the indication is pediatric (children) and the disease is "
            f"neurological or neuromuscular. The answer mentions: {cond_hint!r}. Accept reasonable synonyms "
            f"(e.g., neurodevelopmental disorders, neuromuscular diseases like SMA or DMD, leukodystrophies, "
            f"enzymatic deficiencies affecting the nervous system). If not pediatric and neuro/neuromuscular, "
            f"mark as incorrect."
        )
    )

    # Leaf: Administration_Method (critical)
    admin_leaf = evaluator.add_leaf(
        id="Administration_Method",
        desc="The therapy is approved for intrathecal (into the spinal canal) administration route",
        parent=clinical_node,
        critical=True
    )
    await evaluator.verify(
        claim="The therapy's approved administration route includes intrathecal (into the spinal canal) delivery.",
        node=admin_leaf,
        sources=clinical_sources,
        additional_instruction=(
            "Verify on the FDA page(s) that the administration route is intrathecal. "
            "Allow synonyms/descriptions such as 'intrathecal injection', 'IT administration', 'administered into the CSF', "
            "'given via lumbar puncture', or 'into the spinal canal'."
        )
    )

    # Child 2: Documentation_Requirements (critical, parallel)
    docs_node = evaluator.add_parallel(
        id="Documentation_Requirements",
        desc="All required official documentation and details must be provided",
        parent=root_node,
        critical=True
    )

    # Leaf: Medical_Condition_Identified (critical) - existence in the answer
    cond_exists = bool(extracted.condition and extracted.condition.strip())
    evaluator.add_custom_node(
        result=cond_exists,
        id="Medical_Condition_Identified",
        desc="The specific medical condition being treated is explicitly stated in the answer",
        parent=docs_node,
        critical=True
    )

    # Leaf: FDA_Approval_Date (critical) - existence in the answer
    date_exists = bool(extracted.approval_date and extracted.approval_date.strip())
    evaluator.add_custom_node(
        result=date_exists,
        id="FDA_Approval_Date",
        desc="The specific FDA approval date is stated in the answer",
        parent=docs_node,
        critical=True
    )

    # Leaf: Official_FDA_URL (critical) - verify the provided URL is an official FDA press/approval page
    fda_url_leaf = evaluator.add_leaf(
        id="Official_FDA_URL",
        desc="A direct link to an official FDA press announcement or approval notice is provided",
        parent=docs_node,
        critical=True
    )
    await evaluator.verify(
        claim="This URL is on the official FDA website and is a press announcement or approval notice for this therapy.",
        node=fda_url_leaf,
        sources=extracted.official_fda_url,  # May be None; then simple verification over answer will be used.
        additional_instruction=(
            "Confirm that the URL is on an FDA domain (e.g., fda.gov, www.fda.gov, accessdata.fda.gov, "
            "drugsatfda) and that the page explicitly announces or documents FDA approval for the therapy. "
            "Examples: FDA Newsroom press announcements ('FDA approves ...'), Drugs@FDA approval notices/labels, "
            "biologics approval summaries/letters. If no URL is provided or the URL is non-FDA, mark as incorrect."
        )
    )

    # Leaf: Age_Eligibility (critical) - ensure the answer states '2 years or older' explicitly
    age_leaf = evaluator.add_leaf(
        id="Age_Eligibility",
        desc="The patient age eligibility criterion (2 years or older) is stated",
        parent=docs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the therapy is approved for patients aged 2 years or older (or equivalent phrasing).",
        node=age_leaf,
        sources=None,  # Check presence in the answer itself
        additional_instruction=(
            "Only assess the answer content. Accept synonymous phrasings such as '2 years and older', '≥2 years of age', "
            "'two years and older', '2 years of age and up', 'ages 2+.' If the answer does not clearly say 2 years or older, mark as incorrect."
        )
    )

    # Child 3: Drug_Identification (critical, parallel)
    drug_node = evaluator.add_parallel(
        id="Drug_Identification",
        desc="Complete drug nomenclature must be provided",
        parent=root_node,
        critical=True
    )

    # Leaf: Generic_Drug_Name (critical) - existence in the answer
    generic_exists = bool(extracted.generic_name and extracted.generic_name.strip())
    evaluator.add_custom_node(
        result=generic_exists,
        id="Generic_Drug_Name",
        desc="The generic (nonproprietary) name of the drug is provided",
        parent=drug_node,
        critical=True
    )

    # Leaf: Brand_Name (critical) - existence in the answer
    brand_exists = bool(extracted.brand_name and extracted.brand_name.strip())
    evaluator.add_custom_node(
        result=brand_exists,
        id="Brand_Name",
        desc="The brand (proprietary/trade) name of the drug is provided",
        parent=drug_node,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entrypoint for evaluating the answer for the gene therapy task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_therapy_info(),
        template_class=TherapyInfo,
        extraction_name="therapy_info",
    )

    # Optionally record helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "generic_name": extracted_info.generic_name,
            "brand_name": extracted_info.brand_name,
            "condition": extracted_info.condition,
            "approval_date": extracted_info.approval_date,
            "age_eligibility_statement": extracted_info.age_eligibility_statement,
            "administration_route_statement": extracted_info.administration_route_statement,
            "official_fda_url": extracted_info.official_fda_url,
            "additional_urls_count": len(extracted_info.additional_urls or []),
        },
        info_type="extraction_overview",
    )

    # Build verification tree based on rubric
    await build_verification_tree(evaluator, root, extracted_info)

    # Return the evaluation summary
    return evaluator.get_summary()