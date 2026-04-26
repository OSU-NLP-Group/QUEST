import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "drug_2025_orphan_peds_adult"
TASK_DESCRIPTION = (
    "Identify a novel drug that received FDA approval in 2025 for treating a rare disease in both pediatric and adult patients. "
    "Provide comprehensive information about this drug including: (1) Brand/trade name, (2) Generic name (active ingredient), "
    "(3) Exact FDA approval date, (4) Approved indication (disease/condition treated), (5) Approved age range (minimum age for pediatric use), "
    "(6) Dosage form, (7) Route of administration, (8) Mechanism of action or drug class, (9) Orphan drug designation status (confirmed), "
    "(10) Breakthrough therapy designation status (if applicable), (11) Manufacturer/sponsor company name, and (12) Reference URL to FDA's official approval documentation. "
    "The drug must meet the following requirements: received novel drug approval from FDA in calendar year 2025, has orphan drug designation for a rare disease, "
    "and is approved for use in both pediatric and adult patient populations. All information must be verifiable through official FDA sources or manufacturer documentation."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DrugSources(BaseModel):
    """URLs/sources cited in the answer."""
    fda_official_url: Optional[str] = None
    orphan_designation_urls: List[str] = Field(default_factory=list)
    breakthrough_designation_urls: List[str] = Field(default_factory=list)
    label_or_indication_urls: List[str] = Field(default_factory=list)
    manufacturer_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class DrugInfoExtraction(BaseModel):
    """All fields required by the task, extracted from the agent's answer."""
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    approval_date: Optional[str] = None
    indication: Optional[str] = None
    pediatric_min_age: Optional[str] = None
    dosage_form: Optional[str] = None
    route_of_admin: Optional[str] = None
    mechanism_or_class: Optional[str] = None
    orphan_designation_status: Optional[str] = None  # e.g., "Yes" / "Designated orphan" / textual mention
    breakthrough_designation_status: Optional[str] = None  # e.g., "Yes", "No", "Not applicable"
    manufacturer: Optional[str] = None
    sources: DrugSources = Field(default_factory=DrugSources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
    Extract the complete set of drug information provided in the answer for a single FDA-approved drug in 2025 (novel drug) for a rare disease approved in both pediatric and adult patients.

    Required fields (return null if missing in the answer text):
    1. brand_name: The brand or trade name of the drug.
    2. generic_name: The generic name or active ingredient name.
    3. approval_date: The exact FDA approval date (month/day/year) as stated.
    4. indication: The approved medical indication (disease/condition) described in the answer.
    5. pediatric_min_age: The minimum pediatric age for which the drug is approved (e.g., "≥ 6 months", "2 years", "12 years").
    6. dosage_form: The dosage form (e.g., tablet, capsule, injection, oral suspension).
    7. route_of_admin: The route of administration (e.g., oral, intravenous, subcutaneous).
    8. mechanism_or_class: The mechanism of action or drug class/category.
    9. orphan_designation_status: Whether the drug has orphan drug designation, as stated (e.g., "Yes", "Designated", "No").
    10. breakthrough_designation_status: Whether the drug received breakthrough therapy designation, if stated (e.g., "Yes", "No", "Not applicable").
    11. manufacturer: The manufacturer or sponsor company name.

    Source URLs (extract explicit URLs only; return null for missing single URL fields; empty arrays are allowed for lists):
    sources:
      - fda_official_url: A URL to the official FDA approval documentation (e.g., Drugs@FDA approval letter, CDER news announcement, FDA press release). It must be an FDA domain.
      - orphan_designation_urls: URLs that specifically confirm orphan drug designation (e.g., FDA Orphan Drug Designations and Approvals database entry).
      - breakthrough_designation_urls: URLs supporting breakthrough therapy designation, if claimed.
      - label_or_indication_urls: URLs that show indication, age ranges, label information (can be FDA label PDFs or Drugs@FDA).
      - manufacturer_urls: URLs from the manufacturer or sponsor confirming approval details (optional if FDA page suffices).
      - other_urls: Any other supporting URLs from the answer.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent or infer.
    - For URLs, capture the actual links (plain URLs or markdown links). If a source is mentioned without an explicit URL, return null or empty list accordingly.
    - Do not normalize content. Preserve the wording as it appears in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())

def _unique_urls(urls: List[Optional[str]]) -> List[str]:
    out = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u_str = str(u).strip()
        if not u_str:
            continue
        if u_str not in seen:
            out.append(u_str)
            seen.add(u_str)
    return out

def _merge_sources(drug: DrugInfoExtraction) -> List[str]:
    return _unique_urls(
        [drug.sources.fda_official_url]
        + drug.sources.orphan_designation_urls
        + drug.sources.breakthrough_designation_urls
        + drug.sources.label_or_indication_urls
        + drug.sources.manufacturer_urls
        + drug.sources.other_urls
    )


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_drug_verification_tree(
    evaluator: Evaluator,
    root_node,
    drug: DrugInfoExtraction,
) -> None:
    """
    Build the verification tree per rubric and run key verifications.
    """
    # Main parallel node as per rubric
    main_node = evaluator.add_parallel(
        id="Drug_Information_Completeness",
        desc="All required information about a 2025 FDA-approved drug for rare pediatric/adult disease must be provided with proper verification",
        parent=root_node,
        critical=False
    )

    # ---------- Existence checks (match '..._Provided' rubric leaves) ----------
    brand_node = evaluator.add_custom_node(
        result=_non_empty(drug.brand_name),
        id="Brand_Name_Provided",
        desc="The brand/trade name of the drug is provided",
        parent=main_node,
        critical=True
    )

    generic_node = evaluator.add_custom_node(
        result=_non_empty(drug.generic_name),
        id="Generic_Name_Provided",
        desc="The generic name or active ingredient name is provided",
        parent=main_node,
        critical=True
    )

    approval_date_node = evaluator.add_custom_node(
        result=_non_empty(drug.approval_date),
        id="Approval_Date_Provided",
        desc="The specific FDA approval date (month/day/year) is provided",
        parent=main_node,
        critical=True
    )

    indication_node = evaluator.add_custom_node(
        result=_non_empty(drug.indication),
        id="Indication_Provided",
        desc="The approved medical indication or disease/condition treated is provided",
        parent=main_node,
        critical=True
    )

    age_range_node = evaluator.add_custom_node(
        result=_non_empty(drug.pediatric_min_age),
        id="Age_Range_Provided",
        desc="The approved age range or minimum age for use is provided",
        parent=main_node,
        critical=True
    )

    dosage_form_node = evaluator.add_custom_node(
        result=_non_empty(drug.dosage_form),
        id="Dosage_Form_Provided",
        desc="The dosage form (e.g., tablet, capsule, injection, suspension) is provided",
        parent=main_node,
        critical=True
    )

    route_node = evaluator.add_custom_node(
        result=_non_empty(drug.route_of_admin),
        id="Route_Administration_Provided",
        desc="The route of administration (e.g., oral, intravenous, subcutaneous) is provided",
        parent=main_node,
        critical=True
    )

    mechanism_node = evaluator.add_custom_node(
        result=_non_empty(drug.mechanism_or_class),
        id="Mechanism_Class_Provided",
        desc="The drug's mechanism of action or therapeutic class is provided",
        parent=main_node,
        critical=True
    )

    breakthrough_node = evaluator.add_custom_node(
        result=_non_empty(drug.breakthrough_designation_status),
        id="Breakthrough_Status_Provided",
        desc="Information about whether the drug received breakthrough therapy designation is provided",
        parent=main_node,
        critical=False
    )

    manufacturer_node = evaluator.add_custom_node(
        result=_non_empty(drug.manufacturer),
        id="Manufacturer_Provided",
        desc="The name of the manufacturer or sponsor company is provided",
        parent=main_node,
        critical=False
    )

    # Ensure FDA reference URL is provided and official (fda.gov domain)
    fda_url_provided = _non_empty(drug.sources.fda_official_url)
    fda_url_is_official = fda_url_provided and ("fda.gov" in str(drug.sources.fda_official_url).lower())
    fda_url_node = evaluator.add_custom_node(
        result=fda_url_is_official,
        id="FDA_Reference_URL",
        desc="A reference URL to official FDA documentation about the approval is provided",
        parent=main_node,
        critical=True
    )

    # ---------- Core verification leaves (critical checks) ----------
    core_verif_node = evaluator.add_parallel(
        id="Core_Verifications",
        desc="Core evidence-based verifications (FDA approval in 2025, orphan designation, pediatric and adult approvals)",
        parent=main_node,
        critical=False
    )

    # 1) FDA Approval in 2025 (critical)
    fda_approval_leaf = evaluator.add_leaf(
        id="FDA_Approval_2025",
        desc="The identified drug received novel drug approval from FDA during calendar year 2025",
        parent=core_verif_node,
        critical=True
    )

    # 2) Orphan drug designation (critical)
    orphan_leaf = evaluator.add_leaf(
        id="Orphan_Designation",
        desc="The drug has orphan drug designation for treating a rare disease",
        parent=core_verif_node,
        critical=True
    )

    # 3) Pediatric approval (critical)
    pediatric_leaf = evaluator.add_leaf(
        id="Pediatric_Approval",
        desc="The drug is approved for use in pediatric patients",
        parent=core_verif_node,
        critical=True
    )

    # 4) Adult approval (critical)
    adult_leaf = evaluator.add_leaf(
        id="Adult_Approval",
        desc="The drug is approved for use in adult patients",
        parent=core_verif_node,
        critical=True
    )

    # Prepare claims and sources
    brand = drug.brand_name or ""
    generic = drug.generic_name or ""
    appr_date = drug.approval_date or ""
    ped_min_age = drug.pediatric_min_age or ""
    indication = drug.indication or ""

    # Sources for specific checks
    fda_only_sources = _unique_urls([drug.sources.fda_official_url])
    orphan_sources = _unique_urls(drug.sources.orphan_designation_urls or []) or fda_only_sources
    ped_sources = _unique_urls(drug.sources.label_or_indication_urls or []) or fda_only_sources
    adult_sources = fda_only_sources

    # Build verification tasks concurrently to minimize unintended precondition gating
    verify_tasks = []

    # FDA Approval 2025 claim
    fda_claim = (
        f"The official FDA documentation confirms that the drug {brand} ({generic}) received FDA approval in calendar year 2025, "
        f"with an approval date of {appr_date}."
    )
    verify_tasks.append(
        evaluator.verify(
            claim=fda_claim,
            node=fda_approval_leaf,
            sources=fda_only_sources,
            additional_instruction=(
                "Verify the FDA approval occurred in 2025 and the approval date matches exactly. "
                "Accept if the page clearly indicates an FDA approval in 2025, including approval letters, Drugs@FDA entries, "
                "or FDA announcements. Minor formatting differences are acceptable."
            ),
            extra_prerequisites=[fda_url_node, brand_node, generic_node, approval_date_node]
        )
    )

    # Orphan designation claim
    orphan_claim = (
        f"The drug {brand} ({generic}) has orphan drug designation for a rare disease."
    )
    verify_tasks.append(
        evaluator.verify(
            claim=orphan_claim,
            node=orphan_leaf,
            sources=orphan_sources,
            additional_instruction=(
                "Confirm orphan drug designation using FDA Orphan Drug Designations and Approvals database or "
                "explicit mention on FDA/manufacturer documentation. The page should explicitly indicate 'orphan drug' "
                "designation for this product."
            ),
            extra_prerequisites=[fda_url_node]
        )
    )

    # Pediatric approval claim
    ped_claim = (
        f"The drug {brand} ({generic}) is approved for pediatric patients, with a minimum age of {ped_min_age}."
    )
    verify_tasks.append(
        evaluator.verify(
            claim=ped_claim,
            node=pediatric_leaf,
            sources=ped_sources,
            additional_instruction=(
                "Check labeling or FDA documentation (e.g., Indications and Usage) that clearly states pediatric approval and "
                "the minimum age. Accept equivalent phrasing (e.g., 'patients 12 years and older', '≥ 6 months')."
            ),
            extra_prerequisites=[fda_url_node, age_range_node]
        )
    )

    # Adult approval claim
    adult_claim = (
        f"The drug {brand} ({generic}) is approved for adult patients."
    )
    verify_tasks.append(
        evaluator.verify(
            claim=adult_claim,
            node=adult_leaf,
            sources=adult_sources,
            additional_instruction=(
                "Confirm that the indication or labeling includes adult patients. Accept equivalent phrasing such as "
                "'adults', 'patients 18 years and older', or age ranges that include adults."
            ),
            extra_prerequisites=[fda_url_node]
        )
    )

    # Execute all verifications concurrently
    await asyncio.gather(*verify_tasks)


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
    Evaluate an answer for the 2025 FDA rare disease drug approval task.
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract structured drug info from the answer
    drug_info = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugInfoExtraction,
        extraction_name="drug_info_extraction"
    )

    # Build verification tree and run checks
    await build_drug_verification_tree(evaluator, root, drug_info)

    # Return evaluation summary
    return evaluator.get_summary()