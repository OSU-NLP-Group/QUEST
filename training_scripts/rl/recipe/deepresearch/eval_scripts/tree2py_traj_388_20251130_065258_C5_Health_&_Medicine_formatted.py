import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse
import re

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_2024_first_in_class_public_company"
TASK_DESCRIPTION = (
    "A hospital formulary committee is evaluating breakthrough therapies for potential addition to their treatment "
    "protocols. Identify a novel drug that was approved by the FDA (CDER) in calendar year 2024 and that represents "
    "the first FDA-approved pharmacological treatment for its specific medical indication. The drug must be manufactured "
    "by a publicly traded pharmaceutical company. Provide the following information: (1) Drug brand name and active "
    "ingredient(s), (2) Specific medical indication/condition the drug treats, (3) FDA approval date, (4) Pharmaceutical "
    "company/manufacturer name, (5) Description of the drug's mechanism of action, (6) Name or identifier of a Phase 3 "
    "clinical trial that supported the approval, and (7) Reference URLs from FDA.gov or official company/medical sources "
    "supporting your answer."
)

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class DrugExtraction(BaseModel):
    brand_name: Optional[str] = None
    active_ingredients: List[str] = Field(default_factory=list)
    indication: Optional[str] = None
    fda_approval_date: Optional[str] = None
    manufacturer: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    phase3_trial: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_drug_info() -> str:
    return """
    Extract a single drug candidate as presented in the answer and return the following JSON fields:

    - brand_name: The drug brand name (string).
    - active_ingredients: A list of active ingredient names exactly as written in the answer. If a single ingredient is given, return a one-element list.
    - indication: The specific medical condition or indication treated (string).
    - fda_approval_date: The FDA approval date as written in the answer (string; keep original format).
    - manufacturer: The pharmaceutical company/manufacturer name (string).
    - mechanism_of_action: A brief description of the mechanism of action (string), or null if not provided.
    - phase3_trial: The name or identifier of a Phase 3 clinical trial that supported the approval (string), or null if not provided.
    - reference_urls: A list of all reference URLs explicitly provided in the answer (FDA.gov, company domains, medical sources such as ClinicalTrials.gov, NEJM, JAMA, etc.). Extract only valid URLs that appear in the answer.

    IMPORTANT:
    - If multiple drugs are mentioned, extract only the first primary drug used by the answer to satisfy the task.
    - Do not invent information not present in the answer. If a field is not clearly stated, return null (for strings) or [] (for lists).
    - For active_ingredients, split multiple ingredients if clearly comma- or slash-separated in the answer.
    - For the reference_urls, include all valid URLs that appear in the answer (plain or markdown link targets).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_ALLOWED_MEDICAL_OR_OFFICIAL_DOMAINS = [
    "fda.gov",
    "clinicaltrials.gov",
    "nih.gov",
    "jamanetwork.com",
    "nejm.org",
    "thelancet.com",
    "nature.com",
    "bmj.com",
    "ema.europa.eu",
    "pubmed.ncbi.nlm.nih.gov",
    "who.int",
]

_STOPWORDS_FOR_COMPANY = {
    "inc", "inc.", "corp", "corp.", "corporation", "ltd", "ltd.",
    "limited", "plc", "group", "co", "co.", "company", "sa", "ag",
    "nv", "kk", "gmbh", "holding", "holdings", "pharmaceuticals",
    "pharmaceutical", "pharma", "biosciences", "therapeutics",
    "biotech", "biotechnology", "laboratories", "lab", "labs",
}


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _domain_matches_suffix(netloc: str, suffix: str) -> bool:
    if not netloc or not suffix:
        return False
    return netloc == suffix or netloc.endswith("." + suffix)


def _normalize_company_tokens(company: Optional[str]) -> List[str]:
    if not company:
        return []
    s = company.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in _STOPWORDS_FOR_COMPANY and len(t) >= 2]
    # Deduplicate, keep order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _is_official_company_domain(url: str, company: Optional[str]) -> bool:
    net = _netloc(url)
    if not net or not company:
        return False
    tokens = _normalize_company_tokens(company)
    # Accept if any company token appears in the netloc contiguously.
    # Also accept if token's abbreviation (e.g., "bms" for "bristol myers squibb") is present — too risky to generalize; skip abbreviations.
    for t in tokens:
        if t in net:
            return True
    return False


def _is_acceptable_reference_url(url: str, company: Optional[str]) -> bool:
    net = _netloc(url)
    if not net:
        return False
    # Official/medical/agency sources
    for suf in _ALLOWED_MEDICAL_OR_OFFICIAL_DOMAINS:
        if _domain_matches_suffix(net, suf):
            return True
    # Company domain heuristic
    if _is_official_company_domain(url, company):
        return True
    return False


def _partition_urls_by_acceptability(urls: List[str], company: Optional[str]) -> Tuple[List[str], List[str]]:
    good, bad = [], []
    for u in urls:
        if _is_acceptable_reference_url(u, company):
            good.append(u)
        else:
            bad.append(u)
    return good, bad


def _display_drug_name(data: DrugExtraction) -> str:
    if data.brand_name:
        return data.brand_name
    if data.active_ingredients:
        return ", ".join([x for x in data.active_ingredients if x]) or "the drug"
    return "the drug"


# --------------------------------------------------------------------------- #
# Tree-building and verification                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root, data: DrugExtraction) -> None:
    # Task Completion wrapper (non-critical to allow optional sub-criteria)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify a drug satisfying all mandatory eligibility constraints and provide all mandatory details and acceptable references; include optional details if present.",
        parent=root,
        critical=False
    )

    # ----------------------------- References --------------------------------
    refs_group = evaluator.add_parallel(
        id="References",
        desc="Provides acceptable supporting reference URLs.",
        parent=task_node,
        critical=True
    )

    acceptable_urls, unacceptable_urls = _partition_urls_by_acceptability(data.reference_urls, data.manufacturer)

    refs_ok = len(acceptable_urls) > 0
    refs_leaf = evaluator.add_custom_node(
        result=refs_ok,
        id="Reference_URLs_From_Acceptable_Sources",
        desc="Includes reference URL(s) from FDA.gov or official company/medical sources supporting the answer.",
        parent=refs_group,
        critical=True
    )

    # Record helpful info for debugging
    evaluator.add_custom_info(
        info={
            "all_reference_urls": data.reference_urls,
            "acceptable_urls_used": acceptable_urls,
            "unacceptable_urls": unacceptable_urls
        },
        info_type="reference_urls_check",
        info_name="reference_urls_diagnostics"
    )

    # ---------------------- Mandatory Answer Fields --------------------------
    mandatory_group = evaluator.add_parallel(
        id="Mandatory_Answer_Fields",
        desc="All mandatory fields requested in the constraints are present.",
        parent=task_node,
        critical=True
    )

    # Brand + Active ingredients provided
    brand_and_ai_provided = (
        (data.brand_name is not None and data.brand_name.strip() != "") and
        (len([ai for ai in data.active_ingredients if isinstance(ai, str) and ai.strip() != ""]) >= 1)
    )
    evaluator.add_custom_node(
        result=brand_and_ai_provided,
        id="Brand_Name_And_Active_Ingredients_Provided",
        desc="Provides the drug brand name and active ingredient(s).",
        parent=mandatory_group,
        critical=True
    )

    # Indication Provided
    evaluator.add_custom_node(
        result=(data.indication is not None and data.indication.strip() != ""),
        id="Indication_Provided",
        desc="Provides the specific medical indication/condition treated.",
        parent=mandatory_group,
        critical=True
    )

    # FDA Approval Date Provided
    evaluator.add_custom_node(
        result=(data.fda_approval_date is not None and data.fda_approval_date.strip() != ""),
        id="FDA_Approval_Date_Provided",
        desc="Provides the FDA approval date.",
        parent=mandatory_group,
        critical=True
    )

    # Manufacturer Provided
    evaluator.add_custom_node(
        result=(data.manufacturer is not None and data.manufacturer.strip() != ""),
        id="Manufacturer_Name_Provided",
        desc="Provides the pharmaceutical company/manufacturer name.",
        parent=mandatory_group,
        critical=True
    )

    # ---------------------- Optional Answer Fields ---------------------------
    optional_group = evaluator.add_parallel(
        id="Optional_Answer_Fields",
        desc='Optional ("should") fields included if provided.',
        parent=task_node,
        critical=False
    )

    evaluator.add_custom_node(
        result=(data.mechanism_of_action is not None and data.mechanism_of_action.strip() != ""),
        id="Mechanism_Of_Action_Described",
        desc="Includes a description of the drug’s mechanism of action.",
        parent=optional_group,
        critical=False
    )

    evaluator.add_custom_node(
        result=(data.phase3_trial is not None and data.phase3_trial.strip() != ""),
        id="Phase_3_Trial_Identifier_Provided",
        desc="Provides the name or identifier of a Phase 3 clinical trial that supported the approval.",
        parent=optional_group,
        critical=False
    )

    # ------------------------ Eligibility Criteria ---------------------------
    eligibility_group = evaluator.add_parallel(
        id="Eligibility_Criteria",
        desc="Drug meets all mandatory eligibility constraints.",
        parent=task_node,
        critical=True
    )

    # 1) FDA CDER novel approval in 2024
    novel_node = evaluator.add_leaf(
        id="Novel_2024_FDA_CDER_Approval",
        desc="Drug is a novel FDA (CDER) approval in calendar year 2024 (Jan 1–Dec 31, 2024).",
        parent=eligibility_group,
        critical=True
    )
    drug_label = _display_drug_name(data)
    claim_novel = (
        f"The drug {drug_label} received an FDA approval in the calendar year 2024 under CDER (Center for Drug "
        f"Evaluation and Research). It is a 'novel drug' or first approval/NME in 2024."
    )
    await evaluator.verify(
        claim=claim_novel,
        node=novel_node,
        sources=acceptable_urls if acceptable_urls else data.reference_urls,
        additional_instruction=(
            "Confirm that the drug appears on FDA CDER's 'Novel Drug Approvals 2024' list or has an FDA Drugs@FDA page "
            "showing first approval in 2024 (CDER). Evidence may include FDA.gov novel drug list, approval letter, "
            "or label showing Approval Date in 2024. If the approval is by CBER (biologics), this should not count."
        ),
        extra_prerequisites=[refs_leaf]
    )

    # 2) First pharmacological treatment for the indication
    first_for_ind_node = evaluator.add_leaf(
        id="First_Pharmacologic_Treatment_For_Indication",
        desc="Drug is the first FDA-approved pharmacological treatment for its specific medical indication.",
        parent=eligibility_group,
        critical=True
    )
    indication_text = data.indication or "the specified indication"
    claim_first_for_ind = (
        f"The drug {drug_label} represents the first FDA-approved pharmacological treatment for {indication_text}."
    )
    await evaluator.verify(
        claim=claim_first_for_ind,
        node=first_for_ind_node,
        sources=acceptable_urls if acceptable_urls else data.reference_urls,
        additional_instruction=(
            "Look for explicit language like 'first FDA-approved treatment', 'first pharmacologic therapy', or "
            "'first therapy approved for' the stated indication on FDA.gov or official company/medical sources."
        ),
        extra_prerequisites=[refs_leaf]
    )

    # 3) Manufacturer publicly traded
    public_company_node = evaluator.add_leaf(
        id="Manufacturer_Is_Publicly_Traded",
        desc="Manufacturer is a publicly traded pharmaceutical company.",
        parent=eligibility_group,
        critical=True
    )
    manufacturer_name = data.manufacturer or "the manufacturer"
    claim_public = (
        f"{manufacturer_name} is a publicly traded pharmaceutical company (has a stock ticker on a major exchange such as "
        f"NASDAQ, NYSE, LSE, TSE, SIX, HKEX, etc.)."
    )
    await evaluator.verify(
        claim=claim_public,
        node=public_company_node,
        sources=acceptable_urls if acceptable_urls else data.reference_urls,
        additional_instruction=(
            "Use official company sources if available (e.g., investor relations or press releases mentioning their ticker, "
            "such as 'NASDAQ: XXXX' or 'NYSE: XXXX'). If the provided URLs do not clearly indicate listing, the claim should fail."
        ),
        extra_prerequisites=[refs_leaf]
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2024 FDA novel first-in-indication drug identification task.
    """
    # Initialize evaluator (root node is non-critical parallel aggregator)
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

    # Extract structured info from the answer
    extracted: DrugExtraction = await evaluator.extract(
        prompt=prompt_extract_drug_info(),
        template_class=DrugExtraction,
        extraction_name="drug_extraction"
    )

    # Build verification tree and run verifications
    await build_verification_tree(evaluator, root, extracted)

    # Return standardized summary
    return evaluator.get_summary()