import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "fda_2025_drugs_identification"
TASK_DESCRIPTION = """Identify five distinct FDA-approved novel drug therapies from 2025, where each drug meets one of the following specific criteria. For each drug, provide: (1) brand name, (2) generic/nonproprietary name, (3) exact FDA approval date (formatted as Month Day, Year), (4) approved indication as stated in FDA documentation, and (5) the name of the pharmaceutical company that is the sponsor/manufacturer.

The five drugs you must identify are:

Drug A: A drug that received FDA accelerated approval in September 2025 for treating an ultra-rare mitochondrial disease and has orphan drug designation.

Drug B: A first-in-class oral antibiotic approved by the FDA in 2025 that represents the first new antibiotic class for treating uncomplicated urinary tract infections in nearly 30 years.

Drug C: A cardiac myosin inhibitor approved by the FDA in December 2025 specifically for treating symptomatic obstructive hypertrophic cardiomyopathy in adults.

Drug D: The first and only FDA-approved therapy for hematopoietic stem cell transplant-associated thrombotic microangiopathy (TA-TMA), approved in December 2025.

Drug E: A first-in-class non-opioid analgesic approved by the FDA in January 2025 for treating moderate to severe acute pain in adults.

For each drug, include a reference URL to either the official FDA novel drug approvals page entry, the drug's FDA-approved label, or an official FDA press release about the approval.
"""


# ------------------------------ Data Models -------------------------------- #
class DrugItem(BaseModel):
    label: Optional[str] = None  # "A" | "B" | "C" | "D" | "E"
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    approval_date: Optional[str] = None  # Expected format: "Month Day, Year"
    indication: Optional[str] = None
    sponsor: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class DrugsExtraction(BaseModel):
    drug_A: Optional[DrugItem] = None
    drug_B: Optional[DrugItem] = None
    drug_C: Optional[DrugItem] = None
    drug_D: Optional[DrugItem] = None
    drug_E: Optional[DrugItem] = None


# ---------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_drugs() -> str:
    return """
    Extract structured information for five specific 2025 FDA CDER novel drug therapies labeled A–E from the answer.
    For each of Drug A–E, extract the following fields:
      - label: The label letter ("A", "B", "C", "D", or "E") as referred to in the answer.
      - brand_name: The brand name (trade name) of the drug.
      - generic_name: The generic/nonproprietary name of the drug.
      - approval_date: The exact FDA approval date as stated in official sources, formatted exactly as "Month Day, Year" (e.g., "January 15, 2025").
      - indication: The FDA‑approved indication text (or a faithful paraphrase) from official FDA documentation/label.
      - sponsor: The sponsor/manufacturer company name associated with the FDA approval.
      - references: All source URLs explicitly included in the answer for that drug. These may be plain URLs or markdown links. Include only URLs actually present in the answer text.

    SPECIAL RULES:
    - Extract only what appears in the answer; do not invent or infer missing values. If a field is missing for a drug, set it to null (or an empty list for references).
    - approval_date must be a string exactly formatted as "Month Day, Year". If the answer shows a different format, convert to "Month Day, Year" if it is unambiguous; otherwise set to null.
    - references must be valid URL strings found in the answer (plain or markdown); do not infer. Include FDA press releases, FDA labels (Drugs@FDA), and FDA novel drug approvals entries when provided.
    - Keep each drug's references limited to URLs cited for that specific drug.

    Return a JSON object with keys: drug_A, drug_B, drug_C, drug_D, drug_E. Each key maps to an object containing the fields above. If an entire drug is missing, set that key to null.
    """


# ------------------------------ Helper Utils -------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _pair_key(item: Optional[DrugItem]) -> str:
    if not item:
        return ""
    return f"{_norm(item.brand_name)}|{_norm(item.generic_name)}"


def _collect_items(drugs: DrugsExtraction) -> List[Optional[DrugItem]]:
    return [drugs.drug_A, drugs.drug_B, drugs.drug_C, drugs.drug_D, drugs.drug_E]


def _distinct_all_five(drugs: DrugsExtraction) -> bool:
    items = _collect_items(drugs)
    # All five must be present with at least one identifying name (brand or generic)
    if any(it is None for it in items):
        return False
    keys = []
    for it in items:
        if not it or (not _norm(it.brand_name) and not _norm(it.generic_name)):
            return False
        keys.append(_pair_key(it))
    return len(set(keys)) == 5


def _safe_drug_display(item: Optional[DrugItem]) -> str:
    if not item:
        return "unknown drug"
    b = (item.brand_name or "").strip()
    g = (item.generic_name or "").strip()
    if b and g:
        return f"{b} ({g})"
    return b or g or "unknown drug"


# -------------------------- Per-Drug Verification --------------------------- #
def _criteria_claim(letter: str, item: Optional[DrugItem]) -> Tuple[str, str]:
    """
    Build the criteria claim and additional instruction for each drug letter A–E.
    """
    disp = _safe_drug_display(item)

    if letter == "A":
        claim = (
            f"{disp} received FDA accelerated approval in September 2025, treats an ultra-rare mitochondrial disease, "
            f"and has orphan drug designation."
        )
        addins = (
            "Confirm all parts: (1) accelerated approval; (2) month is September 2025; "
            "(3) indication relates to an ultra-rare mitochondrial disease; (4) orphan drug designation. "
            "Allow reasonable paraphrases. Use official FDA sources or press releases provided."
        )
    elif letter == "B":
        claim = (
            f"{disp} is a first-in-class oral antibiotic approved by FDA in 2025 and represents the first new antibiotic "
            f"class for treating uncomplicated urinary tract infections in nearly 30 years."
        )
        addins = (
            "Look for explicit 'first-in-class' and statements about the first new antibiotic class for UTI in ~30 years. "
            "Approval year must be 2025. Verify from FDA sources or official press releases provided."
        )
    elif letter == "C":
        claim = (
            f"{disp} is a cardiac myosin inhibitor approved in December 2025 for treating symptomatic obstructive "
            f"hypertrophic cardiomyopathy (HCM) in adults."
        )
        addins = (
            "Confirm 'cardiac myosin inhibitor', month/year 'December 2025', and indication 'symptomatic obstructive HCM in adults'. "
            "Use FDA documentation (label, novel drug page, or press release) among the provided URLs."
        )
    elif letter == "D":
        claim = (
            f"{disp} is the first and only FDA-approved therapy for hematopoietic stem cell transplant-associated "
            f"thrombotic microangiopathy (TA-TMA), with FDA approval in December 2025."
        )
        addins = (
            "Confirm 'first and only' therapy for TA-TMA and approval month/year 'December 2025'. "
            "Rely on FDA official sources or company press releases among the provided URLs."
        )
    elif letter == "E":
        claim = (
            f"{disp} is a first-in-class non-opioid analgesic approved in January 2025 for treating moderate to severe acute pain in adults."
        )
        addins = (
            "Confirm 'first-in-class', non-opioid analgesic, month/year 'January 2025', and indication "
            "'moderate to severe acute pain in adults' using the provided official URLs."
        )
    else:
        claim = f"{disp} meets the specified criteria for drug {letter}."
        addins = "Verify the criteria specified for the drug using the provided URLs."

    return claim, addins


def _listed_on_2025_novel_page_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    claim = f"{disp} is listed as a 2025 CDER novel drug therapy on the official FDA 'Novel Drug Approvals 2025' page."
    addins = (
        "This should be supported specifically by the drug's entry on FDA's 'Novel Drug Approvals 2025' page. "
        "If none of the provided URLs is that FDA page entry for this drug, mark as not supported."
    )
    return claim, addins


def _brand_name_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    brand = (item.brand_name or "").strip() if item else ""
    claim = f"The brand name for {disp} is '{brand}'."
    addins = "Verify the brand (trade) name exactly as shown in official FDA sources (label, novel drug page, or press release). Allow minor casing variations."
    return claim, addins


def _generic_name_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    gen = (item.generic_name or "").strip() if item else ""
    claim = f"The generic/nonproprietary name for {disp} is '{gen}'."
    addins = "Verify the generic/nonproprietary name exactly as shown in official FDA sources (label, novel drug page, or press release). Allow minor casing variations."
    return claim, addins


def _approval_date_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    dt = (item.approval_date or "").strip() if item else ""
    claim = f"The FDA approval date for {disp} is '{dt}' (formatted as Month Day, Year) according to official FDA sources."
    addins = (
        "Confirm the exact approval date string and ensure it matches the official FDA page content for this drug. "
        "The date must be in 'Month Day, Year' format (e.g., 'December 12, 2025')."
    )
    return claim, addins


def _indication_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    ind = (item.indication or "").strip() if item else ""
    claim = f"The FDA-approved indication for {disp} is: {ind}"
    addins = (
        "Verify the indication text (or faithful paraphrase) matches the FDA label or official FDA documentation. "
        "Allow close paraphrases, but the clinical meaning must match."
    )
    return claim, addins


def _sponsor_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    sp = (item.sponsor or "").strip() if item else ""
    claim = f"The sponsor/manufacturer for {disp} is '{sp}'."
    addins = (
        "Verify the sponsor/manufacturer from FDA documentation (label, novel drug page) or official company press release. "
        "Allow minor name variants (e.g., Inc., Ltd.)."
    )
    return claim, addins


def _reference_valid_claim(item: Optional[DrugItem]) -> Tuple[str, str]:
    disp = _safe_drug_display(item)
    claim = (
        f"At least one of the provided URLs for {disp} is an official and relevant source "
        f"(FDA novel drug page entry, FDA label/Drugs@FDA, or an FDA/official company press release about the approval)."
    )
    addins = (
        "Treat pages on fda.gov (including Drugs@FDA labels) or official FDA press releases as valid. "
        "Also accept an official company press release specifically about FDA approval of this drug. "
        "The page content must clearly be about this drug’s FDA approval or label."
    )
    return claim, addins


async def verify_drug_item(
    evaluator: Evaluator,
    parent_node,
    item: Optional[DrugItem],
    letter: str,
) -> None:
    """
    Build verification nodes and run verifications for a single drug item (A–E).
    """
    # Parent node for this drug (non-critical to allow partial scoring per item)
    desc_map = {
        "A": "Drug A: accelerated approval in September 2025 for an ultra-rare mitochondrial disease with orphan drug designation; provide required fields and references.",
        "B": "Drug B: first-in-class oral antibiotic approved in 2025; first new class for uncomplicated UTI in nearly 30 years; provide required fields and references.",
        "C": "Drug C: cardiac myosin inhibitor approved in December 2025 for symptomatic obstructive hypertrophic cardiomyopathy in adults; provide required fields and references.",
        "D": "Drug D: first and only FDA-approved therapy for TA-TMA, approved in December 2025; provide required fields and references.",
        "E": "Drug E: first-in-class non-opioid analgesic approved in January 2025 for moderate to severe acute pain in adults; provide required fields and references.",
    }
    group_node = evaluator.add_parallel(
        id=f"Drug_{letter}_Item",
        desc=desc_map.get(letter, f"Drug {letter} verification"),
        parent=parent_node,
        critical=False,
    )

    refs = item.references if item and item.references else []

    # Create leaf nodes
    listed_node = evaluator.add_leaf(
        id=f"{letter}_Listed_As_2025_CDER_Novel_Drug",
        desc="Listed on FDA 2025 CDER Novel Drug Approvals page",
        parent=group_node,
        critical=True,
    )
    meets_node = evaluator.add_leaf(
        id=f"{letter}_Meets_Drug_{letter}_Criteria",
        desc=f"Meets drug {letter} specified criteria",
        parent=group_node,
        critical=True,
    )
    brand_node = evaluator.add_leaf(
        id=f"{letter}_Brand_Name_Correct",
        desc=f"Drug {letter} brand name matches official documentation",
        parent=group_node,
        critical=True,
    )
    generic_node = evaluator.add_leaf(
        id=f"{letter}_Generic_Name_Correct",
        desc=f"Drug {letter} generic/nonproprietary name matches official documentation",
        parent=group_node,
        critical=True,
    )
    approval_node = evaluator.add_leaf(
        id=f"{letter}_Approval_Date_Exact_And_Formatted",
        desc=f"Drug {letter} FDA approval date matches official source and is correctly formatted",
        parent=group_node,
        critical=True,
    )
    indication_node = evaluator.add_leaf(
        id=f"{letter}_Indication_Accurate",
        desc=f"Drug {letter} indication accurately reflects FDA-approved use",
        parent=group_node,
        critical=True,
    )
    sponsor_node = evaluator.add_leaf(
        id=f"{letter}_Sponsor_Manufacturer_Verifiable",
        desc=f"Drug {letter} sponsor/manufacturer verifiable from official sources",
        parent=group_node,
        critical=True,
    )
    ref_valid_node = evaluator.add_leaf(
        id=f"{letter}_Reference_URL_Valid",
        desc=f"Drug {letter} has at least one valid official reference URL",
        parent=group_node,
        critical=True,
    )

    # Prepare claims and sources
    listed_claim, listed_addins = _listed_on_2025_novel_page_claim(item)
    criteria_claim, criteria_addins = _criteria_claim(letter, item)
    brand_claim, brand_addins = _brand_name_claim(item)
    generic_claim, generic_addins = _generic_name_claim(item)
    approval_claim, approval_addins = _approval_date_claim(item)
    indication_claim, indication_addins = _indication_claim(item)
    sponsor_claim, sponsor_addins = _sponsor_claim(item)
    ref_valid_claim, ref_valid_addins = _reference_valid_claim(item)

    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = [
        (listed_claim, refs, listed_node, listed_addins),
        (criteria_claim, refs, meets_node, criteria_addins),
        (brand_claim, refs, brand_node, brand_addins),
        (generic_claim, refs, generic_node, generic_addins),
        (approval_claim, refs, approval_node, approval_addins),
        (indication_claim, refs, indication_node, indication_addins),
        (sponsor_claim, refs, sponsor_node, sponsor_addins),
        (ref_valid_claim, refs, ref_valid_node, ref_valid_addins),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


# ------------------------------ Main Evaluation ----------------------------- #
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
    Entry point for evaluating the FDA 2025 drug identification task.
    """
    # Initialize evaluator (root as parallel aggregator)
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

    # Top-level task node (set non-critical to allow partial scoring across A–E)
    task_node = evaluator.add_parallel(
        id="FDA_Drug_Identification_Task",
        desc="Identify five distinct FDA CDER novel drug therapies from 2025 matching specified A–E criteria with official details and references.",
        parent=root,
        critical=False,
    )

    # Extract structured drug info from answer
    drugs_info = await evaluator.extract(
        prompt=prompt_extract_drugs(),
        template_class=DrugsExtraction,
        extraction_name="extracted_2025_fda_drugs"
    )

    # Critical: All five identified drugs must be distinct
    distinct_node = evaluator.add_custom_node(
        result=_distinct_all_five(drugs_info),
        id="All_Drugs_Are_Distinct",
        desc="All five identified drugs are distinct (no duplicates across A–E).",
        parent=task_node,
        critical=True
    )

    # Build and verify per-drug items (A–E)
    letter_map = [("A", drugs_info.drug_A), ("B", drugs_info.drug_B),
                  ("C", drugs_info.drug_C), ("D", drugs_info.drug_D),
                  ("E", drugs_info.drug_E)]

    # Run per-drug verifications concurrently
    await asyncio.gather(*[
        verify_drug_item(evaluator, task_node, item, letter)
        for letter, item in letter_map
    ])

    # Return standardized summary
    return evaluator.get_summary()