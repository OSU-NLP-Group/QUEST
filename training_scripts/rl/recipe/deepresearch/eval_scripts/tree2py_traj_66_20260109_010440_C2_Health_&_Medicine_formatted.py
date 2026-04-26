import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fda_uuti_novel_oral_antibiotic_2025"
TASK_DESCRIPTION = """
Identify the novel oral antibiotic approved by the FDA in March 2025 for the treatment of uncomplicated urinary tract infections in female patients aged 12 years and older, which represents the first new class of oral antibiotics for this indication in nearly 30 years. Provide the following information: (1) brand name, (2) generic name, (3) manufacturer, (4) FDA approval date, and (5) the ClinicalTrials.gov NCT identifiers for the two phase 3 clinical trials that supported its approval.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AntibioticApprovalExtraction(BaseModel):
    brand_name: Optional[str] = None
    generic_name: Optional[str] = None
    manufacturer: Optional[str] = None
    approval_date: Optional[str] = None
    nct_ids: List[str] = Field(default_factory=list)
    fda_urls: List[str] = Field(default_factory=list)
    ctgov_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_antibiotic_info() -> str:
    return """
    You must extract structured information from the answer about the FDA-approved oral antibiotic for uncomplicated urinary tract infection (uUTI) in March 2025.

    Extract the following fields exactly as stated in the answer (do not invent or infer):
    1. brand_name: The FDA-approved brand name of the drug.
    2. generic_name: The generic (nonproprietary) name of the drug.
    3. manufacturer: The company (sponsor/applicant/manufacturer) associated with the drug per the answer.
    4. approval_date: The specific FDA approval date as presented in the answer (e.g., 'March 12, 2025' or '2025-03-12'). Return the exact string; do not convert formats.
    5. nct_ids: A list of ClinicalTrials.gov NCT identifiers mentioned as phase 3 trials supporting approval. Extract ALL NCT IDs mentioned in the answer. Preserve order of appearance, and include each unique NCT ID only once.

    Also extract and categorize all URLs cited in the answer:
    - fda_urls: All URLs that belong to any 'fda.gov' domain, including 'accessdata.fda.gov', 'drugsatfda.fda.gov', or pages under 'fda.gov'.
    - ctgov_urls: All URLs from ClinicalTrials.gov (domain 'clinicaltrials.gov').
    - other_urls: All remaining URLs that are not FDA or ClinicalTrials.gov (e.g., company press releases, news articles). Include official manufacturer/company domains if cited.

    IMPORTANT:
    - Only include URLs explicitly present in the answer. Do not add or infer URLs.
    - Return null for any missing string field.
    - For lists, return an empty list when nothing is provided.
    - Keep URLs as given (markdown links are acceptable; ensure the URL part is extracted).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _is_specific_date(date_str: Optional[str]) -> bool:
    if not _non_empty(date_str):
        return False
    s = date_str.strip()
    # Accept formats like 'March 12, 2025' or 'Mar 12, 2025'
    month_day_year = re.search(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b[, ]+\d{1,2}[\s,]+(?:20\d{2}|\d{4})",
        s, flags=re.IGNORECASE
    )
    # Accept ISO 'YYYY-MM-DD'
    iso = re.search(r"\b\d{4}-\d{2}-\d{2}\b", s)
    return bool(month_day_year or iso)


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for x in items:
        if x is None:
            continue
        xi = x.strip()
        if not xi or xi in seen:
            continue
        seen.add(xi)
        result.append(xi)
    return result


def _choose_two_ncts(ncts: List[str]) -> List[str]:
    uniq = _unique_preserve_order(ncts)
    return uniq[:2]


def _pick_ctgov_url_for_nct(nct_id: str, ctgov_urls: List[str]) -> Optional[str]:
    nid = nct_id.lower().strip()
    for u in ctgov_urls:
        if nid in u.lower():
            return u
    return None


def _display_drug_name(ex: AntibioticApprovalExtraction) -> str:
    if _non_empty(ex.brand_name) and _non_empty(ex.generic_name):
        return f"{ex.brand_name} ({ex.generic_name})"
    if _non_empty(ex.brand_name):
        return ex.brand_name.strip()
    if _non_empty(ex.generic_name):
        return ex.generic_name.strip()
    return "the identified drug"


def _combine_sources(*lists: List[str]) -> List[str]:
    combined = []
    for lst in lists:
        combined.extend(lst or [])
    return _unique_preserve_order(combined)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identification_criteria(
    evaluator: Evaluator,
    parent: Any,
    ex: AntibioticApprovalExtraction,
) -> None:
    criteria_node = evaluator.add_parallel(
        id="Drug_Matches_Identification_Criteria",
        desc="The identified drug satisfies all identifying constraints from the question/constraints.",
        parent=parent,
        critical=True
    )

    display_name = _display_drug_name(ex)
    fda_sources = ex.fda_urls
    mixed_sources = _combine_sources(ex.fda_urls, ex.other_urls)

    # Create leaf nodes
    novel_node = evaluator.add_leaf(
        id="Novel_Drug_CDER_2025",
        desc="Drug is a novel drug approved by FDA CDER in 2025 (as defined/claimed by publicly available FDA approval information).",
        parent=criteria_node,
        critical=True
    )
    approval_march_node = evaluator.add_leaf(
        id="Approval_In_March_2025",
        desc="Drug FDA approval occurred in March 2025.",
        parent=criteria_node,
        critical=True
    )
    indication_node = evaluator.add_leaf(
        id="Indication_uUTI_Females_Age_12_Plus",
        desc="Drug is indicated for uncomplicated UTI in female patients aged 12 years and older.",
        parent=criteria_node,
        critical=True
    )
    oral_form_node = evaluator.add_leaf(
        id="Oral_Antibiotic_Formulation",
        desc="Drug is available as an oral antibiotic formulation.",
        parent=criteria_node,
        critical=True
    )
    new_class_node = evaluator.add_leaf(
        id="First_New_Oral_Antibiotic_Class_In_Nearly_30_Years",
        desc="Drug represents the first new class of oral antibiotics for this indication in nearly 30 years.",
        parent=criteria_node,
        critical=True
    )

    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = [
        (
            f"FDA CDER documentation lists {display_name} among the Novel Drug Approvals for 2025.",
            fda_sources,
            novel_node,
            "Look for official FDA/CDER 'Novel Drug Approvals' pages or equivalent FDA documentation explicitly listing the drug in 2025."
        ),
        (
            f"FDA documentation indicates that the approval of {display_name} occurred in March 2025.",
            fda_sources,
            approval_march_node,
            "Use official FDA pages such as press announcements, approval letters, or drug database entries to confirm the approval month is March 2025."
        ),
        (
            f"Official product labeling or FDA documentation shows {display_name} is indicated for treatment of uncomplicated urinary tract infections (uUTI) in female patients aged 12 years and older.",
            fda_sources,
            indication_node,
            "Verify indication and population from FDA labeling or FDA drug information pages."
        ),
        (
            f"Official documentation confirms {display_name} is an oral antibiotic (oral dosage form).",
            mixed_sources,
            oral_form_node,
            "Check FDA labeling or official sources for dosage form and antibiotic classification."
        ),
        (
            f"Public documentation describes {display_name} as the first new class of oral antibiotics for uUTI in nearly 30 years.",
            mixed_sources,
            new_class_node,
            "Confirm novelty claim from public documentation (preferably FDA or authoritative sources)."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_reported_fields(
    evaluator: Evaluator,
    parent: Any,
    ex: AntibioticApprovalExtraction,
) -> None:
    fields_node = evaluator.add_parallel(
        id="Required_Reported_Fields",
        desc="All requested fields are provided for the identified drug.",
        parent=parent,
        critical=True
    )

    # Existence checks (custom critical nodes)
    evaluator.add_custom_node(
        result=_non_empty(ex.brand_name),
        id="Brand_Name_Provided",
        desc="Provides the FDA-approved brand name.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(ex.generic_name),
        id="Generic_Name_Provided",
        desc="Provides the generic name.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(ex.manufacturer),
        id="Manufacturer_Provided",
        desc="Provides the manufacturer.",
        parent=fields_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_specific_date(ex.approval_date),
        id="Exact_FDA_Approval_Date_Provided",
        desc="Provides the specific FDA approval date (day-month-year).",
        parent=fields_node,
        critical=True
    )

    # NCT IDs provided: exactly two distinct
    two_ncts = _choose_two_ncts(ex.nct_ids)
    exactly_two_distinct = len(two_ncts) == 2 and len(set(two_ncts)) == 2 and len(_unique_preserve_order(ex.nct_ids)) == 2
    evaluator.add_custom_node(
        result=exactly_two_distinct,
        id="Two_NCT_IDs_Provided",
        desc="Provides exactly two distinct ClinicalTrials.gov NCT identifiers.",
        parent=fields_node,
        critical=True
    )

    # NCT IDs are phase 3 and supported approval (split per-NCT as critical children)
    nct_support_parent = evaluator.add_parallel(
        id="NCT_IDs_Are_Phase3_And_Support_Approval",
        desc="The two provided NCT identifiers correspond to the phase 3 clinical trials that supported the drug’s approval.",
        parent=fields_node,
        critical=True
    )

    display_name = _display_drug_name(ex)
    for idx, nct in enumerate(two_ncts):
        leaf = evaluator.add_leaf(
            id=f"NCT_{idx}_Phase3_Support",
            desc=f"NCT {nct} is a phase 3 trial that supported the approval.",
            parent=nct_support_parent,
            critical=True
        )
        ct_url = _pick_ctgov_url_for_nct(nct, ex.ctgov_urls)
        sources = _combine_sources([ct_url] if ct_url else [], ex.fda_urls)
        claim = f"The clinical trial NCT{nct} is a Phase 3 study that contributed to the FDA approval of {display_name} for uUTI."
        add_ins = (
            "To pass, evidence across provided pages must confirm BOTH: "
            "1) NCT{nct} is Phase 3 (ClinicalTrials.gov page typically states the phase), and "
            "2) the trial supported the FDA approval of the drug (usually cited by FDA documents or announcements)."
        ).replace("{nct}", nct)
        await evaluator.verify(claim=claim, node=leaf, sources=sources, additional_instruction=add_ins)


async def verify_public_documentation(
    evaluator: Evaluator,
    parent: Any,
    ex: AntibioticApprovalExtraction,
) -> None:
    pub_node = evaluator.add_parallel(
        id="Public_Documentation_Evidence",
        desc="Information is supported by publicly available documentation as required by the constraints.",
        parent=parent,
        critical=True
    )

    display_name = _display_drug_name(ex)
    fda_sources = ex.fda_urls

    # FDA source verifications
    fda_novel_node = evaluator.add_leaf(
        id="FDA_Public_Source_For_NovelDrug_CDER_2025",
        desc="Cites publicly accessible FDA documentation supporting that the drug is a CDER-approved novel drug in 2025.",
        parent=pub_node,
        critical=True
    )
    fda_approval_month_node = evaluator.add_leaf(
        id="FDA_Public_Source_For_Approval_Month_March_2025",
        desc="Cites publicly accessible FDA documentation supporting that the drug’s FDA approval occurred in March 2025.",
        parent=pub_node,
        critical=True
    )
    fda_mfr_node = evaluator.add_leaf(
        id="FDA_Public_Source_For_Manufacturer",
        desc="Cites publicly accessible FDA documentation supporting the manufacturer.",
        parent=pub_node,
        critical=True
    )
    fda_date_node = evaluator.add_leaf(
        id="FDA_Public_Source_For_Approval_Date",
        desc="Cites publicly accessible FDA documentation supporting the exact FDA approval date.",
        parent=pub_node,
        critical=True
    )
    fda_ind_node = evaluator.add_leaf(
        id="FDA_Public_Source_For_Indication_And_Population",
        desc="Cites publicly accessible FDA documentation supporting the indication and eligibility (uUTI; females; age ≥12).",
        parent=pub_node,
        critical=True
    )

    claims_and_sources: List[Tuple[str, List[str], Any, Optional[str]]] = [
        (
            f"Official FDA/CDER pages list {display_name} among 2025 Novel Drug Approvals.",
            fda_sources,
            fda_novel_node,
            "Use only FDA webpages. Look for CDER 'Novel Drug Approvals 2025' or equivalent FDA documentation."
        ),
        (
            f"Official FDA pages confirm that the approval for {display_name} occurred in March 2025.",
            fda_sources,
            fda_approval_month_node,
            "Use FDA press releases, approval letters, or drug database entries to verify the approval month."
        ),
        (
            f"Official FDA pages identify {ex.manufacturer or 'the manufacturer'} as the sponsor/applicant/manufacturer of {display_name}.",
            fda_sources,
            fda_mfr_node,
            "Verify manufacturer/sponsor from FDA labeling or FDA drug information pages."
        ),
        (
            f"Official FDA pages show the exact approval date for {display_name} is '{ex.approval_date or ''}'.",
            fda_sources,
            fda_date_node,
            "Verify the specific day-month-year approval date from FDA documents."
        ),
        (
            f"Official FDA documentation confirms {display_name} is indicated for uUTI in female patients aged 12 years and older.",
            fda_sources,
            fda_ind_node,
            "Use FDA product labeling or drug database pages to confirm both indication and population."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)

    # ClinicalTrials.gov public sources for NCT IDs (split per NCT)
    ctgov_parent = evaluator.add_parallel(
        id="ClinicalTrialsGov_Public_Sources_For_NCT_IDs",
        desc="Cites publicly accessible ClinicalTrials.gov pages (or equivalent public references) supporting both NCT identifiers and their phase 3 status.",
        parent=pub_node,
        critical=True
    )
    two_ncts = _choose_two_ncts(ex.nct_ids)
    for idx, nct in enumerate(two_ncts):
        ct_leaf = evaluator.add_leaf(
            id=f"CTGov_NCT_{idx}_Phase3",
            desc=f"ClinicalTrials.gov page supports NCT {nct} and its phase 3 status.",
            parent=ctgov_parent,
            critical=True
        )
        ct_url = _pick_ctgov_url_for_nct(nct, ex.ctgov_urls)
        claim = f"The ClinicalTrials.gov page for NCT{nct} shows it is a Phase 3 clinical trial relevant to {display_name}."
        await evaluator.verify(
            claim=claim,
            node=ct_leaf,
            sources=ct_url,
            additional_instruction="Use the ClinicalTrials.gov page only; check the 'Phase' field and trial details to confirm Phase 3."
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
    Evaluate an answer for the task: FDA novel oral antibiotic for uUTI approved March 2025.
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
        default_model=model,
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_antibiotic_info(),
        template_class=AntibioticApprovalExtraction,
        extraction_name="antibiotic_approval_info",
    )

    # Record some useful custom info
    evaluator.add_custom_info(
        {
            "brand_name": extracted.brand_name,
            "generic_name": extracted.generic_name,
            "manufacturer": extracted.manufacturer,
            "approval_date": extracted.approval_date,
            "nct_ids_all": extracted.nct_ids,
            "nct_ids_used": _choose_two_ncts(extracted.nct_ids),
            "counts": {
                "fda_urls": len(extracted.fda_urls),
                "ctgov_urls": len(extracted.ctgov_urls),
                "other_urls": len(extracted.other_urls),
            }
        },
        info_type="extraction_summary"
    )

    # Build rubric root node (critical as per rubric)
    task_completion = evaluator.add_parallel(
        id="Task_Completion",
        desc="Answer identifies the correct FDA-approved drug matching all stated constraints and provides all requested fields with publicly verifiable documentation.",
        parent=root,
        critical=True
    )

    # Subtrees
    await verify_identification_criteria(evaluator, task_completion, extracted)
    await verify_reported_fields(evaluator, task_completion, extracted)
    await verify_public_documentation(evaluator, task_completion, extracted)

    # Return standardized summary
    return evaluator.get_summary()