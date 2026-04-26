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
TASK_ID = "three_major_food_recalls_2024"
TASK_DESCRIPTION = (
    "Identify three separate major food recalls that occurred in the United States in 2024, "
    "where each recall met ALL of the following criteria: (1) The outbreak resulted in at least one "
    "documented death, (2) The outbreak affected at least 10 U.S. states, (3) The contamination "
    "involved bacterial pathogens (specifically Listeria monocytogenes, Salmonella, or E. coli O157:H7), "
    "(4) There was an official CDC outbreak investigation with a published investigation page, "
    "(5) The recall was nationwide in scope (not limited to a specific region). For each of the three recalls, "
    "provide the following information: company name and facility location (city and state), type of food product "
    "recalled and brand names, production date range for contaminated products, specific bacterial pathogen involved, "
    "total number of confirmed illness cases, number of hospitalizations, number of deaths, number of U.S. states affected, "
    "date range when illnesses occurred, date when the CDC investigation began, current status of the investigation (ongoing or closed), "
    "date of the initial recall announcement, the regulatory agency that issued the recall (FDA or USDA FSIS), direct URL to the official CDC "
    "outbreak investigation page, and direct URL to the official recall notice from FDA or USDA."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RecallItem(BaseModel):
    # Company & product information
    company_name: Optional[str] = None
    facility_city: Optional[str] = None
    facility_state: Optional[str] = None
    product_type: Optional[str] = None
    brand_names: List[str] = Field(default_factory=list)
    production_period: Optional[str] = None  # free-form date range string

    # Contamination details
    pathogen: Optional[str] = None  # e.g., "Listeria monocytogenes", "Salmonella", "E. coli O157:H7"
    strain_type: Optional[str] = None  # optional serotype/strain, if given

    # Health impact
    total_cases: Optional[str] = None
    hospitalizations: Optional[str] = None
    deaths: Optional[str] = None
    states_affected: Optional[str] = None
    illness_onset_period: Optional[str] = None  # date range string

    # CDC investigation
    cdc_investigation_start_date: Optional[str] = None
    cdc_investigation_status: Optional[str] = None  # "ongoing" or "closed"
    cdc_investigation_url: Optional[str] = None

    # Regulatory actions / recall
    initial_recall_date: Optional[str] = None
    recall_agency: Optional[str] = None  # "FDA" or "USDA FSIS"
    recall_notice_url: Optional[str] = None

    # Optional supporting links per section (if provided in the answer)
    company_product_sources: List[str] = Field(default_factory=list)
    health_impact_sources: List[str] = Field(default_factory=list)
    contamination_sources: List[str] = Field(default_factory=list)
    regulatory_sources: List[str] = Field(default_factory=list)


class RecallsExtraction(BaseModel):
    recalls: List[RecallItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_recalls() -> str:
    return """
Extract from the answer up to three (3) separate U.S. food recalls in 2024 that meet the described criteria. 
Return a JSON object with a top-level field "recalls" that is an array of at most 3 items. For each recall item, extract the following fields exactly as written in the answer:

- company_name: The company that issued the recall.
- facility_city: The city of the production facility (if provided).
- facility_state: The U.S. state of the production facility (if provided).
- product_type: The category/type of food product recalled (e.g., "bagged salads", "soft cheese", "ground beef").
- brand_names: An array of brand names under which the recalled products were sold (if provided).
- production_period: The production/manufacturing date range for contaminated products (if provided).

- pathogen: The specific bacterial pathogen (must be one of: "Listeria monocytogenes", "Salmonella", or "E. coli O157:H7").
- strain_type: The serotype or strain identifier if present (optional; set to null if not provided).

- total_cases: The total number of confirmed illness cases (as a string).
- hospitalizations: The number of hospitalizations (as a string).
- deaths: The number of deaths (as a string).
- states_affected: The number of U.S. states with illnesses (as a string).
- illness_onset_period: The date range during which illnesses occurred (as a string).

- cdc_investigation_start_date: The date when CDC investigation began (as a string).
- cdc_investigation_status: "ongoing" or "closed" (as a string).
- cdc_investigation_url: The direct URL to the official CDC outbreak investigation page (if provided in the answer).

- initial_recall_date: The date of the initial recall announcement (as a string).
- recall_agency: The recall agency ("FDA" or "USDA FSIS") (as a string).
- recall_notice_url: The direct URL to the official recall notice from FDA or USDA FSIS (if provided in the answer).

- company_product_sources: Array of additional URLs (if any) that the answer cites for company/product info. If none, return an empty array.
- health_impact_sources: Array of additional URLs (if any) for illness counts, hospitalizations, deaths, states, and illness period. If none, return an empty array.
- contamination_sources: Array of additional URLs (if any) for pathogen/contamination confirmation/testing. If none, return an empty array.
- regulatory_sources: Array of additional URLs (if any) for regulatory actions and timelines. If none, return an empty array.

General rules:
- Only extract information explicitly present in the answer.
- If a field is not mentioned, return null for scalars or an empty array for lists.
- Keep dates and counts as strings to avoid parsing ambiguities.
- For URLs, extract only direct/valid URLs present in the answer text; do not fabricate.
- If the answer lists more than 3 recalls, include only the first 3 as they appear.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _present(val: Optional[str]) -> bool:
    return isinstance(val, str) and val.strip() != ""


def _combine_sources(*lists: Optional[List[str]], extra: Optional[List[str]] = None) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        if lst:
            for u in lst:
                if isinstance(u, str) and u.strip():
                    combined.append(u.strip())
    if extra:
        for u in extra:
            if isinstance(u, str) and u.strip():
                combined.append(u.strip())
    # deduplicate while preserving order
    seen = set()
    result = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def _presence_or_verify(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    value_present: bool,
    claim: str,
    sources: List[str],
    additional_instruction: str,
    critical: bool = True,
):
    """
    Convenience helper:
    - If the value is present, create a leaf and verify against sources.
    - If missing, create a custom node that fails (to reflect missing required info).
    """
    if value_present:
        node = evaluator.add_leaf(
            id=leaf_id,
            desc=desc,
            parent=parent,
            critical=critical
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=leaf_id,
            desc=desc,
            parent=parent,
            critical=critical
        )


# --------------------------------------------------------------------------- #
# Verification per recall                                                     #
# --------------------------------------------------------------------------- #
async def verify_single_recall(
    evaluator: Evaluator,
    parent_node,
    recall: RecallItem,
    idx: int
) -> None:
    """
    Build the verification subtree for a single recall and run verifications.
    """
    # Top-level node for this recall (non-critical to allow partial credit across recalls)
    recall_node = evaluator.add_parallel(
        id=f"recall_{idx}",
        desc=f"Documentation of the {'first' if idx == 0 else 'second' if idx == 1 else 'third'} qualifying food recall",
        parent=parent_node,
        critical=False
    )

    # Build commonly used source lists
    cdc_only = _combine_sources([recall.cdc_investigation_url])
    recall_only = _combine_sources([recall.recall_notice_url])
    both_primary = _combine_sources([recall.cdc_investigation_url, recall.recall_notice_url])

    # ------------------------------------------------------------------- #
    # Company and Product                                                 #
    # ------------------------------------------------------------------- #
    comp_prod_node = evaluator.add_parallel(
        id=f"recall_{idx}_company_product",
        desc="Identification of the manufacturer and recalled products",
        parent=recall_node,
        critical=False  # parent non-critical; leaves can be critical
    )

    # Gate: Sources for company/product info (critical)
    comp_prod_sources_ok = bool(
        (_present(recall.recall_notice_url)) or (_present(recall.cdc_investigation_url)) or (len(recall.company_product_sources) > 0)
    )
    evaluator.add_custom_node(
        result=comp_prod_sources_ok,
        id=f"recall_{idx}_company_product_sources",
        desc="Reference URLs for company and product information",
        parent=comp_prod_node,
        critical=True
    )

    # Company Information subgroup
    company_info_node = evaluator.add_parallel(
        id=f"recall_{idx}_company_info",
        desc="Complete manufacturer identification",
        parent=comp_prod_node,
        critical=False
    )

    # Company name
    await _presence_or_verify(
        evaluator,
        company_info_node,
        leaf_id=f"recall_{idx}_company_name",
        desc="The name of the company that issued the recall",
        value_present=_present(recall.company_name),
        claim=f"The company that issued the recall is '{recall.company_name}'.",
        sources=_combine_sources([recall.recall_notice_url], extra=recall.company_product_sources),
        additional_instruction="Verify the company's name on the official recall notice (FDA or USDA FSIS). Allow minor naming/corporate suffix variations.",
        critical=True
    )

    # Facility location (city and state)
    facility_loc_value_present = _present(recall.facility_city) and _present(recall.facility_state)
    await _presence_or_verify(
        evaluator,
        company_info_node,
        leaf_id=f"recall_{idx}_facility_location",
        desc="The location (city and state) of the facility where contaminated products were produced",
        value_present=facility_loc_value_present,
        claim=f"The recalled products were produced at a facility in {recall.facility_city}, {recall.facility_state}.",
        sources=_combine_sources([recall.recall_notice_url], extra=recall.company_product_sources),
        additional_instruction="Confirm the production facility city and state on the official recall notice. If multiple facilities are listed, ensure the claimed location appears among them.",
        critical=True
    )

    # Product Details subgroup
    product_details_node = evaluator.add_parallel(
        id=f"recall_{idx}_product_details",
        desc="Specific products affected by the recall",
        parent=comp_prod_node,
        critical=False
    )

    # Product type
    await _presence_or_verify(
        evaluator,
        product_details_node,
        leaf_id=f"recall_{idx}_product_type",
        desc="The category or type of food product recalled",
        value_present=_present(recall.product_type),
        claim=f"The recalled product type is '{recall.product_type}'.",
        sources=_combine_sources([recall.recall_notice_url, recall.cdc_investigation_url], extra=recall.company_product_sources),
        additional_instruction="Match the product category/type as described on the recall notice and/or CDC outbreak page. Allow reasonable synonyms (e.g., 'bagged salad' vs 'packaged salad').",
        critical=True
    )

    # Brand names
    brand_names_present = isinstance(recall.brand_names, list) and len(recall.brand_names) > 0
    if brand_names_present:
        brands_str = ", ".join([b.strip() for b in recall.brand_names if isinstance(b, str) and b.strip()])
    else:
        brands_str = ""
    await _presence_or_verify(
        evaluator,
        product_details_node,
        leaf_id=f"recall_{idx}_brand_names",
        desc="All brand names under which the recalled products were sold",
        value_present=brand_names_present,
        claim=f"The brand name(s) for the recalled product(s) include: {brands_str}.",
        sources=_combine_sources([recall.recall_notice_url], extra=recall.company_product_sources),
        additional_instruction="Check that each brand listed in the claim appears on the official recall notice. If the notice lists more brands, it's okay; but all listed in the claim must be present.",
        critical=True
    )

    # Production period
    await _presence_or_verify(
        evaluator,
        product_details_node,
        leaf_id=f"recall_{idx}_production_period",
        desc="The date range during which contaminated products were manufactured",
        value_present=_present(recall.production_period),
        claim=f"The contaminated products were produced between {recall.production_period}.",
        sources=_combine_sources([recall.recall_notice_url], extra=recall.company_product_sources),
        additional_instruction="Verify production/manufacturing date range on the official recall notice. Look for phrases like 'produced between', 'pack dates', or 'lot production dates'.",
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Health Impact                                                       #
    # ------------------------------------------------------------------- #
    health_node = evaluator.add_parallel(
        id=f"recall_{idx}_health_impact",
        desc="Documented health consequences of the outbreak",
        parent=recall_node,
        critical=False
    )

    # Gate: Sources for health impact (critical; usually CDC page)
    health_sources_ok = bool((_present(recall.cdc_investigation_url)) or (len(recall.health_impact_sources) > 0))
    evaluator.add_custom_node(
        result=health_sources_ok,
        id=f"recall_{idx}_health_sources",
        desc="Reference URLs for health impact, geographic, and timeline information",
        parent=health_node,
        critical=True
    )

    # Case statistics subgroup
    case_stats_node = evaluator.add_parallel(
        id=f"recall_{idx}_case_stats",
        desc="Quantitative metrics of outbreak severity",
        parent=health_node,
        critical=False
    )

    # Total cases
    await _presence_or_verify(
        evaluator,
        case_stats_node,
        leaf_id=f"recall_{idx}_total_cases",
        desc="The total number of confirmed illness cases",
        value_present=_present(recall.total_cases),
        claim=f"Total confirmed illnesses reported by CDC: {recall.total_cases}.",
        sources=_combine_sources([recall.cdc_investigation_url], extra=recall.health_impact_sources),
        additional_instruction="Verify the total number of cases on the CDC outbreak investigation page. Allow minor rounding/format variations.",
        critical=True
    )

    # Hospitalizations
    await _presence_or_verify(
        evaluator,
        case_stats_node,
        leaf_id=f"recall_{idx}_hospitalizations",
        desc="The number of people hospitalized",
        value_present=_present(recall.hospitalizations),
        claim=f"Total hospitalizations reported by CDC: {recall.hospitalizations}.",
        sources=_combine_sources([recall.cdc_investigation_url], extra=recall.health_impact_sources),
        additional_instruction="Verify the hospitalizations count on the CDC outbreak page.",
        critical=True
    )

    # Deaths (must be >=1)
    await _presence_or_verify(
        evaluator,
        case_stats_node,
        leaf_id=f"recall_{idx}_deaths",
        desc="The number of deaths attributed to the outbreak (must be at least 1)",
        value_present=_present(recall.deaths),
        claim=f"Total deaths reported by CDC: {recall.deaths}. This number is at least 1.",
        sources=_combine_sources([recall.cdc_investigation_url], extra=recall.health_impact_sources),
        additional_instruction="Confirm the deaths count on the CDC outbreak page and ensure it is at least 1.",
        critical=True
    )

    # Geographic scope subgroup
    geo_node = evaluator.add_parallel(
        id=f"recall_{idx}_geo_scope",
        desc="Geographic extent of the outbreak",
        parent=health_node,
        critical=False
    )

    await _presence_or_verify(
        evaluator,
        geo_node,
        leaf_id=f"recall_{idx}_states_affected",
        desc="The number of U.S. states where illnesses occurred (must be at least 10)",
        value_present=_present(recall.states_affected),
        claim=f"Illnesses were reported in {recall.states_affected} U.S. states, which is at least 10.",
        sources=_combine_sources([recall.cdc_investigation_url], extra=recall.health_impact_sources),
        additional_instruction="Verify the number of affected states on the CDC outbreak page and confirm that it is >= 10.",
        critical=True
    )

    # Outbreak timeline subgroup
    timeline_node = evaluator.add_parallel(
        id=f"recall_{idx}_timeline",
        desc="Temporal boundaries of the outbreak",
        parent=health_node,
        critical=False
    )

    await _presence_or_verify(
        evaluator,
        timeline_node,
        leaf_id=f"recall_{idx}_illness_period",
        desc="The date range during which illnesses occurred",
        value_present=_present(recall.illness_onset_period),
        claim=f"Illnesses occurred between {recall.illness_onset_period}.",
        sources=_combine_sources([recall.cdc_investigation_url], extra=recall.health_impact_sources),
        additional_instruction="Verify the illness onset period on the CDC outbreak investigation page.",
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Contamination                                                       #
    # ------------------------------------------------------------------- #
    contam_node = evaluator.add_parallel(
        id=f"recall_{idx}_contamination",
        desc="Details about the pathogen and contamination source",
        parent=recall_node,
        critical=False
    )

    # Gate: Sources for contamination/testing
    contam_sources_ok = bool((_present(recall.cdc_investigation_url)) or (_present(recall.recall_notice_url)) or (len(recall.contamination_sources) > 0))
    evaluator.add_custom_node(
        result=contam_sources_ok,
        id=f"recall_{idx}_contamination_sources",
        desc="Reference URLs for pathogen and testing information",
        parent=contam_node,
        critical=True
    )

    # Pathogen identification subgroup (parent non-critical to allow optional strain_type)
    pathogen_node = evaluator.add_parallel(
        id=f"recall_{idx}_pathogen_identification",
        desc="Specific bacterial pathogen involved",
        parent=contam_node,
        critical=False
    )

    # Bacterial species (critical)
    await _presence_or_verify(
        evaluator,
        pathogen_node,
        leaf_id=f"recall_{idx}_bacterial_species",
        desc="The species of bacteria that caused the outbreak (must be Listeria monocytogenes, Salmonella, or E. coli O157:H7)",
        value_present=_present(recall.pathogen),
        claim=f"The bacterial pathogen for this outbreak is '{recall.pathogen}', which is one of the allowed pathogens.",
        sources=_combine_sources([recall.cdc_investigation_url], [recall.recall_notice_url], extra=recall.contamination_sources),
        additional_instruction="Verify the named pathogen on the CDC (preferred) or recall notice, and ensure it is one of: Listeria monocytogenes, Salmonella, or E. coli O157:H7.",
        critical=True
    )

    # Strain type (optional, non-critical)
    if _present(recall.strain_type):
        strain_leaf = evaluator.add_leaf(
            id=f"recall_{idx}_strain_type",
            desc="Specific strain or serotype if identified through genome sequencing",
            parent=pathogen_node,
            critical=False
        )
        await evaluator.verify(
            claim=f"The strain or serotype is '{recall.strain_type}'.",
            node=strain_leaf,
            sources=_combine_sources([recall.cdc_investigation_url], extra=recall.contamination_sources),
            additional_instruction="Verify the specific strain/serotype on the CDC outbreak page when available."
        )
    else:
        # Missing optional info is not penalized as critical; mark as failed custom node with non-critical flag
        evaluator.add_custom_node(
            result=False,
            id=f"recall_{idx}_strain_type_missing",
            desc="Specific strain or serotype if identified through genome sequencing",
            parent=pathogen_node,
            critical=False
        )

    # Contamination confirmation subgroup
    contam_conf_node = evaluator.add_parallel(
        id=f"recall_{idx}_contam_confirmation",
        desc="Laboratory verification of contamination",
        parent=contam_node,
        critical=False
    )

    await _presence_or_verify(
        evaluator,
        contam_conf_node,
        leaf_id=f"recall_{idx}_product_testing",
        desc="Confirmation that the pathogen was detected in recalled product samples",
        value_present=True,  # We verify directly on sources; presence is not a separate extracted value
        claim=f"The pathogen {recall.pathogen if _present(recall.pathogen) else 'the pathogen'} was detected in recalled product samples.",
        sources=_combine_sources([recall.recall_notice_url, recall.cdc_investigation_url], extra=recall.contamination_sources),
        additional_instruction="Look for explicit statements like 'pathogen found in product', 'positive sample', or 'product testing confirmed'. If the evidence does not explicitly confirm product detection, this should fail.",
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Regulatory Actions                                                  #
    # ------------------------------------------------------------------- #
    reg_node = evaluator.add_parallel(
        id=f"recall_{idx}_reg_actions",
        desc="Official government responses and investigations",
        parent=recall_node,
        critical=False
    )

    # Gate: Regulatory sources (recall notice URL expected)
    reg_sources_ok = bool((_present(recall.recall_notice_url)) or (len(recall.regulatory_sources) > 0))
    evaluator.add_custom_node(
        result=reg_sources_ok,
        id=f"recall_{idx}_regulatory_sources",
        desc="Reference URLs for regulatory action information",
        parent=reg_node,
        critical=True
    )

    # Recall timeframe subgroup
    timeframe_node = evaluator.add_parallel(
        id=f"recall_{idx}_recall_timeframe",
        desc="Verification that the recall occurred in 2024",
        parent=reg_node,
        critical=False
    )

    # Year 2024 verification (May–Dec 2024 per task guidance)
    y2024_leaf = evaluator.add_leaf(
        id=f"recall_{idx}_year_2024_verification",
        desc="Confirmation that the recall was issued in 2024 (between May and December 2024)",
        parent=timeframe_node,
        critical=True
    )
    await evaluator.verify(
        claim="The initial recall announcement occurred in 2024, between May and December 2024.",
        node=y2024_leaf,
        sources=_combine_sources([recall.recall_notice_url], extra=recall.regulatory_sources),
        additional_instruction="Verify the recall announcement date on the official recall notice; confirm the date is in 2024 and between May 1 and December 31."
    )

    # CDC investigation subgroup
    cdc_invest_node = evaluator.add_parallel(
        id=f"recall_{idx}_cdc_investigation",
        desc="CDC's outbreak investigation details",
        parent=reg_node,
        critical=False
    )

    await _presence_or_verify(
        evaluator,
        cdc_invest_node,
        leaf_id=f"recall_{idx}_cdc_investigation_start",
        desc="The date when the CDC investigation began",
        value_present=_present(recall.cdc_investigation_start_date),
        claim=f"The CDC investigation began on {recall.cdc_investigation_start_date}.",
        sources=cdc_only,
        additional_instruction="Verify the 'Investigation notice posted' or equivalent start date on the CDC outbreak page.",
        critical=True
    )

    await _presence_or_verify(
        evaluator,
        cdc_invest_node,
        leaf_id=f"recall_{idx}_cdc_investigation_status",
        desc="Current status of the investigation (ongoing or closed)",
        value_present=_present(recall.cdc_investigation_status),
        claim=f"The CDC investigation status is '{recall.cdc_investigation_status}'.",
        sources=cdc_only,
        additional_instruction="Verify whether the CDC marks the investigation as 'Ongoing' or 'Closed' on the investigation page.",
        critical=True
    )

    # CDC investigation page URL validity/support
    if _present(recall.cdc_investigation_url):
        cdc_page_leaf = evaluator.add_leaf(
            id=f"recall_{idx}_cdc_investigation_page",
            desc="Direct URL to the official CDC outbreak investigation page",
            parent=cdc_invest_node,
            critical=True
        )
        pathogen_for_claim = recall.pathogen if _present(recall.pathogen) else "a bacterial pathogen"
        await evaluator.verify(
            claim=f"This URL is an official CDC outbreak investigation page for a 2024 U.S. foodborne outbreak involving {pathogen_for_claim}.",
            node=cdc_page_leaf,
            sources=recall.cdc_investigation_url,
            additional_instruction="Verify that the URL is on cdc.gov and clearly represents an 'Outbreak' investigation page with outbreak details."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"recall_{idx}_cdc_investigation_page_missing",
            desc="Direct URL to the official CDC outbreak investigation page",
            parent=cdc_invest_node,
            critical=True
        )

    # Recall notice subgroup
    recall_notice_node = evaluator.add_parallel(
        id=f"recall_{idx}_recall_notice",
        desc="Official FDA or USDA recall announcement",
        parent=reg_node,
        critical=False
    )

    await _presence_or_verify(
        evaluator,
        recall_notice_node,
        leaf_id=f"recall_{idx}_initial_recall_date",
        desc="The date of the initial recall announcement",
        value_present=_present(recall.initial_recall_date),
        claim=f"The initial recall announcement was made on {recall.initial_recall_date}.",
        sources=recall_only if recall_only else _combine_sources(extra=recall.regulatory_sources),
        additional_instruction="Verify the initial announcement date on the official recall notice page.",
        critical=True
    )

    await _presence_or_verify(
        evaluator,
        recall_notice_node,
        leaf_id=f"recall_{idx}_recall_agency",
        desc="The agency that issued the recall notice (FDA or USDA FSIS)",
        value_present=_present(recall.recall_agency),
        claim=f"The recall notice was issued by {recall.recall_agency}.",
        sources=recall_only if recall_only else _combine_sources(extra=recall.regulatory_sources),
        additional_instruction="Confirm whether the recall page is issued by the FDA (fda.gov) or USDA FSIS (fsis.usda.gov). Allow common naming variants.",
        critical=True
    )

    # Nationwide classification verification (does not require extracted value; verify directly)
    nationwide_leaf = evaluator.add_leaf(
        id=f"recall_{idx}_recall_classification",
        desc="Confirmation that the recall was nationwide in scope",
        parent=recall_notice_node,
        critical=True
    )
    await evaluator.verify(
        claim="The recall was nationwide in scope across the United States.",
        node=nationwide_leaf,
        sources=recall_only if recall_only else _combine_sources(extra=recall.regulatory_sources),
        additional_instruction="Check distribution/scope on the recall notice. If distribution lists 'Nationwide' or equivalent, pass; if only certain states/regions are listed, fail."
    )

    # Recall notice URL validity/support
    if _present(recall.recall_notice_url):
        notice_leaf = evaluator.add_leaf(
            id=f"recall_{idx}_recall_notice_url",
            desc="Direct URL to the official recall notice from FDA or USDA",
            parent=recall_notice_node,
            critical=True
        )
        agency_for_claim = recall.recall_agency if _present(recall.recall_agency) else "FDA or USDA FSIS"
        await evaluator.verify(
            claim=f"This URL is an official recall announcement published by {agency_for_claim} on an official domain (fda.gov or fsis.usda.gov).",
            node=notice_leaf,
            sources=recall.recall_notice_url,
            additional_instruction="Verify that the page is an official recall/food safety notice on fda.gov or fsis.usda.gov."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"recall_{idx}_recall_notice_url_missing",
            desc="Direct URL to the official recall notice from FDA or USDA",
            parent=recall_notice_node,
            critical=True
        )


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
    Entry point for evaluating an answer for the 2024 major food recalls task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Recalls evaluated independently
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

    # Create a top-level task node (non-critical to allow partial credit across recalls)
    task_node = evaluator.add_parallel(
        id="three_major_food_recalls_2024",
        desc="Identification and documentation of three separate major food recalls from 2024, each resulting in deaths and multi-state impact",
        parent=root,
        critical=False
    )

    # Extract recalls data
    extracted = await evaluator.extract(
        prompt=prompt_extract_recalls(),
        template_class=RecallsExtraction,
        extraction_name="recalls_extraction"
    )

    # Normalize to exactly 3 items (pad with empty placeholders if fewer)
    recalls: List[RecallItem] = list(extracted.recalls) if extracted and extracted.recalls is not None else []
    while len(recalls) < 3:
        recalls.append(RecallItem())

    # Only evaluate the first three
    recalls = recalls[:3]

    # Verify each recall
    for i, rec in enumerate(recalls):
        await verify_single_recall(evaluator, task_node, rec, i)

    # Return structured summary
    return evaluator.get_summary()