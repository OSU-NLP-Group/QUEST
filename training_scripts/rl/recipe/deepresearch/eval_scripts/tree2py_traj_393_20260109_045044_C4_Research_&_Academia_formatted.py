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
TASK_ID = "he_ria_submission_requirements"
TASK_DESCRIPTION = (
    "A research team is preparing to submit a collaborative research proposal to Horizon Europe's Research and Innovation Actions (RIA) program. "
    "What are the mandatory consortium composition requirements, required proposal documents, and eligibility criteria that must be met for their application? "
    "Specifically identify: (1) the minimum number of partner organizations required, (2) the geographic/country distribution requirement for partners, "
    "(3) whether a data management plan is required, (4) whether a budget document is required, (5) what types of organizations are eligible to participate, "
    "and (6) the name of the official portal where applications must be submitted."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RIARequirementsExtraction(BaseModel):
    minimum_partners: Optional[str] = None
    countries_distribution_requirement: Optional[str] = None
    requires_one_eu_member_state_partner: Optional[str] = None
    other_partners_ms_or_associated: Optional[str] = None
    data_management_plan_required: Optional[str] = None
    budget_document_required: Optional[str] = None
    eligible_legal_entities_geography: Optional[str] = None
    eligible_entity_examples: List[str] = Field(default_factory=list)
    application_portal_name: Optional[str] = None
    application_portal_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ria_requirements() -> str:
    return """
    Extract exactly what the answer states regarding Horizon Europe RIA submission requirements. Return null for any field not clearly mentioned.
    Fields to extract:
    - minimum_partners: The minimum number of partner organizations required (e.g., "at least 3", "minimum three"). Extract literally as stated.
    - countries_distribution_requirement: The requirement on countries (e.g., "from 3 different countries", "each established in different MS/AC").
    - requires_one_eu_member_state_partner: Does the answer state that at least one partner must be from an EU Member State? Return "yes", "no", or null if not stated.
    - other_partners_ms_or_associated: Does the answer state that remaining partners can be from EU Member States or Horizon Europe Associated Countries? Return "yes", "no", or null if not stated.
    - data_management_plan_required: Does the answer state that a Data Management Plan (DMP) is required? Return "yes", "no", or null if not stated.
    - budget_document_required: Does the answer state that budget documentation is required as part of the application? Return "yes", "no", or null if not stated.
    - eligible_legal_entities_geography: The geographic eligibility statement for organizations (e.g., "legal entities from EU Member States and Associated Countries").
    - eligible_entity_examples: List of example organization types explicitly mentioned (e.g., "universities", "research institutions", "SMEs", "companies", "public bodies", "NGOs"). Return an array; empty if none mentioned.
    - application_portal_name: The name of the official submission portal as stated (e.g., "Funding & Tenders Portal", "Funding & Tenders Opportunities Portal").
    - application_portal_urls: Any URLs mentioned that point to the official submission portal.

    Only extract exactly what the answer states; do not infer or add new information.
    """


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: RIARequirementsExtraction) -> None:
    root = evaluator.root

    # Add ground truth (reference expectations) for transparency
    evaluator.add_ground_truth(
        {
            "expected": {
                "minimum_partners": "At least 3 independent legal entities (partners).",
                "countries_distribution": "Partners must be established in 3 different countries.",
                "at_least_one_eu_member_state": "At least one partner must be established in an EU Member State.",
                "remaining_partners_ms_or_ac": "The remaining partners may be established in EU Member States or Horizon Europe Associated Countries.",
                "dmp_required": "A Data Management Plan (DMP) is required in Horizon Europe; it must be addressed in the proposal and delivered early in the project where data are generated.",
                "budget_required": "Budget/financial information must be provided as part of the application (e.g., Part A budget tables).",
                "eligible_geography": "Eligible applicants include legal entities established in EU Member States and Horizon Europe Associated Countries.",
                "portal": "Applications must be submitted via the EU Funding & Tenders Portal (Funding & Tenders Opportunities Portal)."
            }
        },
        gt_type="reference_requirements"
    )

    # Common verification instruction to keep judgments focused on the answer text
    focus_on_answer_only = (
        "Judge only based on the provided answer text. Do not rely on your own knowledge. "
        "Consider reasonable synonyms (e.g., 'beneficiaries' for partners, 'Member States' for EU countries, "
        "'Associated Countries' or 'associated participants'). If the answer does not clearly and explicitly state the item, "
        "return Incorrect."
    )

    # 1) Consortium composition (Critical group)
    consortium_node = evaluator.add_parallel(
        id="consortium_composition",
        desc="Correctly identify mandatory consortium composition requirements",
        parent=root,
        critical=True
    )

    # 1.a Minimum consortium size
    min_size_leaf = evaluator.add_leaf(
        id="minimum_consortium_size",
        desc="State that the consortium must include at least 3 partner organizations",
        parent=consortium_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the consortium must include at least 3 independent partner organizations (minimum three legal entities).",
        node=min_size_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 1.b Partners from three different countries
    three_countries_leaf = evaluator.add_leaf(
        id="partners_from_three_countries",
        desc="State that partner organizations must be from 3 different countries",
        parent=consortium_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the consortium partners must be established in 3 different countries.",
        node=three_countries_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 1.c At least one EU Member State partner
    one_ms_leaf = evaluator.add_leaf(
        id="at_least_one_eu_member_state_partner",
        desc="State that at least one partner organization must be from an EU Member State",
        parent=consortium_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that at least one partner must be established in an EU Member State.",
        node=one_ms_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 1.d Remaining partners from EU MS or Associated Countries
    remaining_ms_ac_leaf = evaluator.add_leaf(
        id="remaining_partners_eu_or_associated",
        desc="State that the remaining partners can be from EU Member States or Horizon Europe Associated Countries",
        parent=consortium_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the remaining partners may be established in EU Member States or in Horizon Europe Associated Countries.",
        node=remaining_ms_ac_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 2) Required proposal documents (Critical group)
    required_docs_node = evaluator.add_parallel(
        id="required_proposal_documents",
        desc="Correctly identify required proposal documents",
        parent=root,
        critical=True
    )

    # 2.a Data Management Plan (DMP) requirement
    dmp_leaf = evaluator.add_leaf(
        id="data_management_plan",
        desc="State that a data management plan (DMP) must be included",
        parent=required_docs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that a Data Management Plan (DMP) is required for Horizon Europe RIA (addressed in the proposal and/or delivered early in the project when data are generated).",
        node=dmp_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 2.b Budget documentation requirement
    budget_leaf = evaluator.add_leaf(
        id="budget_document",
        desc="State that budget documentation must be provided as part of the application",
        parent=required_docs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that budget information or documentation must be provided as part of the application.",
        node=budget_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 3) Eligibility criteria (Mixed group; made non-critical to allow optional examples)
    eligibility_node = evaluator.add_parallel(
        id="eligibility_criteria",
        desc="Correctly identify eligibility criteria for participating organizations",
        parent=root,
        critical=False  # Adjusted to allow the optional examples child to be non-critical without violating framework constraints
    )

    # 3.a Eligible legal entities geography (critical leaf under a non-critical group)
    eligible_geo_leaf = evaluator.add_leaf(
        id="eligible_legal_entities_geography",
        desc="State that eligible applicants include legal entities from EU Member States and Horizon Europe Associated Countries",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that eligible applicants include legal entities established in EU Member States and in Horizon Europe Associated Countries.",
        node=eligible_geo_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 3.b Examples of eligible entities (non-critical, optional)
    examples_leaf = evaluator.add_leaf(
        id="eligible_entity_examples",
        desc="Optionally mention examples of eligible organizations (e.g., universities and research institutions)",
        parent=eligibility_node,
        critical=False
    )
    await evaluator.verify(
        claim="The answer mentions at least one example of eligible organizations, such as universities, research institutions, companies/SMEs, public bodies, or NGOs.",
        node=examples_leaf,
        additional_instruction=focus_on_answer_only
    )

    # 4) Application portal (Critical leaf at root)
    portal_leaf = evaluator.add_leaf(
        id="application_portal",
        desc="Identify the official portal where applications must be submitted (Funding and Tenders portal)",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the official submission portal as the EU Funding & Tenders Portal (also called the Funding & Tenders Opportunities Portal).",
        node=portal_leaf,
        additional_instruction=(
            "Judge only by the answer content. Allow minor naming variants like 'Funding & Tenders Opportunities Portal', "
            "'EU Funding and Tenders', or abbreviations like 'FTOP'. If the answer clearly refers to the European Commission's "
            "Funding & Tenders Portal, mark as Correct; otherwise Incorrect."
        )
    )

    # Optionally record extraction results to the summary for transparency
    evaluator.add_custom_info(
        info={
            "extracted_summary": extraction.dict()
        },
        info_type="extraction_recorder",
        info_name="extracted_ria_requirements"
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
    model: str = "o4-mini"
) -> Dict:
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extraction (helps debugging and transparency; verification uses the raw answer via simple verify)
    extraction = await evaluator.extract(
        prompt=prompt_extract_ria_requirements(),
        template_class=RIARequirementsExtraction,
        extraction_name="ria_requirements_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return structured summary with verification tree and scores
    return evaluator.get_summary()