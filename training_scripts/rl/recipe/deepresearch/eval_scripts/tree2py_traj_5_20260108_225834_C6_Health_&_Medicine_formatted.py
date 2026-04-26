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
TASK_ID = "med_device_company_2025_constraints"
TASK_DESCRIPTION = (
    "Identify the name of the medical device company that meets ALL of the following criteria as of December 31, 2025:\n"
    "1) HQ: Chicago, Illinois; founded in 2020.\n"
    "2) Series A of exactly $15.1M, announced Dec 2022, co-led by Broadview Ventures and Hatteras Venture Partners.\n"
    "3) Founding team developed underlying technologies for more than a decade before founding.\n"
    "4) Primary device received FDA 510(k) clearance number K243566 with FDA decision date July 22, 2025.\n"
    "5) Device classified under cardiovascular medicine and is a multimodal wearable sensor that simultaneously captures exactly three physiological signals: ECG, PPG, and SCG.\n"
    "6) Three co-founders with roles CEO, CTO, and CSO; Amit Gupta is CEO and co-founder.\n"
    "7) FDA Breakthrough Device designation for an AI algorithm related to cardiac monitoring for pulmonary capillary wedge pressure (PCWP) estimation.\n"
    "8) Primary clinical focus on heart failure management and monitoring.\n"
    "Provide the company name and supporting reference URLs by category (company profile, funding, regulatory, leadership, technology/specifications)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AnswerExtraction(BaseModel):
    company_name: Optional[str] = None
    company_profile_urls: List[str] = Field(default_factory=list)
    funding_urls: List[str] = Field(default_factory=list)
    regulatory_urls: List[str] = Field(default_factory=list)
    leadership_urls: List[str] = Field(default_factory=list)
    technology_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_and_references() -> str:
    return """
    Extract the following fields from the answer text.

    Required JSON fields:
    - company_name: The single identified medical device company name (string).
    - company_profile_urls: List of URL(s) that support the company's HQ (Chicago, Illinois), founding year (2020), and the pre-founding technology development duration (> 10 years).
    - funding_urls: List of URL(s) that support the Series A details: EXACT $15.1 million, announced in December 2022, co-led by Broadview Ventures and Hatteras Venture Partners.
    - regulatory_urls: List of URL(s) that support regulatory facts: FDA 510(k) clearance (number K243566, decision date July 22, 2025), cardiovascular classification, and Breakthrough Device designation for PCWP estimation.
    - leadership_urls: List of URL(s) that support the leadership structure (three co-founders in roles CEO/CTO/CSO) and the identity of the CEO/co-founder (Amit Gupta).
    - technology_urls: List of URL(s) that support the device being a multimodal wearable sensor capturing exactly three simultaneous signals (ECG, PPG, SCG) and the heart failure management/monitoring focus.

    Rules:
    - Extract only URLs explicitly present in the answer; do not invent or infer any URLs.
    - Accept plain URLs or markdown links; return the actual URL strings.
    - Return empty lists if a category is missing, but never return null for list fields.
    - The company_name should be a single string if provided, else null.

    Return a single JSON object exactly matching the required fields.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _company_name_or_generic(extracted: AnswerExtraction) -> str:
    return extracted.company_name.strip() if _non_empty_str(extracted.company_name) else "the company"


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_response_requirements(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
) -> Dict[str, Any]:
    node = evaluator.add_parallel(
        id="response_requirements",
        desc="Response includes the required outputs (company name and category references).",
        parent=parent,
        critical=True,
    )

    # Company name provided (existence check)
    evaluator.add_custom_node(
        result=_non_empty_str(extracted.company_name),
        id="company_name_provided",
        desc="Provide the company name (single identified medical device company).",
        parent=node,
        critical=True,
    )

    # References-by-category provided (split into concrete leaves)
    refs_parent = evaluator.add_parallel(
        id="references_by_category_provided",
        desc="Provide supporting reference URL(s) for each category.",
        parent=node,
        critical=True,
    )

    ref_presence_nodes = {
        "company_profile": evaluator.add_custom_node(
            result=_has_urls(extracted.company_profile_urls),
            id="profile_refs_provided",
            desc="Supporting URL(s) for company profile are provided.",
            parent=refs_parent,
            critical=True,
        ),
        "funding": evaluator.add_custom_node(
            result=_has_urls(extracted.funding_urls),
            id="funding_refs_provided",
            desc="Supporting URL(s) for funding details are provided.",
            parent=refs_parent,
            critical=True,
        ),
        "regulatory": evaluator.add_custom_node(
            result=_has_urls(extracted.regulatory_urls),
            id="regulatory_refs_provided",
            desc="Supporting URL(s) for regulatory details are provided.",
            parent=refs_parent,
            critical=True,
        ),
        "leadership": evaluator.add_custom_node(
            result=_has_urls(extracted.leadership_urls),
            id="leadership_refs_provided",
            desc="Supporting URL(s) for leadership details are provided.",
            parent=refs_parent,
            critical=True,
        ),
        "technology": evaluator.add_custom_node(
            result=_has_urls(extracted.technology_urls),
            id="technology_refs_provided",
            desc="Supporting URL(s) for technology/specifications are provided.",
            parent=refs_parent,
            critical=True,
        ),
    }

    return {"ref_presence_nodes": ref_presence_nodes}


async def build_company_profile(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
):
    node = evaluator.add_parallel(
        id="company_profile",
        desc="Verify the company's founding details, HQ location, and pre-founding technology development duration.",
        parent=parent,
        critical=True,
    )

    # Presence for this category (to gate subsequent verifications under this node)
    profile_ref_node = evaluator.add_custom_node(
        result=_has_urls(extracted.company_profile_urls),
        id="company_profile_reference",
        desc="Provide URL reference(s) supporting the company profile requirements (HQ, founding year, and > decade technology development).",
        parent=node,
        critical=True,
    )

    company = _company_name_or_generic(extracted)

    # HQ location
    hq_leaf = evaluator.add_leaf(
        id="headquarters_location",
        desc="The company is headquartered in Chicago, Illinois.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company {company} is headquartered in Chicago, Illinois (Chicago, IL).",
        node=hq_leaf,
        sources=extracted.company_profile_urls,
        extra_prerequisites=[profile_ref_node],
        additional_instruction="Look for explicit statements of headquarters location (e.g., 'Headquartered in Chicago, IL'). Accept minor variants like 'Chicago, Illinois' or 'Chicago, IL'.",
    )

    # Founding year
    founding_leaf = evaluator.add_leaf(
        id="founding_year",
        desc="The company was founded in 2020.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The company {company} was founded (established/incorporated) in 2020.",
        node=founding_leaf,
        sources=extracted.company_profile_urls,
        extra_prerequisites=[profile_ref_node],
        additional_instruction="Accept synonymous phrases: 'founded in 2020', 'established in 2020', or 'incorporated in 2020'.",
    )

    # Technology development history
    tech_history_leaf = evaluator.add_leaf(
        id="technology_development_history",
        desc="The founding team developed the underlying technologies for more than a decade before the company's formal establishment.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Before founding, the team behind {company} developed the underlying technologies for more than a decade (10+ years).",
        node=tech_history_leaf,
        sources=extracted.company_profile_urls,
        extra_prerequisites=[profile_ref_node],
        additional_instruction="Look for phrases like 'over a decade', 'more than ten years', 'decade-long research', or similar statements indicating >10 years prior to the 2020 founding.",
    )


async def build_funding_details(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
):
    node = evaluator.add_parallel(
        id="funding_details",
        desc="Verify the Series A funding round details.",
        parent=parent,
        critical=True,
    )

    # Presence for this category
    funding_ref_node = evaluator.add_custom_node(
        result=_has_urls(extracted.funding_urls),
        id="funding_reference",
        desc="Provide URL reference(s) supporting the funding requirements (amount, timing, and co-leads).",
        parent=node,
        critical=True,
    )

    company = _company_name_or_generic(extracted)

    # Series A amount exactly $15.1M
    amount_leaf = evaluator.add_leaf(
        id="series_a_amount",
        desc="The company completed a Series A funding round of exactly $15.1 million.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company} completed a Series A funding round of exactly $15.1 million (USD 15.1M).",
        node=amount_leaf,
        sources=extracted.funding_urls,
        extra_prerequisites=[funding_ref_node],
        additional_instruction="Confirm that the Series A amount is exactly $15.1 million (accept variants like '$15.1M' or 'USD 15.1 million'). Reject amounts like 15M or 15.0M or 15.2M.",
    )

    # Announcement timing: December 2022
    timing_leaf = evaluator.add_leaf(
        id="funding_announcement_timing",
        desc="The Series A round was announced in December 2022.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Series A round for {company} was announced in December 2022.",
        node=timing_leaf,
        sources=extracted.funding_urls,
        extra_prerequisites=[funding_ref_node],
        additional_instruction="Look for press releases or news with a date in December 2022. Accept statements like 'announced December 2022' or an article/PR published in Dec 2022 describing the Series A.",
    )

    # Co-lead investors
    co_leads_leaf = evaluator.add_leaf(
        id="lead_investors",
        desc="The Series A round was co-led by both Broadview Ventures and Hatteras Venture Partners.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The Series A round for {company} was co-led by Broadview Ventures and Hatteras Venture Partners.",
        node=co_leads_leaf,
        sources=extracted.funding_urls,
        extra_prerequisites=[funding_ref_node],
        additional_instruction="Verify that both Broadview Ventures and Hatteras Venture Partners are cited as co-lead investors.",
    )


async def build_regulatory_clearances(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
):
    node = evaluator.add_parallel(
        id="regulatory_clearances",
        desc="Verify FDA 510(k) clearance and Breakthrough Device designation requirements.",
        parent=parent,
        critical=True,
    )

    # Presence for this category
    regulatory_ref_node = evaluator.add_custom_node(
        result=_has_urls(extracted.regulatory_urls),
        id="regulatory_reference",
        desc="Provide URL reference(s) supporting the regulatory requirements (510(k) details, cardiovascular classification, and Breakthrough designation/PCWP).",
        parent=node,
        critical=True,
    )

    company = _company_name_or_generic(extracted)

    # 510(k) clearance sub-node
    k510_node = evaluator.add_parallel(
        id="fda_510k_clearance",
        desc="Primary device received FDA 510(k) clearance with the specified clearance number and decision date.",
        parent=node,
        critical=True,
    )

    clearance_type_leaf = evaluator.add_leaf(
        id="clearance_type",
        desc="The device received FDA 510(k) clearance (premarket notification pathway).",
        parent=k510_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company}'s primary device received an FDA 510(k) clearance (premarket notification).",
        node=clearance_type_leaf,
        sources=extracted.regulatory_urls,
        extra_prerequisites=[regulatory_ref_node],
        additional_instruction="Verify the clearance pathway is 510(k) (premarket notification). Accept pages stating 'FDA 510(k) clearance' or equivalent.",
    )

    clearance_number_leaf = evaluator.add_leaf(
        id="clearance_number",
        desc="The 510(k) clearance number is K243566.",
        parent=k510_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The FDA 510(k) clearance number is K243566.",
        node=clearance_number_leaf,
        sources=extracted.regulatory_urls,
        extra_prerequisites=[regulatory_ref_node],
        additional_instruction="Confirm the exact 510(k) number 'K243566' appears on the regulatory source.",
    )

    clearance_date_leaf = evaluator.add_leaf(
        id="clearance_date",
        desc="The FDA decision date was July 22, 2025.",
        parent=k510_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The FDA decision date for the 510(k) clearance was July 22, 2025.",
        node=clearance_date_leaf,
        sources=extracted.regulatory_urls,
        extra_prerequisites=[regulatory_ref_node],
        additional_instruction="Verify the FDA 'Decision Date' exactly matches July 22, 2025.",
    )

    # Medical specialty classification
    specialty_leaf = evaluator.add_leaf(
        id="medical_specialty",
        desc="The device is classified under cardiovascular medicine.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The device for {company} is classified under cardiovascular medicine (cardiology).",
        node=specialty_leaf,
        sources=extracted.regulatory_urls,
        extra_prerequisites=[regulatory_ref_node],
        additional_instruction="Accept 'cardiology' or 'cardiovascular' classification terminology indicating cardiovascular medicine.",
    )

    # Breakthrough Device designation for PCWP estimation
    bdp_leaf = evaluator.add_leaf(
        id="breakthrough_designation_pcwp",
        desc="The company received FDA Breakthrough Device designation for an AI algorithm related to cardiac monitoring specifically for pulmonary capillary wedge pressure (PCWP) estimation.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company} received FDA Breakthrough Device designation for an AI algorithm related to cardiac monitoring, specifically for pulmonary capillary wedge pressure (PCWP) estimation.",
        node=bdp_leaf,
        sources=extracted.regulatory_urls,
        extra_prerequisites=[regulatory_ref_node],
        additional_instruction="Look for explicit mention of 'Breakthrough Device designation' and 'pulmonary capillary wedge pressure (PCWP)' estimation.",
    )


async def build_leadership_team(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
):
    node = evaluator.add_parallel(
        id="leadership_team",
        desc="Verify the co-founder leadership structure and CEO identity constraint.",
        parent=parent,
        critical=True,
    )

    # Presence for this category
    leadership_ref_node = evaluator.add_custom_node(
        result=_has_urls(extracted.leadership_urls),
        id="leadership_reference",
        desc="Provide URL reference(s) supporting the leadership requirements (three co-founders with CEO/CTO/CSO roles and Amit Gupta as CEO/co-founder).",
        parent=node,
        critical=True,
    )

    company = _company_name_or_generic(extracted)

    # Co-founder structure
    structure_leaf = evaluator.add_leaf(
        id="co_founder_structure",
        desc="The company has three co-founders holding the positions of CEO, CTO, and Chief Scientific Officer (CSO).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company} has three co-founders with the roles: CEO, CTO, and Chief Scientific Officer (CSO).",
        node=structure_leaf,
        sources=extracted.leadership_urls,
        extra_prerequisites=[leadership_ref_node],
        additional_instruction="Confirm there are exactly three co-founders and that their roles correspond to CEO, CTO, and CSO. Accept minor variants like 'Chief Science Officer' for CSO.",
    )

    # CEO identity
    ceo_leaf = evaluator.add_leaf(
        id="ceo_identity",
        desc="Amit Gupta is the CEO and co-founder.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="Amit Gupta is the CEO and a co-founder of the company.",
        node=ceo_leaf,
        sources=extracted.leadership_urls,
        extra_prerequisites=[leadership_ref_node],
        additional_instruction="Verify that Amit Gupta is both CEO and a co-founder.",
    )


async def build_technology_specifications(
    evaluator: Evaluator,
    parent,
    extracted: AnswerExtraction,
):
    node = evaluator.add_parallel(
        id="technology_specifications",
        desc="Verify device type, signal capture specifications, and clinical focus.",
        parent=parent,
        critical=True,
    )

    # Presence for this category
    tech_ref_node = evaluator.add_custom_node(
        result=_has_urls(extracted.technology_urls),
        id="technology_reference",
        desc="Provide URL reference(s) supporting the technology and clinical focus requirements (multimodal wearable sensor, exact three signals ECG/PPG/SCG simultaneously, heart failure focus).",
        parent=node,
        critical=True,
    )

    company = _company_name_or_generic(extracted)

    # Device type
    device_leaf = evaluator.add_leaf(
        id="device_type",
        desc="The device is a multimodal wearable sensor.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company}'s primary device is a multimodal wearable sensor.",
        node=device_leaf,
        sources=extracted.technology_urls,
        extra_prerequisites=[tech_ref_node],
        additional_instruction="Look for clear identification as a wearable sensor and that it is 'multimodal' (multiple physiological modalities).",
    )

    # Signal capture (exact three: ECG, PPG, SCG; simultaneously)
    signals_leaf = evaluator.add_leaf(
        id="signal_capture_exact_three",
        desc="The device simultaneously captures exactly three physiological signals: ECG, PPG, and SCG (and no additional signal types as part of the simultaneous capture set).",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The device from {company} simultaneously captures exactly three physiological signals: ECG, PPG, and SCG.",
        node=signals_leaf,
        sources=extracted.technology_urls,
        extra_prerequisites=[tech_ref_node],
        additional_instruction=(
            "Confirm simultaneous capture of exactly three signals: ECG, PPG, and SCG. "
            "Reject if more or fewer signals are captured simultaneously. "
            "Allow synonyms for SCG like 'seismocardiography', 'seismocardiogram', or 'chest wall vibration/accelerometry' if clearly equivalent."
        ),
    )

    # Clinical focus
    clinical_leaf = evaluator.add_leaf(
        id="clinical_focus",
        desc="The company's primary clinical focus is on heart failure management and monitoring.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{company}'s primary clinical focus is heart failure management and monitoring.",
        node=clinical_leaf,
        sources=extracted.technology_urls,
        extra_prerequisites=[tech_ref_node],
        additional_instruction="Look for 'heart failure' (HF) as the primary clinical application, including monitoring, management, or remote monitoring contexts.",
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
        default_model=model,
    )

    # Extract company and references
    extracted: AnswerExtraction = await evaluator.extract(
        prompt=prompt_extract_company_and_references(),
        template_class=AnswerExtraction,
        extraction_name="company_and_references",
    )

    # Record additional info for debug/analytics
    evaluator.add_custom_info(
        info={
            "company_name": extracted.company_name,
            "counts": {
                "company_profile_urls": len(extracted.company_profile_urls),
                "funding_urls": len(extracted.funding_urls),
                "regulatory_urls": len(extracted.regulatory_urls),
                "leadership_urls": len(extracted.leadership_urls),
                "technology_urls": len(extracted.technology_urls),
            }
        },
        info_type="extraction_summary",
    )

    # Add task-level ground truth constraints text (for traceability only)
    evaluator.add_ground_truth({
        "constraints_summary": [
            "HQ Chicago, IL; founded 2020; >10-year pre-founding tech development",
            "Series A exactly $15.1M, announced Dec 2022, co-led by Broadview Ventures and Hatteras Venture Partners",
            "FDA 510(k) K243566; decision date July 22, 2025; cardiovascular classification",
            "Breakthrough Device designation for AI PCWP estimation",
            "Three co-founders in CEO/CTO/CSO roles; Amit Gupta is CEO & co-founder",
            "Device: multimodal wearable sensor capturing ECG+PPG+SCG simultaneously",
            "Primary clinical focus: heart failure management/monitoring",
        ]
    })

    # Create a critical wrapper node to enforce all constraints jointly
    task_root = evaluator.add_parallel(
        id="task_root",
        desc="Provide the company name and supporting URLs showing the company satisfies all specified constraints (profile, funding, regulatory, leadership, technology/clinical focus).",
        parent=root,
        critical=True,
    )

    # Build subtrees according to rubric
    await build_response_requirements(evaluator, task_root, extracted)
    await build_company_profile(evaluator, task_root, extracted)
    await build_funding_details(evaluator, task_root, extracted)
    await build_regulatory_clearances(evaluator, task_root, extracted)
    await build_leadership_team(evaluator, task_root, extracted)
    await build_technology_specifications(evaluator, task_root, extracted)

    # Return summary
    return evaluator.get_summary()