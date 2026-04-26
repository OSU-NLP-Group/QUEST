import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "icf_career_coaching_ca"
TASK_DESCRIPTION = """You are planning to start a career coaching business in California and want to pursue International Coaching Federation (ICF) credentials. To develop your business plan, you need comprehensive information about the ICF credential pathway. Please provide the following information:

1. ICF Credential Requirements: For each of the three ICF coaching credential levels (ACC, PCC, and MCC), identify: the minimum coach-specific education hours required, the minimum coaching experience hours required, the mentor coaching hours required, and for MCC, also identify any prerequisite credential required. Provide the official ICF source URL where these requirements are documented.

2. Training Program Costs: What is the cost range (minimum to maximum) for ICF-accredited programs that qualify for the ACC credential pathway? What is the typical cost range (minimum to maximum) for PCC-level training programs? Provide source URLs documenting these cost ranges.

3. Earning Potential: For each credential level (ACC, PCC, and MCC), identify the hourly rate range that coaches typically charge in North America (minimum and maximum hourly rates). Provide source URLs documenting these rate ranges.

4. California Business Setup Requirements: Does California require a professional license to practice as a career coach? What type of business registration is required for coaching practices in California? Is professional liability insurance legally required for coaches in California? What is the recommended professional liability insurance coverage amount for coaches? What is the typical annual cost range for professional liability insurance for coaches or consultants?

5. Credential Renewal Requirements: How frequently must ICF credentials be renewed? How many total Continuing Coach Education (CCE) credits are required for renewal? How many CCE credits must be in Core Competencies? How many CCE credits must be in Coaching Ethics? What is the maximum number of CCE credits allowed in Resource Development? What is the ACC-specific renewal requirement regarding mentor coaching? Provide the official ICF source URL where renewal requirements are documented.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CredentialLevel(BaseModel):
    education_hours: Optional[str] = None
    experience_hours: Optional[str] = None
    mentor_hours: Optional[str] = None
    prerequisite: Optional[str] = None  # Only required/used for MCC
    sources: List[str] = Field(default_factory=list)


class CredentialsExtraction(BaseModel):
    acc: Optional[CredentialLevel] = None
    pcc: Optional[CredentialLevel] = None
    mcc: Optional[CredentialLevel] = None


class TrainingCostExtraction(BaseModel):
    acc_min_cost: Optional[str] = None
    acc_max_cost: Optional[str] = None
    acc_cost_sources: List[str] = Field(default_factory=list)
    pcc_min_cost: Optional[str] = None
    pcc_max_cost: Optional[str] = None
    pcc_cost_sources: List[str] = Field(default_factory=list)


class RateRange(BaseModel):
    min_rate: Optional[str] = None
    max_rate: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EarningPotentialExtraction(BaseModel):
    acc_rates: Optional[RateRange] = None
    pcc_rates: Optional[RateRange] = None
    mcc_rates: Optional[RateRange] = None


class BusinessSetupExtraction(BaseModel):
    license_required: Optional[str] = None
    license_sources: List[str] = Field(default_factory=list)
    business_registration: Optional[str] = None
    registration_sources: List[str] = Field(default_factory=list)
    insurance_legally_required: Optional[str] = None
    insurance_recommended_coverage: Optional[str] = None
    insurance_cost_min: Optional[str] = None
    insurance_cost_max: Optional[str] = None
    insurance_sources: List[str] = Field(default_factory=list)


class RenewalExtraction(BaseModel):
    frequency: Optional[str] = None
    total_cce: Optional[str] = None
    core_competencies_cce: Optional[str] = None
    ethics_cce: Optional[str] = None
    resource_development_max: Optional[str] = None
    acc_mentor_coaching_requirement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_credentials() -> str:
    return """
Extract the ICF credential requirements exactly as stated in the answer text. For each level (ACC, PCC, MCC), provide:

- education_hours: The minimum number of coach-specific education/training hours required (string exactly as written).
- experience_hours: The minimum number of coaching experience hours required (string exactly as written).
- mentor_hours: The mentor coaching hours required (string exactly as written).
- prerequisite: Only for MCC, the prerequisite credential required before applying (string exactly as written; set to null for ACC and PCC).
- sources: A list of official ICF URLs (prefer coachingfederation.org) that document these requirements. If no URL is provided in the answer, return an empty list.

Return JSON with keys: acc, pcc, mcc; each being an object with the above fields.
"""


def prompt_extract_training_costs() -> str:
    return """
Extract cost ranges for ICF-accredited training programs as stated in the answer:

- acc_min_cost: The minimum (lowest) cost described for ACC pathway programs (string as written).
- acc_max_cost: The maximum (highest) cost described for ACC pathway programs (string as written).
- acc_cost_sources: List of URLs that document the ACC program cost range.
- pcc_min_cost: The minimum (lowest) cost described for PCC-level programs (string as written).
- pcc_max_cost: The maximum (highest) cost described for PCC-level programs (string as written).
- pcc_cost_sources: List of URLs that document the PCC program cost range.

Return all values exactly as they appear in the answer. If a URL is missing, return an empty list for that sources field.
"""


def prompt_extract_earning_potential() -> str:
    return """
Extract typical hourly rate ranges in North America by credential level:

For each of acc_rates, pcc_rates, mcc_rates:
- min_rate: The minimum hourly rate coaches typically charge in North America (string as written).
- max_rate: The maximum hourly rate (string as written).
- sources: List of URLs that explicitly describe/hourly rates for that credential level in North America.

Return a JSON with keys acc_rates, pcc_rates, mcc_rates. If URLs are missing, return empty lists for sources.
"""


def prompt_extract_business_setup() -> str:
    return """
Extract California business setup details for a coaching practice:

- license_required: Whether California requires a professional license to practice as a career coach (use a concise 'yes'/'no' style phrase if present, else concise summary).
- license_sources: List of URLs that support the licensing statement (prefer .ca.gov, city/county sites, or reputable legal sources).
- business_registration: Summary of the required business registration steps/requirements in California for a coaching practice (string as written).
- registration_sources: List of URLs that support the registration requirement(s).
- insurance_legally_required: Whether professional liability insurance is legally required for coaches in California (yes/no or concise summary).
- insurance_recommended_coverage: Recommended coverage amount (e.g., "$1M/$2M aggregate") if provided.
- insurance_cost_min: Typical minimum annual cost for professional liability insurance for coaches/consultants.
- insurance_cost_max: Typical maximum annual cost for professional liability insurance for coaches/consultants.
- insurance_sources: List of URLs supporting insurance legal status, recommended coverage, and/or typical cost.

Return all values as short strings exactly as written; return empty lists for missing sources.
"""


def prompt_extract_renewal() -> str:
    return """
Extract ICF credential renewal requirements:

- frequency: How often credentials must be renewed (string as written).
- total_cce: Total CCE credits required.
- core_competencies_cce: Minimum CCE credits required in Core Competencies.
- ethics_cce: Minimum CCE credits required in Coaching Ethics.
- resource_development_max: Maximum CCE credits allowed in Resource Development.
- acc_mentor_coaching_requirement: ACC-specific renewal requirement regarding mentor coaching hours.
- sources: List of official ICF URLs documenting credential renewal requirements (prefer coachingfederation.org).

Return all values exactly as written in the answer; return empty list for sources if missing.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _bool_from_text(s: Optional[str]) -> Optional[bool]:
    if not s:
        return None
    low = s.strip().lower()
    # Very lightweight heuristic for yes/no
    yes_trigs = ["yes", "required", "is required", "legally required", "must", "mandatory"]
    no_trigs = ["no", "not required", "is not required", "optional", "no license", "not legally required"]
    if any(t in low for t in yes_trigs):
        return True
    if any(t in low for t in no_trigs):
        return False
    return None


def _mk_official_icf_claim(level: str) -> str:
    return f"This webpage is an official International Coaching Federation (ICF) page (coachingfederation.org) that documents {level} credential requirements."


def _mk_no_url_instruction(urls: List[str]) -> str:
    if urls:
        return "Use the provided webpage(s) only to evaluate the claim."
    return "No URL(s) were provided in the answer; you must judge this claim as not supported (Incorrect)."


def _mk_only_with_urls_instruction(urls: List[str], extra: str = "") -> str:
    base = "Verify this statement strictly using the provided webpage(s)."
    if not urls:
        return "No URL(s) were provided; mark this claim as not supported (Incorrect)."
    return (base + (" " + extra if extra else "")).strip()


def _sources_gate(evaluator: Evaluator, parent, id_prefix: str, label: str, urls: List[str], critical: bool = True):
    return evaluator.add_custom_node(
        result=bool(urls),
        id=f"{id_prefix}_Sources_Provided",
        desc=f"{label} sources provided",
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_credential_requirements(evaluator: Evaluator, parent, creds: CredentialsExtraction):
    node = evaluator.add_parallel(
        id="Credential_Requirements_Analysis",
        desc="Comprehensive analysis of ICF credential requirements for ACC, PCC, and MCC levels",
        parent=parent,
        critical=False
    )

    # ACC
    acc = creds.acc or CredentialLevel()
    acc_node = evaluator.add_parallel(
        id="ACC_Requirements",
        desc="Complete requirements for ICF Associate Certified Coach (ACC) credential",
        parent=node,
        critical=True
    )
    acc_gate = _sources_gate(
        evaluator, acc_node, "ACC", "ACC requirement", acc.sources, critical=True
    )

    # ACC_Education_Hours
    n = evaluator.add_leaf(
        id="ACC_Education_Hours",
        desc="Minimum coach-specific education hours required for ACC credential",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coach-specific education/training hours required for the ACC credential are {acc.education_hours}.",
        node=n,
        sources=acc.sources if acc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            acc.sources,
            "Look for phrasing such as 'coach-specific education' or 'training hours' for ACC."
        ),
        extra_prerequisites=[acc_gate]
    )

    # ACC_Coaching_Experience_Hours
    n = evaluator.add_leaf(
        id="ACC_Coaching_Experience_Hours",
        desc="Minimum coaching experience hours required for ACC credential",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coaching experience hours required for the ACC credential are {acc.experience_hours}.",
        node=n,
        sources=acc.sources if acc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            acc.sources,
            "Match the minimum total client coaching hours for ACC."
        ),
        extra_prerequisites=[acc_gate]
    )

    # ACC_Mentor_Coaching_Hours
    n = evaluator.add_leaf(
        id="ACC_Mentor_Coaching_Hours",
        desc="Mentor coaching hours required for ACC credential",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The mentor coaching hours required for the ACC credential are {acc.mentor_hours}.",
        node=n,
        sources=acc.sources if acc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            acc.sources,
            "Look for 'mentor coaching' requirement for ACC."
        ),
        extra_prerequisites=[acc_gate]
    )

    # ACC_Requirements_Source
    n = evaluator.add_leaf(
        id="ACC_Requirements_Source",
        desc="Valid ICF official source URL documenting ACC requirements",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=_mk_official_icf_claim("ACC"),
        node=n,
        sources=acc.sources if acc.sources else None,
        additional_instruction=_mk_no_url_instruction(acc.sources),
        extra_prerequisites=[acc_gate]
    )

    # PCC
    pcc = creds.pcc or CredentialLevel()
    pcc_node = evaluator.add_parallel(
        id="PCC_Requirements",
        desc="Complete requirements for ICF Professional Certified Coach (PCC) credential",
        parent=node,
        critical=True
    )
    pcc_gate = _sources_gate(
        evaluator, pcc_node, "PCC", "PCC requirement", pcc.sources, critical=True
    )

    n = evaluator.add_leaf(
        id="PCC_Education_Hours",
        desc="Minimum coach-specific education hours required for PCC credential",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coach-specific education/training hours required for the PCC credential are {pcc.education_hours}.",
        node=n,
        sources=pcc.sources if pcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            pcc.sources,
            "Look for 'coach-specific education' or 'training hours' for PCC."
        ),
        extra_prerequisites=[pcc_gate]
    )

    n = evaluator.add_leaf(
        id="PCC_Coaching_Experience_Hours",
        desc="Minimum coaching experience hours required for PCC credential",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coaching experience hours required for the PCC credential are {pcc.experience_hours}.",
        node=n,
        sources=pcc.sources if pcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            pcc.sources,
            "Match the minimum total client coaching hours for PCC."
        ),
        extra_prerequisites=[pcc_gate]
    )

    n = evaluator.add_leaf(
        id="PCC_Mentor_Coaching_Hours",
        desc="Mentor coaching hours required for PCC credential",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The mentor coaching hours required for the PCC credential are {pcc.mentor_hours}.",
        node=n,
        sources=pcc.sources if pcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            pcc.sources,
            "Look for 'mentor coaching' requirement for PCC."
        ),
        extra_prerequisites=[pcc_gate]
    )

    n = evaluator.add_leaf(
        id="PCC_Requirements_Source",
        desc="Valid ICF official source URL documenting PCC requirements",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=_mk_official_icf_claim("PCC"),
        node=n,
        sources=pcc.sources if pcc.sources else None,
        additional_instruction=_mk_no_url_instruction(pcc.sources),
        extra_prerequisites=[pcc_gate]
    )

    # MCC
    mcc = creds.mcc or CredentialLevel()
    mcc_node = evaluator.add_parallel(
        id="MCC_Requirements",
        desc="Complete requirements for ICF Master Certified Coach (MCC) credential",
        parent=node,
        critical=True
    )
    mcc_gate = _sources_gate(
        evaluator, mcc_node, "MCC", "MCC requirement", mcc.sources, critical=True
    )

    n = evaluator.add_leaf(
        id="MCC_Prerequisite",
        desc="Prerequisite credential required before applying for MCC",
        parent=mcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The prerequisite credential before applying for MCC is {mcc.prerequisite}.",
        node=n,
        sources=mcc.sources if mcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            mcc.sources,
            "Confirm that the MCC application requires holding a specific prior ICF credential."
        ),
        extra_prerequisites=[mcc_gate]
    )

    n = evaluator.add_leaf(
        id="MCC_Education_Hours",
        desc="Minimum coach-specific education hours required for MCC credential",
        parent=mcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coach-specific education/training hours required for the MCC credential are {mcc.education_hours}.",
        node=n,
        sources=mcc.sources if mcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            mcc.sources,
            "Look for 'coach-specific education' or 'training hours' for MCC."
        ),
        extra_prerequisites=[mcc_gate]
    )

    n = evaluator.add_leaf(
        id="MCC_Coaching_Experience_Hours",
        desc="Minimum coaching experience hours required for MCC credential",
        parent=mcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum coaching experience hours required for the MCC credential are {mcc.experience_hours}.",
        node=n,
        sources=mcc.sources if mcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            mcc.sources,
            "Match the minimum total client coaching hours for MCC."
        ),
        extra_prerequisites=[mcc_gate]
    )

    n = evaluator.add_leaf(
        id="MCC_Mentor_Coaching_Hours",
        desc="Mentor coaching hours required for MCC credential",
        parent=mcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The mentor coaching hours required for the MCC credential are {mcc.mentor_hours}.",
        node=n,
        sources=mcc.sources if mcc.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            mcc.sources,
            "Look for 'mentor coaching' requirement for MCC."
        ),
        extra_prerequisites=[mcc_gate]
    )

    n = evaluator.add_leaf(
        id="MCC_Requirements_Source",
        desc="Valid ICF official source URL documenting MCC requirements",
        parent=mcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=_mk_official_icf_claim("MCC"),
        node=n,
        sources=mcc.sources if mcc.sources else None,
        additional_instruction=_mk_no_url_instruction(mcc.sources),
        extra_prerequisites=[mcc_gate]
    )


async def verify_training_costs(evaluator: Evaluator, parent, costs: TrainingCostExtraction):
    node = evaluator.add_parallel(
        id="Training_Cost_Analysis",
        desc="Cost ranges for ICF-accredited coach training programs",
        parent=parent,
        critical=False
    )

    # ACC costs
    acc_node = evaluator.add_parallel(
        id="ACC_Program_Costs",
        desc="Cost range for ICF-accredited ACC pathway programs",
        parent=node,
        critical=True
    )
    acc_gate = _sources_gate(
        evaluator, acc_node, "ACC_Cost", "ACC cost range", costs.acc_cost_sources, critical=True
    )

    n = evaluator.add_leaf(
        id="ACC_Minimum_Cost",
        desc="Minimum cost for ICF-accredited ACC pathway programs",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"ICF-accredited ACC pathway programs can cost as low as {costs.acc_min_cost}.",
        node=n,
        sources=costs.acc_cost_sources if costs.acc_cost_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            costs.acc_cost_sources,
            "The page(s) should reference pricing for ICF-accredited Level 1/ACSTH/ACC-path programs."
        ),
        extra_prerequisites=[acc_gate]
    )

    n = evaluator.add_leaf(
        id="ACC_Maximum_Cost",
        desc="Maximum cost for ICF-accredited ACC pathway programs",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"ICF-accredited ACC pathway programs can cost up to {costs.acc_max_cost}.",
        node=n,
        sources=costs.acc_cost_sources if costs.acc_cost_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            costs.acc_cost_sources,
            "The page(s) should reference pricing for ICF-accredited Level 1/ACSTH/ACC-path programs."
        ),
        extra_prerequisites=[acc_gate]
    )

    n = evaluator.add_leaf(
        id="ACC_Cost_Source",
        desc="Valid source URL documenting ACC program cost ranges",
        parent=acc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage documents price ranges for ICF-accredited ACC pathway training programs.",
        node=n,
        sources=costs.acc_cost_sources if costs.acc_cost_sources else None,
        additional_instruction=_mk_no_url_instruction(costs.acc_cost_sources),
        extra_prerequisites=[acc_gate]
    )

    # PCC costs
    pcc_node = evaluator.add_parallel(
        id="PCC_Program_Costs",
        desc="Cost range for PCC-level training programs",
        parent=node,
        critical=True
    )
    pcc_gate = _sources_gate(
        evaluator, pcc_node, "PCC_Cost", "PCC cost range", costs.pcc_cost_sources, critical=True
    )

    n = evaluator.add_leaf(
        id="PCC_Minimum_Cost",
        desc="Minimum cost for PCC-level training programs",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"PCC-level training programs can cost as low as {costs.pcc_min_cost}.",
        node=n,
        sources=costs.pcc_cost_sources if costs.pcc_cost_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            costs.pcc_cost_sources,
            "The page(s) should reference pricing for ICF-accredited Level 2/PCC-path programs."
        ),
        extra_prerequisites=[pcc_gate]
    )

    n = evaluator.add_leaf(
        id="PCC_Maximum_Cost",
        desc="Maximum cost for PCC-level training programs",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"PCC-level training programs can cost up to {costs.pcc_max_cost}.",
        node=n,
        sources=costs.pcc_cost_sources if costs.pcc_cost_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            costs.pcc_cost_sources,
            "The page(s) should reference pricing for ICF-accredited Level 2/PCC-path programs."
        ),
        extra_prerequisites=[pcc_gate]
    )

    n = evaluator.add_leaf(
        id="PCC_Cost_Source",
        desc="Valid source URL documenting PCC program cost ranges",
        parent=pcc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage documents price ranges for PCC-level coach training programs (ICF-accredited Level 2 or equivalent).",
        node=n,
        sources=costs.pcc_cost_sources if costs.pcc_cost_sources else None,
        additional_instruction=_mk_no_url_instruction(costs.pcc_cost_sources),
        extra_prerequisites=[pcc_gate]
    )


async def verify_earning_potential(evaluator: Evaluator, parent, earning: EarningPotentialExtraction):
    node = evaluator.add_parallel(
        id="Earning_Potential_Analysis",
        desc="Hourly rate ranges by ICF credential level in North America",
        parent=parent,
        critical=False
    )

    # Helper for each level
    async def _verify_level(level_id: str, label: str, rr: RateRange):
        sub = evaluator.add_parallel(
            id=f"{level_id}_Hourly_Rates",
            desc=f"Hourly rate range for {label}-credentialed coaches in North America",
            parent=node,
            critical=True
        )
        gate = _sources_gate(
            evaluator, sub, f"{level_id}_Rates", f"{label} rate range", rr.sources if rr else [], critical=True
        )
        # Min
        leaf = evaluator.add_leaf(
            id=f"{level_id}_Minimum_Rate",
            desc=f"Minimum hourly rate for {label} coaches in North America",
            parent=sub,
            critical=True
        )
        urls = rr.sources if rr and rr.sources else None
        await evaluator.verify(
            claim=f"In North America, {label} coaches typically charge at least {rr.min_rate} per hour.",
            node=leaf,
            sources=urls,
            additional_instruction=_mk_only_with_urls_instruction(
                rr.sources if rr else [],
                "Accept reasonable rounding. Ensure the page addresses North America or the U.S./Canada."
            ),
            extra_prerequisites=[gate]
        )
        # Max
        leaf = evaluator.add_leaf(
            id=f"{level_id}_Maximum_Rate",
            desc=f"Maximum hourly rate for {label} coaches in North America",
            parent=sub,
            critical=True
        )
        await evaluator.verify(
            claim=f"In North America, {label} coaches can charge up to {rr.max_rate} per hour.",
            node=leaf,
            sources=urls,
            additional_instruction=_mk_only_with_urls_instruction(
                rr.sources if rr else [],
                "Accept reasonable rounding. Ensure the page addresses North America or the U.S./Canada."
            ),
            extra_prerequisites=[gate]
        )
        # Source
        leaf = evaluator.add_leaf(
            id=f"{level_id}_Rate_Source",
            desc=f"Valid source URL documenting {label} hourly rate ranges",
            parent=sub,
            critical=True
        )
        await evaluator.verify(
            claim=f"This webpage documents hourly rate ranges for {label} coaches in North America.",
            node=leaf,
            sources=urls,
            additional_instruction=_mk_no_url_instruction(rr.sources if rr else []),
            extra_prerequisites=[gate]
        )

    await _verify_level("ACC", "ACC", earning.acc_rates or RateRange())
    await _verify_level("PCC", "PCC", earning.pcc_rates or RateRange())
    await _verify_level("MCC", "MCC", earning.mcc_rates or RateRange())


async def verify_california_business_setup(evaluator: Evaluator, parent, biz: BusinessSetupExtraction):
    node = evaluator.add_parallel(
        id="California_Business_Setup",
        desc="Legal and business requirements for starting a coaching practice in California",
        parent=parent,
        critical=False
    )

    # License status
    lic_gate = _sources_gate(
        evaluator, node, "CA_License", "California professional license", biz.license_sources, critical=True
    )
    leaf = evaluator.add_leaf(
        id="Professional_License_Status",
        desc="Whether California requires a professional license to practice as a career coach",
        parent=node,
        critical=True
    )
    lic_bool = _bool_from_text(biz.license_required)
    if lic_bool is True:
        lic_claim = "California requires a professional/state license to practice as a career coach."
    elif lic_bool is False:
        lic_claim = "California does not require a professional/state license to practice as a career coach."
    else:
        lic_claim = f"California licensing requirement for career coaches is: {biz.license_required}."
    await evaluator.verify(
        claim=lic_claim,
        node=leaf,
        sources=biz.license_sources if biz.license_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            biz.license_sources,
            "Prioritize official California government or city/county sites (.ca.gov) or reputable legal resources."
        ),
        extra_prerequisites=[lic_gate]
    )

    # Business registration requirement
    reg_gate = _sources_gate(
        evaluator, node, "CA_Registration", "California business registration", biz.registration_sources, critical=True
    )
    leaf = evaluator.add_leaf(
        id="Business_Registration_Requirement",
        desc="Business registration requirement for coaching practices in California",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In California, to operate a coaching practice, the following business registration requirement(s) apply: {biz.business_registration}.",
        node=leaf,
        sources=biz.registration_sources if biz.registration_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            biz.registration_sources,
            "Accept requirements like registering LLC/corp with CA Secretary of State, obtaining a city business tax certificate/license, filing a DBA/FBN for sole proprietors using a fictitious name, and obtaining an EIN if applicable."
        ),
        extra_prerequisites=[reg_gate]
    )

    # Professional Liability Insurance block
    ins_node = evaluator.add_parallel(
        id="Professional_Liability_Insurance",
        desc="Professional liability insurance requirements and recommendations for coaches",
        parent=node,
        critical=True
    )
    ins_gate = _sources_gate(
        evaluator, ins_node, "CA_Insurance", "Insurance", biz.insurance_sources, critical=True
    )

    # Insurance legal status
    leaf = evaluator.add_leaf(
        id="Insurance_Legal_Status",
        desc="Whether professional liability insurance is legally required for coaches in California",
        parent=ins_node,
        critical=True
    )
    ins_bool = _bool_from_text(biz.insurance_legally_required)
    if ins_bool is True:
        ins_claim = "Professional liability insurance is legally required for coaches in California."
    elif ins_bool is False:
        ins_claim = "Professional liability insurance is not legally required for coaches in California."
    else:
        ins_claim = f"Legal status of professional liability insurance for coaches in California is: {biz.insurance_legally_required}."
    await evaluator.verify(
        claim=ins_claim,
        node=leaf,
        sources=biz.insurance_sources if biz.insurance_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            biz.insurance_sources,
            "Government or reputable legal/insurance sources preferred."
        ),
        extra_prerequisites=[ins_gate]
    )

    # Recommended coverage
    leaf = evaluator.add_leaf(
        id="Insurance_Recommended_Coverage",
        desc="Recommended professional liability insurance coverage amount for coaches",
        parent=ins_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The recommended professional liability insurance coverage amount for coaches is {biz.insurance_recommended_coverage}.",
        node=leaf,
        sources=biz.insurance_sources if biz.insurance_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            biz.insurance_sources,
            "Accept industry recommendations (e.g., $1M/$2M aggregate) from credible sources."
        ),
        extra_prerequisites=[ins_gate]
    )

    # Insurance cost range
    leaf = evaluator.add_leaf(
        id="Insurance_Cost_Range",
        desc="Typical annual cost range for professional liability insurance for coaches/consultants",
        parent=ins_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The typical annual cost range for professional liability insurance for coaches/consultants is from {biz.insurance_cost_min} to {biz.insurance_cost_max} per year.",
        node=leaf,
        sources=biz.insurance_sources if biz.insurance_sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            biz.insurance_sources,
            "Accept ranges in USD/year; insurance provider or credible business sources acceptable."
        ),
        extra_prerequisites=[ins_gate]
    )


async def verify_renewal_requirements(evaluator: Evaluator, parent, ren: RenewalExtraction):
    node = evaluator.add_parallel(
        id="Credential_Renewal_Requirements",
        desc="ICF credential renewal and maintenance requirements",
        parent=parent,
        critical=False
    )
    gate = _sources_gate(
        evaluator, node, "Renewal", "ICF renewal", ren.sources, critical=True
    )

    # Renewal_Frequency
    leaf = evaluator.add_leaf(
        id="Renewal_Frequency",
        desc="How often ICF credentials must be renewed",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"ICF credentials must be renewed {ren.frequency}.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Use the official ICF page(s) documenting renewal intervals."
        ),
        extra_prerequisites=[gate]
    )

    # CCE requirements block
    cce_node = evaluator.add_parallel(
        id="CCE_Requirements",
        desc="Continuing Coach Education (CCE) credit requirements for credential renewal",
        parent=node,
        critical=True
    )

    # Total_CCE_Credits
    leaf = evaluator.add_leaf(
        id="Total_CCE_Credits",
        desc="Total CCE credits required for ICF credential renewal",
        parent=cce_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The total number of CCE credits required for ICF credential renewal is {ren.total_cce}.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Match the official ICF total CCE requirement for renewal."
        ),
        extra_prerequisites=[gate]
    )

    # Core_Competency_Credits
    leaf = evaluator.add_leaf(
        id="Core_Competency_Credits",
        desc="Minimum CCE credits required in Core Competencies",
        parent=cce_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least {ren.core_competencies_cce} CCE credits must be in Core Competencies for renewal.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Confirm minimum Core Competency CCE credits."
        ),
        extra_prerequisites=[gate]
    )

    # Ethics_Credits
    leaf = evaluator.add_leaf(
        id="Ethics_Credits",
        desc="Minimum CCE credits required in Coaching Ethics",
        parent=cce_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"At least {ren.ethics_cce} CCE credits must be in Coaching Ethics for renewal.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Confirm specific ethics CCE credits (e.g., 'Coaching Ethics' or similar)."
        ),
        extra_prerequisites=[gate]
    )

    # Resource_Development_Credits
    leaf = evaluator.add_leaf(
        id="Resource_Development_Credits",
        desc="Maximum CCE credits allowed in Resource Development",
        parent=cce_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum number of CCE credits allowed in Resource Development for renewal is {ren.resource_development_max}.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Confirm the cap on Resource Development credits."
        ),
        extra_prerequisites=[gate]
    )

    # ACC-specific renewal mentor coaching
    leaf = evaluator.add_leaf(
        id="ACC_Specific_Renewal",
        desc="ACC-specific renewal requirement for mentor coaching hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"For ACC renewal, mentor coaching requirement is: {ren.acc_mentor_coaching_requirement}.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_only_with_urls_instruction(
            ren.sources,
            "Confirm the ACC-specific mentor coaching requirement for renewal."
        ),
        extra_prerequisites=[gate]
    )

    # Renewal source validity
    leaf = evaluator.add_leaf(
        id="Renewal_Source",
        desc="Valid ICF official source URL documenting credential renewal requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is an official ICF page (coachingfederation.org) that documents the credential renewal requirements.",
        node=leaf,
        sources=ren.sources if ren.sources else None,
        additional_instruction=_mk_no_url_instruction(ren.sources),
        extra_prerequisites=[gate]
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
    Evaluate an answer for the ICF career coaching business plan in California.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel top-level sections
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

    # Extract all required sections (in parallel)
    creds_task = evaluator.extract(
        prompt=prompt_extract_credentials(),
        template_class=CredentialsExtraction,
        extraction_name="credential_requirements",
    )
    costs_task = evaluator.extract(
        prompt=prompt_extract_training_costs(),
        template_class=TrainingCostExtraction,
        extraction_name="training_costs",
    )
    earn_task = evaluator.extract(
        prompt=prompt_extract_earning_potential(),
        template_class=EarningPotentialExtraction,
        extraction_name="earning_potential",
    )
    biz_task = evaluator.extract(
        prompt=prompt_extract_business_setup(),
        template_class=BusinessSetupExtraction,
        extraction_name="california_business_setup",
    )
    ren_task = evaluator.extract(
        prompt=prompt_extract_renewal(),
        template_class=RenewalExtraction,
        extraction_name="renewal_requirements",
    )

    creds, costs, earning, biz, ren = await asyncio.gather(
        creds_task, costs_task, earn_task, biz_task, ren_task
    )

    # Build and verify tree according to rubric
    # Root node is non-critical to allow partial scoring across sections
    await verify_credential_requirements(evaluator, root, creds)
    await verify_training_costs(evaluator, root, costs)
    await verify_earning_potential(evaluator, root, earning)
    await verify_california_business_setup(evaluator, root, biz)
    await verify_renewal_requirements(evaluator, root, ren)

    # Return evaluation summary
    return evaluator.get_summary()