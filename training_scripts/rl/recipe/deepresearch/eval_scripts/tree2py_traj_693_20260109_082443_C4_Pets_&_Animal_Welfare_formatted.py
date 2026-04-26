import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "therapy_org_il"
TASK_DESCRIPTION = (
    "Identify a nationally recognized therapy dog certification organization that operates in Illinois and certifies therapy dog teams for hospital and nursing home visits. "
    "Provide the following information about this organization: (1) the organization's official name, (2) the minimum age requirement for dogs at the time of evaluation, "
    "(3) whether Canine Good Citizen (CGC) certification is required or recommended as a prerequisite, (4) whether the organization provides liability insurance coverage "
    "for certified therapy dog teams, (5) the vaccination documentation required (specifically regarding rabies), (6) whether they offer evaluation or testing services in the "
    "Chicago metropolitan area, (7) a list of facility types that certified teams are authorized to visit, and (8) whether formal handler training is required or provided for volunteers."
)


class FieldSources(BaseModel):
    official_name_urls: List[str] = Field(default_factory=list)
    eligibility_urls: List[str] = Field(default_factory=list)
    minimum_age_urls: List[str] = Field(default_factory=list)
    cgc_urls: List[str] = Field(default_factory=list)
    insurance_urls: List[str] = Field(default_factory=list)
    vaccination_urls: List[str] = Field(default_factory=list)
    chicago_urls: List[str] = Field(default_factory=list)
    facility_urls: List[str] = Field(default_factory=list)
    training_urls: List[str] = Field(default_factory=list)


class OrgExtraction(BaseModel):
    org_name: Optional[str] = None
    org_urls: List[str] = Field(default_factory=list)

    minimum_age: Optional[str] = None
    cgc_policy: Optional[str] = None
    liability_insurance: Optional[str] = None
    vaccination_requirements: Optional[str] = None
    chicago_testing: Optional[str] = None
    facility_types: List[str] = Field(default_factory=list)
    handler_training: Optional[str] = None

    field_sources: FieldSources = Field(default_factory=FieldSources)


def prompt_extract_org_data() -> str:
    return (
        "Extract information about a single therapy dog certification organization mentioned in the answer. "
        "Return the fields listed below exactly from the answer text. Do not infer or invent anything. "
        "Also extract URLs from the answer that support each field.\n\n"
        "Fields to extract:\n"
        "1) org_name: The organization's official name as stated in the answer.\n"
        "2) org_urls: All URLs in the answer that point to the organization's official website or relevant official pages.\n"
        "3) minimum_age: The minimum dog age at the time of evaluation (as text; e.g., 'at least 1 year', '12 months').\n"
        "4) cgc_policy: The organization's CGC prerequisite status as explicitly described in the answer (e.g., 'required', 'recommended', 'not required').\n"
        "5) liability_insurance: The answer's statement about whether liability insurance coverage is provided (text such as 'provided', 'included', 'yes', 'no').\n"
        "6) vaccination_requirements: The vaccination documentation policy as described, including any mention of 'rabies'.\n"
        "7) chicago_testing: The answer's statement about evaluation/testing availability in the Chicago metropolitan area.\n"
        "8) facility_types: A list of facility categories the certified teams are authorized to visit (e.g., ['hospitals','nursing homes','schools']).\n"
        "9) handler_training: The answer's statement regarding formal handler training (e.g., 'required', 'provided', 'online course', 'orientation').\n\n"
        "For each attribute, also extract supporting URLs as 'field_sources' with these arrays:\n"
        "- official_name_urls\n"
        "- eligibility_urls\n"
        "- minimum_age_urls\n"
        "- cgc_urls\n"
        "- insurance_urls\n"
        "- vaccination_urls\n"
        "- chicago_urls\n"
        "- facility_urls\n"
        "- training_urls\n\n"
        "Rules for URL extraction:\n"
        "- Extract only URLs explicitly present in the answer (including markdown links).\n"
        "- If a field has no specific supporting URL in the answer, return an empty array for that field's URLs.\n"
        "- Always include full URLs.\n"
    )


def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _has_text(val: Optional[str]) -> bool:
    return bool(val and str(val).strip())


def _contains_any_term(values: List[str], terms: List[str]) -> bool:
    if not values:
        return False
    low_values = [v.lower() for v in values if v]
    for v in low_values:
        for t in terms:
            if t in v:
                return True
    return False


def _normalize_cgc_policy(policy: Optional[str]) -> Optional[str]:
    if not _has_text(policy):
        return None
    p = policy.lower()
    if "recommend" in p:
        return "recommended"
    if "require" in p or "required" in p or "mandatory" in p:
        return "required"
    return None


def _is_positive_insurance(text: Optional[str]) -> bool:
    if not _has_text(text):
        return False
    t = text.lower()
    positive_terms = ["provided", "provides", "coverage", "insured", "liability insurance", "included", "yes"]
    return any(term in t for term in positive_terms)


async def _verify_official_name(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Official_Organization_Name",
        desc="Provides the organization's official name.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=_has_text(data.org_name),
        id="official_name_provided",
        desc="Official organization name is provided in the answer.",
        parent=mod,
        critical=True
    )
    name_verify = evaluator.add_leaf(
        id="official_name_supported",
        desc="Official organization name is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.official_name_urls, data.org_urls)
    claim = f"The organization's official name is '{data.org_name}'."
    await evaluator.verify(
        claim=claim,
        node=name_verify,
        sources=sources,
        additional_instruction="Confirm the official name on the organization's homepage, header, About page, or official profile."
    )


async def _verify_eligibility(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_parallel(
        id="Organization_Eligibility",
        desc="Establishes (with sufficient support) that the organization (a) is nationally recognized, (b) operates in Illinois, and (c) certifies therapy dog teams.",
        parent=parent,
        critical=True
    )
    src_exists = evaluator.add_custom_node(
        result=len(_merge_sources(data.field_sources.eligibility_urls, data.org_urls)) > 0,
        id="eligibility_sources_present",
        desc="Eligibility sources are provided in the answer.",
        parent=mod,
        critical=True
    )
    nat_leaf = evaluator.add_leaf(
        id="nationally_recognized_supported",
        desc="The organization is nationally recognized (has nationwide scope or presence).",
        parent=mod,
        critical=True
    )
    il_leaf = evaluator.add_leaf(
        id="operates_in_illinois_supported",
        desc="The organization operates in Illinois (has evaluators, events, or registered teams there).",
        parent=mod,
        critical=True
    )
    cert_leaf = evaluator.add_leaf(
        id="certifies_therapy_teams_supported",
        desc="The organization certifies or registers therapy dog teams (handler+dog) for facility visits.",
        parent=mod,
        critical=True
    )

    base_sources = _merge_sources(data.field_sources.eligibility_urls, data.org_urls)
    chicago_sources = _merge_sources(data.field_sources.chicago_urls, base_sources)

    await evaluator.verify(
        claim="This therapy dog organization is nationally recognized and operates nationwide within the United States.",
        node=nat_leaf,
        sources=base_sources,
        additional_instruction="Look for statements indicating nationwide operations, presence across states, or being a national organization."
    )
    await evaluator.verify(
        claim="This therapy dog organization operates in Illinois (e.g., has evaluators, scheduled testing, or registered teams in IL or Chicago).",
        node=il_leaf,
        sources=chicago_sources,
        additional_instruction="Accept pages listing Illinois or Chicago evaluators/teams, schedules, or official coverage in Illinois."
    )
    await evaluator.verify(
        claim="This organization certifies or registers therapy dog teams for visiting facilities.",
        node=cert_leaf,
        sources=base_sources,
        additional_instruction="Certification may be phrased as registration/approval of handler–dog teams; accept reasonable synonyms."
    )


async def _verify_facility_types(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Facility_Types_Authorized_List_Including_Hospitals_And_Nursing_Homes",
        desc="Provides a list of facility types that certified teams are authorized to visit, including hospitals and nursing homes.",
        parent=parent,
        critical=True
    )
    list_exists = evaluator.add_custom_node(
        result=len(data.facility_types) > 0,
        id="facility_types_list_provided",
        desc="Facility types list is provided in the answer.",
        parent=mod,
        critical=True
    )
    includes_hosp = evaluator.add_custom_node(
        result=_contains_any_term(data.facility_types, ["hospital", "hospitals"]),
        id="facility_types_include_hospitals",
        desc="The provided facility list includes hospitals.",
        parent=mod,
        critical=True
    )
    includes_nh = evaluator.add_custom_node(
        result=_contains_any_term(
            data.facility_types,
            ["nursing home", "nursing homes", "skilled nursing", "long-term care", "long term care", "care home", "senior living"]
        ),
        id="facility_types_include_nursing_homes",
        desc="The provided facility list includes nursing homes or equivalent (e.g., skilled nursing/long-term care).",
        parent=mod,
        critical=True
    )
    auth_supported = evaluator.add_leaf(
        id="facility_authorization_supported_by_sources",
        desc="Sources support that certified teams are authorized to visit hospitals and nursing homes.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.facility_urls, data.org_urls)
    claim = "Certified therapy dog teams with this organization are authorized to visit hospitals and nursing homes."
    await evaluator.verify(
        claim=claim,
        node=auth_supported,
        sources=sources,
        additional_instruction="Confirm that hospitals and nursing homes are among permitted facility types for visits."
    )


async def _verify_minimum_age(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Minimum_Dog_Age_At_Evaluation",
        desc="Provides the minimum age requirement for dogs at the time of evaluation.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=_has_text(data.minimum_age),
        id="minimum_age_provided",
        desc="Minimum dog age at evaluation is provided in the answer.",
        parent=mod,
        critical=True
    )
    age_leaf = evaluator.add_leaf(
        id="minimum_age_supported",
        desc="Minimum dog age at evaluation is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.minimum_age_urls, data.org_urls)
    claim = f"The minimum dog age at evaluation for this organization is '{data.minimum_age}'."
    await evaluator.verify(
        claim=claim,
        node=age_leaf,
        sources=sources,
        additional_instruction="Accept reasonable variants (e.g., 'at least 1 year', 'minimum 12 months'). Match the policy text."
    )


async def _verify_cgc_policy(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="CGC_Prerequisite_Policy_Required_Or_Recommended",
        desc="Indicates whether CGC is required or recommended as a prerequisite.",
        parent=parent,
        critical=True
    )
    normalized = _normalize_cgc_policy(data.cgc_policy)
    exists_node = evaluator.add_custom_node(
        result=_has_text(data.cgc_policy) and (normalized in {"required", "recommended"}),
        id="cgc_policy_provided_valid",
        desc="CGC policy is provided and is either 'required' or 'recommended'.",
        parent=mod,
        critical=True
    )
    cgc_leaf = evaluator.add_leaf(
        id="cgc_policy_supported",
        desc="CGC prerequisite policy is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.cgc_urls, data.org_urls)
    policy_word = normalized or "required"
    claim = f"Canine Good Citizen (CGC) is {policy_word} by this organization as a prerequisite."
    await evaluator.verify(
        claim=claim,
        node=cgc_leaf,
        sources=sources,
        additional_instruction="Confirm whether CGC is required or recommended as per the organization's policy."
    )


async def _verify_liability_insurance(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Liability_Insurance_Provided",
        desc="Establishes that the organization provides liability insurance coverage for certified therapy dog teams.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=_is_positive_insurance(data.liability_insurance),
        id="liability_insurance_indicated_positive",
        desc="Answer indicates liability insurance is provided.",
        parent=mod,
        critical=True
    )
    ins_leaf = evaluator.add_leaf(
        id="liability_insurance_supported",
        desc="Liability insurance coverage is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.insurance_urls, data.org_urls)
    claim = "This organization provides liability insurance coverage for certified therapy dog teams."
    await evaluator.verify(
        claim=claim,
        node=ins_leaf,
        sources=sources,
        additional_instruction="Confirm policy details indicating liability insurance coverage is included for teams."
    )


async def _verify_vaccination_rabies(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Rabies_Vaccination_Documentation_Required",
        desc="Establishes that current vaccination documentation is required and that rabies vaccination/documentation is mandatory.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=_has_text(data.vaccination_requirements) and ("rabies" in data.vaccination_requirements.lower()),
        id="rabies_mentioned_in_answer",
        desc="Answer explicitly mentions rabies vaccination/documentation.",
        parent=mod,
        critical=True
    )
    rab_leaf = evaluator.add_leaf(
        id="rabies_requirement_supported",
        desc="Rabies vaccination/documentation requirement is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.vaccination_urls, data.org_urls)
    claim = "Current rabies vaccination documentation is required by this organization for therapy dog teams."
    await evaluator.verify(
        claim=claim,
        node=rab_leaf,
        sources=sources,
        additional_instruction="Verify that rabies vaccination (current) documentation is stated as mandatory."
    )


async def _verify_chicago_access(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Chicago_Metro_Evaluation_Testing_Accessible",
        desc="Establishes that evaluation/testing services are available in the Chicago metropolitan area.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=(
            _has_text(data.chicago_testing) or
            len(data.field_sources.chicago_urls) > 0
        ),
        id="chicago_availability_info_provided",
        desc="Answer provides Chicago metro evaluation/testing availability info or Chicago-specific sources.",
        parent=mod,
        critical=True
    )
    chi_leaf = evaluator.add_leaf(
        id="chicago_availability_supported",
        desc="Chicago metropolitan evaluation/testing availability is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.chicago_urls, data.org_urls)
    claim = "Evaluation/testing services are available in the Chicago metropolitan area for this organization."
    await evaluator.verify(
        claim=claim,
        node=chi_leaf,
        sources=sources,
        additional_instruction="Support may include local evaluator listings, scheduled testing in Chicago area, or Illinois-specific pages naming Chicago."
    )


async def _verify_handler_training(evaluator: Evaluator, parent, data: OrgExtraction) -> None:
    mod = evaluator.add_sequential(
        id="Handler_Training_Required_Or_Provided",
        desc="Establishes that formal handler training is required and/or provided for volunteers.",
        parent=parent,
        critical=True
    )
    exists_node = evaluator.add_custom_node(
        result=_has_text(data.handler_training),
        id="handler_training_info_provided",
        desc="Answer provides handler training requirement/provision information.",
        parent=mod,
        critical=True
    )
    train_leaf = evaluator.add_leaf(
        id="handler_training_supported",
        desc="Handler training requirement/provision is supported by sources.",
        parent=mod,
        critical=True
    )
    sources = _merge_sources(data.field_sources.training_urls, data.org_urls)
    ht = (data.handler_training or "").lower()
    if "require" in ht or "required" in ht or "mandatory" in ht:
        claim = "This organization requires formal handler training for volunteers."
    else:
        claim = "This organization provides formal handler training for volunteers."
    await evaluator.verify(
        claim=claim,
        node=train_leaf,
        sources=sources,
        additional_instruction="Accept structured training, courses, orientations, or mandatory education programs for handlers."
    )


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

    data: OrgExtraction = await evaluator.extract(
        prompt=prompt_extract_org_data(),
        template_class=OrgExtraction,
        extraction_name="organization_info",
    )

    main = evaluator.add_parallel(
        id="Root",
        desc="Evaluate whether the answer identifies a nationally recognized therapy dog certification organization operating in Illinois and provides all requested attributes while satisfying all explicit constraints.",
        parent=root,
        critical=True
    )

    await _verify_official_name(evaluator, main, data)
    await _verify_eligibility(evaluator, main, data)
    await _verify_facility_types(evaluator, main, data)
    await _verify_minimum_age(evaluator, main, data)
    await _verify_cgc_policy(evaluator, main, data)
    await _verify_liability_insurance(evaluator, main, data)
    await _verify_vaccination_rabies(evaluator, main, data)
    await _verify_chicago_access(evaluator, main, data)
    await _verify_handler_training(evaluator, main, data)

    return evaluator.get_summary()