import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "credential_platform_selection"
TASK_DESCRIPTION = """
A large state university in the United States is establishing a new continuing education division that will offer professional development micro-credentials and certificates in various fields, including STEM disciplines. The university needs to select a digital credentialing platform to issue, manage, and verify these credentials. The platform must meet the following requirements: (1) Support the Open Badges standard (version 2.0 or higher), (2) Integrate with at least one major Learning Management System (Canvas, Moodle, or Blackboard), (3) Provide API-based integration capabilities for automated credential issuance, (4) Include verification mechanisms for issued credentials, (5) Support embedding metadata in credentials (issuer information, criteria, achievement details), (6) Allow credentials to be shareable and portable across different platforms and systems, and (7) For STEM-related programs, be compatible with quality assurance standards such as ABET Recognition of Credentials. Identify one digital credentialing platform that satisfies all these requirements, and provide the specific evidence from the platform's official documentation or reliable sources that demonstrates compliance with each requirement.
"""


class RequirementEvidence(BaseModel):
    claim: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class LMSRequirement(BaseModel):
    claim: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    lms_names: List[str] = Field(default_factory=list)


class SelectedPlatformExtraction(BaseModel):
    selected_platform_name: Optional[str] = None
    selected_platform_url: Optional[str] = None
    multiple_platforms_selected: Optional[bool] = None


class RequirementsExtraction(BaseModel):
    open_badges: Optional[RequirementEvidence] = None
    lms_integration: Optional[LMSRequirement] = None
    api_support: Optional[RequirementEvidence] = None
    verification_mechanism: Optional[RequirementEvidence] = None
    metadata_embedding: Optional[RequirementEvidence] = None
    portability: Optional[RequirementEvidence] = None
    qa_compatibility: Optional[RequirementEvidence] = None


def prompt_extract_platform_selection() -> str:
    return """
    Identify the single digital credentialing platform that the answer selects as the proposed solution.

    Return a JSON object with:
    - selected_platform_name: the exact name of the platform the answer recommends (e.g., "Credly", "Badgr", "Parchment", "Open Badge Factory"). If none is clearly selected, return null.
    - selected_platform_url: if the answer provides an official product or homepage URL for the selected platform, extract it; otherwise return null.
    - multiple_platforms_selected: return true if the answer explicitly proposes more than one platform as selected solutions (e.g., recommends two platforms together). Otherwise return false.
    """


def prompt_extract_requirements() -> str:
    return """
    Extract, for the chosen platform, the claim and the specific source URLs (evidence) that the answer cites for each requirement. Only extract URLs explicitly present in the answer.

    Return a JSON object with the following fields (each field contains 'claim' and 'sources' arrays; for LMS also 'lms_names'):
    - open_badges: { claim, sources }
      • The claim should reflect support for Open Badges version 2.0 or higher (e.g., "Supports Open Badges v2.1").
      • sources: list of URLs cited for Open Badges support.

    - lms_integration: { claim, sources, lms_names }
      • The claim should state integration with at least one major LMS (Canvas, Moodle, or Blackboard).
      • lms_names: extract the specific LMS names mentioned (subset of Canvas, Moodle, Blackboard), if any.
      • sources: list of URLs cited for LMS integration.

    - api_support: { claim, sources }
      • The claim should state API-based integration enabling automated credential issuance (e.g., REST API, GraphQL).
      • sources: list of URLs cited for API capabilities.

    - verification_mechanism: { claim, sources }
      • The claim should state that issued credentials can be verified (e.g., verification page, hosted validation, cryptographic signature, blockchain proof).
      • sources: list of URLs cited for verification features.

    - metadata_embedding: { claim, sources }
      • The claim should state support for embedding metadata in credentials (issuer info, criteria, achievement details, evidence).
      • sources: list of URLs cited for metadata capabilities.

    - portability: { claim, sources }
      • The claim should state credentials are shareable/portable across platforms (e.g., Open Badges portability, sharing to LinkedIn).
      • sources: list of URLs cited for portability/interoperability.

    - qa_compatibility: { claim, sources }
      • The claim should state compatibility with STEM quality assurance standards, such as ABET Recognition of Credentials.
      • sources: list of URLs cited for ABET or related QA compatibility.

    If any claim is not explicitly stated in the answer, set claim to null. If no URLs are cited for a requirement, return an empty sources array.
    """


def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    filtered = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s:
                if not (s.startswith("http://") or s.startswith("https://")):
                    s = "http://" + s
                filtered.append(s)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for s in filtered:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _non_empty_sources(urls: Optional[List[str]]) -> bool:
    urls = _normalize_urls(urls or [])
    return len(urls) > 0


def _build_claim(default_template: str, platform_name: Optional[str], fallback_if_missing_claim: Optional[str]) -> str:
    name = (platform_name or "").strip()
    if name:
        return default_template.format(platform=name)
    # If platform unknown, use generic fallback text
    return fallback_if_missing_claim or default_template.format(platform="the selected platform")


async def _add_requirement_check(
    evaluator: Evaluator,
    parent_node,
    req_group_id: str,
    req_group_desc: str,
    meets_id: str,
    meets_desc: str,
    evidence_id: str,
    evidence_desc: str,
    claim_text: Optional[str],
    sources: Optional[List[str]],
    additional_instruction: str,
) -> None:
    # Requirement group (critical, parallel)
    req_group = evaluator.add_parallel(
        id=req_group_id,
        desc=req_group_desc,
        parent=parent_node,
        critical=True
    )

    # Evidence provided (critical, existence check)
    evidence_ok = _non_empty_sources(sources)
    evaluator.add_custom_node(
        result=evidence_ok,
        id=evidence_id,
        desc=evidence_desc,
        parent=req_group,
        critical=True
    )

    # Meets requirement (critical, verified against sources)
    meets_leaf = evaluator.add_leaf(
        id=meets_id,
        desc=meets_desc,
        parent=req_group,
        critical=True
    )
    await evaluator.verify(
        claim=(claim_text or meets_desc),
        node=meets_leaf,
        sources=_normalize_urls(sources or []),
        additional_instruction=additional_instruction
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

    # Extraction: selected platform and requirement-specific claims with sources
    selected_platform = await evaluator.extract(
        prompt=prompt_extract_platform_selection(),
        template_class=SelectedPlatformExtraction,
        extraction_name="selected_platform"
    )

    reqs = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_evidence"
    )

    # Build PlatformSelection node (critical, sequential)
    platform_selection_node = evaluator.add_sequential(
        id="PlatformSelection",
        desc="Evaluate whether the response identifies exactly one digital credentialing platform and demonstrates (with evidence) that it satisfies all stated requirements.",
        parent=root,
        critical=True
    )

    # Leaf: SelectOnePlatform (critical)
    select_one_leaf = evaluator.add_leaf(
        id="SelectOnePlatform",
        desc="Response identifies exactly one specific digital credentialing platform as the proposed solution.",
        parent=platform_selection_node,
        critical=True
    )
    await evaluator.verify(
        claim="The response identifies exactly one specific digital credentialing platform as the proposed solution.",
        node=select_one_leaf,
        additional_instruction=(
            "Check the answer text. It should clearly select one platform as the recommended solution. "
            "Mentioning other platforms for context is acceptable, but the answer must not present multiple platforms as the selected solution."
        )
    )

    # RequirementsWithEvidence (critical, parallel)
    requirements_node = evaluator.add_parallel(
        id="RequirementsWithEvidence",
        desc="For the chosen platform, the response demonstrates compliance with each requirement and provides supporting evidence from official documentation or otherwise reliable sources.",
        parent=platform_selection_node,
        critical=True
    )

    platform_name = selected_platform.selected_platform_name

    # 1) Open Badges support
    ob_claim_default = _build_claim(
        default_template="{platform} supports the Open Badges standard version 2.0 or higher.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    ob_add_ins = (
        "Verify the page explicitly states Open Badges support (v2.0 or higher). "
        "Accept mentions such as 'Open Badges 2.0', 'Open Badges v2.1', 'IMS Global', or '1EdTech Open Badges'."
    )
    ob = reqs.open_badges or RequirementEvidence()
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="OpenBadgesCompliance",
        req_group_desc="Open Badges support requirement is satisfied and evidenced.",
        meets_id="OpenBadgesCompliance_MeetsRequirement",
        meets_desc="Platform supports Open Badges standard version 2.0 or higher.",
        evidence_id="OpenBadgesCompliance_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting Open Badges v2.0+ support.",
        claim_text=ob.claim or ob_claim_default,
        sources=ob.sources,
        additional_instruction=ob_add_ins
    )

    # 2) LMS Integration
    lms = reqs.lms_integration or LMSRequirement()
    if lms.lms_names:
        lms_names_str = ", ".join(lms.lms_names)
        lms_claim_default = _build_claim(
            default_template="{platform} integrates with at least one major LMS, specifically: " + lms_names_str + ".",
            platform_name=platform_name,
            fallback_if_missing_claim=None
        )
    else:
        lms_claim_default = _build_claim(
            default_template="{platform} integrates with at least one major LMS: Canvas, Moodle, or Blackboard.",
            platform_name=platform_name,
            fallback_if_missing_claim=None
        )
    lms_add_ins = (
        "Look for explicit integration mentions with Canvas (Instructure Canvas), Moodle, or Blackboard (Blackboard Learn). "
        "References to LTI-based integrations are acceptable if the page clearly ties them to one of these LMSes."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="LMSIntegration",
        req_group_desc="LMS integration requirement is satisfied and evidenced.",
        meets_id="LMSIntegration_MeetsRequirement",
        meets_desc="Platform integrates with at least one major LMS: Canvas, Moodle, or Blackboard.",
        evidence_id="LMSIntegration_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting the stated LMS integration(s).",
        claim_text=lms.claim or lms_claim_default,
        sources=lms.sources,
        additional_instruction=lms_add_ins
    )

    # 3) API-based integration
    api = reqs.api_support or RequirementEvidence()
    api_claim_default = _build_claim(
        default_template="{platform} provides API-based integration capabilities enabling automated credential issuance.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    api_add_ins = (
        "Verify mentions of APIs (e.g., REST API, GraphQL), endpoints, or automation features specifically enabling credential issuance."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="APISupport",
        req_group_desc="API-based integration requirement is satisfied and evidenced.",
        meets_id="APISupport_MeetsRequirement",
        meets_desc="Platform provides API-based integration capabilities enabling automated credential issuance.",
        evidence_id="APISupport_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting API-based automated issuance capabilities.",
        claim_text=api.claim or api_claim_default,
        sources=api.sources,
        additional_instruction=api_add_ins
    )

    # 4) Verification mechanisms
    ver = reqs.verification_mechanism or RequirementEvidence()
    ver_claim_default = _build_claim(
        default_template="{platform} includes mechanisms to verify issued credentials.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    ver_add_ins = (
        "Check for features such as credential verification/validation pages, hosted verification, cryptographic signatures, or blockchain-based proofs."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="VerificationMechanism",
        req_group_desc="Verification mechanism requirement is satisfied and evidenced.",
        meets_id="VerificationMechanism_MeetsRequirement",
        meets_desc="Platform includes mechanisms to verify issued credentials.",
        evidence_id="VerificationMechanism_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources describing credential verification features.",
        claim_text=ver.claim or ver_claim_default,
        sources=ver.sources,
        additional_instruction=ver_add_ins
    )

    # 5) Metadata embedding
    meta = reqs.metadata_embedding or RequirementEvidence()
    meta_claim_default = _build_claim(
        default_template="{platform} supports embedding metadata in credentials, including issuer information, criteria, and achievement details.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    meta_add_ins = (
        "Look for explicit references to Open Badges metadata fields (issuer, criteria, evidence, alignments) or equivalent metadata support in credentials."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="MetadataEmbedding",
        req_group_desc="Metadata embedding requirement is satisfied and evidenced.",
        meets_id="MetadataEmbedding_MeetsRequirement",
        meets_desc="Platform supports embedding metadata in credentials (issuer information, criteria, achievement details).",
        evidence_id="MetadataEmbedding_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting metadata embedding capabilities.",
        claim_text=meta.claim or meta_claim_default,
        sources=meta.sources,
        additional_instruction=meta_add_ins
    )

    # 6) Shareability/Portability
    port = reqs.portability or RequirementEvidence()
    port_claim_default = _build_claim(
        default_template="{platform} allows credentials to be shareable and portable across different platforms and systems.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    port_add_ins = (
        "Accept explicit statements about interoperability, sharing (e.g., LinkedIn), export, or portability across systems, including Open Badges portability."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="CrossPlatformPortability",
        req_group_desc="Shareability/portability requirement is satisfied and evidenced.",
        meets_id="CrossPlatformPortability_MeetsRequirement",
        meets_desc="Platform allows credentials to be shareable and portable across different platforms/systems.",
        evidence_id="CrossPlatformPortability_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting portability/shareability/interoperability.",
        claim_text=port.claim or port_claim_default,
        sources=port.sources,
        additional_instruction=port_add_ins
    )

    # 7) STEM QA compatibility (ABET)
    qa = reqs.qa_compatibility or RequirementEvidence()
    qa_claim_default = _build_claim(
        default_template="For STEM-related programs, {platform} is compatible with quality assurance standards such as ABET Recognition of Credentials.",
        platform_name=platform_name,
        fallback_if_missing_claim=None
    )
    qa_add_ins = (
        "Verify explicit compatibility or alignment with quality assurance frameworks such as ABET Recognition of Credentials. "
        "Accept clear, direct statements of ABET alignment or compliance, or documentation that the platform supports mappings/reporting required for ABET recognition."
    )
    await _add_requirement_check(
        evaluator=evaluator,
        parent_node=requirements_node,
        req_group_id="QualityAssuranceCompatibility",
        req_group_desc="STEM quality assurance compatibility requirement is satisfied and evidenced.",
        meets_id="QualityAssuranceCompatibility_MeetsRequirement",
        meets_desc="For STEM-related programs, platform is compatible with quality assurance standards such as ABET Recognition of Credentials.",
        evidence_id="QualityAssuranceCompatibility_EvidenceProvided",
        evidence_desc="Provides specific cited evidence (URL/reference) from official documentation or reliable sources supporting the claimed compatibility with ABET Recognition of Credentials or the referenced quality assurance standard.",
        claim_text=qa.claim or qa_claim_default,
        sources=qa.sources,
        additional_instruction=qa_add_ins
    )

    return evaluator.get_summary()