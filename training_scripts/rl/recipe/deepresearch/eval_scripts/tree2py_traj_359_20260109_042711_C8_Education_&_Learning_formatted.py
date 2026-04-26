import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "lms_platform_selection_us_university"
TASK_DESCRIPTION = """
Identify a Learning Management System (LMS) platform suitable for a university in the United States that meets all of the following requirements:

Technical Requirements:
- Supports WCAG 2.1 Level AA accessibility standards
- Guarantees a minimum uptime SLA of 99.5%
- Offers cloud-based hosting option
- Provides native mobile applications for both iOS and Android platforms
- Provides RESTful API access for third-party integrations

Integration Standards:
- Supports LTI 1.3 (Learning Tools Interoperability) standard
- Supports at least one major eLearning content standard (SCORM or xAPI)
- Supports Single Sign-On (SSO) functionality using SAML or similar authentication protocols
- Integrates with at least one major video conferencing platform (Zoom or Microsoft Teams)
- Supports integration with content authoring tools such as H5P

Functional Requirements:
- Provides comprehensive learning analytics and reporting capabilities
- Includes gradebook functionality with grade passback capability
- Supports role-based access control with differentiated permissions for students, instructors, and administrators

Compliance Requirements:
- Compliant with FERPA (Family Educational Rights and Privacy Act) for student data protection

Provide the name of the LMS platform and supporting documentation that demonstrates it meets each of these requirements.
"""


class LMSSourcesExtraction(BaseModel):
    platform_name: Optional[str] = None
    platform_homepage_url: Optional[str] = None

    wcag_docs: List[str] = Field(default_factory=list)
    uptime_sla_docs: List[str] = Field(default_factory=list)
    cloud_hosting_docs: List[str] = Field(default_factory=list)
    mobile_ios_android_docs: List[str] = Field(default_factory=list)
    rest_api_docs: List[str] = Field(default_factory=list)

    lti13_docs: List[str] = Field(default_factory=list)
    content_standard_docs: List[str] = Field(default_factory=list)
    sso_docs: List[str] = Field(default_factory=list)
    video_integration_docs: List[str] = Field(default_factory=list)
    h5p_docs: List[str] = Field(default_factory=list)

    analytics_docs: List[str] = Field(default_factory=list)
    gradebook_docs: List[str] = Field(default_factory=list)
    rbac_docs: List[str] = Field(default_factory=list)

    ferpa_docs: List[str] = Field(default_factory=list)


def prompt_extract_lms_sources() -> str:
    return """
    You must extract the LMS platform name proposed in the answer and group the cited supporting documentation URLs for each requirement. Only extract URLs explicitly present in the answer (including plain URLs or URLs inside markdown links). Do NOT invent or infer any URL.

    Extract the following fields:
    - platform_name: The single, clearly identifiable LMS platform name being proposed. If multiple platforms are mentioned, choose the one explicitly recommended; if ambiguous, choose the first named one. If none provided, return null.
    - platform_homepage_url: If a homepage URL for the LMS is explicitly provided in the answer, extract it; otherwise return null.

    For each requirement below, extract an array of URLs cited in the answer that support the claim. If the answer does not provide any URL for a requirement, return an empty array for that requirement.

    Technical Requirements:
    - wcag_docs: URLs that demonstrate WCAG 2.1 Level AA support/compliance for the LMS.
    - uptime_sla_docs: URLs that state the uptime SLA commitment; aim for 99.5% or higher.
    - cloud_hosting_docs: URLs indicating the LMS offers cloud-based hosting/deployment.
    - mobile_ios_android_docs: URLs indicating native mobile apps for both iOS and Android (App Store/Play Store links or vendor docs).
    - rest_api_docs: URLs indicating RESTful API availability for integrations.

    Integration Standards:
    - lti13_docs: URLs indicating support for LTI 1.3 (or LTI Advantage).
    - content_standard_docs: URLs indicating support for SCORM or xAPI.
    - sso_docs: URLs indicating support for Single Sign-On using SAML or similar protocols (OAuth2/OpenID Connect).
    - video_integration_docs: URLs indicating integration with Zoom or Microsoft Teams (either one is acceptable).
    - h5p_docs: URLs indicating integration with H5P.

    Functional Requirements:
    - analytics_docs: URLs indicating comprehensive learning analytics/reporting capabilities.
    - gradebook_docs: URLs indicating gradebook functionality with grade passback capability (e.g., LTI AGS).
    - rbac_docs: URLs indicating role-based access control with differentiated permissions for students, instructors, administrators.

    Compliance Requirements:
    - ferpa_docs: URLs indicating FERPA compliance for student data protection.

    Return a single JSON object with the fields above. Use empty arrays for any requirement with no URLs provided in the answer.
    """


def _build_additional_instruction(requirement_name: str, platform_name: Optional[str], sources: List[str], extra_hint: str) -> str:
    if not sources:
        return (
            f"For the requirement '{requirement_name}', no source URLs were provided in the answer. "
            f"According to the task, claims must be supported by cited documentation. "
            f"You must mark this claim as NOT SUPPORTED."
        )
    base = (
        f"Verify the claim specifically for the LMS platform '{platform_name}' if named. "
        f"Focus on explicit statements on the provided page(s); allow reasonable synonyms. "
        f"If the webpage(s) do not clearly support the requirement, mark as NOT SUPPORTED. "
    )
    return base + extra_hint


async def _verify_technical_requirements(evaluator: Evaluator, parent, data: LMSSourcesExtraction) -> None:
    tech_node = evaluator.add_parallel(
        id="Technical_Requirements_With_Documentation",
        desc="Response demonstrates (with supporting documentation) that the LMS meets all technical requirements.",
        parent=parent,
        critical=True
    )

    platform = data.platform_name or "the LMS platform"

    wcag_leaf = evaluator.add_leaf(
        id="WCAG_Accessibility_Compliance",
        desc="Provides supporting documentation demonstrating the LMS supports WCAG 2.1 Level AA accessibility standards.",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports WCAG 2.1 Level AA accessibility.",
        node=wcag_leaf,
        sources=data.wcag_docs,
        additional_instruction=_build_additional_instruction(
            "WCAG 2.1 Level AA",
            data.platform_name,
            data.wcag_docs,
            "Prefer explicit mention of 'WCAG 2.1' and 'Level AA'. 'WCAG 2.0' or 'Level A' is insufficient."
        )
    )

    uptime_leaf = evaluator.add_leaf(
        id="Uptime_Service_Level",
        desc="Provides supporting documentation demonstrating the LMS guarantees a minimum uptime SLA of 99.5%.",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} guarantees an uptime SLA of at least 99.5%.",
        node=uptime_leaf,
        sources=data.uptime_sla_docs,
        additional_instruction=_build_additional_instruction(
            "Uptime SLA ≥ 99.5%",
            data.platform_name,
            data.uptime_sla_docs,
            "Accept commitments of 99.5% or higher (e.g., 99.9%). Look for terms like 'SLA', 'uptime', 'availability'."
        )
    )

    cloud_leaf = evaluator.add_leaf(
        id="Cloud_Hosting_Option",
        desc="Provides supporting documentation demonstrating the LMS offers a cloud-based hosting/deployment option.",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} offers a cloud-based hosting option.",
        node=cloud_leaf,
        sources=data.cloud_hosting_docs,
        additional_instruction=_build_additional_instruction(
            "Cloud hosting option",
            data.platform_name,
            data.cloud_hosting_docs,
            "Look for terms like 'cloud', 'SaaS', 'hosted', 'managed cloud', or vendor-operated hosting."
        )
    )

    mobile_leaf = evaluator.add_leaf(
        id="Mobile_Application_Support",
        desc="Provides supporting documentation demonstrating the LMS provides native mobile applications for both iOS and Android.",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} provides native mobile applications for both iOS and Android.",
        node=mobile_leaf,
        sources=data.mobile_ios_android_docs,
        additional_instruction=_build_additional_instruction(
            "Native iOS and Android apps",
            data.platform_name,
            data.mobile_ios_android_docs,
            "Mobile-responsive websites are insufficient; look for App Store/Google Play links or vendor docs stating native apps for both platforms."
        )
    )

    api_leaf = evaluator.add_leaf(
        id="API_Availability",
        desc="Provides supporting documentation demonstrating the LMS provides RESTful API access for third-party integrations.",
        parent=tech_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} provides RESTful API access for third-party integrations.",
        node=api_leaf,
        sources=data.rest_api_docs,
        additional_instruction=_build_additional_instruction(
            "RESTful API availability",
            data.platform_name,
            data.rest_api_docs,
            "Look for 'REST', 'RESTful', 'Open API' or similar phrasing. Purely GraphQL-only APIs do not satisfy this requirement."
        )
    )


async def _verify_integration_requirements(evaluator: Evaluator, parent, data: LMSSourcesExtraction) -> None:
    integ_node = evaluator.add_parallel(
        id="Integration_Standards_With_Documentation",
        desc="Response demonstrates (with supporting documentation) that the LMS supports all required integration standards.",
        parent=parent,
        critical=True
    )

    platform = data.platform_name or "the LMS platform"

    lti_leaf = evaluator.add_leaf(
        id="LTI_Standard_Support",
        desc="Provides supporting documentation demonstrating the LMS supports LTI 1.3.",
        parent=integ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports LTI 1.3 (LTI Advantage).",
        node=lti_leaf,
        sources=data.lti13_docs,
        additional_instruction=_build_additional_instruction(
            "LTI 1.3 support",
            data.platform_name,
            data.lti13_docs,
            "Explicitly look for 'LTI 1.3', 'LTI Advantage', or related conformance statements. 'LTI 1.1' alone is insufficient."
        )
    )

    content_leaf = evaluator.add_leaf(
        id="Content_Standard_Compliance",
        desc="Provides supporting documentation demonstrating the LMS supports at least one major eLearning content standard (SCORM or xAPI).",
        parent=integ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports SCORM or xAPI (Tin Can).",
        node=content_leaf,
        sources=data.content_standard_docs,
        additional_instruction=_build_additional_instruction(
            "Content standards (SCORM/xAPI)",
            data.platform_name,
            data.content_standard_docs,
            "Accept SCORM (1.2/2004) or xAPI (Tin Can). Either standard is sufficient to pass."
        )
    )

    sso_leaf = evaluator.add_leaf(
        id="Single_Sign_On",
        desc="Provides supporting documentation demonstrating the LMS supports SSO using SAML or similar authentication protocols.",
        parent=integ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports Single Sign-On using SAML or similar protocols (such as OAuth2/OpenID Connect).",
        node=sso_leaf,
        sources=data.sso_docs,
        additional_instruction=_build_additional_instruction(
            "SSO via SAML or similar",
            data.platform_name,
            data.sso_docs,
            "Accept SAML, OAuth2, OpenID Connect, or equivalent enterprise SSO mechanisms."
        )
    )

    video_leaf = evaluator.add_leaf(
        id="Video_Conferencing_Integration",
        desc="Provides supporting documentation demonstrating the LMS integrates with at least one major video conferencing platform (Zoom or Microsoft Teams).",
        parent=integ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} integrates with Zoom or Microsoft Teams.",
        node=video_leaf,
        sources=data.video_integration_docs,
        additional_instruction=_build_additional_instruction(
            "Video conferencing integration",
            data.platform_name,
            data.video_integration_docs,
            "Either Zoom or Microsoft Teams integration is sufficient to pass."
        )
    )

    h5p_leaf = evaluator.add_leaf(
        id="Content_Authoring_Tool_Integration",
        desc="Provides supporting documentation demonstrating the LMS supports integration with content authoring tools such as H5P.",
        parent=integ_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports integration with H5P.",
        node=h5p_leaf,
        sources=data.h5p_docs,
        additional_instruction=_build_additional_instruction(
            "H5P integration",
            data.platform_name,
            data.h5p_docs,
            "Look for explicit mention of H5P integration, plugins, or supported content types."
        )
    )


async def _verify_functional_requirements(evaluator: Evaluator, parent, data: LMSSourcesExtraction) -> None:
    func_node = evaluator.add_parallel(
        id="Functional_Requirements_With_Documentation",
        desc="Response demonstrates (with supporting documentation) that the LMS provides all required functional capabilities.",
        parent=parent,
        critical=True
    )

    platform = data.platform_name or "the LMS platform"

    analytics_leaf = evaluator.add_leaf(
        id="Learning_Analytics_Capabilities",
        desc="Provides supporting documentation demonstrating the LMS provides comprehensive learning analytics and reporting capabilities.",
        parent=func_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} provides comprehensive learning analytics and reporting capabilities.",
        node=analytics_leaf,
        sources=data.analytics_docs,
        additional_instruction=_build_additional_instruction(
            "Learning analytics",
            data.platform_name,
            data.analytics_docs,
            "Look for analytics dashboards, reporting tools, engagement metrics, or learning data insights."
        )
    )

    gradebook_leaf = evaluator.add_leaf(
        id="Gradebook_Functionality",
        desc="Provides supporting documentation demonstrating the LMS includes gradebook functionality with grade passback capability.",
        parent=func_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} includes a gradebook and supports grade passback capability.",
        node=gradebook_leaf,
        sources=data.gradebook_docs,
        additional_instruction=_build_additional_instruction(
            "Gradebook + grade passback",
            data.platform_name,
            data.gradebook_docs,
            "Grade passback may be provided via LTI Assignment and Grade Services (AGS) or equivalent; look for 'grade passback', 'LTI AGS', or similar phrasing."
        )
    )

    rbac_leaf = evaluator.add_leaf(
        id="Role_Based_Access_Management",
        desc="Provides supporting documentation demonstrating the LMS supports role-based access control with differentiated permissions for students, instructors, and administrators.",
        parent=func_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} supports role-based access control with differentiated permissions for students, instructors, and administrators.",
        node=rbac_leaf,
        sources=data.rbac_docs,
        additional_instruction=_build_additional_instruction(
            "Role-based access control (RBAC)",
            data.platform_name,
            data.rbac_docs,
            "Look for role definitions and permission differences among students, instructors, and administrators."
        )
    )


async def _verify_compliance_requirements(evaluator: Evaluator, parent, data: LMSSourcesExtraction) -> None:
    comp_node = evaluator.add_parallel(
        id="Compliance_Requirements_With_Documentation",
        desc="Response demonstrates (with supporting documentation) that the LMS meets all compliance requirements.",
        parent=parent,
        critical=True
    )

    platform = data.platform_name or "the LMS platform"

    ferpa_leaf = evaluator.add_leaf(
        id="FERPA_Data_Privacy_Compliance",
        desc="Provides supporting documentation demonstrating the LMS is compliant with FERPA for student data protection.",
        parent=comp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The LMS platform {platform} is compliant with FERPA for student data protection.",
        node=ferpa_leaf,
        sources=data.ferpa_docs,
        additional_instruction=_build_additional_instruction(
            "FERPA compliance",
            data.platform_name,
            data.ferpa_docs,
            "Look for explicit mention of FERPA compliance or adherence to U.S. student privacy regulations."
        )
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
        default_model=model
    )

    extraction = await evaluator.extract(
        prompt=prompt_extract_lms_sources(),
        template_class=LMSSourcesExtraction,
        extraction_name="lms_sources_extraction"
    )

    main_node = evaluator.add_parallel(
        id="LMS_Platform_Selection",
        desc="Identify an LMS platform suitable for a US university and provide supporting documentation demonstrating it meets each stated requirement.",
        parent=root,
        critical=True
    )

    platform_leaf = evaluator.add_custom_node(
        result=(extraction.platform_name is not None and extraction.platform_name.strip() != ""),
        id="Platform_Identification",
        desc="Response provides the name of the LMS platform being proposed (a single, clearly identifiable LMS platform).",
        parent=main_node,
        critical=True
    )

    await _verify_technical_requirements(evaluator, main_node, extraction)
    await _verify_integration_requirements(evaluator, main_node, extraction)
    await _verify_functional_requirements(evaluator, main_node, extraction)
    await _verify_compliance_requirements(evaluator, main_node, extraction)

    return evaluator.get_summary()