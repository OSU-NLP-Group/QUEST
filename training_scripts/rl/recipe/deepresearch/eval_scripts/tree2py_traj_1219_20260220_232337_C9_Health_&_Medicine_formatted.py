import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "preventive_plan_tx_sa"
TASK_DESCRIPTION = """
A 67-year-old female Medicare beneficiary has recently relocated to San Antonio, Texas, and needs to establish comprehensive preventive healthcare services in her new area. She requires assistance in identifying appropriate healthcare facilities and providers in San Antonio or Bexar County, Texas for the following five Medicare-covered preventive care services:

1. Annual Wellness Visit Provider
2. Colonoscopy Screening Facility
3. Mammography Screening Center
4. Cardiovascular Disease Screening Provider
5. Adult Immunization Services Facility

For each, provide the name and address, verify Medicare acceptance (where applicable), confirm the specific preventive service is offered, confirm Medicare coverage at no cost (where applicable), and supply reference URLs confirming location, acceptance, service availability, and coverage policy. All information must be based on publicly available sources.
"""


# ------------------------------ Data Models ------------------------------ #
class AWVInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    medicare_acceptance_urls: List[str] = Field(default_factory=list)
    awv_service_urls: List[str] = Field(default_factory=list)
    coverage_policy_urls: List[str] = Field(default_factory=list)  # e.g., Medicare.gov AWV page


class ColonoscopyInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    medicare_certification_urls: List[str] = Field(default_factory=list)  # e.g., Care Compare, CMS directory
    medicare_acceptance_urls: List[str] = Field(default_factory=list)
    screening_service_urls: List[str] = Field(default_factory=list)
    coverage_policy_urls: List[str] = Field(default_factory=list)  # e.g., Medicare CRC screening policy


class MammographyInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    medicare_certification_urls: List[str] = Field(default_factory=list)
    medicare_acceptance_urls: List[str] = Field(default_factory=list)
    screening_service_urls: List[str] = Field(default_factory=list)
    coverage_policy_urls: List[str] = Field(default_factory=list)  # e.g., Medicare mammography coverage


class CardiovascularInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    medicare_acceptance_urls: List[str] = Field(default_factory=list)
    testing_capability_urls: List[str] = Field(default_factory=list)  # cholesterol, lipid, triglyceride labs
    screening_service_urls: List[str] = Field(default_factory=list)
    coverage_policy_urls: List[str] = Field(default_factory=list)  # e.g., Medicare CV screening every 5 years


class ImmunizationInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    program_participation_urls: List[str] = Field(default_factory=list)  # Medicare vaccines or Texas Adult Safety Net
    accessibility_urls: List[str] = Field(default_factory=list)  # access for Medicare beneficiaries/uninsured adults
    vaccines_offered_urls: List[str] = Field(default_factory=list)
    operations_urls: List[str] = Field(default_factory=list)  # hours/contact/how to access services


class HealthcarePlanExtraction(BaseModel):
    awv: Optional[AWVInfo] = None
    colonoscopy: Optional[ColonoscopyInfo] = None
    mammography: Optional[MammographyInfo] = None
    cardiovascular: Optional[CardiovascularInfo] = None
    immunization: Optional[ImmunizationInfo] = None


# ------------------------------ Extraction Prompt ------------------------------ #
def prompt_extract_healthcare_plan() -> str:
    return """
Extract one provider/facility in San Antonio or Bexar County, Texas for each of the five preventive service categories listed below. Return exactly one item per category (if multiple are mentioned, pick the first clearly suitable one). Extract the following fields for each category, using only information explicitly present in the answer text. If a field is missing, set it to null (for strings) or an empty array (for URLs).

GENERAL URL RULES:
- Extract only valid URLs explicitly present in the answer (plain or markdown link). Do not invent URLs.
- Prefer official or authoritative sources (e.g., Medicare.gov, CMS.gov, state/county health department, provider/facility official sites).
- If a URL lacks protocol, prepend http://.

For each category, extract:

1) awv (Annual Wellness Visit Provider):
- name: provider name
- address: street address (including city/state), if given
- location_urls: URLs confirming the provider location in San Antonio/Bexar County
- medicare_acceptance_urls: URLs confirming the provider accepts Medicare
- awv_service_urls: URLs confirming the provider offers Annual Wellness Visit (Medicare wellness visit/AWV)
- coverage_policy_urls: URLs confirming Medicare policy that AWV is covered at no cost under Part B (prefer .gov)

2) colonoscopy (Colonoscopy Screening Facility):
- name
- address
- location_urls: URLs confirming facility location in San Antonio/Bexar County
- medicare_certification_urls: URLs confirming Medicare certification/participation relevant to colonoscopy (e.g., CMS/Medicare directory, Care Compare)
- medicare_acceptance_urls: URLs confirming the facility accepts Medicare patients
- screening_service_urls: URLs confirming screening colonoscopy services are offered
- coverage_policy_urls: URLs confirming Medicare coverage policy for screening colonoscopy (prefer official sources)

3) mammography (Mammography Screening Center):
- name
- address
- location_urls
- medicare_certification_urls: URLs confirming Medicare certification/participation relevant to mammography (e.g., CMS/Medicare directory, MQSA if mentioned)
- medicare_acceptance_urls
- screening_service_urls: URLs confirming screening mammography services are offered
- coverage_policy_urls: URLs confirming Medicare covers annual screening mammograms at no cost for women ≥40

4) cardiovascular (Cardiovascular Disease Screening Provider):
- name
- address
- location_urls
- medicare_acceptance_urls
- testing_capability_urls: URLs confirming capability to perform blood tests for cholesterol, lipid, triglyceride screening
- screening_service_urls: URLs confirming cardiovascular disease screening blood tests are offered
- coverage_policy_urls: URLs confirming Medicare covers these screening blood tests once every 5 years at no cost

5) immunization (Adult Immunization Services Facility):
- name
- address
- location_urls
- program_participation_urls: URLs confirming participation in Medicare vaccine coverage or Texas Adult Safety Net
- accessibility_urls: URLs confirming services accessible to Medicare beneficiaries or uninsured adults
- vaccines_offered_urls: URLs confirming adult vaccines offered (e.g., flu, pneumonia, ACIP-recommended vaccines)
- operations_urls: URLs with operating hours/contact/how to access services

If the answer lists more than one option per category, choose the first strong match. If fewer categories are present, still return the data for those present; set missing categories to null.
"""


# ------------------------------ Helper for additional instructions ------------------------------ #
def require_sources_instruction(extra: str = "") -> str:
    base = (
        "Only consider the claim supported if the provided URL(s) explicitly and clearly support it. "
        "If no URL source is provided for this verification, treat the claim as NOT supported."
    )
    if extra:
        return base + " " + extra
    return base


# ------------------------------ Verification per Category ------------------------------ #
async def verify_awv(evaluator: Evaluator, parent_node, awv: Optional[AWVInfo]) -> None:
    cat_node = evaluator.add_sequential(
        id="Annual_Wellness_Visit_Service",
        desc="Establish Annual Wellness Visit service with an appropriate provider in San Antonio/Bexar County",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id="AWV_Provider_Identification",
        desc="Identify a healthcare provider that offers Annual Wellness Visits in San Antonio or Bexar County",
        parent=cat_node,
        critical=True
    )
    basic_info_node = evaluator.add_parallel(
        id="AWV_Provider_Information",
        desc="Basic provider identification information",
        parent=ident_node,
        critical=True
    )

    name_addr_ok = (awv is not None) and (awv.name is not None and awv.name.strip()) and (awv.address is not None and awv.address.strip())
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="AWV_Provider_Name_Address",
        desc="Provide the name and address of the healthcare provider",
        parent=basic_info_node,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id="AWV_Location_Reference",
        desc="Valid reference URL from search results confirming the provider location in San Antonio/Bexar County",
        parent=basic_info_node,
        critical=True
    )
    loc_claim = f"The provider '{awv.name if awv else ''}' at address '{awv.address if awv else ''}' is located in San Antonio or Bexar County, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(awv.location_urls if awv else []),
        additional_instruction=require_sources_instruction("Prefer official or authoritative sources. Allow minor formatting differences in the address.")
    )

    # Requirements Verification
    reqs_node = evaluator.add_parallel(
        id="AWV_Requirements_Verification",
        desc="Verify the provider meets all Medicare Annual Wellness Visit requirements",
        parent=cat_node,
        critical=False
    )

    # Authorization
    auth_node = evaluator.add_parallel(
        id="AWV_Authorization_Requirements",
        desc="Verify provider authorization and certification",
        parent=reqs_node,
        critical=True
    )

    acc_leaf = evaluator.add_leaf(
        id="AWV_Medicare_Acceptance",
        desc="The provider accepts Medicare patients",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider accepts Medicare.",
        node=acc_leaf,
        sources=(awv.medicare_acceptance_urls if awv else []),
        additional_instruction=require_sources_instruction("The source should explicitly state Medicare acceptance/participation.")
    )

    cert_leaf = evaluator.add_leaf(
        id="AWV_Service_Certification",
        desc="The provider is certified to perform Annual Wellness Visits as defined by Medicare",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider offers Medicare Annual Wellness Visit (AWV) compliant with Medicare definitions.",
        node=cert_leaf,
        sources=(awv.awv_service_urls if awv else []),
        additional_instruction=require_sources_instruction("Look for explicit mention of 'Annual Wellness Visit' or 'Medicare Wellness Visit' on provider or authoritative pages.")
    )

    # Service Specifications
    svc_node = evaluator.add_parallel(
        id="AWV_Service_Specifications",
        desc="Verify specific service availability and scope",
        parent=reqs_node,
        critical=True
    )

    offered_leaf = evaluator.add_leaf(
        id="AWV_Service_Offered",
        desc="The provider explicitly offers Annual Wellness Visit services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider offers Annual Wellness Visit services.",
        node=offered_leaf,
        sources=(awv.awv_service_urls if awv else []),
        additional_instruction=require_sources_instruction("The page should clearly state AWV or Medicare wellness visit is available.")
    )

    svc_ref_leaf = evaluator.add_leaf(
        id="AWV_Service_Reference",
        desc="Valid reference URL from search results confirming AWV service availability",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm Annual Wellness Visit service availability.",
        node=svc_ref_leaf,
        sources=(awv.awv_service_urls if awv else []),
        additional_instruction=require_sources_instruction("Reject if the URLs do not explicitly mention AWV or equivalent.")
    )

    # Coverage Requirements
    cov_node = evaluator.add_parallel(
        id="AWV_Coverage_Requirements",
        desc="Verify Medicare coverage and cost information",
        parent=reqs_node,
        critical=True
    )

    nocost_leaf = evaluator.add_leaf(
        id="AWV_No_Cost_Coverage",
        desc="The Annual Wellness Visit is covered at no cost under Medicare Part B preventive care benefits",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="Medicare Part B covers the Annual Wellness Visit at no cost as preventive care when provided by a participating provider.",
        node=nocost_leaf,
        sources=(awv.coverage_policy_urls if awv else []),
        additional_instruction=require_sources_instruction("Prefer Medicare.gov/CMS.gov. Minor wording differences are acceptable if the policy is clear.")
    )

    cov_ref_leaf = evaluator.add_leaf(
        id="AWV_Coverage_Reference",
        desc="Valid reference URL confirming Medicare coverage policy",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) describe Medicare's coverage policy for Annual Wellness Visits.",
        node=cov_ref_leaf,
        sources=(awv.coverage_policy_urls if awv else []),
        additional_instruction=require_sources_instruction("Prefer .gov domains (Medicare.gov or CMS.gov).")
    )


async def verify_colonoscopy(evaluator: Evaluator, parent_node, col: Optional[ColonoscopyInfo]) -> None:
    cat_node = evaluator.add_sequential(
        id="Colonoscopy_Screening_Service",
        desc="Establish colonoscopy screening service with an appropriate facility in San Antonio/Bexar County",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id="Colonoscopy_Facility_Identification",
        desc="Identify a healthcare facility that provides colonoscopy screening in San Antonio or Bexar County",
        parent=cat_node,
        critical=True
    )
    basic_info_node = evaluator.add_parallel(
        id="Colonoscopy_Facility_Information",
        desc="Basic facility identification information",
        parent=ident_node,
        critical=True
    )
    name_addr_ok = (col is not None) and (col.name and col.name.strip()) and (col.address and col.address.strip())
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="Colonoscopy_Facility_Name_Address",
        desc="Provide the name and address of the healthcare facility",
        parent=basic_info_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Colonoscopy_Location_Reference",
        desc="Valid reference URL from search results confirming the facility location in San Antonio/Bexar County",
        parent=basic_info_node,
        critical=True
    )
    loc_claim = f"The facility '{col.name if col else ''}' at '{col.address if col else ''}' is located in San Antonio or Bexar County, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(col.location_urls if col else []),
        additional_instruction=require_sources_instruction("Prefer official facility pages or recognized directories.")
    )

    # Requirements Verification
    reqs_node = evaluator.add_parallel(
        id="Colonoscopy_Requirements_Verification",
        desc="Verify the facility meets all Medicare colonoscopy screening requirements",
        parent=cat_node,
        critical=False
    )

    # Authorization
    auth_node = evaluator.add_parallel(
        id="Colonoscopy_Authorization_Requirements",
        desc="Verify facility authorization and certification",
        parent=reqs_node,
        critical=True
    )
    cert_leaf = evaluator.add_leaf(
        id="Colonoscopy_Medicare_Certification",
        desc="The facility is Medicare-certified for colonoscopy procedures",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility participates in/ is certified by Medicare for colonoscopy-related procedures.",
        node=cert_leaf,
        sources=(col.medicare_certification_urls if col else []),
        additional_instruction=require_sources_instruction("Evidence may include Medicare/CMS directories or Care Compare pages.")
    )
    acc_leaf = evaluator.add_leaf(
        id="Colonoscopy_Medicare_Acceptance",
        desc="The facility accepts Medicare patients for colonoscopy screening",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility accepts Medicare patients for colonoscopy screening.",
        node=acc_leaf,
        sources=(col.medicare_acceptance_urls if col else []),
        additional_instruction=require_sources_instruction("The page should explicitly indicate Medicare acceptance/participation.")
    )

    # Service Specifications
    svc_node = evaluator.add_parallel(
        id="Colonoscopy_Service_Specifications",
        desc="Verify specific colonoscopy service availability and scope",
        parent=reqs_node,
        critical=True
    )
    offered_leaf = evaluator.add_leaf(
        id="Colonoscopy_Screening_Offered",
        desc="The facility provides screening colonoscopy services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility provides screening colonoscopy services.",
        node=offered_leaf,
        sources=(col.screening_service_urls if col else []),
        additional_instruction=require_sources_instruction("The page should clearly mention screening colonoscopy availability.")
    )
    svc_ref_leaf = evaluator.add_leaf(
        id="Colonoscopy_Service_Reference",
        desc="Valid reference URL from search results confirming colonoscopy screening services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm the availability of screening colonoscopy services.",
        node=svc_ref_leaf,
        sources=(col.screening_service_urls if col else []),
        additional_instruction=require_sources_instruction()
    )

    # Coverage Requirements
    cov_node = evaluator.add_parallel(
        id="Colonoscopy_Coverage_Requirements",
        desc="Verify Medicare coverage for colonoscopy screening",
        parent=reqs_node,
        critical=True
    )
    age_leaf = evaluator.add_leaf(
        id="Colonoscopy_Age_Coverage",
        desc="The facility offers screening colonoscopy covered by Medicare for individuals aged 45 and older",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="Medicare covers screening colonoscopy for individuals aged 45 and older.",
        node=age_leaf,
        sources=(col.coverage_policy_urls if col else []),
        additional_instruction=require_sources_instruction("Prefer Medicare/CMS sources; minor phrasing differences acceptable.")
    )
    cov_ref_leaf = evaluator.add_leaf(
        id="Colonoscopy_Coverage_Reference",
        desc="Valid reference URL confirming Medicare coverage for screening colonoscopy",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) describe Medicare coverage policy for screening colonoscopy.",
        node=cov_ref_leaf,
        sources=(col.coverage_policy_urls if col else []),
        additional_instruction=require_sources_instruction("Prefer .gov domains.")
    )


async def verify_mammography(evaluator: Evaluator, parent_node, mam: Optional[MammographyInfo]) -> None:
    cat_node = evaluator.add_sequential(
        id="Mammography_Screening_Service",
        desc="Establish mammography screening service with an appropriate center in San Antonio/Bexar County",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id="Mammography_Center_Identification",
        desc="Identify a healthcare center that provides mammography screening in San Antonio or Bexar County",
        parent=cat_node,
        critical=True
    )
    basic_info_node = evaluator.add_parallel(
        id="Mammography_Center_Information",
        desc="Basic center identification information",
        parent=ident_node,
        critical=True
    )
    name_addr_ok = (mam is not None) and (mam.name and mam.name.strip()) and (mam.address and mam.address.strip())
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="Mammography_Center_Name_Address",
        desc="Provide the name and address of the mammography center",
        parent=basic_info_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Mammography_Location_Reference",
        desc="Valid reference URL from search results confirming the center location in San Antonio/Bexar County",
        parent=basic_info_node,
        critical=True
    )
    loc_claim = f"The center '{mam.name if mam else ''}' at '{mam.address if mam else ''}' is located in San Antonio or Bexar County, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(mam.location_urls if mam else []),
        additional_instruction=require_sources_instruction("Prefer official center pages or recognized directories.")
    )

    # Requirements Verification
    reqs_node = evaluator.add_parallel(
        id="Mammography_Requirements_Verification",
        desc="Verify the center meets all Medicare mammography screening requirements",
        parent=cat_node,
        critical=False
    )

    # Authorization
    auth_node = evaluator.add_parallel(
        id="Mammography_Authorization_Requirements",
        desc="Verify center authorization and certification",
        parent=reqs_node,
        critical=True
    )
    cert_leaf = evaluator.add_leaf(
        id="Mammography_Medicare_Certification",
        desc="The center is Medicare-certified for mammography services",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This center participates in/ is certified by Medicare for mammography services.",
        node=cert_leaf,
        sources=(mam.medicare_certification_urls if mam else []),
        additional_instruction=require_sources_instruction("Evidence may include CMS/Medicare directories; MQSA certification mention is acceptable.")
    )
    acc_leaf = evaluator.add_leaf(
        id="Mammography_Medicare_Acceptance",
        desc="The center accepts Medicare patients for mammography screening",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This center accepts Medicare patients for mammography screening.",
        node=acc_leaf,
        sources=(mam.medicare_acceptance_urls if mam else []),
        additional_instruction=require_sources_instruction("Look for explicit Medicare acceptance/participation statements.")
    )

    # Service Specifications
    svc_node = evaluator.add_parallel(
        id="Mammography_Service_Specifications",
        desc="Verify specific mammography service availability and scope",
        parent=reqs_node,
        critical=True
    )
    offered_leaf = evaluator.add_leaf(
        id="Mammography_Screening_Offered",
        desc="The center provides screening mammography services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This center provides screening mammography services.",
        node=offered_leaf,
        sources=(mam.screening_service_urls if mam else []),
        additional_instruction=require_sources_instruction("The page should clearly mention screening mammograms.")
    )
    svc_ref_leaf = evaluator.add_leaf(
        id="Mammography_Service_Reference",
        desc="Valid reference URL from search results confirming mammography screening services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm screening mammography services are available.",
        node=svc_ref_leaf,
        sources=(mam.screening_service_urls if mam else []),
        additional_instruction=require_sources_instruction()
    )

    # Coverage Requirements
    cov_node = evaluator.add_parallel(
        id="Mammography_Coverage_Requirements",
        desc="Verify Medicare coverage for mammography screening",
        parent=reqs_node,
        critical=True
    )
    annual_leaf = evaluator.add_leaf(
        id="Mammography_Annual_Coverage",
        desc="The center offers annual screening mammograms covered by Medicare at no cost for women aged 40 and older",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="Medicare covers annual screening mammograms at no cost for women aged 40 and older.",
        node=annual_leaf,
        sources=(mam.coverage_policy_urls if mam else []),
        additional_instruction=require_sources_instruction("Prefer Medicare/CMS pages; allow small wording variations.")
    )
    cov_ref_leaf = evaluator.add_leaf(
        id="Mammography_Coverage_Reference",
        desc="Valid reference URL confirming Medicare coverage for screening mammography",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) describe Medicare's coverage policy for screening mammography.",
        node=cov_ref_leaf,
        sources=(mam.coverage_policy_urls if mam else []),
        additional_instruction=require_sources_instruction("Prefer .gov domains.")
    )


async def verify_cardiovascular(evaluator: Evaluator, parent_node, cv: Optional[CardiovascularInfo]) -> None:
    cat_node = evaluator.add_sequential(
        id="Cardiovascular_Screening_Service",
        desc="Establish cardiovascular disease screening service with an appropriate provider in San Antonio/Bexar County",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id="Cardiovascular_Provider_Identification",
        desc="Identify a healthcare provider that offers cardiovascular disease screening blood tests in San Antonio or Bexar County",
        parent=cat_node,
        critical=True
    )
    basic_info_node = evaluator.add_parallel(
        id="Cardiovascular_Provider_Information",
        desc="Basic provider identification information",
        parent=ident_node,
        critical=True
    )
    name_addr_ok = (cv is not None) and (cv.name and cv.name.strip()) and (cv.address and cv.address.strip())
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="Cardiovascular_Provider_Name_Address",
        desc="Provide the name and address of the healthcare provider",
        parent=basic_info_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Cardiovascular_Location_Reference",
        desc="Valid reference URL from search results confirming the provider location in San Antonio/Bexar County",
        parent=basic_info_node,
        critical=True
    )
    loc_claim = f"The provider '{cv.name if cv else ''}' at '{cv.address if cv else ''}' is located in San Antonio or Bexar County, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(cv.location_urls if cv else []),
        additional_instruction=require_sources_instruction("Prefer official provider pages or recognized directories.")
    )

    # Requirements Verification
    reqs_node = evaluator.add_parallel(
        id="Cardiovascular_Requirements_Verification",
        desc="Verify the provider meets all Medicare cardiovascular screening requirements",
        parent=cat_node,
        critical=False
    )

    # Authorization
    auth_node = evaluator.add_parallel(
        id="Cardiovascular_Authorization_Requirements",
        desc="Verify provider authorization to perform cardiovascular screening",
        parent=reqs_node,
        critical=True
    )
    acc_leaf = evaluator.add_leaf(
        id="Cardiovascular_Medicare_Acceptance",
        desc="The provider accepts Medicare for cardiovascular screening tests",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider accepts Medicare for cardiovascular screening tests.",
        node=acc_leaf,
        sources=(cv.medicare_acceptance_urls if cv else []),
        additional_instruction=require_sources_instruction("Look for explicit Medicare acceptance/participation statements.")
    )
    cap_leaf = evaluator.add_leaf(
        id="Cardiovascular_Testing_Capability",
        desc="The provider has capability to perform blood tests for cholesterol, lipid, and triglyceride screening",
        parent=auth_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider can perform blood tests for cholesterol, lipid, and triglyceride screening.",
        node=cap_leaf,
        sources=(cv.testing_capability_urls if cv else []),
        additional_instruction=require_sources_instruction("The page should explicitly mention these lab tests or panels.")
    )

    # Service Specifications
    svc_node = evaluator.add_parallel(
        id="Cardiovascular_Service_Specifications",
        desc="Verify specific cardiovascular screening service availability",
        parent=reqs_node,
        critical=True
    )
    offered_leaf = evaluator.add_leaf(
        id="Cardiovascular_Screening_Offered",
        desc="The provider offers cardiovascular disease screening blood tests",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This provider offers cardiovascular disease screening blood tests.",
        node=offered_leaf,
        sources=(cv.screening_service_urls if cv else []),
        additional_instruction=require_sources_instruction("The page should clearly indicate CV screening blood tests availability.")
    )
    svc_ref_leaf = evaluator.add_leaf(
        id="Cardiovascular_Service_Reference",
        desc="Valid reference URL from search results confirming cardiovascular screening services",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm cardiovascular screening blood tests are available.",
        node=svc_ref_leaf,
        sources=(cv.screening_service_urls if cv else []),
        additional_instruction=require_sources_instruction()
    )

    # Coverage Requirements
    cov_node = evaluator.add_parallel(
        id="Cardiovascular_Coverage_Requirements",
        desc="Verify Medicare coverage for cardiovascular screening",
        parent=reqs_node,
        critical=True
    )
    freq_leaf = evaluator.add_leaf(
        id="Cardiovascular_Coverage_Frequency",
        desc="The screening is covered by Medicare once every 5 years at no cost as preventive care",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="Medicare covers screening blood tests for cardiovascular disease once every 5 years at no cost as preventive care.",
        node=freq_leaf,
        sources=(cv.coverage_policy_urls if cv else []),
        additional_instruction=require_sources_instruction("Prefer Medicare/CMS sources.")
    )
    cov_ref_leaf = evaluator.add_leaf(
        id="Cardiovascular_Coverage_Reference",
        desc="Valid reference URL confirming Medicare coverage policy for cardiovascular screening",
        parent=cov_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided source(s) describe Medicare's coverage policy for cardiovascular screening blood tests.",
        node=cov_ref_leaf,
        sources=(cv.coverage_policy_urls if cv else []),
        additional_instruction=require_sources_instruction("Prefer .gov domains.")
    )


async def verify_immunization(evaluator: Evaluator, parent_node, imm: Optional[ImmunizationInfo]) -> None:
    cat_node = evaluator.add_sequential(
        id="Immunization_Services",
        desc="Establish adult immunization services with an appropriate facility in San Antonio/Bexar County",
        parent=parent_node,
        critical=False
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id="Immunization_Facility_Identification",
        desc="Identify a facility that provides adult immunization services in San Antonio or Bexar County",
        parent=cat_node,
        critical=True
    )
    basic_info_node = evaluator.add_parallel(
        id="Immunization_Facility_Information",
        desc="Basic facility identification information",
        parent=ident_node,
        critical=True
    )
    name_addr_ok = (imm is not None) and (imm.name and imm.name.strip()) and (imm.address and imm.address.strip())
    evaluator.add_custom_node(
        result=name_addr_ok,
        id="Immunization_Facility_Name_Address",
        desc="Provide the name and address of the immunization facility",
        parent=basic_info_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="Immunization_Location_Reference",
        desc="Valid reference URL from search results confirming the facility location in San Antonio/Bexar County",
        parent=basic_info_node,
        critical=True
    )
    loc_claim = f"The facility '{imm.name if imm else ''}' at '{imm.address if imm else ''}' is located in San Antonio or Bexar County, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(imm.location_urls if imm else []),
        additional_instruction=require_sources_instruction("Prefer official facility pages or recognized directories.")
    )

    # Requirements Verification
    reqs_node = evaluator.add_parallel(
        id="Immunization_Requirements_Verification",
        desc="Verify the facility meets all immunization service requirements",
        parent=cat_node,
        critical=False
    )

    # Access Requirements
    access_node = evaluator.add_parallel(
        id="Immunization_Access_Requirements",
        desc="Verify facility accessibility for Medicare beneficiaries or uninsured adults",
        parent=reqs_node,
        critical=True
    )
    prog_leaf = evaluator.add_leaf(
        id="Immunization_Program_Participation",
        desc="The facility participates in Medicare vaccine coverage or Texas Adult Safety Net program",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility participates in Medicare vaccine coverage or the Texas Adult Safety Net program.",
        node=prog_leaf,
        sources=(imm.program_participation_urls if imm else []),
        additional_instruction=require_sources_instruction("Look for explicit program participation statements on official pages.")
    )
    access_leaf = evaluator.add_leaf(
        id="Immunization_Accessibility",
        desc="The facility provides immunization services accessible to Medicare beneficiaries or uninsured adults",
        parent=access_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility provides immunization services accessible to Medicare beneficiaries or uninsured adults.",
        node=access_leaf,
        sources=(imm.accessibility_urls if imm else []),
        additional_instruction=require_sources_instruction("The page should clearly state eligibility or access for Medicare beneficiaries/uninsured adults.")
    )

    # Service Specifications
    svc_node = evaluator.add_parallel(
        id="Immunization_Service_Specifications",
        desc="Verify specific immunization service availability",
        parent=reqs_node,
        critical=True
    )
    vaccines_leaf = evaluator.add_leaf(
        id="Immunization_Vaccines_Offered",
        desc="The facility offers adult vaccines including flu, pneumonia, or other ACIP-recommended vaccines",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility offers adult vaccines such as influenza, pneumonia, or other ACIP-recommended vaccines.",
        node=vaccines_leaf,
        sources=(imm.vaccines_offered_urls if imm else []),
        additional_instruction=require_sources_instruction("The page should list adult vaccines or describe adult immunization services.")
    )
    svc_ref_leaf = evaluator.add_leaf(
        id="Immunization_Service_Reference",
        desc="Valid reference URL from search results confirming immunization services availability",
        parent=svc_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources confirm adult immunization services are available.",
        node=svc_ref_leaf,
        sources=(imm.vaccines_offered_urls if imm else []),
        additional_instruction=require_sources_instruction()
    )

    # Operational Requirements
    op_node = evaluator.add_parallel(
        id="Immunization_Operational_Requirements",
        desc="Verify facility operational information and accessibility",
        parent=reqs_node,
        critical=True
    )
    hours_leaf = evaluator.add_leaf(
        id="Immunization_Hours_Availability",
        desc="The facility provides clear information about operating hours and how to access services",
        parent=op_node,
        critical=True
    )
    await evaluator.verify(
        claim="This facility provides clear information about operating hours and how to access immunization services.",
        node=hours_leaf,
        sources=(imm.operations_urls if imm else []),
        additional_instruction=require_sources_instruction("Look for posted hours, contact methods, appointment/access instructions.")
    )
    ops_ref_leaf = evaluator.add_leaf(
        id="Immunization_Operations_Reference",
        desc="Valid reference URL from search results confirming operational details and contact information",
        parent=op_node,
        critical=True
    )
    await evaluator.verify(
        claim="The provided sources present operational details and contact information.",
        node=ops_ref_leaf,
        sources=(imm.operations_urls if imm else []),
        additional_instruction=require_sources_instruction()
    )


# ------------------------------ Main Evaluation Function ------------------------------ #
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
        strategy=AggregationStrategy.PARALLEL,  # Overall plan aggregates five categories independently
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

    # Extract structured plan info from the answer
    plan = await evaluator.extract(
        prompt=prompt_extract_healthcare_plan(),
        template_class=HealthcarePlanExtraction,
        extraction_name="healthcare_plan_extraction"
    )

    # Build the comprehensive plan node (non-critical root already initialized)
    plan_node = evaluator.add_parallel(
        id="Comprehensive_Preventive_Healthcare_Plan",
        desc="Establish a comprehensive preventive healthcare service plan for a Medicare beneficiary in San Antonio, Texas, with identification and verification of appropriate Medicare-certified facilities and providers for five essential preventive care services",
        parent=root,
        critical=False
    )

    # AWV
    await verify_awv(evaluator, plan_node, plan.awv)

    # Colonoscopy
    await verify_colonoscopy(evaluator, plan_node, plan.colonoscopy)

    # Mammography
    await verify_mammography(evaluator, plan_node, plan.mammography)

    # Cardiovascular
    await verify_cardiovascular(evaluator, plan_node, plan.cardiovascular)

    # Immunization
    await verify_immunization(evaluator, plan_node, plan.immunization)

    return evaluator.get_summary()