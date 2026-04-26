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
TASK_ID = "grad_research_infrastructure_2026"
TASK_DESCRIPTION = """
You are a first-year computational neuroscience Ph.D. student (U.S. citizen) at an accredited U.S. university, beginning your research in January 2026. You need to plan your research infrastructure and support for the coming year.

Please provide the following information:

1. Graduate Fellowship Program: Identify one U.S. federal graduate fellowship program specifically designed for computational science that you are eligible for and could support your computational neuroscience Ph.D. research. Include:
   - The fellowship name
   - Eligibility requirements (citizenship/residency, graduate year limitations, enrollment requirements)
   - Financial support details (annual stipend amount, tuition coverage, additional allowances, maximum years of support)
   - Application deadline for the 2026-27 fellowship cycle
   - Required application materials (including any page limits for CV/resume and letter requirements)
   - Official reference URL

2. NSF ACCESS Computing Allocation: Identify the most appropriate NSF ACCESS project allocation type for a graduate student just beginning computational research work. Include:
   - The ACCESS project type name
   - Typical approval timeline for this allocation type
   - Key application requirements (account setup, documentation, CV page limits)
   - Post-approval process for accessing computing resources
   - Official reference URL

3. Target Conference for 2025: Identify one major artificial intelligence or machine learning conference taking place in 2025 that would be suitable for submitting computational neuroscience research. Include:
   - Conference name, location, and dates (month and year)
   - Abstract submission deadline and full paper submission deadline (month, day, year)
   - Paper format requirements (page limit for main body, required format, review type)
   - Any additional requirements (checklists, supplementary materials)
   - Official conference reference URL

4. NSF Data Management Requirements: Describe the data management and sharing requirements for NSF-funded computational research. Include:
   - NSF data management plan requirements (including page limit)
   - Minimum research data retention period after project closeout
   - NSF's 2025 data sharing policy requirements
   - Official reference URL for these requirements

5. Timeline Coordination: Briefly analyze whether the identified fellowship application deadline, computing allocation timeline, and conference submission deadline can be reasonably coordinated for a student starting research in January 2026.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FellowshipEligibility(BaseModel):
    citizenship_residency: Optional[str] = None
    graduate_year_limitations: Optional[str] = None
    enrollment_requirements: Optional[str] = None


class FellowshipFinancial(BaseModel):
    annual_stipend_amount: Optional[str] = None
    tuition_coverage: Optional[str] = None
    additional_allowances: Optional[str] = None
    max_years_support: Optional[str] = None


class FellowshipMaterials(BaseModel):
    required_items: Optional[str] = None
    cv_page_limit: Optional[str] = None
    letter_requirements: Optional[str] = None


class FellowshipInfo(BaseModel):
    name: Optional[str] = None
    eligibility: Optional[FellowshipEligibility] = None
    financial: Optional[FellowshipFinancial] = None
    application_deadline_2026_27: Optional[str] = None
    materials: Optional[FellowshipMaterials] = None
    official_url: Optional[str] = None


class AccessRequirements(BaseModel):
    account_setup: Optional[str] = None
    documentation: Optional[str] = None
    cv_page_limit: Optional[str] = None


class AccessInfo(BaseModel):
    project_type_name: Optional[str] = None
    approval_timeline: Optional[str] = None
    key_requirements: Optional[AccessRequirements] = None
    post_approval_process: Optional[str] = None
    appropriateness_rationale: Optional[str] = None
    official_url: Optional[str] = None


class ConferenceFormat(BaseModel):
    main_body_page_limit: Optional[str] = None
    required_template_format: Optional[str] = None
    review_type: Optional[str] = None


class ConferenceInfo(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    dates: Optional[str] = None
    abstract_deadline: Optional[str] = None
    full_paper_deadline: Optional[str] = None
    paper_format: Optional[ConferenceFormat] = None
    additional_requirements: Optional[str] = None
    official_url: Optional[str] = None


class NSFDataMgmtInfo(BaseModel):
    dmp_page_limit: Optional[str] = None
    minimum_retention_period: Optional[str] = None
    data_sharing_policy_2025: Optional[str] = None
    official_url: Optional[str] = None


class TimelineCoordinationInfo(BaseModel):
    analysis_text: Optional[str] = None


class FullPlanExtraction(BaseModel):
    fellowship: Optional[FellowshipInfo] = None
    access: Optional[AccessInfo] = None
    conference: Optional[ConferenceInfo] = None
    nsf_data_mgmt: Optional[NSFDataMgmtInfo] = None
    timeline_coordination: Optional[TimelineCoordinationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_full() -> str:
    return """
    Extract the requested planning information exactly as stated in the answer. Return a JSON object with the following schema:

    {
      "fellowship": {
        "name": string or null,
        "eligibility": {
          "citizenship_residency": string or null,
          "graduate_year_limitations": string or null,
          "enrollment_requirements": string or null
        } or null,
        "financial": {
          "annual_stipend_amount": string or null,
          "tuition_coverage": string or null,
          "additional_allowances": string or null,
          "max_years_support": string or null
        } or null,
        "application_deadline_2026_27": string or null,
        "materials": {
          "required_items": string or null,
          "cv_page_limit": string or null,
          "letter_requirements": string or null
        } or null,
        "official_url": string or null
      },
      "access": {
        "project_type_name": string or null,
        "approval_timeline": string or null,
        "key_requirements": {
          "account_setup": string or null,
          "documentation": string or null,
          "cv_page_limit": string or null
        } or null,
        "post_approval_process": string or null,
        "appropriateness_rationale": string or null,
        "official_url": string or null
      },
      "conference": {
        "name": string or null,
        "location": string or null,
        "dates": string or null,
        "abstract_deadline": string or null,
        "full_paper_deadline": string or null,
        "paper_format": {
          "main_body_page_limit": string or null,
          "required_template_format": string or null,
          "review_type": string or null
        } or null,
        "additional_requirements": string or null,
        "official_url": string or null
      },
      "nsf_data_mgmt": {
        "dmp_page_limit": string or null,
        "minimum_retention_period": string or null,
        "data_sharing_policy_2025": string or null,
        "official_url": string or null
      },
      "timeline_coordination": {
        "analysis_text": string or null
      }
    }

    Rules:
    - Extract only what appears in the answer text; do not invent values.
    - If an item is missing, set the field to null.
    - For URLs, extract actual URLs mentioned (plain or markdown).
    - Keep amounts, page limits, dates as strings exactly as stated (e.g., "March 15, 2026", "2 pages", "$45,000", "up to 4 years").
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


# --------------------------------------------------------------------------- #
# Verification functions per section                                          #
# --------------------------------------------------------------------------- #
async def verify_fellowship(evaluator: Evaluator, parent_node, fellowship: FellowshipInfo) -> None:
    section = evaluator.add_parallel(
        id="Fellowship_Program_Information",
        desc="One eligible U.S. federal computational-science-focused graduate fellowship with all requested details and an official URL.",
        parent=parent_node,
        critical=True
    )

    # Existence checks
    evaluator.add_custom_node(
        result=bool(fellowship and fellowship.name and fellowship.name.strip()),
        id="Fellowship_Name_Provided",
        desc="States the fellowship program name.",
        parent=section,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(fellowship and fellowship.official_url and fellowship.official_url.strip()),
        id="Fellowship_URL_Provided",
        desc="Fellowship official URL is provided.",
        parent=section,
        critical=True
    )

    # Is U.S. federal and computational science focused
    leaf_federal_comp = evaluator.add_leaf(
        id="Fellowship_Is_US_Federal_And_Computational_Science_Focused",
        desc="Indicates the program is a U.S. federal graduate fellowship specifically designed for computational science.",
        parent=section,
        critical=True
    )
    claim_fc = f"The fellowship '{_safe(fellowship.name)}' is a U.S. federal graduate fellowship that is specifically designed for computational science or computational disciplines."
    await evaluator.verify(
        claim=claim_fc,
        node=leaf_federal_comp,
        sources=fellowship.official_url,
        additional_instruction="Verify on the official program page whether it is a U.S. federal fellowship and explicitly focused on computational science or computational fields."
    )

    # Eligibility matches constraints
    leaf_elig = evaluator.add_leaf(
        id="Fellowship_Eligibility_Matches_Stated_Constraints",
        desc="Provides eligibility requirements and confirms they match the stated constraints (U.S. citizenship; accredited U.S. university enrollment; accepts applicants with ≤1 year of full-time graduate work toward Ph.D.).",
        parent=section,
        critical=True
    )
    elig = fellowship.eligibility or FellowshipEligibility()
    claim_elig = (
        f"Eligibility for '{_safe(fellowship.name)}' requires U.S. citizenship or acceptable residency (stated as '{_safe(elig.citizenship_residency)}'), "
        f"requires enrollment at an accredited U.S. university (stated as '{_safe(elig.enrollment_requirements)}'), "
        f"and accepts applicants with ≤1 year of full-time graduate work toward a Ph.D (stated as '{_safe(elig.graduate_year_limitations)}')."
    )
    await evaluator.verify(
        claim=claim_elig,
        node=leaf_elig,
        sources=fellowship.official_url,
        additional_instruction="Check the official eligibility page for citizenship requirement, accredited U.S. enrollment, and constraints on prior graduate study (e.g., first-year or ≤1 year of full-time graduate work)."
    )

    # Financial details (split into specific checks)
    financial_parent = evaluator.add_parallel(
        id="Fellowship_Financial_Support_Details_Main",
        desc="Financial support details verification (stipend, tuition coverage, allowances, max duration).",
        parent=section,
        critical=True
    )
    fin = fellowship.financial or FellowshipFinancial()

    leaf_stipend = evaluator.add_leaf(
        id="Fellowship_Stipend_Amount_Accurate",
        desc="Annual stipend amount is accurately stated.",
        parent=financial_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The annual stipend amount is {_safe(fin.annual_stipend_amount)}.",
        node=leaf_stipend,
        sources=fellowship.official_url,
        additional_instruction="Verify the stipend amount on the official fellowship page; allow exact or clearly equivalent amounts."
    )

    leaf_tuition = evaluator.add_leaf(
        id="Fellowship_Tuition_Coverage_Accurate",
        desc="Tuition coverage is accurately stated.",
        parent=financial_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Tuition coverage for the fellowship is described as: {_safe(fin.tuition_coverage)}.",
        node=leaf_tuition,
        sources=fellowship.official_url,
        additional_instruction="Confirm tuition/fees coverage (full/partial) as described by the official fellowship documentation."
    )

    leaf_allowances = evaluator.add_leaf(
        id="Fellowship_Allowances_Accurate",
        desc="Additional allowances are accurately stated.",
        parent=financial_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Additional allowances include: {_safe(fin.additional_allowances)}.",
        node=leaf_allowances,
        sources=fellowship.official_url,
        additional_instruction="Verify any additional allowances such as travel, professional development, or equipment if stated on the official page."
    )

    leaf_years = evaluator.add_leaf(
        id="Fellowship_Max_Years_Support_Accurate",
        desc="Maximum years/duration of support is accurately stated.",
        parent=financial_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The maximum years/duration of support is {_safe(fin.max_years_support)}.",
        node=leaf_years,
        sources=fellowship.official_url,
        additional_instruction="Confirm the maximum number of years the fellowship can support (e.g., up to 4 years) on the official page."
    )

    # Deadline for 2026–27 cycle
    leaf_deadline = evaluator.add_leaf(
        id="Fellowship_Deadline_For_2026_27_Cycle",
        desc="Gives the application deadline for the 2026–27 cycle.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The application deadline for the 2026–27 cycle is {_safe(fellowship.application_deadline_2026_27)}.",
        node=leaf_deadline,
        sources=fellowship.official_url,
        additional_instruction="Verify the application deadline for the specified cycle; accept equivalent naming conventions (e.g., academic year or cohort naming)."
    )

    # Application materials and limits (split)
    materials_parent = evaluator.add_parallel(
        id="Fellowship_Application_Materials_And_Limits_Main",
        desc="Required application materials, CV/resume page limits, and letter requirements.",
        parent=section,
        critical=True
    )
    mats = fellowship.materials or FellowshipMaterials()

    leaf_mats_list = evaluator.add_leaf(
        id="Fellowship_Materials_Listed",
        desc="Required application materials are listed.",
        parent=materials_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Required application materials include: {_safe(mats.required_items)}.",
        node=leaf_mats_list,
        sources=fellowship.official_url,
        additional_instruction="Verify that the official page lists these application components."
    )

    leaf_cv_limit = evaluator.add_leaf(
        id="Fellowship_CV_Page_Limit_Accurate",
        desc="CV/resume page limits are accurately stated.",
        parent=materials_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CV/resume page limit is {_safe(mats.cv_page_limit)}.",
        node=leaf_cv_limit,
        sources=fellowship.official_url,
        additional_instruction="Confirm any page limit constraint for CV/resume on the official page."
    )

    leaf_letters = evaluator.add_leaf(
        id="Fellowship_Letter_Requirements_Accurate",
        desc="Letter requirements are accurately stated.",
        parent=materials_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Letter requirements are: {_safe(mats.letter_requirements)}.",
        node=leaf_letters,
        sources=fellowship.official_url,
        additional_instruction="Verify number/type of recommendation letters and any constraints on letter content or providers."
    )

    # Official reference URL verification (officialness)
    leaf_official_url = evaluator.add_leaf(
        id="Fellowship_Official_Reference_URL",
        desc="Provides an official reference URL from the responsible organization/program.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided URL for '{_safe(fellowship.name)}' is an official reference page from the responsible organization/program.",
        node=leaf_official_url,
        sources=fellowship.official_url,
        additional_instruction="Assess whether the page is an official program page (e.g., .gov or recognized agency domain) documenting the fellowship."
    )


async def verify_access(evaluator: Evaluator, parent_node, access: AccessInfo) -> None:
    section = evaluator.add_parallel(
        id="NSF_ACCESS_Computing_Allocation",
        desc="Most appropriate NSF ACCESS allocation type for a beginning graduate researcher, with requested details and an official URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(access and access.project_type_name and access.project_type_name.strip()),
        id="ACCESS_Project_Type_Name_Provided",
        desc="States the ACCESS project allocation type name.",
        parent=section,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(access and access.official_url and access.official_url.strip()),
        id="ACCESS_URL_Provided",
        desc="ACCESS official URL is provided.",
        parent=section,
        critical=True
    )

    # Confirm allocation is from NSF ACCESS
    leaf_is_access = evaluator.add_leaf(
        id="ACCESS_Allocation_Is_From_NSF_ACCESS",
        desc="Confirms the allocation type is an NSF ACCESS program allocation.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The allocation type '{_safe(access.project_type_name)}' is an NSF ACCESS program allocation.",
        node=leaf_is_access,
        sources=access.official_url,
        additional_instruction="Verify that the page describes this allocation type as part of NSF ACCESS."
    )

    # Appropriateness for beginner
    leaf_appropriate = evaluator.add_leaf(
        id="ACCESS_Type_Appropriate_For_Beginner",
        desc="Explains why this allocation type is appropriate for a graduate student beginning computational research (e.g., does not require extensive preliminary data), consistent with constraints.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The allocation type '{_safe(access.project_type_name)}' is appropriate for a beginner graduate researcher; {_safe(access.appropriateness_rationale)}.",
        node=leaf_appropriate,
        sources=access.official_url,
        additional_instruction="Confirm that the official ACCESS page indicates suitability for new users or small/test allocations without extensive prior results."
    )

    # Typical approval timeline
    leaf_timeline = evaluator.add_leaf(
        id="ACCESS_Typical_Approval_Timeline_Stated",
        desc="States the typical approval timeline for the chosen allocation type.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The typical approval timeline for '{_safe(access.project_type_name)}' is {_safe(access.approval_timeline)}.",
        node=leaf_timeline,
        sources=access.official_url,
        additional_instruction="Verify the typical review/approval timeline described on the official ACCESS page."
    )

    # Key application requirements (split)
    req_parent = evaluator.add_parallel(
        id="ACCESS_Key_Application_Requirements_Main",
        desc="Key application requirements (account setup, documentation, CV page limits).",
        parent=section,
        critical=True
    )
    reqs = access.key_requirements or AccessRequirements()

    leaf_acc_setup = evaluator.add_leaf(
        id="ACCESS_Account_Setup_Requirement",
        desc="Account setup requirement is stated.",
        parent=req_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Account setup requirement: {_safe(reqs.account_setup)}.",
        node=leaf_acc_setup,
        sources=access.official_url,
        additional_instruction="Confirm any prerequisite ACCESS account creation or portal registration steps."
    )

    leaf_docs = evaluator.add_leaf(
        id="ACCESS_Documentation_Requirement",
        desc="Documentation requirement is stated.",
        parent=req_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"Required documentation: {_safe(reqs.documentation)}.",
        node=leaf_docs,
        sources=access.official_url,
        additional_instruction="Verify that the official page lists required documentation (proposal forms, usage plans, etc.)."
    )

    leaf_cv = evaluator.add_leaf(
        id="ACCESS_CV_Page_Limit",
        desc="CV page limit requirement is stated.",
        parent=req_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The CV page limit is {_safe(reqs.cv_page_limit)}.",
        node=leaf_cv,
        sources=access.official_url,
        additional_instruction="Confirm any CV/resume page limit stated by ACCESS for this allocation type."
    )

    # Post-approval process
    leaf_post = evaluator.add_leaf(
        id="ACCESS_Post_Approval_Access_Process_Described",
        desc="Describes the post-approval process for accessing computing resources (process description without requiring any specific proprietary step unless stated in the official source).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"Post-approval process: {_safe(access.post_approval_process)}.",
        node=leaf_post,
        sources=access.official_url,
        additional_instruction="Confirm on the official page how users access resources post-approval (e.g., linking accounts, resource provider onboarding, allocation use instructions)."
    )

    # Official reference URL verification
    leaf_official_url = evaluator.add_leaf(
        id="ACCESS_Official_Reference_URL",
        desc="Provides an official reference URL from ACCESS.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL is an official ACCESS reference page documenting this allocation type.",
        node=leaf_official_url,
        sources=access.official_url,
        additional_instruction="Assess whether the page is an official ACCESS site describing the allocation type."
    )


async def verify_conference(evaluator: Evaluator, parent_node, conf: ConferenceInfo) -> None:
    section = evaluator.add_parallel(
        id="Target_Conference_2025",
        desc="One major AI/ML conference in 2025 suitable for computational neuroscience submissions, with deadlines, formatting, and an official URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(conf and conf.name and conf.name.strip()),
        id="Conference_Name_Provided",
        desc="States the conference name.",
        parent=section,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(conf and conf.official_url and conf.official_url.strip()),
        id="Conference_URL_Provided",
        desc="Conference official URL is provided.",
        parent=section,
        critical=True
    )

    # Major AI/ML and held in 2025
    leaf_major_2025 = evaluator.add_leaf(
        id="Conference_Is_Major_AI_ML_And_Held_In_2025",
        desc="Indicates the conference is a major AI/ML conference and is held in 2025.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The conference '{_safe(conf.name)}' is a major AI/ML conference and is held in 2025 (dates: {_safe(conf.dates)}).",
        node=leaf_major_2025,
        sources=conf.official_url,
        additional_instruction="Verify the official site shows it is a major AI/ML venue and that the event dates fall in 2025."
    )

    # Location and dates
    leaf_loc_dates = evaluator.add_leaf(
        id="Conference_Location_And_Dates_Provided",
        desc="Provides the conference location and dates (at least month and year, plus dates if available).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"Location: {_safe(conf.location)}; Dates: {_safe(conf.dates)}.",
        node=leaf_loc_dates,
        sources=conf.official_url,
        additional_instruction="Verify the location and dates on the official conference page."
    )

    # Abstract and full paper deadlines
    leaf_abs = evaluator.add_leaf(
        id="Conference_Abstract_Deadline_Accurate",
        desc="Abstract submission deadline is accurately stated (month/day/year).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"Abstract submission deadline: {_safe(conf.abstract_deadline)}.",
        node=leaf_abs,
        sources=conf.official_url,
        additional_instruction="Verify the abstract deadline as shown on the official call-for-papers or important dates page."
    )

    leaf_full = evaluator.add_leaf(
        id="Conference_Full_Paper_Deadline_Accurate",
        desc="Full paper submission deadline is accurately stated (month/day/year).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"Full paper submission deadline: {_safe(conf.full_paper_deadline)}.",
        node=leaf_full,
        sources=conf.official_url,
        additional_instruction="Verify the paper deadline as shown on the official page."
    )

    # Paper format requirements (split)
    fmt_parent = evaluator.add_parallel(
        id="Conference_Paper_Format_Requirements_Main",
        desc="Paper format requirements including page limit, template/format, and review type.",
        parent=section,
        critical=True
    )
    fmt = conf.paper_format or ConferenceFormat()

    leaf_pages = evaluator.add_leaf(
        id="Conference_Main_Body_Page_Limit_Accurate",
        desc="Main-body page limit is accurately stated.",
        parent=fmt_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The main-body page limit is {_safe(fmt.main_body_page_limit)}.",
        node=leaf_pages,
        sources=conf.official_url,
        additional_instruction="Verify page limit in author instructions (main text excluding references if applicable)."
    )

    leaf_template = evaluator.add_leaf(
        id="Conference_Required_Template_Format_Accurate",
        desc="Required submission format/template is accurately stated.",
        parent=fmt_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The required template/format is {_safe(fmt.required_template_format)}.",
        node=leaf_template,
        sources=conf.official_url,
        additional_instruction="Verify the required submission template/format (e.g., LaTeX style, PDF format) on the official page."
    )

    leaf_review = evaluator.add_leaf(
        id="Conference_Review_Type_Accurate",
        desc="Review type is accurately stated.",
        parent=fmt_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The review type is {_safe(fmt.review_type)}.",
        node=leaf_review,
        sources=conf.official_url,
        additional_instruction="Verify the review type if documented (e.g., double-blind, single-blind, etc.). If the official page documents the review style, use it."
    )

    # Additional requirements
    leaf_additional = evaluator.add_leaf(
        id="Conference_Additional_Requirements",
        desc="Mentions any additional requirements (e.g., checklists, supplementary materials).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"Additional requirements: {_safe(conf.additional_requirements)}.",
        node=leaf_additional,
        sources=conf.official_url,
        additional_instruction="Verify that any author checklist or supplementary material policy is documented on the official site."
    )

    # Official reference URL verification
    leaf_official_url = evaluator.add_leaf(
        id="Conference_Official_Reference_URL",
        desc="Provides an official conference reference URL documenting dates/deadlines/author instructions.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL is an official conference page documenting dates/deadlines/author instructions.",
        node=leaf_official_url,
        sources=conf.official_url,
        additional_instruction="Assess whether the page is an official conference site for dates and author instructions."
    )


async def verify_nsf_data_mgmt(evaluator: Evaluator, parent_node, nsf: NSFDataMgmtInfo) -> None:
    section = evaluator.add_parallel(
        id="NSF_Data_Management_Requirements",
        desc="NSF data management and sharing requirements, including required page limits/retention period, with an official URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(nsf and nsf.official_url and nsf.official_url.strip()),
        id="NSF_Data_URL_Provided",
        desc="NSF data management official URL is provided.",
        parent=section,
        critical=True
    )

    # DMP requirement and page limit
    leaf_dmp = evaluator.add_leaf(
        id="NSF_DMP_Requirement_And_Page_Limit",
        desc="States NSF data management plan requirement and includes the maximum page limit (2 pages).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"NSF requires a Data Management Plan and the maximum page limit is {_safe(nsf.dmp_page_limit)}.",
        node=leaf_dmp,
        sources=nsf.official_url,
        additional_instruction="Verify that NSF DMP is required and that the page limit is 2 pages as documented."
    )

    # Minimum retention period
    leaf_retention = evaluator.add_leaf(
        id="Minimum_Data_Retention_Period",
        desc="Specifies the minimum research data retention period after project closeout (minimum 3 years).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum research data retention period after project closeout is {_safe(nsf.minimum_retention_period)}.",
        node=leaf_retention,
        sources=nsf.official_url,
        additional_instruction="Verify that NSF policy (or referenced federal requirements) specify at least a 3-year retention period."
    )

    # 2025 data sharing policy
    leaf_policy = evaluator.add_leaf(
        id="NSF_2025_Data_Sharing_Policy_Requirements_Described",
        desc="Describes NSF's 2025 data sharing policy requirements (property-based description without hard-coding specific mechanisms not stated in the question/constraints).",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim=f"NSF's 2025 data sharing policy requirements are: {_safe(nsf.data_sharing_policy_2025)}.",
        node=leaf_policy,
        sources=nsf.official_url,
        additional_instruction="Verify the description of NSF's data sharing requirements applicable in 2025 (e.g., public access, data availability consistent with DMP)."
    )

    # Official reference URL verification
    leaf_official_url = evaluator.add_leaf(
        id="NSF_Data_Management_Official_Reference_URL",
        desc="Provides an official NSF reference URL documenting these requirements.",
        parent=section,
        critical=True
    )
    await evaluator.verify(
        claim="The provided URL is an official NSF page documenting data management and sharing requirements.",
        node=leaf_official_url,
        sources=nsf.official_url,
        additional_instruction="Assess whether the page is an official NSF site detailing DMP, retention, and sharing policies."
    )


async def verify_timeline_coordination(
    evaluator: Evaluator,
    parent_node,
    fellowship: FellowshipInfo,
    access: AccessInfo,
    conf: ConferenceInfo,
    timeline: TimelineCoordinationInfo
) -> None:
    section = evaluator.add_parallel(
        id="Timeline_Coordination_Analysis",
        desc="Brief analysis of whether the fellowship deadline, ACCESS allocation timeline, and conference submission deadlines can be coordinated for a January 2026 research start.",
        parent=parent_node,
        critical=True
    )

    # Uses identified dates and timelines
    leaf_uses_dates = evaluator.add_leaf(
        id="Uses_Identified_Dates_And_Timelines",
        desc="References the specific dates/timelines provided in the earlier sections (fellowship deadline, ACCESS approval timeline, conference deadlines) in the analysis.",
        parent=section,
        critical=True
    )
    fellowship_deadline = _safe(fellowship.application_deadline_2026_27)
    access_timeline = _safe(access.approval_timeline)
    conf_abs = _safe(conf.abstract_deadline)
    conf_full = _safe(conf.full_paper_deadline)
    claim_uses = (
        f"The analysis references the fellowship deadline ('{fellowship_deadline}'), the ACCESS approval timeline ('{access_timeline}'), "
        f"and the conference submission deadlines (abstract: '{conf_abs}', full paper: '{conf_full}')."
    )
    await evaluator.verify(
        claim=claim_uses,
        node=leaf_uses_dates,
        additional_instruction="Check within the provided analysis text whether these specific dates/timelines are explicitly referenced."
    )

    # Feasibility assessment for Jan 2026 start
    leaf_feasible = evaluator.add_leaf(
        id="Feasibility_Assessment_For_Jan_2026_Start",
        desc="Provides a reasoned conclusion on coordination feasibility relative to a January 2026 research start (including noting if any selected conference deadlines precede the start).",
        parent=section,
        critical=True
    )
    claim_feasible = (
        "The analysis provides a reasoned conclusion about whether the fellowship application deadline, ACCESS approval timeline, and the conference deadlines "
        "can be coordinated for a January 2026 research start, noting if any selected conference deadlines precede January 2026."
    )
    await evaluator.verify(
        claim=claim_feasible,
        node=leaf_feasible,
        additional_instruction="Verify that the analysis synthesizes the timelines into a conclusion about feasibility for a January 2026 start, explicitly noting if conference deadlines occur before that start date."
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

    # Extract all information in one pass
    extracted = await evaluator.extract(
        prompt=prompt_extract_full(),
        template_class=FullPlanExtraction,
        extraction_name="full_plan_extraction"
    )

    # Build top-level critical node to mirror rubric
    plan_node = evaluator.add_parallel(
        id="Graduate_Research_Infrastructure_Planning",
        desc="Provide required fellowship, ACCESS allocation, 2025 conference, NSF data management requirements, and a brief coordination analysis for a Jan 2026 research start.",
        parent=root,
        critical=True
    )

    # Fellowship
    await verify_fellowship(evaluator, plan_node, extracted.fellowship or FellowshipInfo())

    # ACCESS
    await verify_access(evaluator, plan_node, extracted.access or AccessInfo())

    # Conference
    await verify_conference(evaluator, plan_node, extracted.conference or ConferenceInfo())

    # NSF Data Management
    await verify_nsf_data_mgmt(evaluator, plan_node, extracted.nsf_data_mgmt or NSFDataMgmtInfo())

    # Timeline Coordination (uses other sections)
    await verify_timeline_coordination(
        evaluator,
        plan_node,
        extracted.fellowship or FellowshipInfo(),
        extracted.access or AccessInfo(),
        extracted.conference or ConferenceInfo(),
        extracted.timeline_coordination or TimelineCoordinationInfo()
    )

    # Add a small custom info block to record evaluation date context
    evaluator.add_custom_info({"evaluation_context_date": "2026-01-11"}, info_type="context", info_name="date_info")

    return evaluator.get_summary()