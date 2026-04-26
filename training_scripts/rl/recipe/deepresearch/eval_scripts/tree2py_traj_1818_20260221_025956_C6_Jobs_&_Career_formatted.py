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
TASK_ID = "senior_associate_ad_qualifications"
TASK_DESCRIPTION = (
    "What are the typical minimum qualification requirements for a Senior Associate Athletic Director "
    "position in the Business & Finance area at a NCAA Division I FBS institution? Include: "
    "(1) the minimum and preferred educational credentials (degree level and relevant fields of study), "
    "(2) the minimum years of professional experience required (both total years and years at senior administrative level), "
    "(3) any Division I or FBS-specific experience requirements, "
    "(4) typical positions held in the career progression path leading to this role, and "
    "(5) relevant professional certifications or development programs that strengthen candidacy."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class MastersFieldsPercents(BaseModel):
    sport_management_percent: Optional[str] = None
    education_percent: Optional[str] = None
    physical_education_percent: Optional[str] = None
    business_percent: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MidLevelDirectorAreasPercents(BaseModel):
    development_percent: Optional[str] = None
    marketing_percent: Optional[str] = None
    business_management_percent: Optional[str] = None
    compliance_percent: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EducationSection(BaseModel):
    bachelors_minimum_mentioned: Optional[bool] = None
    bachelors_sources: List[str] = Field(default_factory=list)

    masters_preferred_mentioned: Optional[bool] = None
    masters_preferred_sources: List[str] = Field(default_factory=list)

    masters_fields: MastersFieldsPercents = MastersFieldsPercents()

    cosma_context_mentioned: Optional[bool] = None
    cosma_sources: List[str] = Field(default_factory=list)


class ExperienceSection(BaseModel):
    total_experience_range_mentioned: Optional[bool] = None
    total_experience_sources: List[str] = Field(default_factory=list)

    di_business_dept_10yrs_mentioned: Optional[bool] = None
    di_business_dept_10yrs_sources: List[str] = Field(default_factory=list)

    senior_admin_min_5yrs_mentioned: Optional[bool] = None
    senior_admin_min_5yrs_sources: List[str] = Field(default_factory=list)

    di_fbs_experience_expected_mentioned: Optional[bool] = None
    di_fbs_experience_expected_sources: List[str] = Field(default_factory=list)

    ad_avg_years_10_4_mentioned: Optional[bool] = None
    ad_avg_years_10_4_sources: List[str] = Field(default_factory=list)


class CareerPathSection(BaseModel):
    assistant_or_associate_ad_66_7_mentioned: Optional[bool] = None
    assistant_or_associate_ad_sources: List[str] = Field(default_factory=list)

    graduate_assistant_25_3_mentioned: Optional[bool] = None
    graduate_assistant_sources: List[str] = Field(default_factory=list)

    mid_level_director_areas: MidLevelDirectorAreasPercents = MidLevelDirectorAreasPercents()


class CertificationsSection(BaseModel):
    niaaa_caa_cert_mentioned: Optional[bool] = None
    niaaa_caa_sources: List[str] = Field(default_factory=list)

    caa_requirements_mentioned: Optional[bool] = None
    caa_requirements_sources: List[str] = Field(default_factory=list)


class SeniorADQualificationsExtraction(BaseModel):
    educational_credentials: EducationSection = EducationSection()
    experience_requirements: ExperienceSection = ExperienceSection()
    career_progression_path: CareerPathSection = CareerPathSection()
    professional_certifications: CertificationsSection = CertificationsSection()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_qualifications() -> str:
    return """
    Extract from the answer whether it explicitly includes and cites sources for the following qualification facts
    for NCAA Division I (preferably FBS) Senior Associate/Associate AD roles in Business/Finance. For each item:
    - Mark `mentioned` as true if the answer explicitly states the fact (or a very close equivalent).
    - Collect all URLs cited in the answer that support that specific fact (do NOT invent URLs).
    - For percentage items, extract the exact percentage text mentioned (include the '%' symbol) and collect sources.

    EDUCATION
    1) Minimum_Education_Bachelors:
       - Field: educational_credentials.bachelors_minimum_mentioned (boolean)
       - Field: educational_credentials.bachelors_sources (list of URLs)
       Consider variants: "bachelor’s required", "BA/BS required", "baccalaureate degree required".
    2) Preferred_Education_Masters_Commonality ("> 80% of Division I athletic directors hold a master's"):
       - Field: educational_credentials.masters_preferred_mentioned (boolean)
       - Field: educational_credentials.masters_preferred_sources (list of URLs)
    3) Common_Masters_Fields_With_Percentages (these exact fields/percentages if present in the answer):
       - sport_management_percent -> educational_credentials.masters_fields.sport_management_percent (string, e.g., "35.5%")
       - education_percent -> educational_credentials.masters_fields.education_percent (string, e.g., "25.4%")
       - physical_education_percent -> educational_credentials.masters_fields.physical_education_percent (string, e.g., "12.2%")
       - business_percent -> educational_credentials.masters_fields.business_percent (string, e.g., "13.2%")
       - sources -> educational_credentials.masters_fields.sources (list of URLs)
    4) COSMA_Accreditation_Context:
       - Field: educational_credentials.cosma_context_mentioned (boolean)
       - Field: educational_credentials.cosma_sources (list of URLs)
       Consider wording like "COSMA is the only discipline-specific accreditation body for sport management"
       and references to COSMA accreditation for sport management graduate programs.

    EXPERIENCE
    5) Total_Experience_Typical_Range ("5–10 years" typical for Business & Finance senior associate roles):
       - Field: experience_requirements.total_experience_range_mentioned (boolean)
       - Field: experience_requirements.total_experience_sources (list of URLs)
       Accept supporting examples like "minimum 5 years", "7–10 years", "10 years experience" on DI postings.
    6) Example_DI_Business_Department_Experience ("~10 years in an NCAA Division I business department"):
       - Field: experience_requirements.di_business_dept_10yrs_mentioned (boolean)
       - Field: experience_requirements.di_business_dept_10yrs_sources (list of URLs)
    7) Senior_Admin_Experience_Minimum ("minimum 5 years senior-level administrative experience in intercollegiate athletics"):
       - Field: experience_requirements.senior_admin_min_5yrs_mentioned (boolean)
       - Field: experience_requirements.senior_admin_min_5yrs_sources (list of URLs)
    8) DivisionI_or_FBS_Specific_Experience (DI/FBS experience commonly preferred/expected):
       - Field: experience_requirements.di_fbs_experience_expected_mentioned (boolean)
       - Field: experience_requirements.di_fbs_experience_expected_sources (list of URLs)
    9) AD_Average_Years_Context ("Division I athletic directors average 10.4 years as an AD at any institution"):
       - Field: experience_requirements.ad_avg_years_10_4_mentioned (boolean)
       - Field: experience_requirements.ad_avg_years_10_4_sources (list of URLs)

    CAREER PROGRESSION PATH
    10) Assistant_or_Associate_AD_Predecessor_Stat ("66.7% previously held assistant/associate AD"):
        - Field: career_progression_path.assistant_or_associate_ad_66_7_mentioned (boolean)
        - Field: career_progression_path.assistant_or_associate_ad_sources (list of URLs)
    11) Graduate_Assistant_Step_Stat ("25.3% graduate assistant"):
        - Field: career_progression_path.graduate_assistant_25_3_mentioned (boolean)
        - Field: career_progression_path.graduate_assistant_sources (list of URLs)
    12) Mid_Level_Director_Functional_Areas_With_Percentages:
        - development_percent -> career_progression_path.mid_level_director_areas.development_percent (string, e.g., "25.3%")
        - marketing_percent -> career_progression_path.mid_level_director_areas.marketing_percent (string, e.g., "22.2%")
        - business_management_percent -> career_progression_path.mid_level_director_areas.business_management_percent (string, e.g., "19.2%")
        - compliance_percent -> career_progression_path.mid_level_director_areas.compliance_percent (string, e.g., "16.2%")
        - sources -> career_progression_path.mid_level_director_areas.sources (list of URLs)

    PROFESSIONAL CERTIFICATIONS / DEVELOPMENT
    13) NIAAA_CAA_Certification mention:
        - Field: professional_certifications.niaaa_caa_cert_mentioned (boolean)
        - Field: professional_certifications.niaaa_caa_sources (list of URLs)
    14) CAA_Requirements_As_Listed (CAA requires: bachelor's degree; completion of LTC 501/502/503/504/506; ≥2 years as an athletic administrator; 65 credits):
        - Field: professional_certifications.caa_requirements_mentioned (boolean)
        - Field: professional_certifications.caa_requirements_sources (list of URLs)

    RULES:
    - Only include URLs explicitly present in the answer text (including markdown links).
    - Use true/false for 'mentioned' fields based on explicit statements in the answer.
    - Use strings for percentages exactly as written in the answer (keep the % sign).
    - If an item is NOT mentioned, set the boolean to false and the corresponding URLs to an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _norm_sources(urls: Optional[List[str]]) -> List[str]:
    return urls or []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_education(evaluator: Evaluator, parent_node, edu: EducationSection) -> None:
    # Category node (critical, parallel)
    cat = evaluator.add_parallel(
        id="Educational_Credentials",
        desc="Includes minimum and preferred educational credentials (degree level and relevant fields).",
        parent=parent_node,
        critical=True
    )

    # 1) Bachelor's minimum
    evaluator.add_custom_node(
        result=bool(edu.bachelors_minimum_mentioned) and _has_sources(edu.bachelors_sources),
        id="Minimum_Education_Bachelors_presence",
        desc="Answer explicitly states a bachelor's is the minimum and cites at least one source.",
        parent=cat,
        critical=True
    )
    leaf_bach = evaluator.add_leaf(
        id="Minimum_Education_Bachelors",
        desc="States that a bachelor's degree is the minimum educational requirement (per constraints).",
        parent=cat,
        critical=True
    )
    claim_bach = (
        "A bachelor's degree is commonly the minimum educational requirement for Senior Associate/Associate "
        "Athletic Director roles in Business/Finance within NCAA Division I (preferably FBS) athletic departments."
    )
    await evaluator.verify(
        claim=claim_bach,
        node=leaf_bach,
        sources=_norm_sources(edu.bachelors_sources),
        additional_instruction="Confirm the source explicitly lists a bachelor's (BA/BS/baccalaureate) as a minimum requirement for the role."
    )

    # 2) Master's preferred / >80% hold a master's
    evaluator.add_custom_node(
        result=bool(edu.masters_preferred_mentioned) and _has_sources(edu.masters_preferred_sources),
        id="Preferred_Education_Masters_Commonality_presence",
        desc="Answer states a master's is common/preferred with >80% holding a master's and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_masters_pref = evaluator.add_leaf(
        id="Preferred_Education_Masters_Commonality",
        desc="States that a master's degree is common/preferred among Division I athletic directors (per constraints: >80% hold a master's).",
        parent=cat,
        critical=True
    )
    claim_masters_pref = (
        "Among NCAA Division I athletic directors, more than 80% hold a master's degree, and a master's degree "
        "is commonly preferred for senior administrative roles."
    )
    await evaluator.verify(
        claim=claim_masters_pref,
        node=leaf_masters_pref,
        sources=_norm_sources(edu.masters_preferred_sources),
        additional_instruction="The source should explicitly support 'over 80%' or equivalent phrasing; minor rounding differences are acceptable."
    )

    # 3) Common master's fields with percentages (break into four subchecks)
    fields_node = evaluator.add_parallel(
        id="Common_Masters_Fields_With_Percentages",
        desc="Common master's fields and percentages: Sport Management, Education, Physical Education, Business.",
        parent=cat,
        critical=True
    )

    # Sport Management 35.5%
    evaluator.add_custom_node(
        result=bool(edu.masters_fields.sport_management_percent) and _has_sources(edu.masters_fields.sources),
        id="Common_Masters_Fields_Sport_Management_presence",
        desc="Answer includes Sport Management percentage and cites sources.",
        parent=fields_node,
        critical=True
    )
    leaf_sm = evaluator.add_leaf(
        id="Common_Masters_Fields_Sport_Management",
        desc="Sport Management master's field percentage is correctly stated as 35.5%.",
        parent=fields_node,
        critical=True
    )
    claim_sm = "Sport Management accounts for approximately 35.5% of master's degrees among Division I athletic directors."
    await evaluator.verify(
        claim=claim_sm,
        node=leaf_sm,
        sources=_norm_sources(edu.masters_fields.sources),
        additional_instruction="Verify the source explicitly reports ~35.5% for Sport Management; accept minor rounding."
    )

    # Education 25.4%
    evaluator.add_custom_node(
        result=bool(edu.masters_fields.education_percent) and _has_sources(edu.masters_fields.sources),
        id="Common_Masters_Fields_Education_presence",
        desc="Answer includes Education percentage and cites sources.",
        parent=fields_node,
        critical=True
    )
    leaf_ed = evaluator.add_leaf(
        id="Common_Masters_Fields_Education",
        desc="Education master's field percentage is correctly stated as 25.4%.",
        parent=fields_node,
        critical=True
    )
    claim_ed = "Education accounts for approximately 25.4% of master's degrees among Division I athletic directors."
    await evaluator.verify(
        claim=claim_ed,
        node=leaf_ed,
        sources=_norm_sources(edu.masters_fields.sources),
        additional_instruction="Verify the source explicitly reports ~25.4% for Education; accept minor rounding."
    )

    # Physical Education 12.2%
    evaluator.add_custom_node(
        result=bool(edu.masters_fields.physical_education_percent) and _has_sources(edu.masters_fields.sources),
        id="Common_Masters_Fields_Physical_Education_presence",
        desc="Answer includes Physical Education percentage and cites sources.",
        parent=fields_node,
        critical=True
    )
    leaf_pe = evaluator.add_leaf(
        id="Common_Masters_Fields_Physical_Education",
        desc="Physical Education master's field percentage is correctly stated as 12.2%.",
        parent=fields_node,
        critical=True
    )
    claim_pe = "Physical Education accounts for approximately 12.2% of master's degrees among Division I athletic directors."
    await evaluator.verify(
        claim=claim_pe,
        node=leaf_pe,
        sources=_norm_sources(edu.masters_fields.sources),
        additional_instruction="Verify the source explicitly reports ~12.2% for Physical Education; accept minor rounding."
    )

    # Business 13.2%
    evaluator.add_custom_node(
        result=bool(edu.masters_fields.business_percent) and _has_sources(edu.masters_fields.sources),
        id="Common_Masters_Fields_Business_presence",
        desc="Answer includes Business percentage and cites sources.",
        parent=fields_node,
        critical=True
    )
    leaf_bus = evaluator.add_leaf(
        id="Common_Masters_Fields_Business",
        desc="Business master's field percentage is correctly stated as 13.2%.",
        parent=fields_node,
        critical=True
    )
    claim_bus = "Business accounts for approximately 13.2% of master's degrees among Division I athletic directors."
    await evaluator.verify(
        claim=claim_bus,
        node=leaf_bus,
        sources=_norm_sources(edu.masters_fields.sources),
        additional_instruction="Verify the source explicitly reports ~13.2% for Business; accept minor rounding."
    )

    # 4) COSMA context
    evaluator.add_custom_node(
        result=bool(edu.cosma_context_mentioned) and _has_sources(edu.cosma_sources),
        id="COSMA_Accreditation_Context_presence",
        desc="Answer mentions COSMA context and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_cosma = evaluator.add_leaf(
        id="COSMA_Accreditation_Context",
        desc="Mentions COSMA accreditation and that COSMA is the only discipline-specific accreditation body for sport management.",
        parent=cat,
        critical=True
    )
    claim_cosma = (
        "COSMA (Commission on Sport Management Accreditation) accredits sport management programs and is the only "
        "discipline-specific accreditation body for sport management."
    )
    await evaluator.verify(
        claim=claim_cosma,
        node=leaf_cosma,
        sources=_norm_sources(edu.cosma_sources),
        additional_instruction="Confirm that the source states COSMA is the only discipline-specific accreditor for sport management programs."
    )


async def verify_experience(evaluator: Evaluator, parent_node, exp: ExperienceSection) -> None:
    cat = evaluator.add_parallel(
        id="Experience_Requirements",
        desc="Includes minimum years of experience and Division I/FBS-specific experience expectations.",
        parent=parent_node,
        critical=True
    )

    # Total experience typical range (5–10 years)
    evaluator.add_custom_node(
        result=bool(exp.total_experience_range_mentioned) and _has_sources(exp.total_experience_sources),
        id="Total_Experience_Typical_Range_presence",
        desc="Answer states a typical total experience range (5–10 years) and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_total_range = evaluator.add_leaf(
        id="Total_Experience_Typical_Range",
        desc="Gives the typical total professional experience requirement range for Business & Finance senior associate roles (5–10 years).",
        parent=cat,
        critical=True
    )
    claim_total_range = (
        "For Senior Associate/Associate AD roles in Business/Finance at NCAA Division I, job postings typically require "
        "between 5 and 10 years of relevant professional experience."
    )
    await evaluator.verify(
        claim=claim_total_range,
        node=leaf_total_range,
        sources=_norm_sources(exp.total_experience_sources),
        additional_instruction="Accept examples like 'minimum 5 years', '7-10 years', or '10 years' in DI postings as supporting the 5–10 range."
    )

    # Example DI business department ~10 years
    evaluator.add_custom_node(
        result=bool(exp.di_business_dept_10yrs_mentioned) and _has_sources(exp.di_business_dept_10yrs_sources),
        id="Example_DI_Business_Department_Experience_presence",
        desc="Answer includes ~10 years in a DI business department example with sources.",
        parent=cat,
        critical=True
    )
    leaf_di_10yrs = evaluator.add_leaf(
        id="Example_DI_Business_Department_Experience",
        desc="Includes the example that some postings require ~10 years in an NCAA Division I business department.",
        parent=cat,
        critical=True
    )
    claim_di_10yrs = (
        "Some NCAA Division I athletics job postings for Business/Finance senior associate roles explicitly require "
        "around 10 years of experience in a Division I athletics business department."
    )
    await evaluator.verify(
        claim=claim_di_10yrs,
        node=leaf_di_10yrs,
        sources=_norm_sources(exp.di_business_dept_10yrs_sources),
        additional_instruction="Confirm the source states ~10 years experience specifically in a DI athletics business/finance department."
    )

    # Senior-level admin minimum 5 years
    evaluator.add_custom_node(
        result=bool(exp.senior_admin_min_5yrs_mentioned) and _has_sources(exp.senior_admin_min_5yrs_sources),
        id="Senior_Admin_Experience_Minimum_presence",
        desc="Answer states minimum 5 years senior-level administrative experience with sources.",
        parent=cat,
        critical=True
    )
    leaf_senior_admin = evaluator.add_leaf(
        id="Senior_Admin_Experience_Minimum",
        desc="States the typical minimum senior-level administrative experience requirement in intercollegiate athletics (minimum 5 years).",
        parent=cat,
        critical=True
    )
    claim_senior_admin = (
        "Job postings for Senior Associate/Associate AD roles commonly require at least 5 years of senior-level "
        "administrative experience in intercollegiate athletics."
    )
    await evaluator.verify(
        claim=claim_senior_admin,
        node=leaf_senior_admin,
        sources=_norm_sources(exp.senior_admin_min_5yrs_sources),
        additional_instruction="Look for phrases like 'minimum five years of senior-level administrative experience' in DI postings."
    )

    # Division I / FBS-specific experience expected
    evaluator.add_custom_node(
        result=bool(exp.di_fbs_experience_expected_mentioned) and _has_sources(exp.di_fbs_experience_expected_sources),
        id="DivisionI_or_FBS_Specific_Experience_presence",
        desc="Answer states DI/FBS experience commonly preferred/expected with sources.",
        parent=cat,
        critical=True
    )
    leaf_di_fbs = evaluator.add_leaf(
        id="DivisionI_or_FBS_Specific_Experience",
        desc="States that NCAA Division I FBS-level experience is commonly preferred/expected.",
        parent=cat,
        critical=True
    )
    claim_di_fbs = (
        "NCAA Division I (and specifically FBS) athletics experience is commonly preferred or expected for "
        "Senior Associate/Associate AD roles in Business/Finance."
    )
    await evaluator.verify(
        claim=claim_di_fbs,
        node=leaf_di_fbs,
        sources=_norm_sources(exp.di_fbs_experience_expected_sources),
        additional_instruction="Confirm the posting or policy explicitly prefers or requires Division I/FBS athletics experience."
    )

    # AD average years context (10.4 years)
    evaluator.add_custom_node(
        result=bool(exp.ad_avg_years_10_4_mentioned) and _has_sources(exp.ad_avg_years_10_4_sources),
        id="AD_Average_Years_Context_presence",
        desc="Answer includes 10.4-year average AD experience context with sources.",
        parent=cat,
        critical=True
    )
    leaf_ad_avg = evaluator.add_leaf(
        id="AD_Average_Years_Context",
        desc="Includes that Division I athletic directors average 10.4 years serving as an AD at any institution (context).",
        parent=cat,
        critical=True
    )
    claim_ad_avg = "Division I athletic directors average approximately 10.4 years of total experience serving as an athletic director."
    await evaluator.verify(
        claim=claim_ad_avg,
        node=leaf_ad_avg,
        sources=_norm_sources(exp.ad_avg_years_10_4_sources),
        additional_instruction="Minor rounding is acceptable; confirm the figure is ~10.4 years."
    )


async def verify_career_path(evaluator: Evaluator, parent_node, cp: CareerPathSection) -> None:
    cat = evaluator.add_parallel(
        id="Career_Progression_Path",
        desc="Describes typical prior roles/trajectory leading to the Senior Associate AD level.",
        parent=parent_node,
        critical=True
    )

    # Assistant/Associate AD predecessor stat (66.7%)
    evaluator.add_custom_node(
        result=bool(cp.assistant_or_associate_ad_66_7_mentioned) and _has_sources(cp.assistant_or_associate_ad_sources),
        id="Assistant_or_Associate_AD_Predecessor_Stat_presence",
        desc="Answer includes 66.7% predecessor stat and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_asst_assoc = evaluator.add_leaf(
        id="Assistant_or_Associate_AD_Predecessor_Stat",
        desc="Includes that 66.7% of Division I athletic directors previously held assistant/associate AD positions.",
        parent=cat,
        critical=True
    )
    claim_asst_assoc = "Approximately 66.7% of NCAA Division I athletic directors previously held an assistant or associate athletic director position."
    await evaluator.verify(
        claim=claim_asst_assoc,
        node=leaf_asst_assoc,
        sources=_norm_sources(cp.assistant_or_associate_ad_sources),
        additional_instruction="Confirm the percentage (about 66.7%) is explicitly reported."
    )

    # Graduate Assistant step stat (25.3%)
    evaluator.add_custom_node(
        result=bool(cp.graduate_assistant_25_3_mentioned) and _has_sources(cp.graduate_assistant_sources),
        id="Graduate_Assistant_Step_Stat_presence",
        desc="Answer includes 25.3% graduate assistant stat and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_ga = evaluator.add_leaf(
        id="Graduate_Assistant_Step_Stat",
        desc="Includes that a common early role is graduate assistant (25.3%).",
        parent=cat,
        critical=True
    )
    claim_ga = "Approximately 25.3% of NCAA Division I athletic directors had a graduate assistant role early in their careers."
    await evaluator.verify(
        claim=claim_ga,
        node=leaf_ga,
        sources=_norm_sources(cp.graduate_assistant_sources),
        additional_instruction="Confirm the percentage (~25.3%) is explicitly reported."
    )

    # Mid-level director functional areas with percentages (break into four subchecks)
    mid_node = evaluator.add_parallel(
        id="Mid_Level_Director_Functional_Areas_With_Percentages",
        desc="Mid-level director areas and percentages: development, marketing, business management, compliance.",
        parent=cat,
        critical=True
    )

    # Development 25.3%
    evaluator.add_custom_node(
        result=bool(cp.mid_level_director_areas.development_percent) and _has_sources(cp.mid_level_director_areas.sources),
        id="Mid_Director_Development_presence",
        desc="Answer includes development percentage and cites sources.",
        parent=mid_node,
        critical=True
    )
    leaf_dev = evaluator.add_leaf(
        id="Mid_Director_Development",
        desc="Development (mid-level director area) percentage is correctly stated as 25.3%.",
        parent=mid_node,
        critical=True
    )
    claim_dev = "Development is reported at approximately 25.3% among mid-level director areas in career progression data."
    await evaluator.verify(
        claim=claim_dev,
        node=leaf_dev,
        sources=_norm_sources(cp.mid_level_director_areas.sources),
        additional_instruction="Confirm the source explicitly reports ~25.3% for development."
    )

    # Marketing 22.2%
    evaluator.add_custom_node(
        result=bool(cp.mid_level_director_areas.marketing_percent) and _has_sources(cp.mid_level_director_areas.sources),
        id="Mid_Director_Marketing_presence",
        desc="Answer includes marketing percentage and cites sources.",
        parent=mid_node,
        critical=True
    )
    leaf_mkt = evaluator.add_leaf(
        id="Mid_Director_Marketing",
        desc="Marketing (mid-level director area) percentage is correctly stated as 22.2%.",
        parent=mid_node,
        critical=True
    )
    claim_mkt = "Marketing is reported at approximately 22.2% among mid-level director areas in career progression data."
    await evaluator.verify(
        claim=claim_mkt,
        node=leaf_mkt,
        sources=_norm_sources(cp.mid_level_director_areas.sources),
        additional_instruction="Confirm the source explicitly reports ~22.2% for marketing."
    )

    # Business Management 19.2%
    evaluator.add_custom_node(
        result=bool(cp.mid_level_director_areas.business_management_percent) and _has_sources(cp.mid_level_director_areas.sources),
        id="Mid_Director_Business_Management_presence",
        desc="Answer includes business management percentage and cites sources.",
        parent=mid_node,
        critical=True
    )
    leaf_bm = evaluator.add_leaf(
        id="Mid_Director_Business_Management",
        desc="Business management (mid-level director area) percentage is correctly stated as 19.2%.",
        parent=mid_node,
        critical=True
    )
    claim_bm = "Business management is reported at approximately 19.2% among mid-level director areas in career progression data."
    await evaluator.verify(
        claim=claim_bm,
        node=leaf_bm,
        sources=_norm_sources(cp.mid_level_director_areas.sources),
        additional_instruction="Confirm the source explicitly reports ~19.2% for business management."
    )

    # Compliance 16.2%
    evaluator.add_custom_node(
        result=bool(cp.mid_level_director_areas.compliance_percent) and _has_sources(cp.mid_level_director_areas.sources),
        id="Mid_Director_Compliance_presence",
        desc="Answer includes compliance percentage and cites sources.",
        parent=mid_node,
        critical=True
    )
    leaf_comp = evaluator.add_leaf(
        id="Mid_Director_Compliance",
        desc="Compliance (mid-level director area) percentage is correctly stated as 16.2%.",
        parent=mid_node,
        critical=True
    )
    claim_comp = "Compliance is reported at approximately 16.2% among mid-level director areas in career progression data."
    await evaluator.verify(
        claim=claim_comp,
        node=leaf_comp,
        sources=_norm_sources(cp.mid_level_director_areas.sources),
        additional_instruction="Confirm the source explicitly reports ~16.2% for compliance."
    )


async def verify_certifications(evaluator: Evaluator, parent_node, certs: CertificationsSection) -> None:
    cat = evaluator.add_parallel(
        id="Professional_Certifications_or_Development",
        desc="Identifies relevant certifications or development programs that strengthen candidacy.",
        parent=parent_node,
        critical=True
    )

    # NIAAA CAA certification mention
    evaluator.add_custom_node(
        result=bool(certs.niaaa_caa_cert_mentioned) and _has_sources(certs.niaaa_caa_sources),
        id="NIAAA_CAA_Certification_presence",
        desc="Answer references NIAAA CAA certification and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_caa = evaluator.add_leaf(
        id="NIAAA_CAA_Certification",
        desc="References the NIAAA Certified Athletic Administrator (CAA) certification.",
        parent=cat,
        critical=True
    )
    claim_caa = "The NIAAA offers the Certified Athletic Administrator (CAA) credential for athletic administrators."
    await evaluator.verify(
        claim=claim_caa,
        node=leaf_caa,
        sources=_norm_sources(certs.niaaa_caa_sources),
        additional_instruction="Confirm the source is NIAAA or an official page describing the CAA certification."
    )

    # CAA requirements as listed
    evaluator.add_custom_node(
        result=bool(certs.caa_requirements_mentioned) and _has_sources(certs.caa_requirements_sources),
        id="CAA_Requirements_As_Listed_presence",
        desc="Answer lists CAA requirements and cites sources.",
        parent=cat,
        critical=True
    )
    leaf_caa_reqs = evaluator.add_leaf(
        id="CAA_Requirements_As_Listed",
        desc="States CAA requirements: bachelor's degree or higher; completion of LTC 501/502/503/504/506; ≥2 years as an athletic administrator; 65 credits.",
        parent=cat,
        critical=True
    )
    claim_caa_reqs = (
        "CAA requirements include: a bachelor's degree or higher; completion of LTC 501, 502, 503, 504, and 506; "
        "at least two years' experience as an athletic administrator; and attainment of 65 credits."
    )
    await evaluator.verify(
        claim=claim_caa_reqs,
        node=leaf_caa_reqs,
        sources=_norm_sources(certs.caa_requirements_sources),
        additional_instruction="Verify the NIAAA/official documentation lists these specific requirements."
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
    """
    Evaluate an answer for Senior Associate AD (Business & Finance) qualifications at NCAA Division I FBS institutions.
    """
    # Initialize evaluator (root is non-critical by design; add a critical top-level node under it)
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

    # Create top-level critical node that mirrors the rubric root
    top_node = evaluator.add_parallel(
        id="Senior_Associate_AD_Qualifications",
        desc="Answer includes all requested qualification categories for a Senior Associate Athletic Director (Business & Finance) at an NCAA Division I FBS institution.",
        parent=root,
        critical=True
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_qualifications(),
        template_class=SeniorADQualificationsExtraction,
        extraction_name="qualifications_extraction"
    )

    # Build verification subtrees per category
    await verify_education(evaluator, top_node, extracted.educational_credentials)
    await verify_experience(evaluator, top_node, extracted.experience_requirements)
    await verify_career_path(evaluator, top_node, extracted.career_progression_path)
    await verify_certifications(evaluator, top_node, extracted.professional_certifications)

    # Add ground truth context info (for transparency)
    evaluator.add_ground_truth({
        "expected_masters_fields_percentages": {
            "Sport Management": "35.5%",
            "Education": "25.4%",
            "Physical Education": "12.2%",
            "Business": "13.2%"
        },
        "expected_mid_level_director_percentages": {
            "Development": "25.3%",
            "Marketing": "22.2%",
            "Business Management": "19.2%",
            "Compliance": "16.2%"
        },
        "ad_average_years_context": "Division I athletic directors ~10.4 years as AD (any institution)"
    })

    # Return standardized summary
    return evaluator.get_summary()