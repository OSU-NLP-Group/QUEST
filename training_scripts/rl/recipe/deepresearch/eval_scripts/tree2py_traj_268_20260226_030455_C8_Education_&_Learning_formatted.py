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
TASK_ID = "va_niche_2026_top_district"
TASK_DESCRIPTION = """
According to Niche's 2026 rankings for Best School Districts in Virginia, identify the highest-ranked public school district. Then provide comprehensive documentation about this district including the following information: (1) Official district name, (2) Geographic location (specific city or county in Virginia), (3) Total student enrollment statistics with source, (4) Total number of schools operated by the district, (5) Grade levels served, (6) High school graduation rate, (7) Student demographic information, (8) Accreditation status according to Virginia Department of Education standards, (9) Niche ranking score or rating, (10) Per-student expenditure data, (11) Current superintendent or chief administrator, (12) Official website URL, (13) District contact information, and (14) Virginia School Performance and Support Framework data if applicable. For each piece of information provided, include reference URLs to support your answer.
"""

DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT = (
    "You must rely only on the provided source URLs to judge this claim. "
    "If no source URLs are provided or the sources are irrelevant/inaccessible, you must return 'Incorrect'. "
    "Allow minor naming or formatting variations. If the value is numeric, allow reasonable rounding (e.g., 66.7 ≈ 67)."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DataField(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DistrictIdentification(BaseModel):
    district_name: Optional[str] = None
    niche_urls: List[str] = Field(default_factory=list)


class DistrictReport(BaseModel):
    official_name: Optional[DataField] = None
    location: Optional[DataField] = None
    enrollment: Optional[DataField] = None
    number_of_schools: Optional[DataField] = None
    grade_levels: Optional[DataField] = None
    graduation_rate: Optional[DataField] = None
    demographics: Optional[DataField] = None
    accreditation_status: Optional[DataField] = None
    ranking_score: Optional[DataField] = None
    per_student_expenditure: Optional[DataField] = None
    superintendent: Optional[DataField] = None
    official_website: Optional[DataField] = None
    contact_info: Optional[DataField] = None
    performance_framework: Optional[DataField] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_identification() -> str:
    return """
    From the provided answer, extract the single highest-ranked Virginia public school district according to Niche's 2026 Best School Districts in Virginia ranking.
    Return:
    - district_name: The district name exactly as stated in the answer text (string or null).
    - niche_urls: An array of URLs explicitly cited in the answer that support the Niche 2026 Virginia ranking and the #1 placement of the district. Include only actual URLs present in the answer; do not invent any.
    If the answer mentions multiple districts, pick the one explicitly described as #1/highest-ranked in Virginia per Niche 2026. If none is clearly identified, return null for district_name and an empty array for niche_urls.
    """


def prompt_extract_district_report() -> str:
    return """
    Extract comprehensive information about the identified district from the answer text. For each category below, return an object:
    {
      "value": string or null,
      "sources": [list of URLs explicitly present in the answer that support the value]
    }
    Do NOT invent URLs. Use only URLs that appear in the answer. If none are provided for a category, use an empty array for sources.

    Categories (JSON keys):
    - official_name: The official name of the district.
    - location: The specific Virginia city or county the district serves.
    - enrollment: Total student enrollment numbers (e.g., '12,345' or 'about 12k').
    - number_of_schools: Total number of schools operated by the district.
    - grade_levels: Grade levels served (e.g., 'PK-12', 'KG-12').
    - graduation_rate: High school graduation rate.
    - demographics: Student demographic information (e.g., 'Minority enrollment 35%').
    - accreditation_status: Accreditation status according to Virginia Department of Education standards.
    - ranking_score: Niche ranking score or rating for the district (for 2026 if present).
    - per_student_expenditure: Expenditure per student (e.g., '$13,500').
    - superintendent: Current superintendent or chief administrator name.
    - official_website: Official district website URL in the 'value' field if present; sources should include supporting URLs from the answer (can include the website itself).
    - contact_info: Primary district contact info (address or phone).
    - performance_framework: Virginia School Performance and Support Framework data for schools in the district (if applicable; otherwise value can be null).
    """


# --------------------------------------------------------------------------- #
# Helper functions for verification                                           #
# --------------------------------------------------------------------------- #
async def verify_district_identification(
    evaluator: Evaluator,
    parent_node,
    ident: DistrictIdentification,
) -> None:
    """
    Build and verify the 'District_Identification' sub-tree:
    - Existence check (critical)
    - Niche #1 ranking verification (critical, using URLs)
    """
    node = evaluator.add_sequential(
        id="District_Identification",
        desc="Correctly identify the highest-ranked Virginia public school district according to Niche 2026 rankings",
        parent=parent_node,
        critical=True  # Critical step; if this fails, the rest should be skipped
    )

    district_name = (ident.district_name or "").strip()
    niche_sources = ident.niche_urls if ident.niche_urls else []

    # Existence check: district name present AND at least one Niche URL
    evaluator.add_custom_node(
        result=(bool(district_name) and len(niche_sources) > 0),
        id="District_Identification_Exists",
        desc="District identification and at least one Niche ranking source URL are provided",
        parent=node,
        critical=True
    )

    # Verify #1 ranking according to Niche 2026 Virginia page(s)
    niche_verify_leaf = evaluator.add_leaf(
        id="District_Identification_TopRank_Verification",
        desc="District is ranked #1 in Niche's 2026 Best School Districts in Virginia",
        parent=node,
        critical=True
    )
    claim = (
        f"According to Niche's 2026 Best School Districts in Virginia, '{district_name}' is ranked #1 in Virginia."
    )
    await evaluator.verify(
        claim=claim,
        node=niche_verify_leaf,
        sources=niche_sources,
        additional_instruction=(
            "Verify directly on the provided Niche ranking page(s) for Virginia (2026) that the district is #1. "
            "Allow minor name variations (e.g., capitalization). If the pages are irrelevant or do not show #1, return 'Incorrect'."
        ),
    )


async def _verify_field_leaf(
    evaluator: Evaluator,
    parent_node,
    field: Optional[DataField],
    node_id: str,
    node_desc: str,
    claim_text: str,
    additional_instruction: Optional[str] = None,
    override_sources: Optional[List[str] | str] = None,
) -> None:
    """
    Create a leaf node and attempt verification of a single field claim.
    If sources are missing, still call verify() but instruct the judge to return 'Incorrect'.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit across categories
    )

    # Sources handling
    sources_to_use: Optional[List[str] | str] = None
    if override_sources is not None:
        sources_to_use = override_sources
    else:
        if field and field.sources:
            sources_to_use = field.sources
        else:
            sources_to_use = None  # This triggers simple_verify; we mitigate via instruction

    add_ins = additional_instruction or DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=sources_to_use,
        additional_instruction=add_ins,
    )


async def verify_comprehensive_info(
    evaluator: Evaluator,
    parent_node,
    district_name: str,
    report: DistrictReport
) -> None:
    """
    Build the 'Comprehensive_Verification' parallel sub-tree and verify each category as a single binary leaf.
    """
    comp_node = evaluator.add_parallel(
        id="Comprehensive_Verification",
        desc="Verify comprehensive information about the identified district across multiple categories",
        parent=parent_node,
        critical=False  # Non-critical to enable partial credit across many categories
    )

    # Official Name
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.official_name,
        node_id="Official_Name_Verification",
        node_desc="Provide the official name of the school district",
        claim_text=f"The district's official name is '{(report.official_name.value if report.official_name else '')}'.",
    )

    # Geographic Location
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.location,
        node_id="Geographic_Location",
        node_desc="Specify the city or county served by the district in Virginia",
        claim_text=f"The district serves '{(report.location.value if report.location else '')}' in Virginia.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer official district, Virginia DOE, or authoritative government sources."
    )

    # Enrollment Statistics
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.enrollment,
        node_id="Enrollment_Statistics",
        node_desc="Provide total student enrollment numbers with source citation (NCES or official district source)",
        claim_text=f"The district's total student enrollment is '{(report.enrollment.value if report.enrollment else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer NCES, Virginia DOE, or official district sources; allow reasonable rounding."
    )

    # Number of Schools
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.number_of_schools,
        node_id="Number_of_Schools",
        node_desc="State the total number of schools operated by the district",
        claim_text=f"The district operates '{(report.number_of_schools.value if report.number_of_schools else '')}' schools.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer official district or Virginia DOE sources."
    )

    # Grade Levels Served
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.grade_levels,
        node_id="Grade_Levels_Served",
        node_desc="Specify the range of grade levels served",
        claim_text=f"The district serves grade levels '{(report.grade_levels.value if report.grade_levels else '')}'.",
    )

    # Graduation Rate
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.graduation_rate,
        node_id="Graduation_Rate",
        node_desc="Provide high school graduation rate data with source reference",
        claim_text=f"The district's high school graduation rate is '{(report.graduation_rate.value if report.graduation_rate else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer Virginia DOE or official district sources; allow rounding."
    )

    # Student Demographics
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.demographics,
        node_id="Student_Demographics",
        node_desc="Provide student demographic information such as minority enrollment percentage",
        claim_text=f"Student demographics: '{(report.demographics.value if report.demographics else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer NCES, Virginia DOE, or official district sources."
    )

    # Accreditation Status
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.accreditation_status,
        node_id="Accreditation_Status",
        node_desc="State the district's or its schools' accreditation status according to Virginia DOE standards",
        claim_text=f"Accreditation status according to Virginia DOE: '{(report.accreditation_status.value if report.accreditation_status else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer Virginia DOE sources (e.g., School Quality Profiles or official accreditation pages)."
    )

    # Ranking Score (Niche)
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.ranking_score,
        node_id="Ranking_Score",
        node_desc="Provide the Niche ranking score or rating for the district",
        claim_text=f"Niche 2026 ranking score/rating for the district is '{(report.ranking_score.value if report.ranking_score else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer Niche pages; verify the rating/score pertains to 2026 or the page cited."
    )

    # Per Student Expenditure
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.per_student_expenditure,
        node_id="Per_Student_Expenditure",
        node_desc="Provide financial data on expenditure per student if available",
        claim_text=f"Per-student expenditure is '{(report.per_student_expenditure.value if report.per_student_expenditure else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer official district budget/finance pages, Virginia DOE, or NCES; allow rounding."
    )

    # Superintendent Information
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.superintendent,
        node_id="Superintendent_Information",
        node_desc="Identify the current superintendent or chief administrator of the district",
        claim_text=f"The current superintendent (or chief administrator) is '{(report.superintendent.value if report.superintendent else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer official district leadership pages or credible recent announcements."
    )

    # Official Website (verify the URL itself)
    official_website_url = report.official_website.value if report.official_website else None
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.official_website,
        node_id="Official_Website",
        node_desc="Provide the URL of the district's official website",
        claim_text=f"This URL is the official website of {district_name}: '{official_website_url or ''}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " If the provided value is not a valid URL or belongs to an unrelated entity, return 'Incorrect'.",
        override_sources=official_website_url if official_website_url else None
    )

    # District Contact Information
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.contact_info,
        node_id="District_Contact_Information",
        node_desc="Provide contact information such as main phone number or address",
        claim_text=f"The district's main contact information is '{(report.contact_info.value if report.contact_info else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer official district contact/administration pages."
    )

    # Performance Framework Data
    await _verify_field_leaf(
        evaluator=evaluator,
        parent_node=comp_node,
        field=report.performance_framework,
        node_id="Performance_Framework_Data",
        node_desc="Provide Virginia School Performance and Support Framework scores or ratings for schools in the district (if applicable)",
        claim_text=f"Virginia School Performance and Support Framework data: '{(report.performance_framework.value if report.performance_framework else '')}'.",
        additional_instruction=DEFAULT_VERIFY_INSTRUCTION_WITH_SOURCE_REQUIREMENT +
        " Prefer Virginia DOE School Quality Profiles or VSP&SF official documentation for schools in the district."
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
    Evaluate an answer for the top Virginia school district (Niche 2026) and its comprehensive documentation.
    """
    # Initialize evaluator with root sequential aggregation (non-critical root to allow mixed children criticality)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract identification
    ident = await evaluator.extract(
        prompt=prompt_extract_identification(),
        template_class=DistrictIdentification,
        extraction_name="district_identification"
    )

    # Extract comprehensive report
    report = await evaluator.extract(
        prompt=prompt_extract_district_report(),
        template_class=DistrictReport,
        extraction_name="district_report"
    )

    # Verification: District Identification
    await verify_district_identification(evaluator, root, ident)

    # Verification: Comprehensive Information (skipped automatically if identification fails due to root sequential)
    district_name_for_claims = ident.district_name or "the district"
    await verify_comprehensive_info(evaluator, root, district_name_for_claims, report)

    # Optional: record custom info
    evaluator.add_custom_info(
        info={
            "niche_sources_extracted": ident.niche_urls,
            "district_name_extracted": ident.district_name
        },
        info_type="extraction_metadata",
        info_name="identification_metadata"
    )

    # Return structured summary
    return evaluator.get_summary()