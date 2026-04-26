import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fha_multifamily_va_analysis"
TASK_DESCRIPTION = """
You are planning to purchase a 4-unit multifamily property in Richmond, Virginia, and you intend to use FHA financing while living in one unit and renting out the other three. Provide a comprehensive analysis that includes: (1) the eligibility requirements for using an FHA loan on this type of property, including all owner-occupancy obligations and timelines; (2) the minimum down payment percentage required and the 2025 conforming loan limit for properties in most U.S. areas; (3) the licensing requirements in Virginia if you want to manage the rental units yourself, including specific education hour requirements for different license types and minimum age; (4) the occupancy rate standards that indicate solid property management and what lenders typically require for stabilized properties; and (5) the three internationally accepted property valuation methods and which approach is most appropriate for income-producing rental properties. For each requirement, include supporting reference URLs from your research.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class FHAPropertyType(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class OwnerOccupancy(BaseModel):
    primary_residence_statement: Optional[str] = None
    primary_residence_urls: List[str] = Field(default_factory=list)

    move_in_timeline_statement: Optional[str] = None
    move_in_timeline_urls: List[str] = Field(default_factory=list)

    min_occupancy_duration_statement: Optional[str] = None
    min_occupancy_duration_urls: List[str] = Field(default_factory=list)


class FHAEligibility(BaseModel):
    property_type: Optional[FHAPropertyType] = None
    owner_occupancy: Optional[OwnerOccupancy] = None


class DownPaymentAndLimits(BaseModel):
    min_down_payment_percentage: Optional[str] = None
    min_down_payment_urls: List[str] = Field(default_factory=list)

    conforming_loan_limit_2025_statement: Optional[str] = None
    conforming_loan_limit_2025_urls: List[str] = Field(default_factory=list)


class VAManagementLicensing(BaseModel):
    license_requirement_statement: Optional[str] = None
    license_requirement_urls: List[str] = Field(default_factory=list)

    broker_education_hours: Optional[str] = None
    broker_education_urls: List[str] = Field(default_factory=list)

    salesperson_education_hours: Optional[str] = None
    salesperson_education_urls: List[str] = Field(default_factory=list)

    age_requirement: Optional[str] = None
    age_requirement_urls: List[str] = Field(default_factory=list)

    exam_requirement_statement: Optional[str] = None
    exam_requirement_urls: List[str] = Field(default_factory=list)


class RentalStandards(BaseModel):
    economic_occupancy_standard_statement: Optional[str] = None
    economic_occupancy_urls: List[str] = Field(default_factory=list)

    lender_stabilized_occupancy_requirement_statement: Optional[str] = None
    lender_stabilized_occupancy_urls: List[str] = Field(default_factory=list)


class ValuationMethods(BaseModel):
    three_methods_list: List[str] = Field(default_factory=list)
    valuation_methods_urls: List[str] = Field(default_factory=list)

    income_approach_best_for_rentals_statement: Optional[str] = None
    income_approach_best_for_rentals_urls: List[str] = Field(default_factory=list)


class AnalysisExtraction(BaseModel):
    fha_eligibility: Optional[FHAEligibility] = None
    down_payment: Optional[DownPaymentAndLimits] = None
    va_licensing: Optional[VAManagementLicensing] = None
    rental_standards: Optional[RentalStandards] = None
    valuation_methods: Optional[ValuationMethods] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_analysis() -> str:
    return """
    Extract the structured facts and their supporting source URLs exactly as stated in the answer. Return null for any missing field and an empty array for missing URL lists.

    REQUIRED OUTPUT SCHEMA (field names must match exactly):
    {
      "fha_eligibility": {
        "property_type": {
          "statement": "text stating FHA property type requirement for multifamily (e.g., 'FHA permits 1–4 unit properties'). Use the phrasing from the answer if possible.",
          "urls": ["list of URLs cited for this point"]
        },
        "owner_occupancy": {
          "primary_residence_statement": "text stating that borrower must occupy one unit as primary residence",
          "primary_residence_urls": ["URLs supporting this requirement"],
          "move_in_timeline_statement": "text stating the move-in timeline (e.g., within 60 days of closing)",
          "move_in_timeline_urls": ["URLs supporting this timeline"],
          "min_occupancy_duration_statement": "text stating minimum occupancy duration (e.g., at least one year)",
          "min_occupancy_duration_urls": ["URLs supporting this duration"]
        }
      },
      "down_payment": {
        "min_down_payment_percentage": "text for minimum FHA down payment percentage, e.g., '3.5%' (keep symbols as in the answer)",
        "min_down_payment_urls": ["URLs supporting the minimum down payment"],
        "conforming_loan_limit_2025_statement": "text describing the 2025 conforming loan limit baseline for one-unit properties in most U.S. areas, including the number (e.g., '$806,500')",
        "conforming_loan_limit_2025_urls": ["URLs supporting this 2025 conforming loan limit statement"]
      },
      "va_licensing": {
        "license_requirement_statement": "text describing whether Virginia requires a real estate license for property management activities",
        "license_requirement_urls": ["URLs supporting the licensing requirement statement"],
        "broker_education_hours": "text for required broker pre-licensing education hours (e.g., '180 hours')",
        "broker_education_urls": ["URLs supporting broker education hours"],
        "salesperson_education_hours": "text for required salesperson pre-licensing education hours (e.g., '60 hours')",
        "salesperson_education_urls": ["URLs supporting salesperson education hours"],
        "age_requirement": "text for the minimum age (e.g., '18 years old')",
        "age_requirement_urls": ["URLs supporting the age requirement"],
        "exam_requirement_statement": "text stating the exam requirement (e.g., 'must pass the real estate licensing exam')",
        "exam_requirement_urls": ["URLs supporting the exam requirement"]
      },
      "rental_standards": {
        "economic_occupancy_standard_statement": "text describing what economic occupancy rate signals solid management (e.g., '>90%')",
        "economic_occupancy_urls": ["URLs supporting economic occupancy standard"],
        "lender_stabilized_occupancy_requirement_statement": "text describing lenders' typical stabilized occupancy requirement (e.g., '85–95%')",
        "lender_stabilized_occupancy_urls": ["URLs supporting lenders' occupancy standard"]
      },
      "valuation_methods": {
        "three_methods_list": ["list of the three methods as stated, typically ['Sales Comparison', 'Cost Approach', 'Income Approach']"],
        "valuation_methods_urls": ["URLs supporting the three-methods framework"],
        "income_approach_best_for_rentals_statement": "text stating that Income Approach is most appropriate for income-producing properties",
        "income_approach_best_for_rentals_urls": ["URLs supporting the appropriateness of the income approach"]
      }
    }

    RULES:
    - Extract exactly from the answer text. Do not invent any numbers or URLs.
    - For URLs, return only valid, complete URLs that appear in the answer (plain or markdown).
    - If the answer references a source without a URL, return an empty array for that URL list.
    - Keep numerals and symbols (%, $) exactly as stated in the answer when present.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return urls or []


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


def _add_source_presence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: Optional[List[str]],
    critical: bool = True,
):
    return evaluator.add_custom_node(
        result=_has_sources(urls),
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_fha_eligibility(
    evaluator: Evaluator,
    parent,
    ex: AnalysisExtraction,
):
    fha_node = evaluator.add_parallel(
        id="FHA_Loan_Eligibility",
        desc="FHA loan eligibility requirements for multifamily properties",
        parent=parent,
        critical=True,
    )

    # Property Type Verification
    prop_node = evaluator.add_parallel(
        id="Property_Type_Verification",
        desc="Verify the property must be 1-4 unit residential multifamily (duplex, triplex, or fourplex)",
        parent=fha_node,
        critical=True,
    )
    pt = ex.fha_eligibility.property_type if ex.fha_eligibility and ex.fha_eligibility.property_type else FHAPropertyType()
    _add_source_presence_node(
        evaluator,
        prop_node,
        "URL_Reference_Property_Type_present",
        "At least one URL provided for FHA multifamily property type requirements",
        pt.urls,
        critical=True,
    )
    prop_leaf = evaluator.add_leaf(
        id="URL_Reference_Property_Type",
        desc="Provide URL reference for FHA multifamily property type requirements",
        parent=prop_node,
        critical=True,
    )
    prop_claim = pt.statement or "FHA-insured mortgages can be used for 1–4 unit residential properties (including duplex, triplex, and fourplex) when owner-occupied."
    await evaluator.verify(
        claim=prop_claim,
        node=prop_leaf,
        sources=_safe_list(pt.urls),
        additional_instruction="Verify that the page states FHA financing applies to 1–4 unit residential properties (duplex, triplex, fourplex). Ignore 5+ unit commercial programs.",
    )

    # Owner Occupancy Requirements
    owner_node = evaluator.add_parallel(
        id="Owner_Occupancy_Requirements",
        desc="Owner occupancy requirements for FHA multifamily financing",
        parent=fha_node,
        critical=True,
    )
    oo = ex.fha_eligibility.owner_occupancy if ex.fha_eligibility and ex.fha_eligibility.owner_occupancy else OwnerOccupancy()

    # Primary residence requirement
    _add_source_presence_node(
        evaluator,
        owner_node,
        "URL_Reference_Primary_Residence_present",
        "At least one URL provided for primary residence requirement",
        oo.primary_residence_urls,
        critical=True,
    )
    primary_leaf = evaluator.add_leaf(
        id="URL_Reference_Primary_Residence",
        desc="Provide URL reference for primary residence requirement",
        parent=owner_node,
        critical=True,
    )
    primary_claim = oo.primary_residence_statement or "To use an FHA loan on a 2–4 unit property, the borrower must occupy one unit as their primary residence."
    await evaluator.verify(
        claim=primary_claim,
        node=primary_leaf,
        sources=_safe_list(oo.primary_residence_urls),
        additional_instruction="Look for language that the FHA borrower must occupy the property (one unit) as their primary residence.",
    )

    # Move-in timeline (60 days)
    _add_source_presence_node(
        evaluator,
        owner_node,
        "URL_Reference_Timeline_present",
        "At least one URL provided for move-in timeline requirement",
        oo.move_in_timeline_urls,
        critical=True,
    )
    timeline_leaf = evaluator.add_leaf(
        id="URL_Reference_Timeline",
        desc="Provide URL reference for move-in timeline requirement",
        parent=owner_node,
        critical=True,
    )
    timeline_claim = oo.move_in_timeline_statement or "The borrower must move into the property within 60 days of closing."
    await evaluator.verify(
        claim=timeline_claim,
        node=timeline_leaf,
        sources=_safe_list(oo.move_in_timeline_urls),
        additional_instruction="Verify the source indicates an occupancy timeline around 'within 60 days' after closing for FHA owner-occupancy.",
    )

    # Minimum occupancy duration (1 year)
    _add_source_presence_node(
        evaluator,
        owner_node,
        "URL_Reference_Duration_present",
        "At least one URL provided for minimum occupancy duration",
        oo.min_occupancy_duration_urls,
        critical=True,
    )
    duration_leaf = evaluator.add_leaf(
        id="URL_Reference_Duration",
        desc="Provide URL reference for minimum occupancy duration",
        parent=owner_node,
        critical=True,
    )
    duration_claim = oo.min_occupancy_duration_statement or "The borrower must live in the property as their primary residence for at least one year after closing."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=_safe_list(oo.min_occupancy_duration_urls),
        additional_instruction="Verify minimum occupancy duration (commonly at least 12 months or one year) for FHA owner-occupancy.",
    )


async def build_down_payment_and_limits(
    evaluator: Evaluator,
    parent,
    ex: AnalysisExtraction,
):
    dp_node = evaluator.add_parallel(
        id="Down_Payment_Requirements",
        desc="FHA down payment requirements for multifamily properties",
        parent=parent,
        critical=True,
    )
    dp = ex.down_payment if ex.down_payment else DownPaymentAndLimits()

    # Minimum down payment percentage
    _add_source_presence_node(
        evaluator,
        dp_node,
        "URL_Reference_Down_Payment_present",
        "At least one URL provided for FHA minimum down payment requirement",
        dp.min_down_payment_urls,
        critical=True,
    )
    down_leaf = evaluator.add_leaf(
        id="URL_Reference_Down_Payment",
        desc="Provide URL reference for FHA minimum down payment requirement",
        parent=dp_node,
        critical=True,
    )
    down_claim = dp.min_down_payment_percentage or "The minimum down payment for FHA-qualified buyers is 3.5% (subject to standard credit and underwriting criteria)."
    await evaluator.verify(
        claim=down_claim,
        node=down_leaf,
        sources=_safe_list(dp.min_down_payment_urls),
        additional_instruction="Verify the stated minimum FHA down payment (commonly 3.5% for borrowers meeting credit thresholds). If the source mentions credit score conditions, that's acceptable.",
    )

    # 2025 conforming loan limit (most areas)
    _add_source_presence_node(
        evaluator,
        dp_node,
        "URL_Reference_Loan_Limit_present",
        "At least one URL provided for 2025 conforming loan limits",
        dp.conforming_loan_limit_2025_urls,
        critical=True,
    )
    limit_leaf = evaluator.add_leaf(
        id="URL_Reference_Loan_Limit",
        desc="Provide URL reference for 2025 conforming loan limits",
        parent=dp_node,
        critical=True,
    )
    limit_claim = dp.conforming_loan_limit_2025_statement or "The 2025 baseline conforming loan limit for one-unit properties in most U.S. areas is as stated in the answer."
    await evaluator.verify(
        claim=limit_claim,
        node=limit_leaf,
        sources=_safe_list(dp.conforming_loan_limit_2025_urls),
        additional_instruction="Verify the 2025 baseline conforming loan limit (one-unit, most U.S. counties). Accept authoritative FHFA/Fannie Mae/Freddie Mac references. Focus on baseline (not high-cost).",
    )


async def build_va_licensing(
    evaluator: Evaluator,
    parent,
    ex: AnalysisExtraction,
):
    va_node = evaluator.add_sequential(
        id="Property_Management_Licensing_Virginia",
        desc="Property management licensing requirements in Virginia",
        parent=parent,
        critical=True,
    )
    va = ex.va_licensing if ex.va_licensing else VAManagementLicensing()

    # Step 1: License requirement
    _add_source_presence_node(
        evaluator,
        va_node,
        "URL_Reference_License_Requirement_present",
        "At least one URL provided for Virginia property management license requirement",
        va.license_requirement_urls,
        critical=True,
    )
    lic_req_leaf = evaluator.add_leaf(
        id="URL_Reference_License_Requirement",
        desc="Provide URL reference for Virginia property management license requirement",
        parent=va_node,
        critical=True,
    )
    lic_req_claim = va.license_requirement_statement or "In Virginia, engaging in property management activities for others and for compensation requires a real estate license."
    await evaluator.verify(
        claim=lic_req_claim,
        node=lic_req_leaf,
        sources=_safe_list(va.license_requirement_urls),
        additional_instruction="Verify via Virginia DPOR or Virginia Code that property management for others for compensation requires a real estate license; self-management of one's own property may be exempt.",
    )

    # Step 2: License type options (parallel)
    types_node = evaluator.add_parallel(
        id="License_Type_Options",
        desc="Can obtain either broker license or salesperson license (under broker supervision)",
        parent=va_node,
        critical=True,
    )

    # Broker education hours
    _add_source_presence_node(
        evaluator,
        types_node,
        "URL_Reference_Broker_Education_present",
        "At least one URL provided for broker education hours requirement",
        va.broker_education_urls,
        critical=True,
    )
    broker_leaf = evaluator.add_leaf(
        id="URL_Reference_Broker_Education",
        desc="Provide URL reference for broker education hours requirement",
        parent=types_node,
        critical=True,
    )
    broker_claim = va.broker_education_hours or "Virginia broker license requires 180 hours of approved pre-licensing education."
    await evaluator.verify(
        claim=broker_claim,
        node=broker_leaf,
        sources=_safe_list(va.broker_education_urls),
        additional_instruction="Verify the total required pre-licensing education hours for a Virginia real estate broker (commonly 180 hours).",
    )

    # Salesperson education hours
    _add_source_presence_node(
        evaluator,
        types_node,
        "URL_Reference_Salesperson_Education_present",
        "At least one URL provided for salesperson education hours requirement",
        va.salesperson_education_urls,
        critical=True,
    )
    sales_leaf = evaluator.add_leaf(
        id="URL_Reference_Salesperson_Education",
        desc="Provide URL reference for salesperson education hours requirement",
        parent=types_node,
        critical=True,
    )
    sales_claim = va.salesperson_education_hours or "Virginia salesperson license requires 60 hours of approved pre-licensing coursework."
    await evaluator.verify(
        claim=sales_claim,
        node=sales_leaf,
        sources=_safe_list(va.salesperson_education_urls),
        additional_instruction="Verify the required pre-licensing hours for a Virginia real estate salesperson (commonly 60 hours).",
    )

    # Age requirement
    _add_source_presence_node(
        evaluator,
        types_node,
        "Age_Requirement_source_present",
        "At least one URL provided for minimum age requirement",
        va.age_requirement_urls,
        critical=True,
    )
    age_leaf = evaluator.add_leaf(
        id="Age_Requirement",
        desc="Must be at least 18 years old",
        parent=types_node,
        critical=True,
    )
    age_claim = va.age_requirement or "You must be at least 18 years old to qualify for a Virginia real estate license."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        sources=_safe_list(va.age_requirement_urls),
        additional_instruction="Verify the minimum age requirement for Virginia real estate licensing (commonly 18 years old).",
    )

    # Exam requirement
    _add_source_presence_node(
        evaluator,
        types_node,
        "Exam_Requirement_source_present",
        "At least one URL provided for real estate exam requirement",
        va.exam_requirement_urls,
        critical=True,
    )
    exam_leaf = evaluator.add_leaf(
        id="Exam_Requirement",
        desc="Must pass the real estate exam",
        parent=types_node,
        critical=True,
    )
    exam_claim = va.exam_requirement_statement or "You must pass the Virginia real estate licensing exam."
    await evaluator.verify(
        claim=exam_claim,
        node=exam_leaf,
        sources=_safe_list(va.exam_requirement_urls),
        additional_instruction="Verify that passing the Virginia real estate licensing exam is required.",
    )


async def build_rental_operations(
    evaluator: Evaluator,
    parent,
    ex: AnalysisExtraction,
):
    rent_node = evaluator.add_parallel(
        id="Rental_Operations_Standards",
        desc="Standards and requirements for rental operations in multifamily properties",
        parent=parent,
        critical=True,
    )
    rs = ex.rental_standards if ex.rental_standards else RentalStandards()

    # Economic occupancy rate standard
    _add_source_presence_node(
        evaluator,
        rent_node,
        "URL_Reference_Occupancy_present",
        "At least one URL provided for occupancy rate standards",
        rs.economic_occupancy_urls,
        critical=True,
    )
    econ_leaf = evaluator.add_leaf(
        id="URL_Reference_Occupancy",
        desc="Provide URL reference for occupancy rate standards",
        parent=rent_node,
        critical=True,
    )
    econ_claim = rs.economic_occupancy_standard_statement or "An economic occupancy rate above 90% indicates solid property management performance."
    await evaluator.verify(
        claim=econ_claim,
        node=econ_leaf,
        sources=_safe_list(rs.economic_occupancy_urls),
        additional_instruction="Verify that sources suggest ~90%+ economic occupancy is indicative of strong/stable performance. Allow minor variations in phrasing.",
    )

    # Lender occupancy requirements for stabilized properties
    _add_source_presence_node(
        evaluator,
        rent_node,
        "URL_Reference_Lender_Standards_present",
        "At least one URL provided for lender occupancy requirements",
        rs.lender_stabilized_occupancy_urls,
        critical=True,
    )
    lender_leaf = evaluator.add_leaf(
        id="URL_Reference_Lender_Standards",
        desc="Provide URL reference for lender occupancy requirements",
        parent=rent_node,
        critical=True,
    )
    lender_claim = rs.lender_stabilized_occupancy_requirement_statement or "Commercial lenders typically require 85–95% minimum occupancy for stabilized multifamily properties."
    await evaluator.verify(
        claim=lender_claim,
        node=lender_leaf,
        sources=_safe_list(rs.lender_stabilized_occupancy_urls),
        additional_instruction="Verify typical lender stabilized occupancy requirements (often around 85%–95%). Accept ranges or typical thresholds.",
    )


async def build_valuation_methods(
    evaluator: Evaluator,
    parent,
    ex: AnalysisExtraction,
):
    val_node = evaluator.add_parallel(
        id="Property_Valuation_Method",
        desc="Appropriate property valuation method for multifamily investment properties",
        parent=parent,
        critical=True,
    )
    vm = ex.valuation_methods if ex.valuation_methods else ValuationMethods()

    # Three accepted methods
    three_node = evaluator.add_parallel(
        id="Three_Accepted_Methods",
        desc="Three internationally accepted methods: Sales Comparison, Cost Approach, Income Approach",
        parent=val_node,
        critical=True,
    )
    _add_source_presence_node(
        evaluator,
        three_node,
        "URL_Reference_Valuation_Methods_present",
        "At least one URL provided for the three property valuation methods",
        vm.valuation_methods_urls,
        critical=True,
    )
    methods_leaf = evaluator.add_leaf(
        id="URL_Reference_Valuation_Methods",
        desc="Provide URL reference for the three property valuation methods",
        parent=three_node,
        critical=True,
    )
    reported_methods = vm.three_methods_list or ["Sales Comparison", "Cost Approach", "Income Approach"]
    methods_claim = f"The three generally accepted real estate appraisal approaches are: {reported_methods}."
    await evaluator.verify(
        claim=methods_claim,
        node=methods_leaf,
        sources=_safe_list(vm.valuation_methods_urls),
        additional_instruction="Verify that the source lists the three standard appraisal approaches: Sales Comparison (aka Market or Comparative), Cost Approach, and Income (Income Capitalization) Approach. Allow synonyms.",
    )

    # Income approach most appropriate for rentals
    _add_source_presence_node(
        evaluator,
        val_node,
        "Income_Approach_For_Rental_source_present",
        "At least one URL provided for 'Income Approach is most appropriate for rental/investment properties'",
        vm.income_approach_best_for_rentals_urls,
        critical=True,
    )
    income_leaf = evaluator.add_leaf(
        id="Income_Approach_For_Rental",
        desc="Income Approach is most appropriate for rental/investment properties",
        parent=val_node,
        critical=True,
    )
    income_claim = vm.income_approach_best_for_rentals_statement or "For income-producing rental properties, the Income Approach is generally the most appropriate primary valuation method."
    await evaluator.verify(
        claim=income_claim,
        node=income_leaf,
        sources=_safe_list(vm.income_approach_best_for_rentals_urls),
        additional_instruction="Verify that the source states or strongly implies the income approach is the most appropriate/relied-upon approach for income-producing rental or commercial properties.",
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
    Evaluate an answer for the FHA-financed owner-occupied multifamily investment analysis task.
    """
    # Initialize evaluator (root is a neutral container)
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

    # Extract structured information from the answer
    extracted: AnalysisExtraction = await evaluator.extract(
        prompt=prompt_extract_analysis(),
        template_class=AnalysisExtraction,
        extraction_name="analysis_extraction",
    )

    # Build top-level critical analysis node (since initialize() root is non-critical)
    analysis_root = evaluator.add_parallel(
        id="FHA_Multifamily_Investment_Analysis",
        desc="Complete analysis of FHA-financed owner-occupied multifamily property investment requirements and standards",
        parent=root,
        critical=True,
    )

    # Build and verify each section
    await build_fha_eligibility(evaluator, analysis_root, extracted)
    await build_down_payment_and_limits(evaluator, analysis_root, extracted)
    await build_va_licensing(evaluator, analysis_root, extracted)
    await build_rental_operations(evaluator, analysis_root, extracted)
    await build_valuation_methods(evaluator, analysis_root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()