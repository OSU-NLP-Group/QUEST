import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "tx_dc_tier3_compliance"
TASK_DESCRIPTION = (
    "A technology company is planning to develop a new data center facility in Travis County, Texas, with the following specifications: "
    "facility size of 150,000 square feet, target certification of Uptime Institute Tier III, power capacity requirement of 10 megawatts (MW), "
    "target timeline of 48 months from project initiation to operational status, and intended designation as a Texas Qualifying Data Center for sales tax exemption. "
    "Provide a comprehensive compliance checklist that includes: "
    "(1) Texas Regulatory Requirements - identify all minimum facility requirements under Texas Comptroller qualifying data center regulations, including minimum square footage, "
    "occupancy structure rules, capital investment eligibility criteria, and sales tax exemption qualifications; "
    "(2) Employment Obligations - calculate the exact employment requirements, including the minimum number of qualifying jobs that must be created, the minimum weekly wage for each job "
    "based on current Travis County average weekly wage data, the required annual hours per job, the retention period for each job, and the timeframe within which all jobs must be created; "
    "(3) Tier III Technical Requirements - list all Uptime Institute Tier III topology requirements, including concurrent maintainability standards, redundant distribution path specifications, "
    "and all required redundant capacity components for power systems, cooling systems, and support systems; "
    "(4) Power and Cooling Specifications - define the electrical infrastructure requirements for a 10 MW data center (including voltage and phase requirements), confirm the facility's size category, "
    "specify grid reliability requirements, define the temperature control range, confirm continuous cooling requirements, and specify cooling redundancy levels; "
    "(5) Development Timeline - provide estimated duration ranges for each standard development phase (site selection, permitting and approvals, design, and construction), state the total typical development timeline range, "
    "assess whether the 48-month target is feasible, and identify any timeline risks or considerations. For each major section, provide at least one authoritative URL reference that supports the requirements stated."
)


# ----------------------------- Extraction Models ----------------------------- #
class TexasRegulatoryExtraction(BaseModel):
    minimum_facility_size_statement: Optional[str] = None
    single_occupant_rule_statement: Optional[str] = None
    capital_investment_eligibility_statement: Optional[str] = None
    electricity_use_threshold_statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class EmploymentExtraction(BaseModel):
    minimum_job_count_statement: Optional[str] = None
    job_creation_window_statement: Optional[str] = None
    travis_avg_weekly_wage_value: Optional[str] = None
    mentions_120_percent: Optional[bool] = None
    computed_min_weekly_wage_value: Optional[str] = None
    annual_hour_requirement_statement: Optional[str] = None
    retention_period_statement: Optional[str] = None
    vacancy_fill_window_statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class Tier3Extraction(BaseModel):
    concurrent_maintainability_statement: Optional[str] = None
    independent_distribution_paths_statement: Optional[str] = None
    redundant_power_components: List[str] = Field(default_factory=list)
    redundant_cooling_components: List[str] = Field(default_factory=list)
    redundant_support_components: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class PowerCoolingExtraction(BaseModel):
    electrical_phase_statement: Optional[str] = None
    voltage_requirement_statement: Optional[str] = None
    grid_reliability_statement: Optional[str] = None
    size_category_statement: Optional[str] = None
    temperature_range_statement: Optional[str] = None
    continuous_cooling_statement: Optional[str] = None
    cooling_redundancy_statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class TimelineExtraction(BaseModel):
    site_selection_range_statement: Optional[str] = None
    permitting_range_statement: Optional[str] = None
    design_range_statement: Optional[str] = None
    construction_range_statement: Optional[str] = None
    total_timeline_range_statement: Optional[str] = None
    feasibility_assessment_statement: Optional[str] = None
    timeline_risks_statement: Optional[str] = None
    references: List[str] = Field(default_factory=list)


class DataCenterChecklistExtraction(BaseModel):
    texas_regulatory: Optional[TexasRegulatoryExtraction] = None
    employment: Optional[EmploymentExtraction] = None
    tier3: Optional[Tier3Extraction] = None
    power_cooling: Optional[PowerCoolingExtraction] = None
    timeline: Optional[TimelineExtraction] = None


# --------------------------- Extraction Prompt --------------------------- #
def prompt_extract_compliance_checklist() -> str:
    return (
        "Extract a structured compliance checklist from the answer organized into five major sections. "
        "For each section, capture the requested statements exactly as phrased and list any authoritative URL references mentioned. "
        "Return a JSON with fields: texas_regulatory, employment, tier3, power_cooling, timeline. "
        "Within texas_regulatory, extract: "
        "- minimum_facility_size_statement (text describing the 100,000 sq ft single building or portion requirement), "
        "- single_occupant_rule_statement (text stating one occupant per qualifying data center), "
        "- capital_investment_eligibility_statement (text stating IRC Section 179/1245/1250 property eligibility), "
        "- electricity_use_threshold_statement (text stating >50% electricity used for data center operations for sales tax exemption), "
        "- references (all URLs provided for this section). "
        "Within employment, extract: "
        "- minimum_job_count_statement (text stating minimum 20 qualifying jobs), "
        "- job_creation_window_statement (text stating jobs created within first five years of certification), "
        "- travis_avg_weekly_wage_value (the numeric or textual value used for Travis County average weekly wage), "
        "- mentions_120_percent (true if the answer explicitly mentions 120% or equivalent 1.2x requirement), "
        "- computed_min_weekly_wage_value (the numeric or textual computed 120% minimum weekly wage), "
        "- annual_hour_requirement_statement (text stating at least 1,820 hours/year), "
        "- retention_period_statement (text stating at least five years retention), "
        "- vacancy_fill_window_statement (text stating vacancies filled within 120 days), "
        "- references (URLs for employment requirements and/or wage data). "
        "Within tier3, extract: "
        "- concurrent_maintainability_statement (text stating maintenance without shutdown of critical environment), "
        "- independent_distribution_paths_statement (text stating multiple independent distribution paths for power and cooling), "
        "- redundant_power_components (list of component names included; expect generators, energy storage, UPS modules), "
        "- redundant_cooling_components (list; expect chillers, cooling units, heat rejection equipment), "
        "- redundant_support_components (list; expect pumps, fuel tanks, fuel cells), "
        "- references (URLs supporting Tier III requirements). "
        "Within power_cooling, extract: "
        "- electrical_phase_statement (text stating three-phase service), "
        "- voltage_requirement_statement (text that includes at least one explicit voltage level or range in V or kV), "
        "- grid_reliability_statement (text stating access to reliable grid with adequate capacity), "
        "- size_category_statement (text providing a size-category label for a 10 MW data center, e.g., small/medium/large), "
        "- temperature_range_statement (text stating 70–75°F (21–24°C) or ASHRAE Class A3 5–40°C), "
        "- continuous_cooling_statement (text stating 24/7 continuous cooling for Tier III), "
        "- cooling_redundancy_statement (text stating redundancy sufficient for concurrent maintainability), "
        "- references (URLs supporting power/cooling specs). "
        "Within timeline, extract: "
        "- site_selection_range_statement (text stating 6–12 months), "
        "- permitting_range_statement (text stating 6–18 months), "
        "- design_range_statement (text stating 3–6 months), "
        "- construction_range_statement (text stating 6–18 months), "
        "- total_timeline_range_statement (text stating 36–72 months or 3–6 years), "
        "- feasibility_assessment_statement (text assessing feasibility of 48-month target), "
        "- timeline_risks_statement (text listing at least one timeline risk/consideration), "
        "- references (URLs supporting timeline ranges and discussion). "
        "If any item is missing in the answer, return null for that field; for references, return an empty list if none are present. "
        "Extract only URLs explicitly present in the answer."
    )


# --------------------------- Verification Builders --------------------------- #
async def build_texas_regulatory_section(evaluator: Evaluator, parent_node, data: Optional[TexasRegulatoryExtraction]) -> None:
    section_node = evaluator.add_parallel(
        id="TexasRegulatoryCompliance",
        desc="Texas Comptroller qualifying data center regulatory requirements are correctly included.",
        parent=parent_node,
        critical=True,
    )

    leaf_min_size = evaluator.add_leaf(
        id="MinimumFacilitySize",
        desc="States the Texas Comptroller minimum facility size requirement: 100,000 sq ft in a single building or portion of a building.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Texas Comptroller minimum facility size requirement is 100,000 square feet in a single building or portion of a building.",
        node=leaf_min_size,
    )

    leaf_single_occupant = evaluator.add_leaf(
        id="SingleOccupantRule",
        desc="States that each qualifying data center may have only one occupant.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that each qualifying data center may have only one occupant.",
        node=leaf_single_occupant,
    )

    leaf_capital_investment = evaluator.add_leaf(
        id="CapitalInvestmentEligibility",
        desc="States that capital investment must qualify as IRC Section 179, Section 1245, or Section 1250 property.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that capital investment must qualify as IRC Section 179, Section 1245, or Section 1250 property.",
        node=leaf_capital_investment,
    )

    leaf_electricity_threshold = evaluator.add_leaf(
        id="ElectricityUseThresholdForExemption",
        desc="States that electricity used for data center operations must exceed 50% of total electricity use to qualify for the sales tax exemption.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that electricity used for data center operations must exceed 50% of total electricity use to qualify for the sales tax exemption.",
        node=leaf_electricity_threshold,
    )

    refs = data.references if data and data.references else []
    leaf_reg_refs = evaluator.add_leaf(
        id="TexasRegulatoryReference",
        desc="Provides at least one authoritative URL reference supporting the Texas qualifying data center regulatory requirements stated in this section.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced webpage(s) support the Texas qualifying data center requirements stated (minimum facility size 100,000 sq ft single building/portion, single-occupant rule, IRC 179/1245/1250 eligibility, and >50% electricity use threshold).",
        node=leaf_reg_refs,
        sources=refs,
        additional_instruction="If no URLs are provided, judge this claim as Incorrect because an authoritative reference is required.",
    )


async def build_employment_section(evaluator: Evaluator, parent_node, data: Optional[EmploymentExtraction]) -> None:
    section_node = evaluator.add_parallel(
        id="EmploymentObligations",
        desc="Employment/job obligations are correctly calculated and stated per constraints.",
        parent=parent_node,
        critical=True,
    )

    leaf_job_count = evaluator.add_leaf(
        id="MinimumJobCount",
        desc="States the minimum number of qualifying jobs required: 20.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the minimum number of qualifying jobs required is 20.",
        node=leaf_job_count,
    )

    leaf_job_window = evaluator.add_leaf(
        id="JobCreationWindow",
        desc="States the timeframe to create all qualifying jobs: within the first five years of certification.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that all qualifying jobs must be created within the first five years of certification.",
        node=leaf_job_window,
    )

    leaf_weekly_wage_calc = evaluator.add_leaf(
        id="MinimumWeeklyWageCalculatedFromCountyData",
        desc="Provides (a) the Travis County average weekly wage value used (numeric), (b) applies the 120% requirement, and (c) states the resulting computed minimum weekly wage.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer provides the Travis County average weekly wage value used, explicitly applies a 120% requirement (or 1.2x), and states the resulting computed minimum weekly wage.",
        node=leaf_weekly_wage_calc,
        additional_instruction="Judge Correct only if all three are present in the answer: the county average weekly wage value (numeric), the explicit 120% requirement, and the computed 120% weekly wage (numeric).",
    )

    leaf_annual_hours = evaluator.add_leaf(
        id="AnnualHourRequirement",
        desc="States the minimum annual hours per qualifying job: at least 1,820 hours/year.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a minimum of at least 1,820 hours per qualifying job per year.",
        node=leaf_annual_hours,
    )

    leaf_retention = evaluator.add_leaf(
        id="RetentionPeriod",
        desc="States the job retention period: at least five years from creation date.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that each qualifying job must be retained for at least five years from its creation date.",
        node=leaf_retention,
    )

    leaf_vacancy = evaluator.add_leaf(
        id="VacancyFillWindow",
        desc="States that vacant positions must be filled within 120 days to continue counting toward job requirements.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that vacant qualifying positions must be filled within 120 days to continue counting toward job requirements.",
        node=leaf_vacancy,
    )

    refs = data.references if data and data.references else []
    leaf_emp_refs = evaluator.add_leaf(
        id="EmploymentReference",
        desc="Provides at least one authoritative URL reference supporting the employment requirements and/or wage data used in this section.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced webpage(s) support the employment obligations and/or wage data stated in this section (job counts, creation window, wage calculations, hours, retention, vacancy window).",
        node=leaf_emp_refs,
        sources=refs,
        additional_instruction="If no URLs are provided, judge this claim as Incorrect because an authoritative reference is required.",
    )


async def build_tier3_section(evaluator: Evaluator, parent_node, data: Optional[Tier3Extraction]) -> None:
    section_node = evaluator.add_parallel(
        id="TierIIITechnicalRequirements",
        desc="Uptime Institute Tier III requirements requested are stated.",
        parent=parent_node,
        critical=True,
    )

    leaf_cm = evaluator.add_leaf(
        id="ConcurrentMaintainability",
        desc="States Tier III concurrent maintainability requirement: maintenance without shutdown of the critical environment.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states the Tier III concurrent maintainability requirement: maintenance can be performed without shutdown of the critical environment.",
        node=leaf_cm,
    )

    leaf_ind_paths = evaluator.add_leaf(
        id="IndependentDistributionPaths",
        desc="States Tier III requires multiple independent distribution paths for power and cooling serving the critical environment.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that Tier III requires multiple independent distribution paths for power and cooling serving the critical environment.",
        node=leaf_ind_paths,
    )

    red_caps_node = evaluator.add_parallel(
        id="RedundantCapacityComponents",
        desc="Includes the redundant capacity components specified in the constraints, grouped by system type.",
        parent=section_node,
        critical=True,
    )

    leaf_power_comps = evaluator.add_leaf(
        id="PowerSystemComponents",
        desc="Includes generators, energy storage, and UPS modules as redundant power components.",
        parent=red_caps_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes all of the following as redundant power components for Tier III: generators, energy storage, and UPS modules.",
        node=leaf_power_comps,
        additional_instruction="Judge Correct only if all three terms appear: generators; energy storage; UPS modules.",
    )

    leaf_cooling_comps = evaluator.add_leaf(
        id="CoolingSystemComponents",
        desc="Includes chillers, cooling units, and heat rejection equipment as redundant cooling components.",
        parent=red_caps_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes all of the following as redundant cooling components for Tier III: chillers, cooling units, and heat rejection equipment.",
        node=leaf_cooling_comps,
        additional_instruction="Judge Correct only if all three terms appear: chillers; cooling units; heat rejection.",
    )

    leaf_support_comps = evaluator.add_leaf(
        id="SupportSystemComponents",
        desc="Includes pumps, fuel tanks, and fuel cells as redundant support components.",
        parent=red_caps_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes all of the following as redundant support components for Tier III: pumps, fuel tanks, and fuel cells.",
        node=leaf_support_comps,
        additional_instruction="Judge Correct only if all three terms appear: pumps; fuel tanks; fuel cells.",
    )

    refs = data.references if data and data.references else []
    leaf_tier3_refs = evaluator.add_leaf(
        id="TierIIIReference",
        desc="Provides at least one authoritative URL reference supporting the Tier III requirements stated in this section.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced webpage(s) support the Tier III requirements stated (concurrent maintainability, independent distribution paths, and redundant components).",
        node=leaf_tier3_refs,
        sources=refs,
        additional_instruction="If no URLs are provided, judge this claim as Incorrect because an authoritative reference is required.",
    )


async def build_power_cooling_section(evaluator: Evaluator, parent_node, data: Optional[PowerCoolingExtraction]) -> None:
    section_node = evaluator.add_parallel(
        id="PowerAndCoolingSpecifications",
        desc="Power and cooling specs for a 10 MW facility include the requested attributes.",
        parent=parent_node,
        critical=True,
    )

    electrical_node = evaluator.add_parallel(
        id="ElectricalInfrastructureRequirements",
        desc="Electrical infrastructure requirements for a 10 MW data center are stated, including phase, voltage, and grid reliability.",
        parent=section_node,
        critical=True,
    )

    leaf_phase = evaluator.add_leaf(
        id="PhaseRequirement",
        desc="States three-phase electrical power service requirement (per constraints).",
        parent=electrical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a three-phase electrical power service requirement for the 10 MW data center.",
        node=leaf_phase,
    )

    leaf_voltage = evaluator.add_leaf(
        id="VoltageRequirementIncluded",
        desc="Includes at least one explicit voltage level or voltage range (in V or kV) as part of the electrical infrastructure description for the 10 MW facility.",
        parent=electrical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer includes at least one explicit voltage level or voltage range (in V or kV) for the 10 MW facility.",
        node=leaf_voltage,
        additional_instruction="Look for numeric voltage values or ranges like 480V, 120/208V, 13.2 kV, 34.5 kV, etc. General statements without numbers do NOT satisfy this requirement.",
    )

    leaf_grid = evaluator.add_leaf(
        id="GridReliabilityRequirement",
        desc="States the requirement to have access to a reliable power grid with adequate capacity.",
        parent=electrical_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states the requirement to have access to a reliable power grid with adequate capacity.",
        node=leaf_grid,
    )

    leaf_size_category = evaluator.add_leaf(
        id="FacilitySizeCategoryStatement",
        desc="States a size-category label for a 10 MW data center (e.g., small/medium/large).",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer provides a size-category label for a 10 MW data center (for example, small, medium, or large).",
        node=leaf_size_category,
        additional_instruction="Judge Correct if the answer includes any reasonable size-category label associated with 10 MW (e.g., small, medium, large).",
    )

    cooling_node = evaluator.add_parallel(
        id="CoolingSystemRequirements",
        desc="Cooling requirements requested are stated (temperature guidance, continuous cooling, redundancy).",
        parent=section_node,
        critical=True,
    )

    leaf_temp = evaluator.add_leaf(
        id="TemperatureControlRange",
        desc="States the temperature control guidance from the constraints: 70–75°F (21–24°C) or ASHRAE Class A3 range (5–40°C).",
        parent=cooling_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states temperature control guidance as either 70–75°F (21–24°C) or ASHRAE Class A3 range (5–40°C).",
        node=leaf_temp,
    )

    leaf_continuous = evaluator.add_leaf(
        id="ContinuousCoolingRequirement",
        desc="States continuous cooling requirement (24/7) for Tier III facilities (per constraints).",
        parent=cooling_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a continuous cooling requirement (24/7) for Tier III facilities.",
        node=leaf_continuous,
    )

    leaf_cool_redund = evaluator.add_leaf(
        id="CoolingRedundancyLevel",
        desc="States cooling redundancy sufficient to support concurrent maintainability (per constraints).",
        parent=cooling_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states cooling redundancy sufficient to support concurrent maintainability.",
        node=leaf_cool_redund,
    )

    refs = data.references if data and data.references else []
    leaf_power_cool_refs = evaluator.add_leaf(
        id="PowerCoolingReference",
        desc="Provides at least one authoritative URL reference supporting the power/cooling specifications stated in this section.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced webpage(s) support the stated power and cooling specifications for the 10 MW facility (phase, voltage, grid reliability, temperature guidance, continuous cooling, redundancy).",
        node=leaf_power_cool_refs,
        sources=refs,
        additional_instruction="If no URLs are provided, judge this claim as Incorrect because an authoritative reference is required.",
    )


async def build_timeline_section(evaluator: Evaluator, parent_node, data: Optional[TimelineExtraction]) -> None:
    section_node = evaluator.add_parallel(
        id="DevelopmentTimelinePhases",
        desc="Development timeline includes phase ranges, total range, 48-month feasibility assessment, and risks/considerations.",
        parent=parent_node,
        critical=True,
    )

    leaf_site_sel = evaluator.add_leaf(
        id="SiteSelectionPhase",
        desc="States typical site selection duration range: 6–12 months.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a typical site selection duration range of 6–12 months.",
        node=leaf_site_sel,
    )

    leaf_permitting = evaluator.add_leaf(
        id="PermittingPhase",
        desc="States typical permitting and approvals duration range: 6–18 months.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a typical permitting and approvals duration range of 6–18 months.",
        node=leaf_permitting,
    )

    leaf_design = evaluator.add_leaf(
        id="DesignPhase",
        desc="States typical design duration range: 3–6 months.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a typical design duration range of 3–6 months.",
        node=leaf_design,
    )

    leaf_construction = evaluator.add_leaf(
        id="ConstructionPhase",
        desc="States typical construction duration range: 6–18 months.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states a typical construction duration range of 6–18 months.",
        node=leaf_construction,
    )

    leaf_total_range = evaluator.add_leaf(
        id="TotalTimelineRange",
        desc="States the total typical development timeline range: 36–72 months (3–6 years).",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states the total typical development timeline range as 36–72 months (3–6 years).",
        node=leaf_total_range,
    )

    leaf_feasibility = evaluator.add_leaf(
        id="FeasibilityAssessmentProvided",
        desc="Provides an explicit assessment of whether the 48-month target is feasible relative to the typical total range.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly assesses whether the 48-month target is feasible relative to the stated typical total development timeline range.",
        node=leaf_feasibility,
    )

    leaf_risks = evaluator.add_leaf(
        id="TimelineRisksOrConsiderations",
        desc="Identifies at least one timeline risk or consideration affecting feasibility.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer identifies at least one timeline risk or consideration that could affect feasibility of the development timeline.",
        node=leaf_risks,
    )

    refs = data.references if data and data.references else []
    leaf_timeline_refs = evaluator.add_leaf(
        id="TimelineReference",
        desc="Provides at least one authoritative URL reference supporting the timeline ranges and/or timeline discussion in this section.",
        parent=section_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The referenced webpage(s) support the stated development timeline ranges and/or timeline considerations.",
        node=leaf_timeline_refs,
        sources=refs,
        additional_instruction="If no URLs are provided, judge this claim as Incorrect because an authoritative reference is required.",
    )


# --------------------------- Main Evaluation Entry --------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_compliance_checklist(),
        template_class=DataCenterChecklistExtraction,
        extraction_name="compliance_checklist_extraction",
    )

    top_node = evaluator.add_parallel(
        id="DataCenterComplianceEvaluation",
        desc="Evaluate whether the response provides a comprehensive compliance checklist covering all required sections and constraints for a Travis County, TX Tier III data center project.",
        parent=root,
        critical=True,
    )

    await build_texas_regulatory_section(evaluator, top_node, extracted.texas_regulatory)
    await build_employment_section(evaluator, top_node, extracted.employment)
    await build_tier3_section(evaluator, top_node, extracted.tier3)
    await build_power_cooling_section(evaluator, top_node, extracted.power_cooling)
    await build_timeline_section(evaluator, top_node, extracted.timeline)

    return evaluator.get_summary()