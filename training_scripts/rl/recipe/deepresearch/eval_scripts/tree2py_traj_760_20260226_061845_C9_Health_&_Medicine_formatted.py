import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "bristol_explosion_comprehensive"
TASK_DESCRIPTION = (
    "On December 23, 2025, a catastrophic natural gas explosion occurred at a skilled nursing facility in "
    "Bristol Township, Pennsylvania, resulting in multiple casualties and extensive property damage. The incident "
    "has prompted comprehensive investigation by federal authorities and raised serious questions about emergency "
    "preparedness protocols and gas system safety in long-term care facilities.\n\n"
    "As a regulatory analyst preparing a comprehensive incident report, document all verifiable facts about this explosion, including:\n\n"
    "1. Complete facility identification: Provide the current facility name, any former names, exact physical address (street, municipality, county, state), regulatory classification, licensed bed capacity, and number of people present at the time of the incident.\n\n"
    "2. Detailed chronological timeline: Document the complete sequence of events from the initial gas odor report through the conclusion of search and rescue operations, including: the date and time of each significant event, weather conditions, locations where gas odors were detected, arrival times of all responding personnel and agencies, time when gas flow was isolated, and timing of all investigation activities.\n\n"
    "3. Complete human impact assessment: Provide exact counts of immediate fatalities, any delayed deaths (with specific dates), total deaths, total injuries, and categories of victims.\n\n"
    "4. Technical infrastructure analysis: Document the gas service provider (including parent company), complete specifications of the gas distribution system (service line diameter, material, and location), meter system details (type and location), and the specific equipment where the leak was identified.\n\n"
    "5. Official investigation status: Identify the lead investigating agency, the date of the preliminary report release, all physical evidence collected and analyzed, and the planned scope of future investigation activities.\n\n"
    "Each fact must be supported by at least one verifiable URL reference from a reliable source."
)

# A general additional instruction for most verifications
DEFAULT_VERIFY_ADDITIONAL_INSTRUCTION = (
    "Use the provided source URLs to verify the claim exactly as stated. Allow minor wording differences, "
    "reasonable synonyms, and standard abbreviations. For date/time statements, accept reasonable approximations "
    "when the claim says 'approximately' or 'around' (e.g., ±10 minutes). For names, allow minor formatting "
    "differences (e.g., capitalization, middle initials)."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ValueWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ListWithSources(BaseModel):
    items: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class FacilityIdentificationExtraction(BaseModel):
    current_facility_name: Optional[ValueWithSources] = None
    former_facility_name: Optional[ValueWithSources] = None
    facility_address: Optional[ValueWithSources] = None
    regulatory_classification: Optional[ValueWithSources] = None
    licensed_bed_capacity: Optional[ValueWithSources] = None
    people_present_approx: Optional[ValueWithSources] = None


class IncidentTimelineExtraction(BaseModel):
    incident_date: Optional[ValueWithSources] = None
    explosion_time: Optional[ValueWithSources] = None
    weather_conditions: Optional[ValueWithSources] = None
    initial_odor_report_time: Optional[ValueWithSources] = None
    initial_odor_location_1: Optional[ValueWithSources] = None
    initial_odor_location_2: Optional[ValueWithSources] = None
    exelon_technician_arrival: Optional[ValueWithSources] = None
    meter_services_technician_arrival: Optional[ValueWithSources] = None
    odor_present_before_explosion_basement: Optional[ValueWithSources] = None
    odor_present_before_explosion_first_floor: Optional[ValueWithSources] = None
    odor_present_before_explosion_second_floor: Optional[ValueWithSources] = None
    fire_rescue_dispatch_time: Optional[ValueWithSources] = None
    first_responder_arrival_time: Optional[ValueWithSources] = None
    exelon_emergency_responder_arrival: Optional[ValueWithSources] = None
    gas_flow_isolated_time: Optional[ValueWithSources] = None
    bar_hole_testing_time: Optional[ValueWithSources] = None
    puc_directed_bar_hole_tests: Optional[ValueWithSources] = None
    subsurface_gas_outside_building: Optional[ValueWithSources] = None
    search_rescue_conclusion: Optional[ValueWithSources] = None
    responding_agencies: Optional[ListWithSources] = None


class HumanImpactExtraction(BaseModel):
    immediate_fatalities_count: Optional[ValueWithSources] = None
    immediate_fatality_category_resident: Optional[ValueWithSources] = None
    immediate_fatality_category_employee: Optional[ValueWithSources] = None
    delayed_death_count_and_date: Optional[ValueWithSources] = None
    total_deaths: Optional[ValueWithSources] = None
    total_injuries: Optional[ValueWithSources] = None


class TechnicalInfrastructureExtraction(BaseModel):
    gas_provider: Optional[ValueWithSources] = None
    gas_provider_parent_company: Optional[ValueWithSources] = None
    service_line_diameter: Optional[ValueWithSources] = None
    service_line_material: Optional[ValueWithSources] = None
    service_line_location: Optional[ValueWithSources] = None
    meter_type: Optional[ValueWithSources] = None
    meter_location: Optional[ValueWithSources] = None
    leak_equipment: Optional[ValueWithSources] = None
    leak_location: Optional[ValueWithSources] = None
    personnel_experience_foreman: Optional[ValueWithSources] = None
    personnel_experience_meter_services_technician: Optional[ValueWithSources] = None


class InvestigationStatusExtraction(BaseModel):
    lead_investigating_agency: Optional[ValueWithSources] = None
    preliminary_report_date: Optional[ValueWithSources] = None

    recovered_indoor_meter_set: Optional[ValueWithSources] = None
    excavated_service_line_portions: Optional[ValueWithSources] = None
    delivered_components_to_materials_lab: Optional[ValueWithSources] = None

    focus_pipeline_safety_management_system: Optional[ValueWithSources] = None
    focus_personnel_training: Optional[ValueWithSources] = None
    focus_operator_qualifications: Optional[ValueWithSources] = None
    focus_task_specific_procedures: Optional[ValueWithSources] = None
    focus_odor_complaint_response: Optional[ValueWithSources] = None
    focus_documentation: Optional[ValueWithSources] = None
    focus_emergency_response: Optional[ValueWithSources] = None

    entity_phmsa: Optional[ValueWithSources] = None
    entity_osha: Optional[ValueWithSources] = None
    entity_bristol_township_fire_marshal: Optional[ValueWithSources] = None
    entity_ibew_local_614: Optional[ValueWithSources] = None
    entity_saber_healthcare_group: Optional[ValueWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_identification() -> str:
    return """
Extract the facility identification details mentioned in the answer. For each field, return a JSON object with:
- value: the exact value as stated in the answer text
- sources: an array of URLs explicitly cited in the answer that support that exact fact

Fields:
- current_facility_name
- former_facility_name
- facility_address  (can be a single string like "Tower Road, Bristol Township, Bucks County, Pennsylvania" or a fuller address)
- regulatory_classification  (e.g., "skilled nursing facility regulated under 42 CFR Part 483")
- licensed_bed_capacity  (e.g., "174 beds")
- people_present_approx  (e.g., "approximately 180")

Return a JSON object of type FacilityIdentificationExtraction.
If a value is missing, set value=null. If no source URLs are present for a field in the answer, set sources=[].
    """.strip()


def prompt_extract_incident_timeline() -> str:
    return """
Extract detailed timeline information as presented in the answer. For each field, return:
- value: the exact phrasing (e.g., time/date/description) as in the answer
- sources: array of URLs explicitly cited that support the fact

Fields:
- incident_date
- explosion_time
- weather_conditions
- initial_odor_report_time
- initial_odor_location_1
- initial_odor_location_2
- exelon_technician_arrival
- meter_services_technician_arrival
- odor_present_before_explosion_basement
- odor_present_before_explosion_first_floor
- odor_present_before_explosion_second_floor
- fire_rescue_dispatch_time
- first_responder_arrival_time
- exelon_emergency_responder_arrival
- gas_flow_isolated_time
- bar_hole_testing_time
- puc_directed_bar_hole_tests
- subsurface_gas_outside_building
- search_rescue_conclusion

Also extract responding agencies:
- responding_agencies.items: list of agency names provided in the answer
- responding_agencies.sources: array of URLs that support the involvement of these agencies

Return a JSON object of type IncidentTimelineExtraction.
If a value is missing, set value=null. If no URLs exist in the answer for that field, set sources=[].
    """.strip()


def prompt_extract_human_impact() -> str:
    return """
Extract the human impact figures as stated. For each, provide:
- value: the exact phrase (e.g., "two", "approximately 20", "January 5, 2026")
- sources: array of URLs cited supporting that fact

Fields:
- immediate_fatalities_count
- immediate_fatality_category_resident
- immediate_fatality_category_employee
- delayed_death_count_and_date
- total_deaths
- total_injuries

Return a JSON object of type HumanImpactExtraction.
If a value is missing, set value=null. If no URLs exist in the answer for that field, set sources=[].
    """.strip()


def prompt_extract_technical_infrastructure() -> str:
    return """
Extract technical infrastructure information about the gas service and equipment. For each field, return:
- value: exact phrase as in the answer
- sources: array of URLs cited supporting the fact

Fields:
- gas_provider
- gas_provider_parent_company
- service_line_diameter
- service_line_material
- service_line_location
- meter_type
- meter_location
- leak_equipment
- leak_location
- personnel_experience_foreman
- personnel_experience_meter_services_technician

Return a JSON object of type TechnicalInfrastructureExtraction.
If a value is missing, set value=null. If no URLs exist for that field, set sources=[].
    """.strip()


def prompt_extract_investigation_status() -> str:
    return """
Extract investigation status details. For each field, return:
- value: exact phrase as in the answer
- sources: array of URLs cited supporting the fact

Fields:
- lead_investigating_agency
- preliminary_report_date

Physical evidence collected/analyzed:
- recovered_indoor_meter_set
- excavated_service_line_portions
- delivered_components_to_materials_lab

Future investigation focus areas (each boolean-like or short phrase acceptable as 'value'):
- focus_pipeline_safety_management_system
- focus_personnel_training
- focus_operator_qualifications
- focus_task_specific_procedures
- focus_odor_complaint_response
- focus_documentation
- focus_emergency_response

Other entities involved:
- entity_phmsa
- entity_osha
- entity_bristol_township_fire_marshal
- entity_ibew_local_614
- entity_saber_healthcare_group

Return a JSON object of type InvestigationStatusExtraction.
If a value is missing, set value=null. If no URLs exist in the answer for that field, set sources=[].
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _src_list(srcs: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize sources: return list if non-empty, else None."""
    if not srcs:
        return None
    # Remove obvious empties/whitespace
    cleaned = [s for s in (srcs or []) if isinstance(s, str) and s.strip()]
    return cleaned if cleaned else None


def _vw_has_sources(vw: Optional[ValueWithSources]) -> bool:
    return bool(vw and vw.sources and len([s for s in vw.sources if isinstance(s, str) and s.strip()]))


def _lws_has_sources(lws: Optional[ListWithSources]) -> bool:
    return bool(lws and lws.sources and len([s for s in lws.sources if isinstance(s, str) and s.strip()]))


def _add_fact_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: Optional[str] = None,
    critical: bool = True
):
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    return (claim, sources, leaf, additional_instruction or DEFAULT_VERIFY_ADDITIONAL_INSTRUCTION)


# --------------------------------------------------------------------------- #
# Build tree sections                                                         #
# --------------------------------------------------------------------------- #
async def build_facility_identification(
    evaluator: Evaluator,
    parent,
    fi: FacilityIdentificationExtraction
):
    node = evaluator.add_parallel(
        id="Facility_Identification",
        desc="Facility identification details as specified.",
        parent=parent,
        critical=True
    )

    claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    # Current facility name
    claims.append(_add_fact_leaf(
        evaluator, node, "Current_Facility_Name",
        "State the current facility name: Bristol Health & Rehab Center.",
        "The current facility name is Bristol Health & Rehab Center.",
        _src_list(fi.current_facility_name.sources if fi and fi.current_facility_name else None)
    ))

    # Former facility name
    claims.append(_add_fact_leaf(
        evaluator, node, "Former_Facility_Name",
        "State the former facility name: Silver Lake Nursing Home.",
        "The former facility name was Silver Lake Nursing Home.",
        _src_list(fi.former_facility_name.sources if fi and fi.former_facility_name else None)
    ))

    # Facility address (key elements)
    claims.append(_add_fact_leaf(
        evaluator, node, "Facility_Address",
        "Provide the facility location details: Tower Road, Bristol Township, Bucks County, Pennsylvania.",
        "The facility is located on Tower Road, Bristol Township, Bucks County, Pennsylvania.",
        _src_list(fi.facility_address.sources if fi and fi.facility_address else None)
    ))

    # Regulatory classification
    claims.append(_add_fact_leaf(
        evaluator, node, "Regulatory_Classification",
        "Identify the facility as a skilled nursing facility regulated under 42 CFR Part 483.",
        "The facility is a skilled nursing facility regulated under 42 CFR Part 483.",
        _src_list(fi.regulatory_classification.sources if fi and fi.regulatory_classification else None)
    ))

    # Licensed bed capacity
    claims.append(_add_fact_leaf(
        evaluator, node, "Licensed_Bed_Capacity",
        "Provide the licensed bed capacity: 174 beds.",
        "The facility has a licensed bed capacity of 174 beds.",
        _src_list(fi.licensed_bed_capacity.sources if fi and fi.licensed_bed_capacity else None)
    ))

    # People present approx
    claims.append(_add_fact_leaf(
        evaluator, node, "People_Present_Approx",
        "Provide the approximate number of people on site at the time: approximately 180.",
        "Approximately 180 people were on site at the time of the incident.",
        _src_list(fi.people_present_approx.sources if fi and fi.people_present_approx else None)
    ))

    await evaluator.batch_verify(claims)


async def build_incident_timeline(
    evaluator: Evaluator,
    parent,
    tl: IncidentTimelineExtraction
):
    node = evaluator.add_parallel(
        id="Incident_Timeline",
        desc="Timeline facts as specified from odor report through rescue conclusion, plus specified response/investigation timings.",
        parent=parent,
        critical=True
    )

    claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    claims.append(_add_fact_leaf(
        evaluator, node, "Incident_Date",
        "Explosion date: December 23, 2025.",
        "The explosion occurred on December 23, 2025.",
        _src_list(tl.incident_date.sources if tl and tl.incident_date else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Explosion_Time",
        "Explosion time: approximately 2:15 PM local time.",
        "The explosion occurred at approximately 2:15 PM local time.",
        _src_list(tl.explosion_time.sources if tl and tl.explosion_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Weather_Conditions",
        "Weather at time of incident: 38°F, cloudy, with light and variable winds.",
        "At the time of the incident, weather was about 38°F, cloudy, with light and variable winds.",
        _src_list(tl.weather_conditions.sources if tl and tl.weather_conditions else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Initial_Odor_Report_Time",
        "Gas odor was first reported shortly after 11:00 AM.",
        "Gas odor was first reported shortly after 11:00 AM.",
        _src_list(tl.initial_odor_report_time.sources if tl and tl.initial_odor_report_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Initial_Odor_Location_1",
        "Gas odor was reported in the basement boiler room (initial report).",
        "Gas odor was reported in the basement boiler room (initial report).",
        _src_list(tl.initial_odor_location_1.sources if tl and tl.initial_odor_location_1 else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Initial_Odor_Location_2",
        "Gas odor was reported in the first-floor hallway (initial report).",
        "Gas odor was reported in the first-floor hallway (initial report).",
        _src_list(tl.initial_odor_location_2.sources if tl and tl.initial_odor_location_2 else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Exelon_Technician_Arrival",
        "Exelon energy technician arrived onsite around 11:50 AM.",
        "An Exelon energy technician arrived onsite around 11:50 AM.",
        _src_list(tl.exelon_technician_arrival.sources if tl and tl.exelon_technician_arrival else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Meter_Services_Technician_Arrival",
        "Meter services technician arrived around 1:20 PM to perform repairs.",
        "A meter services technician arrived around 1:20 PM to perform repairs.",
        _src_list(tl.meter_services_technician_arrival.sources if tl and tl.meter_services_technician_arrival else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Odor_Present_Shortly_Before_Explosion_Basement",
        "Staff smelled gas odorant in the basement shortly before the explosion.",
        "Staff smelled gas odorant in the basement shortly before the explosion.",
        _src_list(tl.odor_present_before_explosion_basement.sources if tl and tl.odor_present_before_explosion_basement else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Odor_Present_Shortly_Before_Explosion_First_Floor",
        "Staff smelled gas odorant on the first floor shortly before the explosion.",
        "Staff smelled gas odorant on the first floor shortly before the explosion.",
        _src_list(tl.odor_present_before_explosion_first_floor.sources if tl and tl.odor_present_before_explosion_first_floor else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Odor_Present_Shortly_Before_Explosion_Second_Floor",
        "Staff smelled gas odorant on the second floor shortly before the explosion.",
        "Staff smelled gas odorant on the second floor shortly before the explosion.",
        _src_list(tl.odor_present_before_explosion_second_floor.sources if tl and tl.odor_present_before_explosion_second_floor else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Fire_Rescue_Dispatch_Time",
        "Fire/rescue units were dispatched at approximately 2:17 PM.",
        "Fire/rescue units were dispatched at approximately 2:17 PM.",
        _src_list(tl.fire_rescue_dispatch_time.sources if tl and tl.fire_rescue_dispatch_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "First_Responder_Arrival_Time",
        "First responders arrived within approximately 1 minute of dispatch.",
        "First responders arrived within approximately 1 minute of dispatch.",
        _src_list(tl.first_responder_arrival_time.sources if tl and tl.first_responder_arrival_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Exelon_Emergency_Responder_Arrival",
        "Exelon emergency responders arrived at approximately 2:42 PM.",
        "Exelon emergency responders arrived at approximately 2:42 PM.",
        _src_list(tl.exelon_emergency_responder_arrival.sources if tl and tl.exelon_emergency_responder_arrival else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Gas_Flow_Isolated_Time",
        "Natural gas flow was isolated at approximately 3:50 PM.",
        "Natural gas flow was isolated at approximately 3:50 PM.",
        _src_list(tl.gas_flow_isolated_time.sources if tl and tl.gas_flow_isolated_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Bar_Hole_Testing_Time",
        "Bar hole tests were conducted at approximately 5:00 PM.",
        "Bar hole tests were conducted at approximately 5:00 PM.",
        _src_list(tl.bar_hole_testing_time.sources if tl and tl.bar_hole_testing_time else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "PUC_Directed_Bar_Hole_Tests",
        "The Pennsylvania Public Utility Commission directed Exelon to conduct bar hole tests.",
        "The Pennsylvania Public Utility Commission directed Exelon to conduct bar hole tests.",
        _src_list(tl.puc_directed_bar_hole_tests.sources if tl and tl.puc_directed_bar_hole_tests else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Subsurface_Gas_Outside_Building",
        "Bar hole tests identified subsurface gas outside the building.",
        "Bar hole tests identified subsurface gas outside the building.",
        _src_list(tl.subsurface_gas_outside_building.sources if tl and tl.subsurface_gas_outside_building else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Search_Rescue_Conclusion",
        "Search and rescue operation concluded approximately 6 hours after the incident.",
        "Search and rescue operation concluded approximately 6 hours after the incident.",
        _src_list(tl.search_rescue_conclusion.sources if tl and tl.search_rescue_conclusion else None)
    ))

    # Run batch for the simple event leaves
    await evaluator.batch_verify(claims)

    # Responding agencies (parallel, critical)
    agencies_node = evaluator.add_parallel(
        id="Responding_Agencies",
        desc="Identify the specified responding agencies.",
        parent=node,
        critical=True
    )

    agencies_claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    agencies_sources = _src_list(tl.responding_agencies.sources if tl and tl.responding_agencies else None)

    agencies_claims.append(_add_fact_leaf(
        evaluator, agencies_node, "Responding_Agency_Third_District_Fire_Company",
        "Includes Third District Fire Company as a responding agency.",
        "The Third District Fire Company responded to the incident.",
        agencies_sources
    ))
    agencies_claims.append(_add_fact_leaf(
        evaluator, agencies_node, "Responding_Agency_Bristol_Township_Fire_Rescue",
        "Includes Bristol Township Fire Rescue as a responding agency.",
        "Bristol Township Fire Rescue responded to the incident.",
        agencies_sources
    ))
    agencies_claims.append(_add_fact_leaf(
        evaluator, agencies_node, "Responding_Agency_Bucks_County_Rescue_Squad",
        "Includes Bucks County Rescue Squad as a responding agency.",
        "The Bucks County Rescue Squad responded to the incident.",
        agencies_sources
    ))
    agencies_claims.append(_add_fact_leaf(
        evaluator, agencies_node, "Responding_Agency_Bristol_Township_Police",
        "Includes Bristol Township Police as a responding agency.",
        "Bristol Township Police responded to the incident.",
        agencies_sources
    ))

    await evaluator.batch_verify(agencies_claims)


async def build_human_impact(
    evaluator: Evaluator,
    parent,
    hi: HumanImpactExtraction
):
    node = evaluator.add_parallel(
        id="Human_Impact_Assessment",
        desc="Casualty and injury counts and victim categories as specified.",
        parent=parent,
        critical=True
    )

    claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    claims.append(_add_fact_leaf(
        evaluator, node, "Immediate_Fatalities_Count",
        "Immediate fatalities count: two people died immediately in the explosion.",
        "Two people died immediately in the explosion.",
        _src_list(hi.immediate_fatalities_count.sources if hi and hi.immediate_fatalities_count else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Immediate_Fatality_Category_Resident",
        "Immediate fatalities included one resident.",
        "The immediate fatalities included one resident.",
        _src_list(hi.immediate_fatality_category_resident.sources if hi and hi.immediate_fatality_category_resident else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Immediate_Fatality_Category_Employee",
        "Immediate fatalities included one employee.",
        "The immediate fatalities included one employee.",
        _src_list(hi.immediate_fatality_category_employee.sources if hi and hi.immediate_fatality_category_employee else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Delayed_Death_Count_And_Date",
        "Delayed death: one additional person died on January 5, 2026 from injuries sustained in the incident.",
        "One additional person died on January 5, 2026 from injuries sustained in the incident.",
        _src_list(hi.delayed_death_count_and_date.sources if hi and hi.delayed_death_count_and_date else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Total_Deaths",
        "Total deaths: three deaths resulted from the incident.",
        "Three deaths resulted from the incident.",
        _src_list(hi.total_deaths.sources if hi and hi.total_deaths else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Total_Injuries",
        "Total injuries: approximately 20 people were injured.",
        "Approximately 20 people were injured.",
        _src_list(hi.total_injuries.sources if hi and hi.total_injuries else None)
    ))

    await evaluator.batch_verify(claims)


async def build_technical_infrastructure(
    evaluator: Evaluator,
    parent,
    ti: TechnicalInfrastructureExtraction
):
    node = evaluator.add_parallel(
        id="Technical_Infrastructure_Analysis",
        desc="Gas provider, system specifications, meter details, and leak equipment location as specified.",
        parent=parent,
        critical=True
    )

    claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    claims.append(_add_fact_leaf(
        evaluator, node, "Gas_Provider",
        "Natural gas was provided by PECO.",
        "Natural gas was provided by PECO.",
        _src_list(ti.gas_provider.sources if ti and ti.gas_provider else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Gas_Provider_Parent_Company",
        "PECO is a subsidiary of Exelon Corporation.",
        "PECO is a subsidiary of Exelon Corporation.",
        _src_list(ti.gas_provider_parent_company.sources if ti and ti.gas_provider_parent_company else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Service_Line_Diameter",
        "Service line diameter: 1.25-inch.",
        "The service line diameter was 1.25-inch.",
        _src_list(ti.service_line_diameter.sources if ti and ti.service_line_diameter else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Service_Line_Material",
        "Service line material: coated steel.",
        "The service line material was coated steel.",
        _src_list(ti.service_line_material.sources if ti and ti.service_line_material else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Service_Line_Location",
        "Service line location: underground.",
        "The service line was located underground.",
        _src_list(ti.service_line_location.sources if ti and ti.service_line_location else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Meter_Type",
        "Meter type: rotary meter set.",
        "The meter type was a rotary meter set.",
        _src_list(ti.meter_type.sources if ti and ti.meter_type else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Meter_Location",
        "Meter location: in the basement.",
        "The meter was located in the basement.",
        _src_list(ti.meter_location.sources if ti and ti.meter_location else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Leak_Equipment",
        "Leak was identified on a meter set valve.",
        "The leak was identified on a meter set valve.",
        _src_list(ti.leak_equipment.sources if ti and ti.leak_equipment else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Leak_Location",
        "Leak location: basement boiler room.",
        "The leak was located in the basement boiler room.",
        _src_list(ti.leak_location.sources if ti and ti.leak_location else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Personnel_Experience_Foreman",
        "The foreman had less than 1 year of experience in the current role.",
        "The foreman had less than 1 year of experience in the current role.",
        _src_list(ti.personnel_experience_foreman.sources if ti and ti.personnel_experience_foreman else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Personnel_Experience_Meter_Services_Technician",
        "The meter services technician had less than 1 year of experience in the current role.",
        "The meter services technician had less than 1 year of experience in the current role.",
        _src_list(ti.personnel_experience_meter_services_technician.sources if ti and ti.personnel_experience_meter_services_technician else None)
    ))

    await evaluator.batch_verify(claims)


async def build_investigation_status(
    evaluator: Evaluator,
    parent,
    inv: InvestigationStatusExtraction
):
    node = evaluator.add_parallel(
        id="Official_Investigation_Status",
        desc="Lead agency, preliminary report date, evidence collected, and planned future investigation scope as specified.",
        parent=parent,
        critical=True
    )

    claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []

    claims.append(_add_fact_leaf(
        evaluator, node, "Lead_Investigating_Agency",
        "Identify the lead investigating agency: National Transportation Safety Board (NTSB).",
        "The lead investigating agency was the National Transportation Safety Board (NTSB).",
        _src_list(inv.lead_investigating_agency.sources if inv and inv.lead_investigating_agency else None)
    ))
    claims.append(_add_fact_leaf(
        evaluator, node, "Preliminary_Report_Date",
        "Preliminary report release date: January 28, 2026.",
        "The preliminary report was released on January 28, 2026.",
        _src_list(inv.preliminary_report_date.sources if inv and inv.preliminary_report_date else None)
    ))

    # Physical Evidence (parallel)
    phys_node = evaluator.add_parallel(
        id="Physical_Evidence_Collected_And_Analyzed",
        desc="State the specified physical evidence collected/analyzed actions.",
        parent=node,
        critical=True
    )
    phys_claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []
    phys_claims.append(_add_fact_leaf(
        evaluator, phys_node, "Recovered_Indoor_Meter_Set",
        "NTSB recovered the indoor meter set for examination.",
        "NTSB recovered the indoor meter set for examination.",
        _src_list(inv.recovered_indoor_meter_set.sources if inv and inv.recovered_indoor_meter_set else None)
    ))
    phys_claims.append(_add_fact_leaf(
        evaluator, phys_node, "Excavated_Service_Line_Portions",
        "Portions of the service line that did not hold pressure during testing were excavated.",
        "Portions of the service line that did not hold pressure during testing were excavated.",
        _src_list(inv.excavated_service_line_portions.sources if inv and inv.excavated_service_line_portions else None)
    ))
    phys_claims.append(_add_fact_leaf(
        evaluator, phys_node, "Delivered_Components_To_Materials_Lab",
        "Components were delivered to the NTSB Materials Laboratory for examination.",
        "Components were delivered to the NTSB Materials Laboratory for examination.",
        _src_list(inv.delivered_components_to_materials_lab.sources if inv and inv.delivered_components_to_materials_lab else None)
    ))
    await evaluator.batch_verify(phys_claims)

    # Future Investigation Focus Areas (parallel)
    focus_node = evaluator.add_parallel(
        id="Future_Investigation_Focus_Areas",
        desc="State the specified future investigation focus areas.",
        parent=node,
        critical=True
    )
    focus_claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Pipeline_Safety_Management_System",
        "Future investigation will focus on Exelon's pipeline safety management system.",
        "Future investigation will focus on Exelon's pipeline safety management system.",
        _src_list(inv.focus_pipeline_safety_management_system.sources if inv and inv.focus_pipeline_safety_management_system else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Personnel_Training",
        "Future investigation will focus on personnel training.",
        "Future investigation will focus on personnel training.",
        _src_list(inv.focus_personnel_training.sources if inv and inv.focus_personnel_training else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Operator_Qualifications",
        "Future investigation will focus on operator qualifications.",
        "Future investigation will focus on operator qualifications.",
        _src_list(inv.focus_operator_qualifications.sources if inv and inv.focus_operator_qualifications else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Task_Specific_Procedures",
        "Future investigation will focus on task-specific procedures.",
        "Future investigation will focus on task-specific procedures.",
        _src_list(inv.focus_task_specific_procedures.sources if inv and inv.focus_task_specific_procedures else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Odor_Complaint_Response",
        "Future investigation will focus on odor complaint response.",
        "Future investigation will focus on odor complaint response.",
        _src_list(inv.focus_odor_complaint_response.sources if inv and inv.focus_odor_complaint_response else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Documentation",
        "Future investigation will focus on documentation.",
        "Future investigation will focus on documentation.",
        _src_list(inv.focus_documentation.sources if inv and inv.focus_documentation else None)
    ))
    focus_claims.append(_add_fact_leaf(
        evaluator, focus_node, "Focus_Emergency_Response",
        "Future investigation will focus on emergency response.",
        "Future investigation will focus on emergency response.",
        _src_list(inv.focus_emergency_response.sources if inv and inv.focus_emergency_response else None)
    ))
    await evaluator.batch_verify(focus_claims)

    # Other Entities Involved (parallel)
    ent_node = evaluator.add_parallel(
        id="Other_Entities_Involved",
        desc="Identify the additional involved entities specified.",
        parent=node,
        critical=True
    )
    ent_claims: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []
    ent_claims.append(_add_fact_leaf(
        evaluator, ent_node, "Entity_PHMSA",
        "Includes Pipeline and Hazardous Materials Safety Administration (PHMSA) as involved.",
        "The Pipeline and Hazardous Materials Safety Administration (PHMSA) was involved.",
        _src_list(inv.entity_phmsa.sources if inv and inv.entity_phmsa else None)
    ))
    ent_claims.append(_add_fact_leaf(
        evaluator, ent_node, "Entity_OSHA",
        "Includes Occupational Safety and Health Administration (OSHA) as involved.",
        "The Occupational Safety and Health Administration (OSHA) was involved.",
        _src_list(inv.entity_osha.sources if inv and inv.entity_osha else None)
    ))
    ent_claims.append(_add_fact_leaf(
        evaluator, ent_node, "Entity_Bristol_Township_Fire_Marshal",
        "Includes Bristol Township Office of the Fire Marshal as involved.",
        "The Bristol Township Office of the Fire Marshal was involved.",
        _src_list(inv.entity_bristol_township_fire_marshal.sources if inv and inv.entity_bristol_township_fire_marshal else None)
    ))
    ent_claims.append(_add_fact_leaf(
        evaluator, ent_node, "Entity_IBEW_Local_Union_614",
        "Includes International Brotherhood of Electrical Workers Local Union 614 as involved.",
        "The International Brotherhood of Electrical Workers Local Union 614 was involved.",
        _src_list(inv.entity_ibew_local_614.sources if inv and inv.entity_ibew_local_614 else None)
    ))
    ent_claims.append(_add_fact_leaf(
        evaluator, ent_node, "Entity_Saber_Healthcare_Group",
        "Includes Saber Healthcare Group as involved.",
        "Saber Healthcare Group was involved.",
        _src_list(inv.entity_saber_healthcare_group.sources if inv and inv.entity_saber_healthcare_group else None)
    ))
    await evaluator.batch_verify(ent_claims)


# --------------------------------------------------------------------------- #
# Citation support global check                                               #
# --------------------------------------------------------------------------- #
def compute_citation_coverage(
    fi: FacilityIdentificationExtraction,
    tl: IncidentTimelineExtraction,
    hi: HumanImpactExtraction,
    ti: TechnicalInfrastructureExtraction,
    inv: InvestigationStatusExtraction
) -> Dict[str, bool]:
    """
    Build a map from field identifier -> has_sources(bool), for all rubric facts we verify.
    """
    coverage: Dict[str, bool] = {}

    # Facility Identification
    coverage["Current_Facility_Name"] = _vw_has_sources(fi.current_facility_name) if fi else False
    coverage["Former_Facility_Name"] = _vw_has_sources(fi.former_facility_name) if fi else False
    coverage["Facility_Address"] = _vw_has_sources(fi.facility_address) if fi else False
    coverage["Regulatory_Classification"] = _vw_has_sources(fi.regulatory_classification) if fi else False
    coverage["Licensed_Bed_Capacity"] = _vw_has_sources(fi.licensed_bed_capacity) if fi else False
    coverage["People_Present_Approx"] = _vw_has_sources(fi.people_present_approx) if fi else False

    # Incident Timeline
    if tl:
        coverage["Incident_Date"] = _vw_has_sources(tl.incident_date)
        coverage["Explosion_Time"] = _vw_has_sources(tl.explosion_time)
        coverage["Weather_Conditions"] = _vw_has_sources(tl.weather_conditions)
        coverage["Initial_Odor_Report_Time"] = _vw_has_sources(tl.initial_odor_report_time)
        coverage["Initial_Odor_Location_1"] = _vw_has_sources(tl.initial_odor_location_1)
        coverage["Initial_Odor_Location_2"] = _vw_has_sources(tl.initial_odor_location_2)
        coverage["Exelon_Technician_Arrival"] = _vw_has_sources(tl.exelon_technician_arrival)
        coverage["Meter_Services_Technician_Arrival"] = _vw_has_sources(tl.meter_services_technician_arrival)
        coverage["Odor_Present_Shortly_Before_Explosion_Basement"] = _vw_has_sources(tl.odor_present_before_explosion_basement)
        coverage["Odor_Present_Shortly_Before_Explosion_First_Floor"] = _vw_has_sources(tl.odor_present_before_explosion_first_floor)
        coverage["Odor_Present_Shortly_Before_Explosion_Second_Floor"] = _vw_has_sources(tl.odor_present_before_explosion_second_floor)
        coverage["Fire_Rescue_Dispatch_Time"] = _vw_has_sources(tl.fire_rescue_dispatch_time)
        coverage["First_Responder_Arrival_Time"] = _vw_has_sources(tl.first_responder_arrival_time)
        coverage["Exelon_Emergency_Responder_Arrival"] = _vw_has_sources(tl.exelon_emergency_responder_arrival)
        coverage["Gas_Flow_Isolated_Time"] = _vw_has_sources(tl.gas_flow_isolated_time)
        coverage["Bar_Hole_Testing_Time"] = _vw_has_sources(tl.bar_hole_testing_time)
        coverage["PUC_Directed_Bar_Hole_Tests"] = _vw_has_sources(tl.puc_directed_bar_hole_tests)
        coverage["Subsurface_Gas_Outside_Building"] = _vw_has_sources(tl.subsurface_gas_outside_building)
        coverage["Search_Rescue_Conclusion"] = _vw_has_sources(tl.search_rescue_conclusion)
        coverage["Responding_Agencies"] = _lws_has_sources(tl.responding_agencies)
    else:
        # If no timeline at all, mark all timeline facts missing
        for k in [
            "Incident_Date", "Explosion_Time", "Weather_Conditions", "Initial_Odor_Report_Time",
            "Initial_Odor_Location_1", "Initial_Odor_Location_2", "Exelon_Technician_Arrival",
            "Meter_Services_Technician_Arrival", "Odor_Present_Shortly_Before_Explosion_Basement",
            "Odor_Present_Shortly_Before_Explosion_First_Floor", "Odor_Present_Shortly_Before_Explosion_Second_Floor",
            "Fire_Rescue_Dispatch_Time", "First_Responder_Arrival_Time", "Exelon_Emergency_Responder_Arrival",
            "Gas_Flow_Isolated_Time", "Bar_Hole_Testing_Time", "PUC_Directed_Bar_Hole_Tests",
            "Subsurface_Gas_Outside_Building", "Search_Rescue_Conclusion", "Responding_Agencies"
        ]:
            coverage[k] = False

    # Human Impact
    coverage["Immediate_Fatalities_Count"] = _vw_has_sources(hi.immediate_fatalities_count) if hi else False
    coverage["Immediate_Fatality_Category_Resident"] = _vw_has_sources(hi.immediate_fatality_category_resident) if hi else False
    coverage["Immediate_Fatality_Category_Employee"] = _vw_has_sources(hi.immediate_fatality_category_employee) if hi else False
    coverage["Delayed_Death_Count_And_Date"] = _vw_has_sources(hi.delayed_death_count_and_date) if hi else False
    coverage["Total_Deaths"] = _vw_has_sources(hi.total_deaths) if hi else False
    coverage["Total_Injuries"] = _vw_has_sources(hi.total_injuries) if hi else False

    # Technical Infrastructure
    if ti:
        coverage["Gas_Provider"] = _vw_has_sources(ti.gas_provider)
        coverage["Gas_Provider_Parent_Company"] = _vw_has_sources(ti.gas_provider_parent_company)
        coverage["Service_Line_Diameter"] = _vw_has_sources(ti.service_line_diameter)
        coverage["Service_Line_Material"] = _vw_has_sources(ti.service_line_material)
        coverage["Service_Line_Location"] = _vw_has_sources(ti.service_line_location)
        coverage["Meter_Type"] = _vw_has_sources(ti.meter_type)
        coverage["Meter_Location"] = _vw_has_sources(ti.meter_location)
        coverage["Leak_Equipment"] = _vw_has_sources(ti.leak_equipment)
        coverage["Leak_Location"] = _vw_has_sources(ti.leak_location)
        coverage["Personnel_Experience_Foreman"] = _vw_has_sources(ti.personnel_experience_foreman)
        coverage["Personnel_Experience_Meter_Services_Technician"] = _vw_has_sources(ti.personnel_experience_meter_services_technician)
    else:
        for k in [
            "Gas_Provider", "Gas_Provider_Parent_Company", "Service_Line_Diameter", "Service_Line_Material",
            "Service_Line_Location", "Meter_Type", "Meter_Location", "Leak_Equipment", "Leak_Location",
            "Personnel_Experience_Foreman", "Personnel_Experience_Meter_Services_Technician"
        ]:
            coverage[k] = False

    # Investigation Status
    if inv:
        coverage["Lead_Investigating_Agency"] = _vw_has_sources(inv.lead_investigating_agency)
        coverage["Preliminary_Report_Date"] = _vw_has_sources(inv.preliminary_report_date)

        coverage["Recovered_Indoor_Meter_Set"] = _vw_has_sources(inv.recovered_indoor_meter_set)
        coverage["Excavated_Service_Line_Portions"] = _vw_has_sources(inv.excavated_service_line_portions)
        coverage["Delivered_Components_To_Materials_Lab"] = _vw_has_sources(inv.delivered_components_to_materials_lab)

        coverage["Focus_Pipeline_Safety_Management_System"] = _vw_has_sources(inv.focus_pipeline_safety_management_system)
        coverage["Focus_Personnel_Training"] = _vw_has_sources(inv.focus_personnel_training)
        coverage["Focus_Operator_Qualifications"] = _vw_has_sources(inv.focus_operator_qualifications)
        coverage["Focus_Task_Specific_Procedures"] = _vw_has_sources(inv.focus_task_specific_procedures)
        coverage["Focus_Odor_Complaint_Response"] = _vw_has_sources(inv.focus_odor_complaint_response)
        coverage["Focus_Documentation"] = _vw_has_sources(inv.focus_documentation)
        coverage["Focus_Emergency_Response"] = _vw_has_sources(inv.focus_emergency_response)

        coverage["Entity_PHMSA"] = _vw_has_sources(inv.entity_phmsa)
        coverage["Entity_OSHA"] = _vw_has_sources(inv.entity_osha)
        coverage["Entity_Bristol_Township_Fire_Marshal"] = _vw_has_sources(inv.entity_bristol_township_fire_marshal)
        coverage["Entity_IBEW_Local_Union_614"] = _vw_has_sources(inv.entity_ibew_local_614)
        coverage["Entity_Saber_Healthcare_Group"] = _vw_has_sources(inv.entity_saber_healthcare_group)
    else:
        for k in [
            "Lead_Investigating_Agency", "Preliminary_Report_Date",
            "Recovered_Indoor_Meter_Set", "Excavated_Service_Line_Portions", "Delivered_Components_To_Materials_Lab",
            "Focus_Pipeline_Safety_Management_System", "Focus_Personnel_Training", "Focus_Operator_Qualifications",
            "Focus_Task_Specific_Procedures", "Focus_Odor_Complaint_Response", "Focus_Documentation", "Focus_Emergency_Response",
            "Entity_PHMSA", "Entity_OSHA", "Entity_Bristol_Township_Fire_Marshal", "Entity_IBEW_Local_Union_614", "Entity_Saber_Healthcare_Group"
        ]:
            coverage[k] = False

    return coverage


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Bristol Township explosion comprehensive documentation task.
    """
    # Initialize evaluator
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

    # Create top-level rubric node (critical, parallel)
    main_node = evaluator.add_parallel(
        id="Bristol_Explosion_Comprehensive_Documentation",
        desc="Complete and accurate documentation of the specified verifiable facts about the Dec 23, 2025 natural gas explosion at the Bristol Township skilled nursing facility, with supporting URLs.",
        parent=root,
        critical=True
    )

    # Run extractions concurrently
    fi_task = evaluator.extract(
        prompt=prompt_extract_facility_identification(),
        template_class=FacilityIdentificationExtraction,
        extraction_name="facility_identification"
    )
    tl_task = evaluator.extract(
        prompt=prompt_extract_incident_timeline(),
        template_class=IncidentTimelineExtraction,
        extraction_name="incident_timeline"
    )
    hi_task = evaluator.extract(
        prompt=prompt_extract_human_impact(),
        template_class=HumanImpactExtraction,
        extraction_name="human_impact"
    )
    ti_task = evaluator.extract(
        prompt=prompt_extract_technical_infrastructure(),
        template_class=TechnicalInfrastructureExtraction,
        extraction_name="technical_infrastructure"
    )
    inv_task = evaluator.extract(
        prompt=prompt_extract_investigation_status(),
        template_class=InvestigationStatusExtraction,
        extraction_name="investigation_status"
    )

    fi, tl, hi, ti, inv = await asyncio.gather(fi_task, tl_task, hi_task, ti_task, inv_task)

    # Global citation-support check (critical leaf)
    coverage = compute_citation_coverage(fi, tl, hi, ti, inv)
    missing = [k for k, ok in coverage.items() if not ok]
    all_supported = (len(missing) == 0)

    # Log missing coverage in custom info for transparency
    evaluator.add_custom_info(
        info={"missing_citation_support_fields": missing, "total_facts_checked": len(coverage)},
        info_type="citation_coverage"
    )

    evaluator.add_custom_node(
        result=all_supported,
        id="Citation_Support_For_All_Facts",
        desc="Every factual claim included in the report is supported by at least one verifiable URL reference from a reliable source.",
        parent=main_node,
        critical=True
    )

    # Build and verify each section
    await build_facility_identification(evaluator, main_node, fi)
    await build_incident_timeline(evaluator, main_node, tl)
    await build_human_impact(evaluator, main_node, hi)
    await build_technical_infrastructure(evaluator, main_node, ti)
    await build_investigation_status(evaluator, main_node, inv)

    return evaluator.get_summary()