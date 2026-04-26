import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "telecom_bc_emergency"
TASK_DESCRIPTION = """
Identify four telecommunications carriers or network service providers that operate in the United States and meet ALL of the following requirements:

1. Geographic Coverage: The carrier must provide wireless or wireline telecommunications services in at least two of the following states: California, Texas, Florida, or New York.

2. Backup Power Compliance: The carrier must publicly document that their network infrastructure includes backup power systems, with specific mention of battery backup or generator systems for their cell towers or network facilities.

3. Business Continuity Certification: The carrier must hold or publicly claim compliance with ISO 22301 (Business Continuity Management Systems) or demonstrate an equivalent business continuity management framework.

4. Emergency Telecommunications Participation: The carrier must be an authorized participant in at least one federal emergency telecommunications priority program (such as WPS - Wireless Priority Service, GETS - Government Emergency Telecommunications Service, or TSP - Telecommunications Service Priority).

5. 24/7 Network Operations Center: The carrier must operate or publicly document having a Network Operations Center (NOC) that provides 24/7/365 network monitoring and management.

6. FCC Outage Reporting Compliance: The carrier must be subject to and comply with FCC's Network Outage Reporting System (NORS) requirements as a registered telecommunications provider.

For each identified carrier, provide:
- The carrier's official name and corporate website
- Specific documentation or webpage URL confirming each of the six requirements above
- A brief description (2-3 sentences) explaining how the carrier meets each requirement
"""

ALLOWED_STATES = {"California", "Texas", "Florida", "New York"}
ABBREV_MAP = {"ca": "California", "tx": "Texas", "fl": "Florida", "ny": "New York"}


class CarrierData(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None

    geographic_states: List[str] = Field(default_factory=list)
    geographic_description: Optional[str] = None
    geographic_urls: List[str] = Field(default_factory=list)

    backup_power_description: Optional[str] = None
    backup_power_urls: List[str] = Field(default_factory=list)

    business_continuity_description: Optional[str] = None
    business_continuity_urls: List[str] = Field(default_factory=list)

    emergency_programs: List[str] = Field(default_factory=list)  # e.g., ["WPS", "GETS"]
    emergency_program_description: Optional[str] = None
    emergency_program_urls: List[str] = Field(default_factory=list)

    noc_description: Optional[str] = None
    noc_urls: List[str] = Field(default_factory=list)

    fcc_compliance_description: Optional[str] = None
    fcc_compliance_urls: List[str] = Field(default_factory=list)


class CarriersExtraction(BaseModel):
    carriers: List[CarrierData] = Field(default_factory=list)


def prompt_extract_carriers() -> str:
    return """
    Extract up to four U.S. telecommunications carriers or network service providers from the provided answer.
    For each carrier, extract the following fields. If anything is missing in the answer text, return null for that field or an empty list for URLs:

    - name: Official registered name of the carrier.
    - website: Official corporate website URL (prefer the top-level corporate domain, not social media).
    - geographic_states: List of states among [California, Texas, Florida, New York] explicitly claimed in the answer where the carrier provides wireless or wireline telecommunications services. Use either full names or common abbreviations (CA, TX, FL, NY). Only include states that are explicitly mentioned in the answer.
    - geographic_description: 2-3 sentence description explaining the coverage claim as presented in the answer.
    - geographic_urls: All URLs cited that support geographic coverage in those states, including coverage maps, service availability pages, or product pages clearly tied to these states.

    - backup_power_description: 1-2 sentence description of the backup power systems (battery backup or generators) as presented in the answer.
    - backup_power_urls: All URLs cited that document such backup power systems for network facilities or cell sites.

    - business_continuity_description: 1-2 sentence description of the carrier’s ISO 22301 certification/compliance, or equivalent BCMS framework, as presented in the answer.
    - business_continuity_urls: All URLs cited that confirm ISO 22301 certification/compliance or an equivalent business continuity management system.

    - emergency_programs: List program names explicitly mentioned among [WPS, GETS, TSP] that the carrier participates in.
    - emergency_program_description: 1-2 sentence description of the participation claim as presented in the answer.
    - emergency_program_urls: All URLs cited that confirm participation in WPS, GETS, or TSP (could be carrier pages or official DHS/CISA program pages listing the carrier as authorized).

    - noc_description: 1-2 sentence description explaining the NOC (Network Operations Center) operations and 24/7/365 monitoring as presented in the answer.
    - noc_urls: All URLs cited that confirm having a 24/7/365 NOC.

    - fcc_compliance_description: 1-2 sentence description explaining FCC NORS (Network Outage Reporting System) compliance as presented in the answer.
    - fcc_compliance_urls: All URLs cited that confirm being subject to and compliant with FCC NORS requirements.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text. Do not invent URLs.
    - Return full URLs (include http:// or https://). If protocol missing, prepend http://.
    - Include URLs from official carrier sites or authoritative sources (FCC, DHS/CISA), if these are cited in the answer.

    Return a JSON object with a single field:
    - carriers: an array of up to four CarrierData items in the same order they appear in the answer.
    """


def normalize_state_name(s: str) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in ABBREV_MAP:
        return ABBREV_MAP[t]
    if "calif" in t or t == "california":
        return "California"
    if t == "texas" or t == "tx":
        return "Texas"
    if t == "florida" or t == "fl":
        return "Florida"
    if "new york" in t or t == "ny":
        return "New York"
    return None


def normalize_allowed_states(states: List[str]) -> List[str]:
    result = []
    for s in states:
        ns = normalize_state_name(s)
        if ns and ns in ALLOWED_STATES and ns not in result:
            result.append(ns)
    return result


def first_two_allowed(states: List[str]) -> List[str]:
    norm = normalize_allowed_states(states)
    return norm[:2]


async def verify_geographic_coverage(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    cov_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_Geographic_Coverage",
        desc="The carrier provides telecommunications services in at least two of the specified states (California, Texas, Florida, or New York)",
        parent=parent,
        critical=True
    )
    desc_present = bool(carrier.geographic_description and carrier.geographic_description.strip())
    evaluator.add_custom_node(
        result=desc_present,
        id=f"Carrier_{idx}_Geographic_Description",
        desc="Describe which states the carrier operates in and the type of services provided",
        parent=cov_node,
        critical=True  # Adjusted to satisfy critical parent constraints and the task requires description
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_Geographic_URL",
        desc="Provide URL reference confirming geographic coverage",
        parent=cov_node,
        critical=True
    )
    states_two = first_two_allowed(carrier.geographic_states)
    if len(states_two) >= 2 and carrier.name:
        claim = f"{carrier.name} provides wireless or wireline telecommunications services in {states_two[0]} and {states_two[1]}."
        add_ins = "Verify the webpage(s) show service coverage or operations explicitly in the named states. Accept coverage maps, availability pages, or official service descriptions. Allow abbreviations (CA, TX, FL, NY). If no URLs are provided, mark as not supported."
    else:
        claim = "This carrier provides wireless or wireline telecommunications services in at least two of the following states: California, Texas, Florida, New York."
        add_ins = "Check if the webpage(s) show coverage or operations in any two of the listed states. Allow maps or explicit state listings. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.geographic_urls if carrier.geographic_urls else None,
        additional_instruction=add_ins
    )


async def verify_backup_power(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    bp_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_Backup_Power",
        desc="The carrier publicly documents backup power systems including battery backup or generators for network facilities",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.backup_power_description and carrier.backup_power_description.strip()),
        id=f"Carrier_{idx}_Backup_Power_Description",
        desc="Describe the backup power systems documented by the carrier",
        parent=bp_node,
        critical=True
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_Backup_Power_URL",
        desc="Provide URL reference confirming backup power documentation",
        parent=bp_node,
        critical=True
    )
    name = carrier.name or "The carrier"
    claim = f"{name}'s network infrastructure includes backup power systems such as battery backups or generators for cell towers or network facilities."
    add_ins = "Confirm that the webpage(s) explicitly mention battery backup, generators, or equivalent power backup for network facilities or cell sites. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.backup_power_urls if carrier.backup_power_urls else None,
        additional_instruction=add_ins
    )


async def verify_business_continuity(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    bc_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_Business_Continuity",
        desc="The carrier holds ISO 22301 certification or demonstrates equivalent business continuity management framework",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.business_continuity_description and carrier.business_continuity_description.strip()),
        id=f"Carrier_{idx}_Business_Continuity_Description",
        desc="Describe the business continuity certification or equivalent framework the carrier maintains",
        parent=bc_node,
        critical=True
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_Business_Continuity_URL",
        desc="Provide URL reference confirming business continuity certification or framework",
        parent=bc_node,
        critical=True
    )
    name = carrier.name or "The carrier"
    claim = f"{name} holds ISO 22301 certification or publicly claims compliance with an equivalent Business Continuity Management System framework."
    add_ins = "Support may include ISO 22301 certificates, compliance statements, audits, or equivalent BCMS documentation. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.business_continuity_urls if carrier.business_continuity_urls else None,
        additional_instruction=add_ins
    )


async def verify_emergency_program(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    ep_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_Emergency_Program",
        desc="The carrier participates in at least one federal emergency telecommunications priority program (WPS, GETS, or TSP)",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.emergency_program_description and carrier.emergency_program_description.strip()),
        id=f"Carrier_{idx}_Emergency_Program_Description",
        desc="Describe which federal emergency telecommunications program(s) the carrier participates in",
        parent=ep_node,
        critical=True
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_Emergency_Program_URL",
        desc="Provide URL reference confirming emergency program participation",
        parent=ep_node,
        critical=True
    )
    prog = carrier.emergency_programs[0] if carrier.emergency_programs else None
    name = carrier.name or "The carrier"
    if prog:
        claim = f"{name} participates in {prog}, a federal emergency telecommunications priority program."
    else:
        claim = f"{name} participates in at least one of WPS (Wireless Priority Service), GETS (Government Emergency Telecommunications Service), or TSP (Telecommunications Service Priority)."
    add_ins = "Accept carrier pages or authoritative DHS/CISA program pages indicating the carrier's authorized participation. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.emergency_program_urls if carrier.emergency_program_urls else None,
        additional_instruction=add_ins
    )


async def verify_noc(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    noc_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_NOC",
        desc="The carrier operates a Network Operations Center providing 24/7/365 network monitoring and management",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.noc_description and carrier.noc_description.strip()),
        id=f"Carrier_{idx}_NOC_Description",
        desc="Describe the NOC operations and monitoring capabilities",
        parent=noc_node,
        critical=True
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_NOC_URL",
        desc="Provide URL reference confirming NOC operations",
        parent=noc_node,
        critical=True
    )
    name = carrier.name or "The carrier"
    claim = f"{name} operates a Network Operations Center (NOC) that provides 24/7/365 network monitoring and management."
    add_ins = "Confirm the page explicitly states 24/7 (or 24x7x365) NOC operations or equivalent continuous monitoring. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.noc_urls if carrier.noc_urls else None,
        additional_instruction=add_ins
    )


async def verify_fcc_compliance(evaluator: Evaluator, parent, carrier: CarrierData, idx: int):
    fcc_node = evaluator.add_parallel(
        id=f"Carrier_{idx}_FCC_Compliance",
        desc="The carrier is subject to and complies with FCC Network Outage Reporting System (NORS) requirements",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.fcc_compliance_description and carrier.fcc_compliance_description.strip()),
        id=f"Carrier_{idx}_FCC_Compliance_Description",
        desc="Describe how the carrier demonstrates compliance with FCC NORS requirements",
        parent=fcc_node,
        critical=True
    )
    url_leaf = evaluator.add_leaf(
        id=f"Carrier_{idx}_FCC_Compliance_URL",
        desc="Provide URL reference confirming FCC NORS compliance",
        parent=fcc_node,
        critical=True
    )
    name = carrier.name or "The carrier"
    claim = f"{name} is subject to and complies with FCC Network Outage Reporting System (NORS) requirements as a registered telecommunications provider."
    add_ins = "Accept FCC pages describing NORS obligations for carriers or the carrier's own compliance statements. If no URLs are provided, mark as not supported."
    await evaluator.verify(
        claim=claim,
        node=url_leaf,
        sources=carrier.fcc_compliance_urls if carrier.fcc_compliance_urls else None,
        additional_instruction=add_ins
    )


async def verify_carrier(evaluator: Evaluator, root_node, carrier: CarrierData, ordinal: int):
    car_node = evaluator.add_sequential(
        id=f"Carrier_{ordinal}",
        desc=f"{['First','Second','Third','Fourth'][ordinal-1]} telecommunications carrier meeting all requirements",
        parent=root_node,
        critical=False
    )
    ident_node = evaluator.add_parallel(
        id=f"Carrier_{ordinal}_Identification",
        desc="Correctly identify a valid US telecommunications carrier with official name and corporate website",
        parent=car_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.name and carrier.name.strip()),
        id=f"Carrier_{ordinal}_Name",
        desc="Provide the official registered name of the telecommunications carrier",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(carrier.website and carrier.website.strip()),
        id=f"Carrier_{ordinal}_Website",
        desc="Provide the official corporate website URL of the carrier",
        parent=ident_node,
        critical=True
    )
    reqs_node = evaluator.add_parallel(
        id=f"Carrier_{ordinal}_Requirements",
        desc="Verify that the carrier meets all six operational and regulatory requirements",
        parent=car_node,
        critical=True
    )
    await verify_geographic_coverage(evaluator, reqs_node, carrier, ordinal)
    await verify_backup_power(evaluator, reqs_node, carrier, ordinal)
    await verify_business_continuity(evaluator, reqs_node, carrier, ordinal)
    await verify_emergency_program(evaluator, reqs_node, carrier, ordinal)
    await verify_noc(evaluator, reqs_node, carrier, ordinal)
    await verify_fcc_compliance(evaluator, reqs_node, carrier, ordinal)


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
        strategy=AggregationStrategy.PARALLEL,  # Parallel across carriers
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_carriers(),
        template_class=CarriersExtraction,
        extraction_name="carriers_structured"
    )
    carriers_list = extracted.carriers if extracted and extracted.carriers else []
    # Ensure exactly 4 carriers (pad with empty placeholders if fewer)
    while len(carriers_list) < 4:
        carriers_list.append(CarrierData())
    for i in range(4):
        await verify_carrier(evaluator, root, carriers_list[i], i + 1)
    evaluator.add_custom_info(
        {"allowed_states": list(ALLOWED_STATES)},
        info_type="helper",
        info_name="allowed_states_for_geographic_requirement"
    )
    return evaluator.get_summary()