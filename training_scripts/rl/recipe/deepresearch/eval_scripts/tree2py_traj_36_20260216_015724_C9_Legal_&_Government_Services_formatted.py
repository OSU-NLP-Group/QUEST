import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "sd_county_services_2026"
TASK_DESCRIPTION = (
    "I am researching South Dakota county government service accessibility for a civic engagement project. "
    "I need to identify four different South Dakota counties where each county provides a comprehensive set of essential government services with proper public accessibility.\n\n"
    "For each of the four counties, please verify and document that the county provides ALL of the following services:\n\n"
    "1. Property Tax Assessment Appeal Services: The county must observe the March 12, 2026 deadline for property tax assessment appeals to the local board for properties located in cities, towns, or organized townships. "
    "The county must accept written appeals filed with the clerk of the local board and provide public information about the appeal process.\n\n"
    "2. Vehicle Registration Renewal Services: The county Treasurer's office must process vehicle registration renewals following South Dakota's alphabetical system "
    "(where renewal month is based on the first letter of the registrant's last name). The county must provide online vehicle registration renewal services through my605Drive or the county's own portal.\n\n"
    "3. Vital Records Services: The county's Register of Deeds office must be authorized to issue certified copies of South Dakota vital records, including birth records, death records, and marriage records.\n\n"
    "4. Emergency Management Services: The county must maintain either a dedicated Office of Emergency Management or a designated Emergency Management Director, "
    "with publicly available contact information (phone number) and a specified physical office address.\n\n"
    "For each of the four counties you identify, please provide:\n"
    "- The county name\n"
    "- For each of the four service categories listed above, provide the official county website URL that documents that specific service, along with a brief description confirming the service meets the stated requirements."
)

APPEAL_DEADLINE_TEXT = "March 12, 2026"
STATE_NAME = "South Dakota"

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CountyServices(BaseModel):
    county_name: Optional[str] = None

    property_tax_urls: List[str] = Field(default_factory=list)
    property_tax_desc: Optional[str] = None

    vehicle_registration_urls: List[str] = Field(default_factory=list)
    vehicle_registration_desc: Optional[str] = None

    vital_records_urls: List[str] = Field(default_factory=list)
    vital_records_desc: Optional[str] = None

    emergency_mgmt_urls: List[str] = Field(default_factory=list)
    emergency_mgmt_desc: Optional[str] = None


class CountiesExtraction(BaseModel):
    counties: List[CountyServices] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_counties() -> str:
    return """
    Extract up to four South Dakota counties and the official documentation URLs for each of the following service categories per county:
    1) Property tax assessment appeal services
    2) Vehicle registration renewal services
    3) Vital records services (Register of Deeds: birth, death, marriage)
    4) Emergency management services (office or director with contacts)

    Requirements:
    - Return a JSON object with a 'counties' array of at most four entries.
    - For each county, include:
        county_name: string (e.g., "Minnehaha County")
        property_tax_urls: array of URLs (official county page(s) detailing property tax appeal procedures and deadlines)
        property_tax_desc: brief text extracted from the answer describing the appeal process or deadline
        vehicle_registration_urls: array of URLs (county Treasurer/DMV page(s); may include my605Drive for online renewal)
        vehicle_registration_desc: brief text extracted from the answer describing the renewal process or online capability
        vital_records_urls: array of URLs (Register of Deeds official page(s) that mention issuing certified birth/death/marriage records)
        vital_records_desc: brief text extracted from the answer confirming the service
        emergency_mgmt_urls: array of URLs (official county page(s) showing emergency management office or director and contacts)
        emergency_mgmt_desc: brief text extracted from the answer confirming the office/director and contact/address

    Rules:
    - Extract only URLs explicitly mentioned in the answer. Include full URLs (with http:// or https://).
    - Prefer official county domains (e.g., *.sd.gov county subpages or county-owned domains). For vehicle online renewal, my605Drive links are acceptable.
    - If the answer lists more than four counties, include only the first four. If fewer than four are provided, include as many as available; leave missing entries out (do not invent).
    - If a field is missing for a county, set it to null or an empty array as appropriate.

    Output strictly as JSON.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def pad_to_four_counties(extraction: CountiesExtraction) -> List[CountyServices]:
    """Pad or clip the counties list to exactly four entries."""
    counties = extraction.counties[:4]
    while len(counties) < 4:
        counties.append(CountyServices())
    return counties


def urls_exist(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_property_tax_checks(
    evaluator: Evaluator,
    county_node,
    county_name: Optional[str],
    urls: List[str],
) -> None:
    """
    Build verification nodes for the Property Tax Assessment Appeal Services.
    """
    # Service category node (critical)
    svc_node = evaluator.add_parallel(
        id=f"{county_node.id}_property_tax_services",
        desc="County provides property tax assessment appeal services with proper deadlines and procedures",
        parent=county_node,
        critical=True,
    )

    # Requirements subgroup (critical)
    req_node = evaluator.add_parallel(
        id=f"{svc_node.id}_requirements",
        desc="County meets all property tax appeal process requirements",
        parent=svc_node,
        critical=True,
    )

    # Existence gate for requirements
    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{req_node.id}_urls_provided",
        desc="Property tax appeal documentation URLs are provided",
        parent=req_node,
        critical=True,
    )

    # 1) Appeal deadline compliance
    deadline_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_appeal_deadline_compliance",
        desc=f"County observes the {APPEAL_DEADLINE_TEXT} deadline for property tax assessment appeals to local board for properties in cities, towns, or organized townships",
        parent=req_node,
        critical=True,
    )
    deadline_claim = (
        f"On official county page(s), {county_name or 'the county'} indicates that appeals to the local board "
        f"for properties located in cities, towns, or organized townships must be filed by {APPEAL_DEADLINE_TEXT}."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the page states or clearly implies the specific deadline 'March 12, 2026' "
            "for local board equalization appeals (cities/towns/organized townships). "
            "Look for terms like 'local board of equalization', 'appeal deadline', '2026 calendar', or explicit dates."
        ),
    )

    # 2) Written appeal process
    written_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_written_appeal_process",
        desc="County accepts written appeals filed with the clerk of the local board",
        parent=req_node,
        critical=True,
    )
    written_claim = (
        f"The official county page(s) state that {county_name or 'the county'} accepts written appeals "
        "filed with the clerk of the local board."
    )
    await evaluator.verify(
        claim=written_claim,
        node=written_leaf,
        sources=urls,
        additional_instruction=(
            "Look for language such as 'file a written appeal with the clerk of the local board', "
            "'written notice of appeal', or similar phrasing that explicitly mentions written appeals "
            "and the clerk."
        ),
    )

    # 3) Appeal information availability
    info_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_appeal_information_availability",
        desc="County provides public information about the property tax appeal process",
        parent=req_node,
        critical=True,
    )
    info_claim = (
        f"The official county page(s) provide public information describing the property tax assessment appeal process in {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=info_claim,
        node=info_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm that the page(s) describe steps or guidance for appealing property assessments, "
            "including timeline, board names, or instructions."
        ),
    )

    # Documentation subgroup (critical)
    doc_node = evaluator.add_parallel(
        id=f"{svc_node.id}_documentation",
        desc="County provides official documentation of property tax appeal services",
        parent=svc_node,
        critical=True,
    )

    # Existence gate for documentation
    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{doc_node.id}_urls_provided",
        desc="Property tax appeal documentation URLs are provided",
        parent=doc_node,
        critical=True,
    )

    doc_leaf = evaluator.add_leaf(
        id=f"{doc_node.id}_property_tax_reference",
        desc="Official county website URL documenting property tax appeal procedures and deadlines",
        parent=doc_node,
        critical=True,
    )
    doc_claim = (
        "This county page documents property tax assessment appeal procedures and deadlines."
    )
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=urls,
        additional_instruction=(
            "Verify the page is an official county resource and it covers property tax appeal procedures and deadlines."
        ),
    )


async def build_vehicle_registration_checks(
    evaluator: Evaluator,
    county_node,
    county_name: Optional[str],
    urls: List[str],
) -> None:
    """
    Build verification nodes for Vehicle Registration Renewal Services.
    """
    svc_node = evaluator.add_parallel(
        id=f"{county_node.id}_vehicle_registration_services",
        desc="County Treasurer's office processes vehicle registration renewals according to South Dakota alphabetical system",
        parent=county_node,
        critical=True,
    )

    req_node = evaluator.add_parallel(
        id=f"{svc_node.id}_requirements",
        desc="County meets all vehicle registration renewal requirements",
        parent=svc_node,
        critical=True,
    )

    # Existence gate
    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{req_node.id}_urls_provided",
        desc="Vehicle registration documentation URLs are provided",
        parent=req_node,
        critical=True,
    )

    # 1) Alphabetical renewal system
    alpha_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_alphabetical_renewal_system",
        desc="County follows South Dakota's system where renewal month is based on first letter of registrant's last name",
        parent=req_node,
        critical=True,
    )
    alpha_claim = (
        f"The official county page(s) indicate that renewal months are determined by the first letter "
        f"of the registrant's last name for {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=alpha_claim,
        node=alpha_leaf,
        sources=urls,
        additional_instruction=(
            "Look for an alphabetical schedule or text explaining the last-name letter determines the renewal month. "
            "Common phrasing: 'Renewal month is based on the first letter of the last name'."
        ),
    )

    # 2) Online renewal capability
    online_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_online_renewal_capability",
        desc="County provides online vehicle registration renewal services through my605Drive or county portal",
        parent=req_node,
        critical=True,
    )
    online_claim = (
        f"The official county page(s) provide online vehicle registration renewal via my605Drive "
        f"or the county's own online portal for {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=online_claim,
        node=online_leaf,
        sources=urls,
        additional_instruction=(
            "Pass if the page includes a link or instruction to renew online using my605Drive (https://my605drive.sd.gov) "
            "or a county-hosted online renewal portal."
        ),
    )

    # 3) County residence requirement
    resident_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_county_residence_requirement",
        desc="County processes renewals for vehicles registered to residents of the county",
        parent=req_node,
        critical=True,
    )
    resident_claim = (
        f"The official county page(s) indicate that renewals are processed for vehicles registered to residents "
        f"of {county_name or 'the county'} (county of residence requirement)."
    )
    await evaluator.verify(
        claim=resident_claim,
        node=resident_leaf,
        sources=urls,
        additional_instruction=(
            "Look for wording that renewals are handled by the Treasurer/DMV of the registrant's county of residence, "
            "or instructions to renew in one's county of residence."
        ),
    )

    # Documentation subgroup
    doc_node = evaluator.add_parallel(
        id=f"{svc_node.id}_documentation",
        desc="County provides official documentation of vehicle registration services",
        parent=svc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{doc_node.id}_urls_provided",
        desc="Vehicle registration documentation URLs are provided",
        parent=doc_node,
        critical=True,
    )

    doc_leaf = evaluator.add_leaf(
        id=f"{doc_node.id}_vehicle_registration_reference",
        desc="Official county website URL documenting vehicle registration renewal services and procedures",
        parent=doc_node,
        critical=True,
    )
    doc_claim = "This county page documents vehicle registration renewal services and procedures."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=urls,
        additional_instruction="Verify the page is an official county resource covering registration renewal steps/policies.",
    )


async def build_vital_records_checks(
    evaluator: Evaluator,
    county_node,
    county_name: Optional[str],
    urls: List[str],
) -> None:
    """
    Build verification nodes for Vital Records Services (Register of Deeds).
    """
    svc_node = evaluator.add_parallel(
        id=f"{county_node.id}_vital_records_services",
        desc="County Register of Deeds office provides certified vital records services",
        parent=county_node,
        critical=True,
    )

    req_node = evaluator.add_parallel(
        id=f"{svc_node.id}_requirements",
        desc="County Register of Deeds meets all vital records issuance requirements",
        parent=svc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{req_node.id}_urls_provided",
        desc="Vital records documentation URLs are provided",
        parent=req_node,
        critical=True,
    )

    # Birth
    birth_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_birth_record_issuance",
        desc="Register of Deeds office is authorized to issue certified copies of South Dakota birth records",
        parent=req_node,
        critical=True,
    )
    birth_claim = (
        f"The official county page(s) indicate that the Register of Deeds issues certified copies of {STATE_NAME} birth records for {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit phrasing like 'certified birth records' or 'issue certified copies of birth certificates' "
            "handled by the Register of Deeds."
        ),
    )

    # Death
    death_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_death_record_issuance",
        desc="Register of Deeds office is authorized to issue certified copies of South Dakota death records",
        parent=req_node,
        critical=True,
    )
    death_claim = (
        f"The official county page(s) indicate that the Register of Deeds issues certified copies of {STATE_NAME} death records for {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=death_claim,
        node=death_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit phrasing like 'certified death records' or 'issue certified copies of death certificates'."
        ),
    )

    # Marriage
    marriage_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_marriage_record_issuance",
        desc="Register of Deeds office is authorized to issue certified copies of South Dakota marriage records",
        parent=req_node,
        critical=True,
    )
    marriage_claim = (
        f"The official county page(s) indicate that the Register of Deeds issues certified copies of {STATE_NAME} marriage records for {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=marriage_claim,
        node=marriage_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit phrasing like 'certified marriage records' or 'issue certified copies of marriage certificates'."
        ),
    )

    # Documentation subgroup
    doc_node = evaluator.add_parallel(
        id=f"{svc_node.id}_documentation",
        desc="County provides official documentation of vital records services",
        parent=svc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{doc_node.id}_urls_provided",
        desc="Vital records documentation URLs are provided",
        parent=doc_node,
        critical=True,
    )

    doc_leaf = evaluator.add_leaf(
        id=f"{doc_node.id}_vital_records_reference",
        desc="Official county website URL documenting vital records services provided by Register of Deeds",
        parent=doc_node,
        critical=True,
    )
    doc_claim = "This county page documents vital records services (birth, death, marriage) provided by the Register of Deeds."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=urls,
        additional_instruction="Verify the page is an official county Register of Deeds resource covering certified vital records.",
    )


async def build_emergency_mgmt_checks(
    evaluator: Evaluator,
    county_node,
    county_name: Optional[str],
    urls: List[str],
) -> None:
    """
    Build verification nodes for Emergency Management Services.
    """
    svc_node = evaluator.add_parallel(
        id=f"{county_node.id}_emergency_management_presence",
        desc="County maintains a dedicated emergency management office or designated emergency manager",
        parent=county_node,
        critical=True,
    )

    req_node = evaluator.add_parallel(
        id=f"{svc_node.id}_requirements",
        desc="County meets all emergency management office requirements",
        parent=svc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{req_node.id}_urls_provided",
        desc="Emergency management documentation URLs are provided",
        parent=req_node,
        critical=True,
    )

    # Existence of office/director
    existence_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_emergency_office_existence",
        desc="County has an established Office of Emergency Management or designated Emergency Management Director",
        parent=req_node,
        critical=True,
    )
    existence_claim = (
        f"The official county page(s) show that {county_name or 'the county'} maintains an Office of Emergency Management "
        "or has a designated Emergency Management Director."
    )
    await evaluator.verify(
        claim=existence_claim,
        node=existence_leaf,
        sources=urls,
        additional_instruction=(
            "Look for titles such as 'Emergency Management', 'Emergency Management Director', 'OEM', or similar."
        ),
    )

    # Public contact information (phone)
    contact_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_emergency_contact_information",
        desc="County provides public contact information for emergency management office including phone number",
        parent=req_node,
        critical=True,
    )
    contact_claim = (
        f"The official county page(s) provide a public phone number for the emergency management office/director in {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm presence of a phone number on the emergency management page (e.g., patterns like (605) xxx-xxxx)."
        ),
    )

    # Physical office address
    address_leaf = evaluator.add_leaf(
        id=f"{req_node.id}_emergency_office_location",
        desc="County specifies physical address for emergency management office",
        parent=req_node,
        critical=True,
    )
    address_claim = (
        f"The official county page(s) specify a physical street address for the emergency management office in {county_name or 'the county'}."
    )
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm a street address is present (e.g., number + street, city, SD, ZIP)."
        ),
    )

    # Documentation subgroup
    doc_node = evaluator.add_parallel(
        id=f"{svc_node.id}_documentation",
        desc="County provides official documentation of emergency management services",
        parent=svc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=urls_exist(urls),
        id=f"{doc_node.id}_urls_provided",
        desc="Emergency management documentation URLs are provided",
        parent=doc_node,
        critical=True,
    )

    doc_leaf = evaluator.add_leaf(
        id=f"{doc_node.id}_emergency_management_reference",
        desc="Official county website URL documenting emergency management office and contact information",
        parent=doc_node,
        critical=True,
    )
    doc_claim = "This county page documents the emergency management office/director and public contact/location information."
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=urls,
        additional_instruction="Verify the page is an official county resource showing emergency management office/director, phone, and address.",
    )


async def verify_one_county(
    evaluator: Evaluator,
    parent_node,
    county_info: CountyServices,
    county_index: int,
) -> None:
    """
    Build the verification subtree for a single county with all required service categories.
    """
    county_label = county_info.county_name or f"County #{county_index + 1}"
    county_node = evaluator.add_parallel(
        id=f"county_{county_index+1}",
        desc=f"{county_label} - qualifying county meeting all service and accessibility requirements",
        parent=parent_node,
        critical=False,  # Allow partial credit across counties at the root level
    )

    # Property Tax
    await build_property_tax_checks(
        evaluator=evaluator,
        county_node=county_node,
        county_name=county_info.county_name,
        urls=county_info.property_tax_urls or [],
    )

    # Vehicle Registration
    await build_vehicle_registration_checks(
        evaluator=evaluator,
        county_node=county_node,
        county_name=county_info.county_name,
        urls=county_info.vehicle_registration_urls or [],
    )

    # Vital Records
    await build_vital_records_checks(
        evaluator=evaluator,
        county_node=county_node,
        county_name=county_info.county_name,
        urls=county_info.vital_records_urls or [],
    )

    # Emergency Management
    await build_emergency_mgmt_checks(
        evaluator=evaluator,
        county_node=county_node,
        county_name=county_info.county_name,
        urls=county_info.emergency_mgmt_urls or [],
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
    Evaluate an answer for the South Dakota county government services accessibility task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel to allow independent county verification
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

    # Extract structured county/service references
    extraction = await evaluator.extract(
        prompt=prompt_extract_counties(),
        template_class=CountiesExtraction,
        extraction_name="sd_county_services_struct",
    )
    counties = pad_to_four_counties(extraction)

    # Add custom info helpful for auditing
    evaluator.add_custom_info(
        info={"appeal_deadline": APPEAL_DEADLINE_TEXT, "state": STATE_NAME},
        info_type="constants",
        info_name="deadline_and_state",
    )

    # Build verification for up to four counties
    for idx in range(4):
        await verify_one_county(
            evaluator=evaluator,
            parent_node=root,
            county_info=counties[idx],
            county_index=idx,
        )

    return evaluator.get_summary()