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
TASK_ID = "chicago_office_conversion_2024_2025"
TASK_DESCRIPTION = (
    "Identify a specific office-to-residential conversion project in Chicago that was approved or is under development "
    "in 2024 or 2025 as part of the city's initiative to repurpose downtown office buildings. For the project you identify, "
    "provide: (1) The complete street address of the property in Chicago; (2) The name of the development company or property owner "
    "responsible for the conversion, along with the headquarters address of that development company in Chicago; "
    "(3) The total number of residential units being created in the conversion, the current development status of the project "
    "(approved, under construction, or completed), and a reference URL from an official city government or credible news source "
    "documenting all this information. All information must be grounded in verifiable sources accessible through the provided reference URL."
)

# Allowed status values for normalization
ALLOWED_STATUSES = {"approved", "under construction", "completed"}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProjectExtraction(BaseModel):
    project_name: Optional[str] = None
    property_address: Optional[str] = None
    developer_name: Optional[str] = None
    developer_hq_address: Optional[str] = None
    total_residential_units: Optional[str] = None
    current_status: Optional[str] = None
    approval_or_initiation_year: Optional[str] = None
    initiative_linkage_phrase: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_project() -> str:
    return (
        "From the provided answer, extract exactly one (the first) Chicago office-to-residential conversion project "
        "that the answer describes. Return the following fields:\n"
        "1) project_name: The project's commonly referenced name or building name (if given); otherwise null.\n"
        "2) property_address: The complete street address of the property in Chicago exactly as written in the answer.\n"
        "3) developer_name: The development company or property owner responsible for the conversion.\n"
        "4) developer_hq_address: The developer/company headquarters street address in Chicago as stated in the answer (if provided); otherwise null.\n"
        "5) total_residential_units: The total number of residential units being created (extract exactly as stated; keep as string to allow ranges or approximations).\n"
        "6) current_status: One of: approved, under construction, or completed (extract exactly as stated in the answer; use these words if possible).\n"
        "7) approval_or_initiation_year: The year 2024 or 2025 if the answer states the approval/initiative year; otherwise null.\n"
        "8) initiative_linkage_phrase: Any phrase the answer uses to indicate linkage to the city's initiative to repurpose downtown office buildings "
        "(e.g., 'LaSalle Street Reimagined', 'city office-to-residential initiative'); otherwise null.\n"
        "9) reference_urls: All URLs cited in the answer as sources for this project (include city government pages or credible news if present). "
        "Extract actual URLs only; include markdown link targets.\n"
        "If any field is missing in the answer, set it to null (or empty list for URLs). Do not invent or infer new details."
    )


# --------------------------------------------------------------------------- #
# Helper: additional instruction builders                                     #
# --------------------------------------------------------------------------- #
def instruction_allowed_source_types() -> str:
    return (
        "Determine whether the page is from an official City of Chicago government source or a credible news outlet.\n"
        "• Official City sources typically include domains like 'chicago.gov' or pages of City departments (e.g., Department of Planning and Development), "
        "City Council/Plan Commission items, official city press releases.\n"
        "• Credible news outlets include recognized mainstream or reputable local journalism such as Chicago Tribune, Chicago Sun-Times, Crain's Chicago Business, "
        "Block Club Chicago, WBEZ, ABC7 Chicago, NBC Chicago, CBS Chicago, WGN, or other comparably reputable outlets.\n"
        "When verifying a single URL, decide if that specific page qualifies. For multi-URL verification, it is sufficient if at least one URL qualifies."
    )


def instruction_address_support(property_address: Optional[str]) -> str:
    return (
        "Verify the page explicitly identifies the property's address or clearly supports the stated location in Chicago. "
        "Minor formatting or abbreviations are acceptable (e.g., 'Chicago, IL' variants). "
        f"Target address: '{property_address or ''}'."
    )


def instruction_developer_support(developer_name: Optional[str]) -> str:
    return (
        "Verify the page explicitly identifies the responsible developer/company or property owner for the conversion. "
        f"Target developer: '{developer_name or ''}'. Allow minor name variations (LLC/Inc.)."
    )


def instruction_hq_support(developer_hq_address: Optional[str]) -> str:
    return (
        "Verify the page explicitly states the developer/company headquarters street address. "
        f"Target HQ address: '{developer_hq_address or ''}'. Minor formatting differences are acceptable."
    )


def instruction_hq_chicago() -> str:
    return (
        "Verify that the stated developer/company headquarters address is located in Chicago, Illinois (city-level). "
        "Accept reasonable variants such as 'Chicago, IL'."
    )


def instruction_units_support(units: Optional[str]) -> str:
    return (
        "Verify the page supports the stated total number of residential units being created. "
        f"Target units: '{units or ''}'. Allow minor rounding (e.g., 370 vs. 372)."
    )


def instruction_status_support(status: Optional[str]) -> str:
    return (
        "Verify that the page supports the stated current project status. "
        f"Target status: '{status or ''}'. Accept synonyms for 'approved' (e.g., Plan Commission/City Council approved) or "
        "'under construction' (e.g., construction underway) or 'completed' (e.g., project finished/opened)."
    )


def instruction_chicago_location(property_address: Optional[str]) -> str:
    return (
        "Confirm the project/property is located in Chicago, Illinois. "
        f"If the full address appears, check that it includes Chicago. Address given: '{property_address or ''}'."
    )


def instruction_conversion() -> str:
    return (
        "Confirm this is an office-to-residential conversion (adaptive reuse of an existing office building, not new construction). "
        "Look for keywords/phrases like 'office-to-residential', 'adaptive reuse', 'conversion of office building', "
        "'former office tower becoming apartments'."
    )


def instruction_year_eligibility(year_str: Optional[str]) -> str:
    return (
        "Confirm the project was approved or initiated/under development in 2024 or 2025. "
        f"If the year in the answer is '{year_str or ''}', verify that this is reflected on the page. "
        "Look for dates on approvals, program selections, press releases, or credible reporting in 2024 or 2025."
    )


def instruction_initiative_linkage(phrase: Optional[str]) -> str:
    return (
        "Confirm the source presents the project as part of the city's initiative to repurpose downtown office buildings. "
        "Common phrasings include 'LaSalle Street Reimagined', 'downtown office-to-residential initiative', "
        "'city office conversion program'. "
        f"If a specific phrase is provided in the answer ('{phrase or ''}'), check for it or an equivalent on the page."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction functions                                    #
# --------------------------------------------------------------------------- #
async def build_project_eligibility(
    evaluator: Evaluator,
    parent_node,
    data: ProjectExtraction,
) -> None:
    """
    Build and verify the Project Eligibility subtree.
    """
    eligibility_node = evaluator.add_parallel(
        id="Project_Eligibility",
        desc="Verify the identified project satisfies the eligibility criteria in the prompt",
        parent=parent_node,
        critical=True,
    )

    # Chicago_Location
    chicago_node = evaluator.add_leaf(
        id="Chicago_Location",
        desc="Project/property is located in Chicago, Illinois",
        parent=eligibility_node,
        critical=True,
    )
    chicago_claim = f"The property at '{data.property_address or ''}' is located in Chicago, Illinois."
    await evaluator.verify(
        claim=chicago_claim,
        node=chicago_node,
        sources=data.reference_urls,
        additional_instruction=instruction_chicago_location(data.property_address),
    )

    # Office_to_Residential_Conversion
    conversion_node = evaluator.add_leaf(
        id="Office_to_Residential_Conversion",
        desc="Project is an office-to-residential conversion (adaptive reuse of an existing building; not new construction)",
        parent=eligibility_node,
        critical=True,
    )
    conversion_claim = (
        "This project converts an existing office building into residential units (office-to-residential adaptive reuse), not new construction."
    )
    await evaluator.verify(
        claim=conversion_claim,
        node=conversion_node,
        sources=data.reference_urls,
        additional_instruction=instruction_conversion(),
    )

    # Approved_or_Initiated_2024_2025
    year_node = evaluator.add_leaf(
        id="Approved_or_Initiated_2024_2025",
        desc="Project was approved or initiated/under development in 2024 or 2025",
        parent=eligibility_node,
        critical=True,
    )
    if data.approval_or_initiation_year in {"2024", "2025"}:
        year_claim = f"The project was approved or initiated/under development in {data.approval_or_initiation_year}."
    else:
        year_claim = "The project was approved or initiated/under development in 2024 or 2025."
    await evaluator.verify(
        claim=year_claim,
        node=year_node,
        sources=data.reference_urls,
        additional_instruction=instruction_year_eligibility(data.approval_or_initiation_year),
    )

    # Downtown_Initiative_Linkage
    initiative_node = evaluator.add_leaf(
        id="Downtown_Initiative_Linkage",
        desc="Project is presented as part of the city's initiative to repurpose downtown office buildings (as stated/attested in the cited source[s])",
        parent=eligibility_node,
        critical=True,
    )
    initiative_claim = (
        "The cited source presents this project as part of the City's initiative to repurpose downtown office buildings "
        "(e.g., LaSalle Street Reimagined or similar city-led downtown office-to-residential initiative)."
    )
    await evaluator.verify(
        claim=initiative_claim,
        node=initiative_node,
        sources=data.reference_urls,
        additional_instruction=instruction_initiative_linkage(data.initiative_linkage_phrase),
    )


async def build_required_details(
    evaluator: Evaluator,
    parent_node,
    data: ProjectExtraction,
) -> None:
    """
    Build the Required Project and Developer Details subtree.
    Each check is a binary existence/format check (content support is verified in the sourcing subtree).
    """
    required_node = evaluator.add_parallel(
        id="Required_Project_and_Developer_Details",
        desc="Provide all required attributes for the identified project and developer/owner",
        parent=parent_node,
        critical=True,
    )

    # Complete_Property_Street_Address (existence)
    evaluator.add_custom_node(
        result=bool(data.property_address and data.property_address.strip()),
        id="Complete_Property_Street_Address",
        desc="Provide the complete street address of the property in Chicago",
        parent=required_node,
        critical=True,
    )

    # Developer_or_Property_Owner_Name (existence)
    evaluator.add_custom_node(
        result=bool(data.developer_name and data.developer_name.strip()),
        id="Developer_or_Property_Owner_Name",
        desc="Identify the development company or property owner responsible for the conversion",
        parent=required_node,
        critical=True,
    )

    # Developer_HQ_Address_in_Chicago (existence only; Chicago confirmation will be verified under sourcing)
    evaluator.add_custom_node(
        result=bool(data.developer_hq_address and data.developer_hq_address.strip()),
        id="Developer_HQ_Address_in_Chicago",
        desc="Provide the headquarters street address of the developer/company and confirm it is located in Chicago",
        parent=required_node,
        critical=True,
    )

    # Total_Residential_Units (existence)
    evaluator.add_custom_node(
        result=bool(data.total_residential_units and data.total_residential_units.strip()),
        id="Total_Residential_Units",
        desc="State the total number of residential units being created in the conversion",
        parent=required_node,
        critical=True,
    )

    # Current_Project_Status (existence + membership check)
    normalized_status = (data.current_status or "").strip().lower()
    evaluator.add_custom_node(
        result=bool(normalized_status in ALLOWED_STATUSES),
        id="Current_Project_Status",
        desc="Indicate the current project status as one of: approved, under construction, or completed",
        parent=required_node,
        critical=True,
    )


async def build_sourcing_and_verifiability(
    evaluator: Evaluator,
    parent_node,
    data: ProjectExtraction,
) -> None:
    """
    Build the Sourcing and Verifiability subtree, including:
    - At least one allowed source type URL.
    - Each reported detail supported by the cited URL(s).
    """
    sourcing_node = evaluator.add_parallel(
        id="Sourcing_and_Verifiability",
        desc="Ensure the response is verifiable from provided sources",
        parent=parent_node,
        critical=True,
    )

    # Reference_URL_From_Allowed_Source_Type
    allowed_source_node = evaluator.add_leaf(
        id="Reference_URL_From_Allowed_Source_Type",
        desc="Provide at least one reference URL from an official city government source or a credible news outlet",
        parent=sourcing_node,
        critical=True,
    )
    allowed_source_claim = "This webpage is an official City of Chicago government source or a credible news outlet."
    # Multi-URL verification: PASS if any URL qualifies
    await evaluator.verify(
        claim=allowed_source_claim,
        node=allowed_source_node,
        sources=data.reference_urls,
        additional_instruction=instruction_allowed_source_types(),
    )

    # All_Claims_Supported_By_Cited_URLs (expand into concrete child checks)
    supported_all_node = evaluator.add_parallel(
        id="All_Claims_Supported_By_Cited_URLs",
        desc="All reported required details (address, developer/owner, developer HQ address, unit count, and status) are supported by the cited reference URL(s)",
        parent=sourcing_node,
        critical=True,
    )

    # Address supported
    addr_supported_leaf = evaluator.add_leaf(
        id="Address_Supported",
        desc="The property street address is supported by the cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    addr_claim = f"The property's street address is '{data.property_address or ''}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_supported_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_address_support(data.property_address),
    )

    # Developer name supported
    developer_supported_leaf = evaluator.add_leaf(
        id="Developer_Supported",
        desc="The developer/company or property owner is supported by the cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    developer_claim = f"The developer/company (or property owner) responsible is '{data.developer_name or ''}'."
    await evaluator.verify(
        claim=developer_claim,
        node=developer_supported_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_developer_support(data.developer_name),
    )

    # Developer HQ address supported
    hq_supported_leaf = evaluator.add_leaf(
        id="Developer_HQ_Address_Supported",
        desc="The developer/company HQ street address is supported by the cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    hq_claim = f"The developer/company headquarters street address is '{data.developer_hq_address or ''}'."
    await evaluator.verify(
        claim=hq_claim,
        node=hq_supported_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_hq_support(data.developer_hq_address),
    )

    # HQ is in Chicago supported
    hq_chicago_leaf = evaluator.add_leaf(
        id="Developer_HQ_In_Chicago_Supported",
        desc="The HQ address is confirmed to be located in Chicago, Illinois, supported by cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    hq_chicago_claim = "The developer/company headquarters address is located in Chicago, Illinois."
    await evaluator.verify(
        claim=hq_chicago_claim,
        node=hq_chicago_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_hq_chicago(),
    )

    # Units supported
    units_supported_leaf = evaluator.add_leaf(
        id="Units_Supported",
        desc="The total number of residential units is supported by the cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    units_claim = f"The total residential unit count for this conversion project is '{data.total_residential_units or ''}'."
    await evaluator.verify(
        claim=units_claim,
        node=units_supported_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_units_support(data.total_residential_units),
    )

    # Status supported
    status_supported_leaf = evaluator.add_leaf(
        id="Status_Supported",
        desc="The current project status (approved / under construction / completed) is supported by the cited source(s)",
        parent=supported_all_node,
        critical=True,
    )
    status_claim = f"The current project status is '{(data.current_status or '').strip()}'."
    await evaluator.verify(
        claim=status_claim,
        node=status_supported_leaf,
        sources=data.reference_urls,
        additional_instruction=instruction_status_support(data.current_status),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Chicago office-to-residential conversion project task.
    """
    # Initialize evaluator (root is a non-critical container)
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

    # Extract structured project info from the answer
    data = await evaluator.extract(
        prompt=prompt_extract_project(),
        template_class=ProjectExtraction,
        extraction_name="extracted_project",
    )

    # Top-level critical node mirroring rubric root
    main_node = evaluator.add_parallel(
        id="Chicago_Office_Conversion_Project",
        desc="Identify and document a Chicago office-to-residential conversion project from 2024–2025 and provide required project, developer, and sourcing details",
        parent=root,
        critical=True,
    )

    # Build subtrees
    await build_project_eligibility(evaluator, main_node, data)
    await build_required_details(evaluator, main_node, data)
    await build_sourcing_and_verifiability(evaluator, main_node, data)

    # Optionally record custom info for debugging
    evaluator.add_custom_info(
        info={
            "extracted_project_name": data.project_name,
            "extracted_property_address": data.property_address,
            "extracted_developer": data.developer_name,
            "extracted_developer_hq_address": data.developer_hq_address,
            "extracted_units": data.total_residential_units,
            "extracted_status": data.current_status,
            "extracted_year": data.approval_or_initiation_year,
            "extracted_initiative_phrase": data.initiative_linkage_phrase,
            "reference_urls_count": len(data.reference_urls),
        },
        info_type="extraction_summary",
    )

    # Return final structured summary
    return evaluator.get_summary()