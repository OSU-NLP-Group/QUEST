import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_llc_compliance_2026"
TASK_DESCRIPTION = (
    "I am planning to form a new limited liability company (LLC) in Louisiana in 2026 and need to create a comprehensive "
    "compliance budget and timeline for the first year of operation. Please provide detailed information about all state-level "
    "requirements, including: (1) Initial Formation Costs: What are the mandatory filing fees for Articles of Organization with "
    "the Louisiana Secretary of State? If filing online, are there any additional processing fees? (2) Annual Report Obligations: "
    "What is the filing fee for the annual report, when is it due, and what is the filing window (how far in advance can it be filed)? "
    "(3) Registered Agent Requirements: What are the legal requirements for a registered agent in Louisiana, specifically regarding "
    "address type and availability? (4) Optional Services: What is the cost and validity period for reserving a business name, and "
    "what are the fees for expedited processing options (both 24-hour and priority in-person)? For each category, please include the "
    "specific dollar amounts, timeframes, and reference URLs from official Louisiana government sources to support the information."
)


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_official_la_url(url: str) -> bool:
    """
    Determine whether a URL belongs to an official Louisiana government domain.

    Acceptable examples include:
    - *.la.gov
    - louisiana.gov
    - sos.la.gov
    - geauxbiz.sos.la.gov

    Returns True if host contains 'la.gov' or 'louisiana.gov'.
    """
    try:
        host = urlparse(url).netloc.lower()
        return ("la.gov" in host) or ("louisiana.gov" in host)
    except Exception:
        return False


def any_official_url(urls: List[str]) -> bool:
    """Return True if at least one URL is identified as an official Louisiana government URL."""
    return any(is_official_la_url(u) for u in urls)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FormationInfo(BaseModel):
    articles_filing_fee: Optional[str] = None  # e.g., "$100"
    online_processing_fee: Optional[str] = None  # e.g., "$5"
    formation_urls: List[str] = Field(default_factory=list)


class AnnualReportInfo(BaseModel):
    annual_report_fee: Optional[str] = None  # e.g., "$30"
    filing_deadline_desc: Optional[str] = None  # e.g., "anniversary date of formation"
    filing_window_desc: Optional[str] = None  # e.g., "within 30 days prior to due date"
    annual_report_urls: List[str] = Field(default_factory=list)


class RegisteredAgentInfo(BaseModel):
    physical_address_requirement_text: Optional[str] = None  # e.g., "must have a physical street address in Louisiana"
    po_box_restriction_text: Optional[str] = None  # e.g., "P.O. boxes are not acceptable"
    business_hours_availability_text: Optional[str] = None  # e.g., "available during normal business hours"
    agent_urls: List[str] = Field(default_factory=list)


class NameReservationInfo(BaseModel):
    reservation_fee: Optional[str] = None  # e.g., "$25"
    reservation_duration: Optional[str] = None  # e.g., "120 days"
    reservation_urls: List[str] = Field(default_factory=list)


class ExpeditedProcessingInfo(BaseModel):
    expedited_24_hour_fee: Optional[str] = None  # e.g., "$30"
    expedited_24_hour_label: Optional[str] = None  # e.g., "24-hour expedited"
    priority_in_person_fee: Optional[str] = None  # e.g., "$50"
    priority_in_person_label: Optional[str] = None  # e.g., "priority in-person / while-you-wait"
    processing_urls: List[str] = Field(default_factory=list)


class LouisianaLLCExtraction(BaseModel):
    formation: Optional[FormationInfo] = None
    annual_report: Optional[AnnualReportInfo] = None
    registered_agent: Optional[RegisteredAgentInfo] = None
    name_reservation: Optional[NameReservationInfo] = None
    expedited_processing: Optional[ExpeditedProcessingInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_louisiana_llc() -> str:
    return """
    Extract the Louisiana LLC first-year compliance information from the answer.
    Return a JSON with the following structure and fields strictly based on what the answer explicitly states:

    {
      "formation": {
        "articles_filing_fee": string | null,                 // e.g., "$100"
        "online_processing_fee": string | null,               // e.g., "$5" (credit card/online processing fee)
        "formation_urls": string[]                            // URLs cited in the answer for formation fees (extract all URLs exactly as they appear)
      },
      "annual_report": {
        "annual_report_fee": string | null,                   // e.g., "$30"
        "filing_deadline_desc": string | null,                // e.g., "due on the anniversary date of formation"
        "filing_window_desc": string | null,                  // e.g., "can only be filed within 30 days prior to due date"
        "annual_report_urls": string[]                        // URLs cited in the answer for annual report (extract all URLs exactly as they appear)
      },
      "registered_agent": {
        "physical_address_requirement_text": string | null,   // e.g., "must have a physical street address in Louisiana"
        "po_box_restriction_text": string | null,             // e.g., "P.O. boxes are not acceptable"
        "business_hours_availability_text": string | null,    // e.g., "must be available during normal business hours"
        "agent_urls": string[]                                // URLs cited in the answer for registered agent requirements
      },
      "name_reservation": {
        "reservation_fee": string | null,                     // e.g., "$25"
        "reservation_duration": string | null,                // e.g., "120 days"
        "reservation_urls": string[]                          // URLs cited for name reservation
      },
      "expedited_processing": {
        "expedited_24_hour_fee": string | null,               // e.g., "$30"
        "expedited_24_hour_label": string | null,             // e.g., "24-hour expedited"
        "priority_in_person_fee": string | null,              // e.g., "$50"
        "priority_in_person_label": string | null,            // e.g., "priority in-person" or "while-you-wait"
        "processing_urls": string[]                           // URLs cited for expedited processing options and fees
      }
    }

    Rules:
    - Extract only what the answer explicitly states. Do not infer or invent.
    - For amounts, keep them as strings exactly as shown (including $ and any text).
    - For timeframes/descriptions, extract the phrasing present in the answer.
    - For each URLs list, extract all URLs that the answer associates with the corresponding category, in any format (plain or markdown).
    - If a field is not present in the answer, set it to null. If no URLs are present, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_formation_costs(evaluator: Evaluator, parent_node, info: Optional[FormationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Formation_Costs",
        desc="Mandatory costs for initial LLC formation filing with the Louisiana Secretary of State, including official sources",
        parent=parent_node,
        critical=True
    )

    urls = (info.formation_urls if info and info.formation_urls else [])

    # Articles filing fee = $100
    leaf_articles_fee = evaluator.add_leaf(
        id="Articles_Filing_Fee",
        desc="Identifies the base state filing fee for Articles of Organization as $100",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The filing fee for Louisiana LLC Articles of Organization is $100.",
        node=leaf_articles_fee,
        sources=urls,
        additional_instruction="Verify on official Louisiana Secretary of State / GeauxBiz pages whether the Articles of Organization fee is $100."
    )

    # Online processing fee = $5
    leaf_online_fee = evaluator.add_leaf(
        id="Online_Processing_Fee",
        desc="Identifies that online filing has an additional $5 credit card processing fee (total $105)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Online filing for a Louisiana LLC includes an additional $5 credit card/processing fee.",
        node=leaf_online_fee,
        sources=urls,
        additional_instruction="Confirm the presence of a $5 additional credit card/processing/transaction fee for online filings."
    )

    # Reference URL presence (official)
    evaluator.add_custom_node(
        result=any_official_url(urls),
        id="Formation_Reference_URL",
        desc="Provides at least one official Louisiana government reference URL supporting the formation fee information",
        parent=node,
        critical=True
    )


async def verify_annual_report(evaluator: Evaluator, parent_node, info: Optional[AnnualReportInfo]) -> None:
    node = evaluator.add_parallel(
        id="Annual_Report_Obligations",
        desc="Annual report filing requirements including fee, due date, filing window, and official sources",
        parent=parent_node,
        critical=True
    )

    urls = (info.annual_report_urls if info and info.annual_report_urls else [])

    # Fee = $30
    leaf_fee = evaluator.add_leaf(
        id="Annual_Report_Fee",
        desc="Identifies the annual report filing fee as $30",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Louisiana LLC annual report filing fee is $30.",
        node=leaf_fee,
        sources=urls,
        additional_instruction="Verify on official Louisiana SOS / GeauxBiz pages that the annual report fee is $30."
    )

    # Deadline: anniversary date
    leaf_deadline = evaluator.add_leaf(
        id="Filing_Deadline",
        desc="States the annual report is due on the anniversary date of the LLC's formation",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The annual report for a Louisiana LLC is due on the anniversary date of the LLC's formation.",
        node=leaf_deadline,
        sources=urls,
        additional_instruction="Confirm wording indicating the annual report due date aligns with the entity’s anniversary date."
    )

    # Filing window: within 30 days prior
    leaf_window = evaluator.add_leaf(
        id="Filing_Window",
        desc="States annual reports can only be filed within 30 days prior to the due date",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Louisiana LLC annual reports can only be filed within 30 days prior to the due date.",
        node=leaf_window,
        sources=urls,
        additional_instruction="Verify the filing window or earliest filing timeframe is limited to 30 days prior to the due date."
    )

    # Reference URL presence (official)
    evaluator.add_custom_node(
        result=any_official_url(urls),
        id="Annual_Report_Reference_URL",
        desc="Provides at least one official Louisiana government reference URL supporting the annual report requirements",
        parent=node,
        critical=True
    )


async def verify_registered_agent(evaluator: Evaluator, parent_node, info: Optional[RegisteredAgentInfo]) -> None:
    node = evaluator.add_parallel(
        id="Registered_Agent_Requirements",
        desc="Registered agent legal requirements (address type and availability) plus official sources",
        parent=parent_node,
        critical=True
    )

    urls = (info.agent_urls if info and info.agent_urls else [])

    # Physical address requirement in LA
    leaf_physical = evaluator.add_leaf(
        id="Physical_Address_Requirement",
        desc="States the registered agent must have a physical street address in Louisiana",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A Louisiana registered agent must maintain a physical street address in Louisiana.",
        node=leaf_physical,
        sources=urls,
        additional_instruction="Confirm that the registered agent must have a physical street address in Louisiana (not virtual or solely mailing)."
    )

    # P.O. Box restriction
    leaf_pobox = evaluator.add_leaf(
        id="PO_Box_Restriction",
        desc="States that P.O. boxes are not acceptable as registered agent addresses",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="P.O. boxes are not acceptable as registered agent addresses in Louisiana.",
        node=leaf_pobox,
        sources=urls,
        additional_instruction="Verify explicit prohibition of P.O. boxes for registered agent address."
    )

    # Availability during business hours
    leaf_hours = evaluator.add_leaf(
        id="Business_Hours_Availability",
        desc="States the registered agent must be available during normal business hours",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The registered agent must be available during normal business hours to accept service of process.",
        node=leaf_hours,
        sources=urls,
        additional_instruction="Confirm language requiring availability during normal business hours."
    )

    # Reference URL presence (official)
    evaluator.add_custom_node(
        result=any_official_url(urls),
        id="Agent_Reference_URL",
        desc="Provides at least one official Louisiana government reference URL supporting registered agent requirements",
        parent=node,
        critical=True
    )


async def verify_name_reservation(evaluator: Evaluator, parent_node, info: Optional[NameReservationInfo]) -> None:
    node = evaluator.add_parallel(
        id="Name_Reservation_Information",
        desc="Optional name reservation cost and validity period, plus official sources",
        parent=parent_node,
        critical=True
    )

    urls = (info.reservation_urls if info and info.reservation_urls else [])

    # Fee = $25
    leaf_fee = evaluator.add_leaf(
        id="Reservation_Fee",
        desc="Identifies the name reservation fee as $25",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The Louisiana business name reservation fee is $25.",
        node=leaf_fee,
        sources=urls,
        additional_instruction="Verify the name reservation fee amount is $25 on official Louisiana sources."
    )

    # Duration = 120 days
    leaf_duration = evaluator.add_leaf(
        id="Reservation_Duration",
        desc="Identifies the name reservation validity period as 120 days",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A Louisiana business name reservation is valid for 120 days.",
        node=leaf_duration,
        sources=urls,
        additional_instruction="Confirm the validity period for a name reservation in Louisiana is 120 days."
    )

    # Reference URL presence (official)
    evaluator.add_custom_node(
        result=any_official_url(urls),
        id="Reservation_Reference_URL",
        desc="Provides at least one official Louisiana government reference URL supporting the name reservation details",
        parent=node,
        critical=True
    )


async def verify_expedited_processing(evaluator: Evaluator, parent_node, info: Optional[ExpeditedProcessingInfo]) -> None:
    node = evaluator.add_parallel(
        id="Expedited_Processing_Options",
        desc="Expedited processing options (24-hour and priority in-person) fees (and the option type as stated) plus official sources",
        parent=parent_node,
        critical=True
    )

    urls = (info.processing_urls if info and info.processing_urls else [])

    # 24-hour expedited = +$30
    leaf_24h = evaluator.add_leaf(
        id="Expedited_24_Hour",
        desc="Identifies the 24-hour expedited processing option fee as an additional $30 (and indicates it is the 24-hour option)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The 24-hour expedited processing option adds an additional $30 fee.",
        node=leaf_24h,
        sources=urls,
        additional_instruction="Verify that there is a 24-hour expedited option and that its additional fee is $30."
    )

    # Priority in-person (while-you-wait) = +$50
    leaf_priority = evaluator.add_leaf(
        id="Priority_In_Person",
        desc="Identifies the priority in-person (while-you-wait) option fee as an additional $50 (and indicates it is the priority in-person option)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Priority in-person (while-you-wait) processing adds an additional $50 fee.",
        node=leaf_priority,
        sources=urls,
        additional_instruction="Verify that a priority in-person while-you-wait option exists and that its additional fee is $50."
    )

    # Reference URL presence (official)
    evaluator.add_custom_node(
        result=any_official_url(urls),
        id="Processing_Reference_URL",
        desc="Provides at least one official Louisiana government reference URL supporting expedited processing options and fees",
        parent=node,
        critical=True
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
    Evaluate an answer for the Louisiana LLC first-year compliance task.

    Returns a structured summary containing extraction info and the verification tree with a final score.
    """
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_louisiana_llc(),
        template_class=LouisianaLLCExtraction,
        extraction_name="louisiana_llc_compliance_extraction"
    )

    # Ground truth expectations (for reference only; verification uses official sources)
    evaluator.add_ground_truth({
        "expected_values": {
            "formation_articles_fee": "$100",
            "online_processing_fee": "$5",
            "annual_report_fee": "$30",
            "annual_report_deadline": "anniversary date of formation",
            "annual_report_window": "30 days prior to due date",
            "registered_agent_physical_address": True,
            "registered_agent_no_po_box": True,
            "registered_agent_business_hours": True,
            "name_reservation_fee": "$25",
            "name_reservation_duration": "120 days",
            "expedited_24h_fee": "$30",
            "priority_in_person_fee": "$50"
        },
        "official_domain_criteria": "URL host contains 'la.gov' or 'louisiana.gov'"
    })

    # Build verification tree by categories (all critical)
    await verify_formation_costs(evaluator, root, extraction.formation)
    await verify_annual_report(evaluator, root, extraction.annual_report)
    await verify_registered_agent(evaluator, root, extraction.registered_agent)
    await verify_name_reservation(evaluator, root, extraction.name_reservation)
    await verify_expedited_processing(evaluator, root, extraction.expedited_processing)

    return evaluator.get_summary()