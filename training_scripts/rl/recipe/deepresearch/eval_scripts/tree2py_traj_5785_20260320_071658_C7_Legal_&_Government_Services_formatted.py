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
TASK_ID = "mecklenburg_county_tax_office_info"
TASK_DESCRIPTION = """
You are registering a new limited liability company (LLC) in Mecklenburg County, North Carolina, and need to understand the property tax obligations and procedures for your business. Gather comprehensive information about the Mecklenburg County Tax Administration Office (also known as the Office of the Tax Collector) to ensure you can properly manage your business property tax responsibilities.

Provide the following information:

1. The complete physical street address of the Tax Administration office where in-person services are provided
2. The regular business hours when the office is open to the public (including days of the week and specific times)
3. The primary phone number to contact the office
4. The official website URL for the Mecklenburg County Tax Administration
5. The annual due date for property taxes in Mecklenburg County
6. The date after which unpaid property taxes become delinquent and begin accruing interest charges
7. The available methods for paying property taxes (such as online, in-person, by mail, or by phone)
8. Whether property taxes can be paid online through the official website
9. The mailing address for sending tax payments or correspondence (if different from the physical address)
10. An official email address for contacting the Tax Administration office
11. The name and title of the current Tax Collector or Director of Tax Administration for Mecklenburg County
12. Whether appointments are required, recommended, or if walk-ins are accepted for in-person services
13. Information about which holidays the office is closed (either a reference to their holiday schedule or specific holidays listed)

For each piece of information, provide a reference URL from the official Mecklenburg County government website or the Tax Administration office website that supports your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TaxOfficeExtraction(BaseModel):
    # 1. Physical address
    physical_address: Optional[str] = None
    physical_address_sources: List[str] = Field(default_factory=list)

    # 2. Office hours
    office_hours: Optional[str] = None
    office_hours_sources: List[str] = Field(default_factory=list)

    # 3. Primary phone number
    primary_phone: Optional[str] = None
    primary_phone_sources: List[str] = Field(default_factory=list)

    # 4. Official website URL
    official_website_url: Optional[str] = None
    official_website_sources: List[str] = Field(default_factory=list)

    # 5. Property tax due date
    property_tax_due_date: Optional[str] = None
    property_tax_due_date_sources: List[str] = Field(default_factory=list)

    # 6. Property tax delinquent date
    property_tax_delinquent_date: Optional[str] = None
    property_tax_delinquent_date_sources: List[str] = Field(default_factory=list)

    # 7. Tax payment methods
    tax_payment_methods: List[str] = Field(default_factory=list)
    tax_payment_methods_sources: List[str] = Field(default_factory=list)

    # 8. Online payment availability ("yes" / "no" / "unknown")
    online_payment_availability: Optional[str] = None
    online_payment_availability_sources: List[str] = Field(default_factory=list)

    # 9. Mailing address
    mailing_address: Optional[str] = None
    mailing_address_sources: List[str] = Field(default_factory=list)

    # 10. Email contact
    email_contact: Optional[str] = None
    email_contact_sources: List[str] = Field(default_factory=list)

    # 11. Department leadership (name/title)
    department_leadership_name: Optional[str] = None
    department_leadership_title: Optional[str] = None
    department_leadership_sources: List[str] = Field(default_factory=list)

    # 12. Appointment requirements
    appointment_requirements: Optional[str] = None
    appointment_requirements_sources: List[str] = Field(default_factory=list)

    # 13. Holiday closure info
    holiday_closure_info: Optional[str] = None
    holiday_closure_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tax_office_info() -> str:
    return """
    Extract the following information exactly as presented in the provided answer text. For each item, also extract the list of supporting URLs that the answer cites for that specific item (if any). Do not invent or infer information not present in the answer.

    Fields to extract:
    1) physical_address: string (complete physical street address for in-person services)
    2) physical_address_sources: array of URLs supporting the physical address

    3) office_hours: string (regular public business hours, including days/times)
    4) office_hours_sources: array of URLs supporting the office hours

    5) primary_phone: string (main phone number)
    6) primary_phone_sources: array of URLs supporting the phone number

    7) official_website_url: string (the official Mecklenburg County Tax Administration website URL)
    8) official_website_sources: array of URLs supporting that this is the official website

    9) property_tax_due_date: string (the annual date when property taxes are due)
    10) property_tax_due_date_sources: array of URLs supporting the due date

    11) property_tax_delinquent_date: string (the date when unpaid taxes become delinquent and start accruing interest)
    12) property_tax_delinquent_date_sources: array of URLs supporting the delinquent date

    13) tax_payment_methods: array of strings (e.g., "online", "in-person", "by mail", "by phone", etc.)
    14) tax_payment_methods_sources: array of URLs supporting the listed methods

    15) online_payment_availability: string, one of ["yes", "no", "unknown"] (whether property taxes can be paid online)
    16) online_payment_availability_sources: array of URLs supporting this

    17) mailing_address: string (mailing address for payments/correspondence, if different)
    18) mailing_address_sources: array of URLs supporting the mailing address

    19) email_contact: string (official email address for contacting the office)
    20) email_contact_sources: array of URLs supporting the email contact

    21) department_leadership_name: string (name of the current Tax Collector or Director of Tax Administration)
    22) department_leadership_title: string (the corresponding title)
    23) department_leadership_sources: array of URLs supporting the leadership info

    24) appointment_requirements: string (e.g., "appointments required", "appointments recommended", "walk-ins accepted", etc.)
    25) appointment_requirements_sources: array of URLs supporting this

    26) holiday_closure_info: string (either a summary of specific holidays or a reference to the holiday schedule)
    27) holiday_closure_sources: array of URLs supporting the holiday closure info

    URL extraction rules:
    - Only include URLs explicitly present in the answer text for each item.
    - Accept plain URLs or markdown links; extract the actual URL.
    - If a URL is missing a protocol, prepend "http://".
    - If the answer provides no URL for a particular item, return an empty array for that item's sources.

    If any field's value is missing in the answer, set it to null (or [] for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s is not None and isinstance(s, str) and s.strip() != "")


def _unique_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        us = u.strip()
        if not us:
            continue
        if us not in seen:
            seen.add(us)
            out.append(us)
    return out


def _bool_from_str(s: Optional[str]) -> Optional[bool]:
    if s is None:
        return None
    val = s.strip().lower()
    if val in {"yes", "true", "y", "t"}:
        return True
    if val in {"no", "false", "n", "f"}:
        return False
    return None


def _combine_sources(*url_lists: List[str]) -> List[str]:
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u not in combined:
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_item_string_field(
    evaluator: Evaluator,
    root,
    *,
    node_id: str,
    desc: str,
    value: Optional[str],
    sources: List[str],
    critical_item: bool,
    claim_template: str,
    add_ins: str,
) -> None:
    """Generic builder for string fields with existence + source support verification."""
    item_node = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=root,
        critical=critical_item
    )

    exist_ok = _non_empty_str(value) and len(_unique_urls(sources)) > 0
    evaluator.add_custom_node(
        result=exist_ok,
        id=f"{node_id}_exists",
        desc=f"{desc} - value present and sources provided",
        parent=item_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - supported by cited source(s)",
        parent=item_node,
        critical=True  # Gate within the item
    )

    claim = claim_template.format(value=value or "")
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_unique_urls(sources),
        additional_instruction=add_ins
    )


async def build_official_website(
    evaluator: Evaluator,
    root,
    *,
    url_value: Optional[str],
    sources: List[str],
    critical_item: bool,
) -> None:
    node_id = "official_website_url"
    desc = "The official website URL for Mecklenburg County Tax Administration"
    # Treat the URL itself as a valid evidence if also provided as value
    merged_sources = _unique_urls(_combine_sources([url_value] if _non_empty_str(url_value) else [], sources))

    item_node = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=root,
        critical=critical_item
    )

    exist_ok = _non_empty_str(url_value) and len(merged_sources) > 0
    evaluator.add_custom_node(
        result=exist_ok,
        id=f"{node_id}_exists",
        desc=f"{desc} - URL present and source(s) provided",
        parent=item_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - confirmed as official",
        parent=item_node,
        critical=True
    )

    claim = f"The URL '{url_value or ''}' is the official website for the Mecklenburg County Tax Administration (Office of the Tax Collector)."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=merged_sources,
        additional_instruction="Verify that the page represents the Mecklenburg County Tax Administration (or Office of the Tax Collector) as an official site or landing page."
    )


async def build_payment_methods(
    evaluator: Evaluator,
    root,
    *,
    methods: List[str],
    sources: List[str],
    critical_item: bool,
) -> None:
    node_id = "tax_payment_methods"
    desc = "Accepted methods for paying property taxes (online, in-person, by mail, by phone, etc.)"

    item_node = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=root,
        critical=critical_item
    )

    exist_ok = (isinstance(methods, list) and len(methods) > 0) and (len(_unique_urls(sources)) > 0)
    evaluator.add_custom_node(
        result=exist_ok,
        id=f"{node_id}_exists",
        desc=f"{desc} - methods listed and sources provided",
        parent=item_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - supported by cited source(s)",
        parent=item_node,
        critical=True if critical_item else False
    )

    methods_str = ", ".join(methods) if methods else ""
    claim = f"The available payment methods for Mecklenburg County property taxes include: {methods_str}. It is acceptable if there are additional methods not listed here; at minimum, these listed methods should be supported by the cited sources."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_unique_urls(sources),
        additional_instruction="Confirm that the cited page(s) explicitly list or clearly imply each of the methods included in the statement. Minor wording differences are acceptable."
    )


async def build_online_payment_availability(
    evaluator: Evaluator,
    root,
    *,
    availability_str: Optional[str],
    sources: List[str],
    critical_item: bool,
) -> None:
    node_id = "online_payment_availability"
    desc = "Whether property taxes can be paid online through the official website"

    item_node = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=root,
        critical=critical_item
    )

    bool_val = _bool_from_str(availability_str)
    exist_ok = (bool_val is not None) and (len(_unique_urls(sources)) > 0)
    evaluator.add_custom_node(
        result=exist_ok,
        id=f"{node_id}_exists",
        desc=f"{desc} - value present and sources provided",
        parent=item_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - supported by cited source(s)",
        parent=item_node,
        critical=True if critical_item else False
    )

    if bool_val is True:
        claim = "Mecklenburg County property taxes can be paid online through the official website."
    elif bool_val is False:
        claim = "Mecklenburg County property taxes cannot be paid online through the official website."
    else:
        # Fallback: neutral phrasing if unknown; this will likely fail if existence check fails
        claim = "Information about whether Mecklenburg County property taxes can be paid online is explicitly stated on the cited page(s)."

    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_unique_urls(sources),
        additional_instruction="Verify the statement exactly as phrased (can or cannot). Focus on explicit evidence from the cited page(s)."
    )


async def build_department_leadership(
    evaluator: Evaluator,
    root,
    *,
    name: Optional[str],
    title: Optional[str],
    sources: List[str],
    critical_item: bool,
) -> None:
    node_id = "department_leadership"
    desc = "Current Tax Collector or Director of Tax Administration (name and title)"

    item_node = evaluator.add_parallel(
        id=node_id,
        desc=desc,
        parent=root,
        critical=critical_item
    )

    exist_ok = _non_empty_str(name) and _non_empty_str(title) and len(_unique_urls(sources)) > 0
    evaluator.add_custom_node(
        result=exist_ok,
        id=f"{node_id}_exists",
        desc=f"{desc} - name/title present and sources provided",
        parent=item_node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{desc} - supported by cited source(s)",
        parent=item_node,
        critical=True if critical_item else False
    )

    claim = f"The current Mecklenburg County Tax Administration leader is {name or ''}, with the title {title or ''} (e.g., Tax Collector or Director of Tax Administration)."
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=_unique_urls(sources),
        additional_instruction="Confirm that the cited page(s) list the person's name and the stated title, or clearly indicate their leadership role for the Mecklenburg County Tax Administration (Office of the Tax Collector)."
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
    Evaluate an answer for the Mecklenburg County Tax Administration information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks
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

    # Extract structured information from the answer
    extracted: TaxOfficeExtraction = await evaluator.extract(
        prompt=prompt_extract_tax_office_info(),
        template_class=TaxOfficeExtraction,
        extraction_name="mecklenburg_tax_office_info"
    )

    # Build verification nodes per rubric
    # Critical items per rubric JSON
    critical_map = {
        "office_physical_address": True,
        "office_hours": True,
        "primary_phone_number": True,
        "official_website_url": True,
        "property_tax_due_date": True,
        "property_tax_delinquent_date": True,
        # Non-critical below
        "tax_payment_methods": False,
        "online_payment_availability": False,
        "mailing_address": False,
        "email_contact": False,
        "department_leadership": False,
        "appointment_requirements": False,
        "holiday_closure_information": False,
    }

    # 1) Office Physical Address
    await build_item_string_field(
        evaluator,
        root,
        node_id="office_physical_address",
        desc="The complete physical street address of the Mecklenburg County Tax Administration office (in-person services)",
        value=extracted.physical_address,
        sources=_unique_urls(extracted.physical_address_sources),
        critical_item=critical_map["office_physical_address"],
        claim_template="The physical street address for in-person services at the Mecklenburg County Tax Administration is: {value}.",
        add_ins="Confirm the page shows the precise street address for the Tax Administration (or Office of the Tax Collector). Minor formatting differences are acceptable."
    )

    # 2) Office Hours
    await build_item_string_field(
        evaluator,
        root,
        node_id="office_hours",
        desc="Regular public business hours (days and times)",
        value=extracted.office_hours,
        sources=_unique_urls(extracted.office_hours_sources),
        critical_item=critical_map["office_hours"],
        claim_template="The regular public business hours for the Mecklenburg County Tax Administration are: {value}.",
        add_ins="Verify that the page lists public-facing office hours (include days and specific times). Formatting variations are acceptable."
    )

    # 3) Primary Phone Number
    await build_item_string_field(
        evaluator,
        root,
        node_id="primary_phone_number",
        desc="Primary phone number to contact the office",
        value=extracted.primary_phone,
        sources=_unique_urls(extracted.primary_phone_sources),
        critical_item=critical_map["primary_phone_number"],
        claim_template="The primary phone number for the Mecklenburg County Tax Administration is: {value}.",
        add_ins="Check that the number is presented as a main/public contact number for the office. Allow different formatting (dashes, spaces, parentheses)."
    )

    # 4) Official Website URL
    await build_official_website(
        evaluator,
        root,
        url_value=extracted.official_website_url,
        sources=_unique_urls(extracted.official_website_sources),
        critical_item=critical_map["official_website_url"]
    )

    # 5) Property Tax Due Date
    await build_item_string_field(
        evaluator,
        root,
        node_id="property_tax_due_date",
        desc="Annual due date for property taxes in Mecklenburg County",
        value=extracted.property_tax_due_date,
        sources=_unique_urls(extracted.property_tax_due_date_sources),
        critical_item=critical_map["property_tax_due_date"],
        claim_template="In Mecklenburg County, property taxes are due on: {value} (annually).",
        add_ins="Verify that the page explicitly mentions the annual due date for property taxes."
    )

    # 6) Property Tax Delinquent Date
    await build_item_string_field(
        evaluator,
        root,
        node_id="property_tax_delinquent_date",
        desc="Date when unpaid property taxes become delinquent and interest begins",
        value=extracted.property_tax_delinquent_date,
        sources=_unique_urls(extracted.property_tax_delinquent_date_sources),
        critical_item=critical_map["property_tax_delinquent_date"],
        claim_template="Unpaid Mecklenburg County property taxes become delinquent and begin accruing interest on: {value}.",
        add_ins="Verify that the page explicitly mentions the date delinquency/interest starts for unpaid property taxes."
    )

    # 7) Tax Payment Methods (non-critical)
    await build_payment_methods(
        evaluator,
        root,
        methods=extracted.tax_payment_methods or [],
        sources=_unique_urls(extracted.tax_payment_methods_sources),
        critical_item=critical_map["tax_payment_methods"]
    )

    # 8) Online Payment Availability (non-critical)
    await build_online_payment_availability(
        evaluator,
        root,
        availability_str=extracted.online_payment_availability,
        sources=_unique_urls(extracted.online_payment_availability_sources),
        critical_item=critical_map["online_payment_availability"]
    )

    # 9) Mailing Address (non-critical)
    await build_item_string_field(
        evaluator,
        root,
        node_id="mailing_address",
        desc="Mailing address for tax payments or correspondence",
        value=extracted.mailing_address,
        sources=_unique_urls(extracted.mailing_address_sources),
        critical_item=critical_map["mailing_address"],
        claim_template="The mailing address for Mecklenburg County tax payments or correspondence is: {value}.",
        add_ins="Verify that the cited page(s) explicitly present a mailing address for tax payments or correspondence."
    )

    # 10) Email Contact (non-critical)
    await build_item_string_field(
        evaluator,
        root,
        node_id="email_contact",
        desc="Official email address for contacting the Tax Administration office",
        value=extracted.email_contact,
        sources=_unique_urls(extracted.email_contact_sources),
        critical_item=critical_map["email_contact"],
        claim_template="The official email address for contacting the Mecklenburg County Tax Administration is: {value}.",
        add_ins="Verify the email appears on the cited page(s) as an official contact for the Tax Administration. Case-insensitive match is acceptable."
    )

    # 11) Department Leadership (non-critical)
    await build_department_leadership(
        evaluator,
        root,
        name=extracted.department_leadership_name,
        title=extracted.department_leadership_title,
        sources=_unique_urls(extracted.department_leadership_sources),
        critical_item=critical_map["department_leadership"]
    )

    # 12) Appointment Requirements (non-critical)
    await build_item_string_field(
        evaluator,
        root,
        node_id="appointment_requirements",
        desc="Whether appointments are required/recommended or if walk-ins are accepted",
        value=extracted.appointment_requirements,
        sources=_unique_urls(extracted.appointment_requirements_sources),
        critical_item=critical_map["appointment_requirements"],
        claim_template="For in-person services at the Mecklenburg County Tax Administration, the policy is: {value}.",
        add_ins="Confirm that the cited page(s) explicitly state whether appointments are required, recommended, or whether walk-ins are accepted."
    )

    # 13) Holiday Closure Information (non-critical)
    await build_item_string_field(
        evaluator,
        root,
        node_id="holiday_closure_information",
        desc="Holiday closure information (specific holidays or reference to schedule)",
        value=extracted.holiday_closure_info,
        sources=_unique_urls(extracted.holiday_closure_sources),
        critical_item=critical_map["holiday_closure_information"],
        claim_template="The Mecklenburg County Tax Administration office holiday closure information is as follows: {value}.",
        add_ins="Verify that the cited page(s) provide a holiday schedule reference or list specific holidays when the office is closed."
    )

    # Return evaluation summary
    return evaluator.get_summary()