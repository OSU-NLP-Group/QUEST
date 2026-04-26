import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "hr_contacts_verification"
TASK_DESCRIPTION = (
    "I am researching employment opportunities at several educational institutions and need to contact their Human Resources departments directly. "
    "Please find the following contact information for each institution's HR or employment office from their official websites:\n\n"
    "1. Carroll County Public Schools, Maryland\n"
    "2. Forsyth County School District, Georgia\n"
    "3. Arlington Independent School District, Texas\n"
    "4. University of Wisconsin-Madison\n\n"
    "For each institution, provide:\n"
    "- The HR department's phone number\n"
    "- The HR department's email address\n"
    "- The physical address of the HR department/office\n\n"
    "Please ensure all information is sourced from the official institutional websites and include the reference URL where you found each piece of information."
)

# Official domain constraints for each institution
OFFICIAL_DOMAINS = {
    "Carroll County Public Schools, Maryland": "carrollk12.org",
    "Forsyth County School District, Georgia": "forsyth.k12.ga.us",
    "Arlington Independent School District, Texas": "aisd.net",
    "University of Wisconsin-Madison": "wisc.edu",
}

# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class InstitutionContact(BaseModel):
    phone: FieldWithSources = Field(default_factory=FieldWithSources)
    email: FieldWithSources = Field(default_factory=FieldWithSources)
    address: FieldWithSources = Field(default_factory=FieldWithSources)


class HRContactsExtraction(BaseModel):
    ccps_md: Optional[InstitutionContact] = None            # Carroll County Public Schools, Maryland
    fcs_ga: Optional[InstitutionContact] = None             # Forsyth County School District, Georgia
    aisd_tx: Optional[InstitutionContact] = None            # Arlington Independent School District, Texas
    uw_madison: Optional[InstitutionContact] = None         # University of Wisconsin-Madison


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_hr_contacts() -> str:
    return """
    Extract HR contact information for the following institutions exactly as presented in the answer text. For EACH institution, you must extract:
    - phone: { value, urls[] }  — 'value' is the phone number string; 'urls' are the explicit reference URLs listed in the answer where the phone number was found.
    - email: { value, urls[] }  — 'value' is the email address string; 'urls' are the explicit reference URLs listed in the answer where the email was found.
    - address: { value, urls[] }  — 'value' is the physical address string; 'urls' are the explicit reference URLs listed in the answer where the address was found.

    IMPORTANT:
    - Only include URLs that are explicitly present in the provided answer. Do not invent or infer URLs.
    - If a field is missing, set its 'value' to null and 'urls' to [].
    - If URLs are given via markdown links, extract the actual URL.
    - Return a single JSON object with the following keys, each mapping to an InstitutionContact object:
      {
        "ccps_md": { "phone": {...}, "email": {...}, "address": {...} },
        "fcs_ga": { "phone": {...}, "email": {...}, "address": {...} },
        "aisd_tx": { "phone": {...}, "email": {...}, "address": {...} },
        "uw_madison": { "phone": {...}, "email": {...}, "address": {...} }
      }

    Institution mapping:
    - ccps_md: Carroll County Public Schools, Maryland
    - fcs_ga: Forsyth County School District, Georgia
    - aisd_tx: Arlington Independent School District, Texas
    - uw_madison: University of Wisconsin-Madison

    Keep values exactly as shown in the answer (do not normalize formatting).
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _collect_all_urls(contact: Optional[InstitutionContact]) -> List[str]:
    urls: List[str] = []
    if not contact:
        return urls
    urls.extend(contact.phone.urls or [])
    urls.extend(contact.email.urls or [])
    urls.extend(contact.address.urls or [])
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def _verify_field_with_sources(
    evaluator: Evaluator,
    parent_node,
    institution_label: str,
    field_key: str,  # "Phone", "Email", "Address"
    field: FieldWithSources,
    critical: bool = False
) -> None:
    """
    Build a sequential node for a single field (phone/email/address):
    - Existence check (value present and at least one source URL)
    - Source-supported verification via verify_by_urls
    """
    node_id_prefix = institution_label.replace(" ", "_").replace(",", "").replace("-", "_")
    field_node = evaluator.add_sequential(
        id=f"{node_id_prefix}_{field_key}",
        desc=f"Provide the HR department {field_key.lower()} for {institution_label}",
        parent=parent_node,
        critical=critical
    )

    # Existence check for value and sources
    exists = (field.value is not None and field.value.strip() != "") and (len(field.urls) > 0)
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id_prefix}_{field_key}_exists",
        desc=f"{field_key} value and at least one reference URL are provided for {institution_label}",
        parent=field_node,
        critical=True
    )

    # Verification against provided sources
    verify_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_{field_key}_supported",
        desc=f"The {field_key.lower()} is supported by the cited official page(s)",
        parent=field_node,
        critical=True
    )

    claim_text = ""
    if field_key.lower() == "phone":
        claim_text = f"The HR department phone number for {institution_label} is {field.value}."
        add_ins = (
            "Verify that the cited page(s) explicitly show this phone number and that it refers to HR/Human Resources/Employee Services/Employment office. "
            "Allow minor formatting variations (spaces, parentheses, hyphens)."
        )
    elif field_key.lower() == "email":
        claim_text = f"The HR department email address for {institution_label} is {field.value}."
        add_ins = (
            "Verify that the cited page(s) explicitly show this email address and that it refers to HR/Human Resources/Employee Services/Employment office. "
            "Allow case-insensitive comparisons and minor formatting variations."
        )
    else:  # address
        claim_text = f"The physical address of the HR department for {institution_label} is {field.value}."
        add_ins = (
            "Verify that the cited page(s) explicitly show this address for HR/Human Resources/Employee Services/Employment office. "
            "Allow minor formatting differences and standard abbreviations."
        )

    await evaluator.verify(
        claim=claim_text,
        node=verify_leaf,
        sources=field.urls,
        additional_instruction=add_ins
    )


async def _verify_institution(
    evaluator: Evaluator,
    parent_node,
    institution_label: str,
    domain_required: str,
    contact: Optional[InstitutionContact],
    ids_prefix: str
) -> None:
    """
    Build verification subtree for one institution:
    - Reference URL checks (existence + domain compliance, critical)
    - Phone, Email, Address checks (non-critical, each gated by existence and supported claims)
    """
    inst_node = evaluator.add_parallel(
        id=ids_prefix,
        desc=f"Verify HR contact information for {institution_label}",
        parent=parent_node,
        critical=False
    )

    # Gather and verify reference URLs
    all_urls = _collect_all_urls(contact)
    ref_node = evaluator.add_sequential(
        id=f"{ids_prefix}_Reference_URLs",
        desc=f"Provide reference URL(s) from {institution_label}'s official website ({domain_required} domain) where the HR contact information was found",
        parent=inst_node,
        critical=True
    )

    # Existence of at least one URL
    evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"{ids_prefix}_RefURLs_exist",
        desc=f"At least one reference URL is provided for {institution_label}",
        parent=ref_node,
        critical=True
    )

    # Domain compliance check for all URLs
    domain_leaf = evaluator.add_leaf(
        id=f"{ids_prefix}_RefURLs_domain_ok",
        desc=f"All provided reference URLs are under the official domain {domain_required} (including subdomains)",
        parent=ref_node,
        critical=True
    )

    domain_claim = (
        f"All of the following URLs are under domain '{domain_required}' or its subdomains: {all_urls}."
    )
    await evaluator.verify(
        claim=domain_claim,
        node=domain_leaf,
        additional_instruction=(
            "Judge purely by the URL strings: Accept subdomains (e.g., hr.{domain}), 'www.' prefixes, "
            "and different paths. Do not rely on page content; this is strictly a domain check."
        )
    )

    # Field verifications (non-critical each)
    if contact is None:
        # Add placeholders to maintain structure even if nothing was extracted
        contact = InstitutionContact()

    await _verify_field_with_sources(
        evaluator=evaluator,
        parent_node=inst_node,
        institution_label=institution_label,
        field_key="Phone",
        field=contact.phone,
        critical=False
    )

    await _verify_field_with_sources(
        evaluator=evaluator,
        parent_node=inst_node,
        institution_label=institution_label,
        field_key="Email",
        field=contact.email,
        critical=False
    )

    await _verify_field_with_sources(
        evaluator=evaluator,
        parent_node=inst_node,
        institution_label=institution_label,
        field_key="Address",
        field=contact.address,
        critical=False
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Entry point: Build the verification tree, perform extraction, and run all checks.
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
        default_model=model
    )

    # Add top-level node to reflect rubric root
    hr_root = evaluator.add_parallel(
        id="HR_Contact_Information",
        desc="Verify that HR contact information (phone, email, address) is provided for all four educational institutions with proper reference URLs from official websites",
        parent=root,
        critical=False
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_hr_contacts(),
        template_class=HRContactsExtraction,
        extraction_name="hr_contacts_extraction"
    )

    # Institution-specific verifications
    await _verify_institution(
        evaluator=evaluator,
        parent_node=hr_root,
        institution_label="Carroll County Public Schools, Maryland",
        domain_required=OFFICIAL_DOMAINS["Carroll County Public Schools, Maryland"],
        contact=extracted.ccps_md,
        ids_prefix="Carroll_County_Public_Schools_MD"
    )

    await _verify_institution(
        evaluator=evaluator,
        parent_node=hr_root,
        institution_label="Forsyth County School District, Georgia",
        domain_required=OFFICIAL_DOMAINS["Forsyth County School District, Georgia"],
        contact=extracted.fcs_ga,
        ids_prefix="Forsyth_County_Schools_GA"
    )

    await _verify_institution(
        evaluator=evaluator,
        parent_node=hr_root,
        institution_label="Arlington Independent School District, Texas",
        domain_required=OFFICIAL_DOMAINS["Arlington Independent School District, Texas"],
        contact=extracted.aisd_tx,
        ids_prefix="Arlington_ISD_TX"
    )

    await _verify_institution(
        evaluator=evaluator,
        parent_node=hr_root,
        institution_label="University of Wisconsin-Madison",
        domain_required=OFFICIAL_DOMAINS["University of Wisconsin-Madison"],
        contact=extracted.uw_madison,
        ids_prefix="University_of_Wisconsin_Madison"
    )

    # Return structured summary
    return evaluator.get_summary()