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
TASK_ID = "in_apra_4agencies"
TASK_DESCRIPTION = (
    "Document the public records request procedures under Indiana's Access to Public Records Act (APRA) for the following four "
    "Indiana state agencies: Department of Revenue (DOR), Bureau of Motor Vehicles (BMV), Professional Licensing Agency (PLA), "
    "and Department of Natural Resources (DNR). For each agency, provide: (1) The official contact information (email address or "
    "mailing address) for submitting APRA requests; (2) The stated response timeframes for APRA requests, which must comply with "
    "Indiana Code 5-14-3 statutory requirements (24 hours for in-person or telephone requests; 7 calendar days for requests "
    "submitted by mail, fax, or email); (3) The fee schedule for copying public records, which for state executive branch agencies "
    "must follow Department of Administration standards ($0.10 per page for black-and-white copies, $0.25 per page for color copies); "
    "(4) Whether the agency provides an online or downloadable APRA request form. All information must be sourced from official agency "
    "websites or state government resources and include reference URLs."
)

AGENCY_META = {
    "dor": {
        "agg_id": "agency_1_dor",
        "agg_desc": "Department of Revenue APRA procedures documentation",
        "full_name": "Indiana Department of Revenue",
        "leaf_ids": {
            "contact": "dor_contact",
            "response": "dor_response_time",
            "fee": "dor_fee_schedule",
            "form": "dor_request_form",
        },
        "leaf_desc": {
            "contact": "Provide the official APRA request contact information (email or mailing address) for the Department of Revenue",
            "response": "Document the response timeframes stated by the Department of Revenue for APRA requests (must align with IC 5-14-3 statutory requirements: 24 hours for in-person/phone, 7 days for mail/fax/email)",
            "fee": "Provide the fee schedule for copying public records from the Department of Revenue (must comply with state standards: $0.10 per page for black-and-white, $0.25 for color)",
            "form": "Indicate whether the Department of Revenue provides an online or downloadable APRA request form",
        },
    },
    "bmv": {
        "agg_id": "agency_2_bmv",
        "agg_desc": "Bureau of Motor Vehicles APRA procedures documentation",
        "full_name": "Indiana Bureau of Motor Vehicles",
        "leaf_ids": {
            "contact": "bmv_contact",
            "response": "bmv_response_time",
            "fee": "bmv_fee_schedule",
            "form": "bmv_request_form",
        },
        "leaf_desc": {
            "contact": "Provide the official APRA request contact information (email or mailing address) for the Bureau of Motor Vehicles",
            "response": "Document the response timeframes stated by the Bureau of Motor Vehicles for APRA requests (must align with IC 5-14-3 statutory requirements: 24 hours for in-person/phone, 7 days for mail/fax/email)",
            "fee": "Provide the fee schedule for copying public records from the Bureau of Motor Vehicles (must comply with state standards: $0.10 per page for black-and-white, $0.25 for color)",
            "form": "Indicate whether the Bureau of Motor Vehicles provides an online or downloadable APRA request form",
        },
    },
    "pla": {
        "agg_id": "agency_3_pla",
        "agg_desc": "Professional Licensing Agency APRA procedures documentation",
        "full_name": "Indiana Professional Licensing Agency",
        "leaf_ids": {
            "contact": "pla_contact",
            "response": "pla_response_time",
            "fee": "pla_fee_schedule",
            "form": "pla_request_form",
        },
        "leaf_desc": {
            "contact": "Provide the official APRA request contact information (email or mailing address) for the Professional Licensing Agency",
            "response": "Document the response timeframes stated by the Professional Licensing Agency for APRA requests (must align with IC 5-14-3 statutory requirements: 24 hours for in-person/phone, 7 days for mail/fax/email)",
            "fee": "Provide the fee schedule for copying public records from the Professional Licensing Agency (must comply with state standards: $0.10 per page for black-and-white, $0.25 for color)",
            "form": "Indicate whether the Professional Licensing Agency provides an online or downloadable APRA request form",
        },
    },
    "dnr": {
        "agg_id": "agency_4_dnr",
        "agg_desc": "Department of Natural Resources APRA procedures documentation",
        "full_name": "Indiana Department of Natural Resources",
        "leaf_ids": {
            "contact": "dnr_contact",
            "response": "dnr_response_time",
            "fee": "dnr_fee_schedule",
            "form": "dnr_request_form",
        },
        "leaf_desc": {
            "contact": "Provide the official APRA request contact information (email or mailing address) for the Department of Natural Resources",
            "response": "Document the response timeframes stated by the Department of Natural Resources for APRA requests (must align with IC 5-14-3 statutory requirements: 24 hours for in-person/phone, 7 days for mail/fax/email)",
            "fee": "Provide the fee schedule for copying public records from the Department of Natural Resources (must comply with state standards: $0.10 per page for black-and-white, $0.25 for color)",
            "form": "Indicate whether the Department of Natural Resources provides an online or downloadable APRA request form",
        },
    },
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AgencyAPRAInfo(BaseModel):
    contact_info: Optional[str] = None
    contact_urls: List[str] = Field(default_factory=list)

    response_time_desc: Optional[str] = None
    response_urls: List[str] = Field(default_factory=list)

    fee_schedule_desc: Optional[str] = None
    fee_urls: List[str] = Field(default_factory=list)

    form_availability: Optional[str] = None  # e.g., "online form", "downloadable form", "no form", "yes", "no"
    form_urls: List[str] = Field(default_factory=list)


class AgenciesAPRAExtraction(BaseModel):
    dor: Optional[AgencyAPRAInfo] = None
    bmv: Optional[AgencyAPRAInfo] = None
    pla: Optional[AgencyAPRAInfo] = None
    dnr: Optional[AgencyAPRAInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_apra_info() -> str:
    return """
Extract the APRA (Access to Public Records Act) procedures information exactly as presented in the answer for the following four Indiana state agencies: Department of Revenue (DOR), Bureau of Motor Vehicles (BMV), Professional Licensing Agency (PLA), and Department of Natural Resources (DNR).

For each agency (keys: 'dor', 'bmv', 'pla', 'dnr'), extract the following fields:
- contact_info: The official contact for APRA/public records requests (an email address OR a mailing address block). Return exactly as written in the answer.
- contact_urls: An array of all URLs in the answer that specifically support the contact_info.
- response_time_desc: The stated APRA response timeframes as given in the answer (e.g., “24 hours for in-person/phone; 7 days for mail/fax/email” or a description referencing IC 5-14-3).
- response_urls: An array of URLs that support the response timeframes stated.
- fee_schedule_desc: The copying fee schedule as given in the answer (e.g., “$0.10 per page black-and-white; $0.25 per page color” or a citation to Department of Administration standards).
- fee_urls: An array of URLs that support the fee schedule.
- form_availability: Whether the agency provides an APRA request form, as described in the answer (e.g., “online form”, “downloadable form”, “no form”, “yes”, or “no”).
- form_urls: An array of URLs that directly link to the form or to a page explicitly indicating the presence/absence of a form.

Rules:
- Extract only what is explicitly stated in the answer. Do not infer or invent any values.
- For URLs, include only valid, complete URLs explicitly present in the answer (plain URLs or in markdown links). If none are present, return an empty array for the URL fields.
- If a field is missing for an agency, set it to null (for strings) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_official_source(urls: List[str]) -> bool:
    """
    Returns True if at least one URL appears to be an official Indiana government resource.
    We consider URLs containing 'in.gov' as official (e.g., www.in.gov, iga.in.gov, secure.in.gov, forms.in.gov).
    """
    if not urls:
        return False
    for u in urls:
        try:
            if "in.gov" in u.lower():
                return True
        except Exception:
            continue
    return False


def _fail_leaf(node, reason: str, evaluator: Evaluator, info_name: str, agency_key: str):
    """
    Mark a leaf as failed and record a brief custom info entry for debugging.
    """
    node.score = 0.0
    node.status = "failed"
    evaluator.add_custom_info(
        {
            "agency": agency_key,
            "item": info_name,
            "reason": reason
        },
        info_type="failure_reason"
    )


# --------------------------------------------------------------------------- #
# Verification logic per agency                                               #
# --------------------------------------------------------------------------- #
async def verify_agency(
    evaluator: Evaluator,
    parent_node,
    agency_key: str,
    agency_meta: Dict[str, Any],
    agency_info: Optional[AgencyAPRAInfo],
) -> None:
    """
    Build verification nodes and run checks for one agency.
    """
    # Create aggregator node for the agency (parallel aggregation, non-critical)
    agg_node = evaluator.add_parallel(
        id=agency_meta["agg_id"],
        desc=agency_meta["agg_desc"],
        parent=parent_node,
        critical=False
    )

    # Prepare easy references
    full_name = agency_meta["full_name"]
    leaf_ids = agency_meta["leaf_ids"]
    leaf_desc = agency_meta["leaf_desc"]

    # 1) Contact (CRITICAL)
    contact_leaf = evaluator.add_leaf(
        id=leaf_ids["contact"],
        desc=leaf_desc["contact"],
        parent=agg_node,
        critical=True
    )

    if not agency_info or not agency_info.contact_info or not agency_info.contact_info.strip():
        _fail_leaf(contact_leaf, "Missing contact_info in answer", evaluator, "contact", agency_key)
    elif not agency_info.contact_urls:
        _fail_leaf(contact_leaf, "No contact_urls provided for sourcing", evaluator, "contact", agency_key)
    elif not _has_official_source(agency_info.contact_urls):
        _fail_leaf(contact_leaf, "No official in.gov source provided for contact", evaluator, "contact", agency_key)
    else:
        contact_claim = (
            f"The official APRA (Access to Public Records Act) request contact for {full_name} is: "
            f"{agency_info.contact_info}"
        )
        await evaluator.verify(
            claim=contact_claim,
            node=contact_leaf,
            sources=agency_info.contact_urls,
            additional_instruction=(
                "Verify that the cited official Indiana government page(s) show this APRA/public records request contact. "
                "Contact can be an email address or a mailing address block. Minor formatting differences are acceptable."
            ),
        )

    # 2) Response timeframes (CRITICAL)
    response_leaf = evaluator.add_leaf(
        id=leaf_ids["response"],
        desc=leaf_desc["response"],
        parent=agg_node,
        critical=True
    )

    if not agency_info or not agency_info.response_time_desc or not agency_info.response_time_desc.strip():
        _fail_leaf(response_leaf, "Missing response_time_desc in answer", evaluator, "response", agency_key)
    elif not agency_info.response_urls:
        _fail_leaf(response_leaf, "No response_urls provided for sourcing", evaluator, "response", agency_key)
    elif not _has_official_source(agency_info.response_urls):
        _fail_leaf(response_leaf, "No official in.gov source provided for response timeframes", evaluator, "response", agency_key)
    else:
        # The statutory standard to be checked in the content
        standard_text = (
            "24 hours for in-person or telephone requests; and 7 calendar days for requests by mail, fax, or email"
        )
        response_claim = (
            f"{full_name} states APRA response timeframes that align with IC 5-14-3: {standard_text}."
        )
        await evaluator.verify(
            claim=response_claim,
            node=response_leaf,
            sources=agency_info.response_urls,
            additional_instruction=(
                "The webpage should explicitly state the APRA response timeframes OR clearly reference IC 5-14-3 "
                "in a way that entails these timeframes. Accept if the page states the 24-hour and 7-day standards directly, "
                "or unambiguously references them. Minor wording differences are acceptable (e.g., 'within 7 days' "
                "for mail/email)."
            ),
        )

    # 3) Fee schedule (CRITICAL)
    fee_leaf = evaluator.add_leaf(
        id=leaf_ids["fee"],
        desc=leaf_desc["fee"],
        parent=agg_node,
        critical=True
    )

    if not agency_info or not agency_info.fee_schedule_desc or not agency_info.fee_schedule_desc.strip():
        _fail_leaf(fee_leaf, "Missing fee_schedule_desc in answer", evaluator, "fee", agency_key)
    elif not agency_info.fee_urls:
        _fail_leaf(fee_leaf, "No fee_urls provided for sourcing", evaluator, "fee", agency_key)
    elif not _has_official_source(agency_info.fee_urls):
        _fail_leaf(fee_leaf, "No official in.gov source provided for fee schedule", evaluator, "fee", agency_key)
    else:
        fee_claim = (
            f"The copying fee schedule for APRA records at {full_name} is $0.10 per page for black-and-white copies "
            f"and $0.25 per page for color copies, consistent with Indiana Department of Administration standards."
        )
        await evaluator.verify(
            claim=fee_claim,
            node=fee_leaf,
            sources=agency_info.fee_urls,
            additional_instruction=(
                "Verify that the cited official Indiana government page(s) show these copying fees "
                "($0.10 per page for black-and-white; $0.25 per page for color) or reference the Department of Administration standards "
                "with these rates. Minor currency formatting differences are acceptable."
            ),
        )

    # 4) Form availability (NON-CRITICAL)
    form_leaf = evaluator.add_leaf(
        id=leaf_ids["form"],
        desc=leaf_desc["form"],
        parent=agg_node,
        critical=False
    )

    # For form availability, we try to verify either the presence or explicit absence of a form.
    # We require at least one official source to ground the claim; otherwise we mark failed (non-critical).
    # If the answer states 'no form' but provides a source stating that, we verify the 'no form' claim.
    form_avail_val = (agency_info.form_availability or "").strip().lower() if agency_info else ""

    if not agency_info or form_avail_val == "":
        _fail_leaf(form_leaf, "Missing form_availability in answer", evaluator, "form", agency_key)
    elif not agency_info.form_urls:
        _fail_leaf(form_leaf, "No form_urls provided for sourcing", evaluator, "form", agency_key)
    elif not _has_official_source(agency_info.form_urls):
        _fail_leaf(form_leaf, "No official in.gov source provided for form availability", evaluator, "form", agency_key)
    else:
        if any(term in form_avail_val for term in ["no", "not provided", "none"]):
            form_claim = f"{full_name} does not provide an online or downloadable APRA request form."
            add_ins = (
                "Pass if the cited official page clearly indicates there is no standardized APRA request form or that "
                "requests must be made without a provided form (e.g., via email or mail)."
            )
        else:
            form_claim = f"{full_name} provides an online or downloadable APRA request form."
            add_ins = (
                "Pass if the cited official page provides either: "
                "1) an online submission form for APRA/public records requests; or "
                "2) a downloadable template (PDF/Word) for APRA/public records requests."
            )

        await evaluator.verify(
            claim=form_claim,
            node=form_leaf,
            sources=agency_info.form_urls,
            additional_instruction=add_ins
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
    Evaluate an answer for Indiana APRA procedure documentation across four agencies.
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

    # Extract structured APRA info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_apra_info(),
        template_class=AgenciesAPRAExtraction,
        extraction_name="apra_extraction"
    )

    # Optional: record statutory standards for reference in summary
    evaluator.add_ground_truth({
        "statute_requirements": {
            "response_time": "24 hours (in-person/telephone); 7 calendar days (mail/fax/email) per IC 5-14-3",
            "copy_fees": "$0.10/page B&W; $0.25/page color (Indiana Department of Administration standards)"
        }
    }, gt_type="standards")

    # Build verification for each agency (parallel)
    await verify_agency(evaluator, root, "dor", AGENCY_META["dor"], extracted.dor)
    await verify_agency(evaluator, root, "bmv", AGENCY_META["bmv"], extracted.bmv)
    await verify_agency(evaluator, root, "pla", AGENCY_META["pla"], extracted.pla)
    await verify_agency(evaluator, root, "dnr", AGENCY_META["dnr"], extracted.dnr)

    # Return full evaluation summary
    return evaluator.get_summary()