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
TASK_ID = "nc_rhododendron_vendor_2026"
TASK_DESCRIPTION = (
    "I'm a woodworking artisan from Charlotte, North Carolina, who makes handmade wooden items "
    "(cutting boards, decorative boxes, and small furniture pieces). I want to apply as a vendor at the "
    "NC Rhododendron Festival Craft Fair in June 2026. Since this is my first time at this festival and I'm not a local "
    "Mitchell County resident, I need to gather all the essential information to prepare properly. Please provide "
    "comprehensive details about: the event schedule and participation requirements, booth size specifications, vendor fees, "
    "North Carolina sales tax obligations and certificate requirements, setup and departure timelines, and the required personal "
    "protective equipment for woodworking activities to ensure safety compliance. Include specific reference URLs for each piece of information."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EventInfo(BaseModel):
    schedule_text: Optional[str] = None
    schedule_urls: List[str] = Field(default_factory=list)
    two_day_commitment_text: Optional[str] = None
    two_day_commitment_urls: List[str] = Field(default_factory=list)


class BoothInfo(BaseModel):
    tent_size: Optional[str] = None
    tent_size_urls: List[str] = Field(default_factory=list)
    electricity_policy: Optional[str] = None
    electricity_urls: List[str] = Field(default_factory=list)


class FeeInfo(BaseModel):
    standard_fee: Optional[str] = None
    standard_fee_urls: List[str] = Field(default_factory=list)
    reduced_fee: Optional[str] = None
    reduced_fee_urls: List[str] = Field(default_factory=list)


class TaxInfo(BaseModel):
    tax_rate: Optional[str] = None
    tax_rate_urls: List[str] = Field(default_factory=list)
    certificate_requirement_text: Optional[str] = None
    certificate_requirement_urls: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    setup_windows_text: Optional[str] = None
    setup_deadline_text: Optional[str] = None
    setup_urls: List[str] = Field(default_factory=list)
    remain_until_friday_end_text: Optional[str] = None
    saturday_depart_by_text: Optional[str] = None
    departure_urls: List[str] = Field(default_factory=list)


class PPEInfo(BaseModel):
    eye_protection_text: Optional[str] = None
    eye_sources: List[str] = Field(default_factory=list)
    hearing_protection_text: Optional[str] = None
    hearing_sources: List[str] = Field(default_factory=list)
    respiratory_text: Optional[str] = None
    respiratory_sources: List[str] = Field(default_factory=list)
    gloves_text: Optional[str] = None
    gloves_sources: List[str] = Field(default_factory=list)
    first_aid_text: Optional[str] = None
    first_aid_sources: List[str] = Field(default_factory=list)


class VendorPreparationExtraction(BaseModel):
    event: Optional[EventInfo] = None
    booth: Optional[BoothInfo] = None
    fees: Optional[FeeInfo] = None
    taxes: Optional[TaxInfo] = None
    timeline: Optional[TimelineInfo] = None
    ppe: Optional[PPEInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_vendor_preparation() -> str:
    return """
    Extract the vendor preparation details for the NC Rhododendron Festival Craft Fair in June 2026 exactly as provided in the answer.
    You must capture the statements and the specific URLs (as provided in the answer) that support each piece of information.
    If any piece is missing from the answer, set the corresponding text to null and the URL list to an empty array.

    EVENT & PARTICIPATION:
    - event.schedule_text: The complete schedule (dates and times) for the craft fair as stated in the answer (e.g., "Friday June 19, 2026, 12pm–5pm; Saturday June 20, 2026, 10am–4pm").
    - event.schedule_urls: All URLs cited that support the schedule.
    - event.two_day_commitment_text: The statement about the requirement to participate both days (two-day commitment), if present.
    - event.two_day_commitment_urls: URLs cited that support the two-day participation requirement.

    BOOTH REQUIREMENTS:
    - booth.tent_size: The stated booth/tent size requirement (e.g., "10'x10' tents only").
    - booth.tent_size_urls: URLs cited that support the tent size requirement.
    - booth.electricity_policy: The statement about electricity availability/limitations and any requirement to request/indicate it on the application.
    - booth.electricity_urls: URLs cited that support the electricity policy.

    VENDOR FEES:
    - fees.standard_fee: The standard booth fee for artists/craftspeople (e.g., "$100"), as stated in the answer.
    - fees.standard_fee_urls: URLs cited for the standard fee.
    - fees.reduced_fee: The reduced fee for local Mitchell County or previous-year vendors (e.g., "$80"), as stated in the answer.
    - fees.reduced_fee_urls: URLs cited for the reduced fee.

    TAX OBLIGATIONS:
    - taxes.tax_rate: The NC sales tax rate vendors must collect (as stated in the answer, e.g., "7.75%").
    - taxes.tax_rate_urls: URLs cited that support the stated tax rate.
    - taxes.certificate_requirement_text: The requirement to display a Sales & Use Tax Certificate (e.g., mentioning N.C. Gen. Stat. § 66-255), as stated in the answer.
    - taxes.certificate_requirement_urls: URLs cited that support the certificate requirement.

    LOGISTICS (SETUP/DEPARTURE):
    - timeline.setup_windows_text: The allowed setup time windows (e.g., "Thursday 5–7pm or Friday 8–10am").
    - timeline.setup_deadline_text: The setup completion deadline (e.g., "must be completely set up by 10am Friday").
    - timeline.setup_urls: URLs cited that support setup windows and deadlines.
    - timeline.remain_until_friday_end_text: The requirement to remain until the show ends at 5pm Friday (no early teardown), as stated in the answer.
    - timeline.saturday_depart_by_text: The requirement to depart by a specific time on Saturday (e.g., "by 5:30 PM Saturday"), as stated in the answer.
    - timeline.departure_urls: URLs cited that support the departure requirements.

    SAFETY EQUIPMENT (PPE) FOR WOODWORKING:
    - ppe.eye_protection_text: Statement that safety glasses or goggles are required/recommended.
    - ppe.eye_sources: URLs cited that support eye protection.
    - ppe.hearing_protection_text: Statement that hearing protection (earmuffs/earplugs) is required/recommended.
    - ppe.hearing_sources: URLs cited that support hearing protection.
    - ppe.respiratory_text: Statement that a dust mask or respirator is required/recommended.
    - ppe.respiratory_sources: URLs cited that support respiratory protection.
    - ppe.gloves_text: Statement that work gloves are required/recommended.
    - ppe.gloves_sources: URLs cited that support glove use.
    - ppe.first_aid_text: Statement that a first aid kit is required/recommended.
    - ppe.first_aid_sources: URLs cited that support having a first aid kit.

    IMPORTANT:
    - Extract only what the answer explicitly states; do not invent values.
    - For URLs, extract the actual URL strings present in the answer (plain or markdown). If missing protocol, prepend http://
    - If something is not present in the answer, set the corresponding text to null and the URL list to an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u:
            continue
        s = str(u).strip()
        if not s:
            continue
        out.append(s)
    # keep order but remove dups
    seen = set()
    uniq = []
    for u in out:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(_clean_list(lst))
    # deduplicate preserving order
    seen = set()
    uniq = []
    for u in combined:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_festival_basics(evaluator: Evaluator, parent, data: VendorPreparationExtraction) -> None:
    basics_node = evaluator.add_parallel(
        id="Festival_Basics",
        desc="Core event and booth information",
        parent=parent,
        critical=False
    )

    # Event Information (Critical)
    event_node = evaluator.add_parallel(
        id="Event_Information",
        desc="Event schedule and participation requirements with supporting reference URL(s)",
        parent=basics_node,
        critical=True
    )
    event = data.event or EventInfo()

    # Schedule presence and sources
    evaluator.add_custom_node(
        result=_has_text(event.schedule_text),
        id="Schedule_Text_Present",
        desc="Schedule text is provided in the answer",
        parent=event_node,
        critical=True
    )
    schedule_sources = _clean_list(event.schedule_urls)
    evaluator.add_custom_node(
        result=len(schedule_sources) > 0,
        id="Schedule_Sources_Provided",
        desc="Schedule has supporting reference URL(s)",
        parent=event_node,
        critical=True
    )

    # Verify schedule against sources
    schedule_leaf = evaluator.add_leaf(
        id="Schedule_and_Commitment",
        desc="Schedule matches the referenced official page(s) for the 2026 craft fair",
        parent=event_node,
        critical=True
    )
    schedule_claim = f"The craft fair schedule stated in the answer is: {event.schedule_text}."
    await evaluator.verify(
        claim=schedule_claim,
        node=schedule_leaf,
        sources=schedule_sources,
        additional_instruction=(
            "Verify that the referenced webpage(s) explicitly list the same dates and times for the NC Rhododendron Festival "
            "Craft Fair schedule in June 2026 as stated in the claim. Allow minor formatting differences. Prefer the official festival "
            "or organizer website pages for this information; if a source is clearly not official and does not authoritatively list the schedule, consider it unsupported."
        )
    )

    # Two-day commitment presence and sources
    evaluator.add_custom_node(
        result=_has_text(event.two_day_commitment_text),
        id="Two_Day_Commitment_Text_Present",
        desc="Two-day commitment requirement is stated in the answer",
        parent=event_node,
        critical=True
    )
    commitment_sources = _clean_list(event.two_day_commitment_urls)
    evaluator.add_custom_node(
        result=len(commitment_sources) > 0,
        id="Two_Day_Commitment_Sources_Provided",
        desc="Two-day commitment has supporting reference URL(s)",
        parent=event_node,
        critical=True
    )

    # Verify two-day commitment against sources
    two_day_leaf = evaluator.add_leaf(
        id="Two_Day_Commitment_Required",
        desc="Two-day participation requirement is supported by the referenced official page(s)",
        parent=event_node,
        critical=True
    )
    two_day_claim = (
        "Vendors must commit to participate both days of the craft fair (Friday and Saturday); no one-day participation."
    )
    await evaluator.verify(
        claim=two_day_claim,
        node=two_day_leaf,
        sources=commitment_sources,
        additional_instruction=(
            "Confirm the page(s) clearly state that vendors must participate both days (a two-day commitment is required). "
            "Prefer information from the official festival or organizer website."
        )
    )

    # Booth Requirements (Critical)
    booth_node = evaluator.add_parallel(
        id="Booth_Requirements",
        desc="Physical booth specifications with supporting reference URL(s)",
        parent=basics_node,
        critical=True
    )
    booth = data.booth or BoothInfo()
    tent_urls = _clean_list(booth.tent_size_urls)
    elec_urls = _clean_list(booth.electricity_urls)
    booth_urls = _combine_urls(tent_urls, elec_urls)

    # Presence checks
    evaluator.add_custom_node(
        result=_has_text(booth.tent_size),
        id="Tent_Size_Text_Present",
        desc="Booth tent size requirement is stated in the answer",
        parent=booth_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(booth_urls) > 0,
        id="Booth_Sources_Provided",
        desc="Booth requirements have supporting reference URL(s)",
        parent=booth_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(booth.electricity_policy),
        id="Electricity_Text_Present",
        desc="Electricity availability/limitations are stated in the answer",
        parent=booth_node,
        critical=True
    )

    # Verify tent size
    tent_leaf = evaluator.add_leaf(
        id="Tent_Size_and_Electricity_TentSize",
        desc="Tent size requirement is correctly supported by referenced page(s)",
        parent=booth_node,
        critical=True
    )
    tent_claim = f"The craft fair booth tent size requirement is as stated: {booth.tent_size}."
    await evaluator.verify(
        claim=tent_claim,
        node=tent_leaf,
        sources=booth_urls,
        additional_instruction=(
            "Verify that the referenced page(s) state the booth/tent size requirement exactly or equivalently (e.g., 10'x10' tents only). "
            "Prefer the official festival or organizer website."
        )
    )

    # Verify electricity policy
    elec_leaf = evaluator.add_leaf(
        id="Tent_Size_and_Electricity_Electricity",
        desc="Electricity availability/limitations are correctly supported by referenced page(s)",
        parent=booth_node,
        critical=True
    )
    elec_claim = f"The craft fair electricity policy is as stated: {booth.electricity_policy}."
    await evaluator.verify(
        claim=elec_claim,
        node=elec_leaf,
        sources=booth_urls,
        additional_instruction=(
            "Confirm that electricity availability is limited and that vendors must indicate or request it on the application if stated; "
            "prefer the official festival or organizer website."
        )
    )


async def build_financial_compliance(evaluator: Evaluator, parent, data: VendorPreparationExtraction) -> None:
    fin_node = evaluator.add_parallel(
        id="Financial_Compliance",
        desc="Cost and tax requirements",
        parent=parent,
        critical=False
    )

    # Vendor Fees (Critical)
    fees_node = evaluator.add_parallel(
        id="Vendor_Fees",
        desc="Booth rental costs with supporting reference URL(s)",
        parent=fin_node,
        critical=True
    )
    fees = data.fees or FeeInfo()
    std_urls = _clean_list(fees.standard_fee_urls)
    red_urls = _clean_list(fees.reduced_fee_urls)
    fees_urls = _combine_urls(std_urls, red_urls)

    evaluator.add_custom_node(
        result=_has_text(fees.standard_fee),
        id="Standard_Fee_Text_Present",
        desc="Standard booth fee is stated in the answer",
        parent=fees_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(fees.reduced_fee),
        id="Reduced_Fee_Text_Present",
        desc="Reduced booth fee (local or prior-year) is stated in the answer",
        parent=fees_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(fees_urls) > 0,
        id="Fees_Sources_Provided",
        desc="Vendor fees have supporting reference URL(s)",
        parent=fees_node,
        critical=True
    )

    std_leaf = evaluator.add_leaf(
        id="Booth_Fee_Structure_Standard",
        desc="Standard booth fee is correctly supported by referenced page(s)",
        parent=fees_node,
        critical=True
    )
    std_claim = f"The standard booth fee for artists/craftspeople is {fees.standard_fee}."
    await evaluator.verify(
        claim=std_claim,
        node=std_leaf,
        sources=fees_urls,
        additional_instruction=(
            "Confirm that the referenced page(s) list the standard booth fee as stated. Prefer official festival/organizer website."
        )
    )

    red_leaf = evaluator.add_leaf(
        id="Booth_Fee_Structure_Reduced",
        desc="Reduced booth fee (local Mitchell County or previous-year vendors) is correctly supported by referenced page(s)",
        parent=fees_node,
        critical=True
    )
    red_claim = f"The reduced fee for local Mitchell County or previous-year vendors is {fees.reduced_fee}."
    await evaluator.verify(
        claim=red_claim,
        node=red_leaf,
        sources=fees_urls,
        additional_instruction=(
            "Confirm that the referenced page(s) list the reduced vendor fee as stated (for local Mitchell County or prior year vendors). Prefer official site."
        )
    )

    # Tax Obligations (Critical)
    tax_node = evaluator.add_parallel(
        id="Tax_Obligations",
        desc="Sales tax and certificate requirements with supporting reference URL(s)",
        parent=fin_node,
        critical=True
    )
    taxes = data.taxes or TaxInfo()
    tax_urls = _clean_list(taxes.tax_rate_urls)
    cert_urls = _clean_list(taxes.certificate_requirement_urls)
    tax_all_urls = _combine_urls(tax_urls, cert_urls)

    evaluator.add_custom_node(
        result=_has_text(taxes.tax_rate),
        id="Tax_Rate_Text_Present",
        desc="Tax rate is stated in the answer",
        parent=tax_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(tax_all_urls) > 0,
        id="Tax_Sources_Provided",
        desc="Tax obligations have supporting reference URL(s)",
        parent=tax_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(taxes.certificate_requirement_text),
        id="Certificate_Text_Present",
        desc="Certificate requirement is stated in the answer",
        parent=tax_node,
        critical=True
    )

    # Verify tax rate
    tax_leaf = evaluator.add_leaf(
        id="Sales_Tax_and_Certificate_Rate",
        desc="NC sales tax rate stated in the answer is supported by referenced page(s)",
        parent=tax_node,
        critical=True
    )
    tax_claim = f"The NC sales tax rate vendors must collect for sales at the craft fair is {taxes.tax_rate}."
    await evaluator.verify(
        claim=tax_claim,
        node=tax_leaf,
        sources=tax_all_urls,
        additional_instruction=(
            "Verify that the page(s) explicitly indicate the same tax rate as stated. Prefer the official festival site if it lists vendor tax obligations; "
            "otherwise ensure the source is an official organizer page providing vendor requirements."
        )
    )

    # Verify certificate requirement
    cert_leaf = evaluator.add_leaf(
        id="Sales_Tax_and_Certificate_Requirement",
        desc="Sales & Use Tax Certificate requirement is supported by referenced page(s)",
        parent=tax_node,
        critical=True
    )
    cert_claim = (
        "Vendors must display a North Carolina Sales & Use Tax Certificate (as required, e.g., per N.C. Gen. Stat. § 66-255)."
    )
    await evaluator.verify(
        claim=cert_claim,
        node=cert_leaf,
        sources=tax_all_urls,
        additional_instruction=(
            "Confirm the referenced page(s) explicitly state the requirement to display a North Carolina Sales & Use Tax Certificate, "
            "ideally referencing the statute or clearly stating the rule. Prefer official festival/organizer pages."
        )
    )


async def build_logistics(evaluator: Evaluator, parent, data: VendorPreparationExtraction) -> None:
    logistics_node = evaluator.add_parallel(
        id="Logistics",
        desc="Setup and departure timeline requirements",
        parent=parent,
        critical=False
    )
    timeline_node = evaluator.add_parallel(
        id="Timeline_Requirements",
        desc="Critical timing information for vendor operations with supporting reference URL(s)",
        parent=logistics_node,
        critical=True
    )
    tl = data.timeline or TimelineInfo()

    # Setup presence/sources
    setup_urls = _clean_list(tl.setup_urls)
    evaluator.add_custom_node(
        result=_has_text(tl.setup_windows_text),
        id="Setup_Windows_Text_Present",
        desc="Setup windows are stated in the answer",
        parent=timeline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(tl.setup_deadline_text),
        id="Setup_Deadline_Text_Present",
        desc="Setup completion deadline is stated in the answer",
        parent=timeline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(setup_urls) > 0,
        id="Setup_Sources_Provided",
        desc="Setup requirements have supporting reference URL(s)",
        parent=timeline_node,
        critical=True
    )

    # Verify setup windows
    setup_windows_leaf = evaluator.add_leaf(
        id="Setup_Windows_and_Deadline_Windows",
        desc="Setup time windows are supported by referenced page(s)",
        parent=timeline_node,
        critical=True
    )
    setup_windows_claim = f"The setup windows are as stated: {tl.setup_windows_text}."
    await evaluator.verify(
        claim=setup_windows_claim,
        node=setup_windows_leaf,
        sources=setup_urls,
        additional_instruction=(
            "Confirm the page(s) explicitly list the same setup time windows (e.g., Thursday 5–7pm and/or Friday 8–10am). "
            "Prefer official festival/organizer pages."
        )
    )

    # Verify setup deadline
    setup_deadline_leaf = evaluator.add_leaf(
        id="Setup_Windows_and_Deadline_Deadline",
        desc="Setup completion deadline is supported by referenced page(s)",
        parent=timeline_node,
        critical=True
    )
    setup_deadline_claim = f"The setup completion deadline is as stated: {tl.setup_deadline_text}."
    await evaluator.verify(
        claim=setup_deadline_claim,
        node=setup_deadline_leaf,
        sources=setup_urls,
        additional_instruction=(
            "Confirm the page(s) explicitly state the same setup completion deadline (e.g., must be completely set up by 10am Friday). "
            "Prefer official festival/organizer pages."
        )
    )

    # Departure presence/sources
    depart_urls = _clean_list(tl.departure_urls)
    evaluator.add_custom_node(
        result=_has_text(tl.remain_until_friday_end_text),
        id="Departure_FridayEnd_Text_Present",
        desc="No early teardown (remain until 5pm Friday) is stated in the answer",
        parent=timeline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(tl.saturday_depart_by_text),
        id="Departure_SatBy_Text_Present",
        desc="Saturday depart-by time is stated in the answer",
        parent=timeline_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(depart_urls) > 0,
        id="Departure_Sources_Provided",
        desc="Departure requirements have supporting reference URL(s)",
        parent=timeline_node,
        critical=True
    )

    # Verify no early teardown on Friday
    depart_friday_leaf = evaluator.add_leaf(
        id="Departure_Requirements_NoEarlyTeardown",
        desc="No early teardown until show ends at 5pm Friday is supported by referenced page(s)",
        parent=timeline_node,
        critical=True
    )
    depart_friday_claim = "Vendors must remain until the show ends at 5pm on Friday; no early teardown is permitted."
    await evaluator.verify(
        claim=depart_friday_claim,
        node=depart_friday_leaf,
        sources=depart_urls,
        additional_instruction=(
            "Verify the page(s) explicitly state that vendors must remain until 5pm Friday and cannot tear down early. Prefer official festival pages."
        )
    )

    # Verify Saturday depart-by time
    depart_sat_leaf = evaluator.add_leaf(
        id="Departure_Requirements_SaturdayBy",
        desc="Vendors must depart by the stated time on Saturday is supported by referenced page(s)",
        parent=timeline_node,
        critical=True
    )
    depart_sat_claim = f"Vendors must depart by the stated time on Saturday: {tl.saturday_depart_by_text}."
    await evaluator.verify(
        claim=depart_sat_claim,
        node=depart_sat_leaf,
        sources=depart_urls,
        additional_instruction=(
            "Confirm the page(s) specify the same required departure time on Saturday (e.g., by 5:30 PM). Prefer official festival pages."
        )
    )


async def build_safety_equipment(evaluator: Evaluator, parent, data: VendorPreparationExtraction) -> None:
    safety_node = evaluator.add_parallel(
        id="Safety_Equipment",
        desc="Required personal protective equipment for woodworking activities",
        parent=parent,
        critical=False
    )
    ppe_node = evaluator.add_parallel(
        id="Required_PPE",
        desc="Essential safety gear for woodworking vendors with supporting reference URL(s)",
        parent=safety_node,
        critical=True
    )
    ppe = data.ppe or PPEInfo()

    # Eye protection
    eye_urls = _clean_list(ppe.eye_sources)
    evaluator.add_custom_node(
        result=_has_text(ppe.eye_protection_text),
        id="Eye_Protection_Text_Present",
        desc="Eye protection requirement/recommendation is stated in the answer",
        parent=ppe_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(eye_urls) > 0,
        id="Eye_Protection_Sources_Provided",
        desc="Eye protection has supporting reference URL(s)",
        parent=ppe_node,
        critical=True
    )
    eye_leaf = evaluator.add_leaf(
        id="Eye_Protection_Verified",
        desc="Safety glasses/goggles requirement/recommendation is supported by reputable woodworking safety source(s)",
        parent=ppe_node,
        critical=True
    )
    eye_claim = "Safety glasses or goggles are required or strongly recommended for woodworking tasks due to flying debris and dust."
    await evaluator.verify(
        claim=eye_claim,
        node=eye_leaf,
        sources=eye_urls,
        additional_instruction=(
            "Confirm that the page(s) (e.g., OSHA/NIOSH/CDC, major universities, or established safety organizations) explicitly state the need for eye protection."
        )
    )

    # Hearing protection
    hearing_urls = _clean_list(ppe.hearing_sources)
    evaluator.add_custom_node(
        result=_has_text(ppe.hearing_protection_text),
        id="Hearing_Protection_Text_Present",
        desc="Hearing protection requirement/recommendation is stated in the answer",
        parent=ppe_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(hearing_urls) > 0,
        id="Hearing_Protection_Sources_Provided",
        desc="Hearing protection has supporting reference URL(s)",
        parent=ppe_node,
        critical=True
    )
    hearing_leaf = evaluator.add_leaf(
        id="Hearing_Protection_Verified",
        desc="Hearing protection requirement/recommendation is supported by reputable woodworking safety source(s)",
        parent=ppe_node,
        critical=True
    )
    hearing_claim = "Hearing protection (earmuffs or earplugs) is required or strongly recommended when operating woodworking machinery due to hazardous noise levels."
    await evaluator.verify(
        claim=hearing_claim,
        node=hearing_leaf,
        sources=hearing_urls,
        additional_instruction=(
            "Confirm that the page(s) explicitly state the need for hearing protection in woodworking contexts; authoritative safety sources only."
        )
    )

    # Respiratory protection
    resp_urls = _clean_list(ppe.respiratory_sources)
    evaluator.add_custom_node(
        result=_has_text(ppe.respiratory_text),
        id="Respiratory_Protection_Text_Present",
        desc="Respiratory protection requirement/recommendation is stated in the answer",
        parent=ppe_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(resp_urls) > 0,
        id="Respiratory_Protection_Sources_Provided",
        desc="Respiratory protection has supporting reference URL(s)",
        parent=ppe_node,
        critical=True
    )
    resp_leaf = evaluator.add_leaf(
        id="Respiratory_Protection_Verified",
        desc="Dust mask or respirator requirement/recommendation is supported by reputable woodworking safety source(s)",
        parent=ppe_node,
        critical=True
    )
    resp_claim = "A dust mask or respirator is required or strongly recommended for woodworking to protect against fine wood dust."
    await evaluator.verify(
        claim=resp_claim,
        node=resp_leaf,
        sources=resp_urls,
        additional_instruction=(
            "Confirm that the page(s) explicitly state the need for a dust mask or respirator for woodworking; authoritative safety sources only."
        )
    )

    # Work gloves
    gloves_urls = _clean_list(ppe.gloves_sources)
    evaluator.add_custom_node(
        result=_has_text(ppe.gloves_text),
        id="Gloves_Text_Present",
        desc="Work gloves requirement/recommendation is stated in the answer",
        parent=ppe_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(gloves_urls) > 0,
        id="Gloves_Sources_Provided",
        desc="Work gloves have supporting reference URL(s)",
        parent=ppe_node,
        critical=True
    )
    gloves_leaf = evaluator.add_leaf(
        id="Gloves_Verified",
        desc="Work gloves recommendation is supported by reputable woodworking safety source(s)",
        parent=ppe_node,
        critical=True
    )
    gloves_claim = "Work gloves are recommended for handling rough lumber or materials to protect against cuts and splinters in woodworking contexts."
    await evaluator.verify(
        claim=gloves_claim,
        node=gloves_leaf,
        sources=gloves_urls,
        additional_instruction=(
            "Confirm that the page(s) recommend or require appropriate work gloves for woodworking tasks; authoritative safety sources only."
        )
    )

    # First aid kit
    first_aid_urls = _clean_list(ppe.first_aid_sources)
    evaluator.add_custom_node(
        result=_has_text(ppe.first_aid_text),
        id="FirstAid_Text_Present",
        desc="First aid kit requirement/recommendation is stated in the answer",
        parent=ppe_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(first_aid_urls) > 0,
        id="FirstAid_Sources_Provided",
        desc="First aid kit has supporting reference URL(s)",
        parent=ppe_node,
        critical=True
    )
    first_aid_leaf = evaluator.add_leaf(
        id="FirstAid_Verified",
        desc="Having a first aid kit on hand is supported by reputable woodworking safety source(s)",
        parent=ppe_node,
        critical=True
    )
    first_aid_claim = "A first aid kit should be available at woodworking work areas to respond to minor injuries and emergencies."
    await evaluator.verify(
        claim=first_aid_claim,
        node=first_aid_leaf,
        sources=first_aid_urls,
        additional_instruction=(
            "Confirm that the page(s) recommend having a first aid kit available for woodworking operations; authoritative safety sources only."
        )
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
    Evaluate an answer for the NC Rhododendron Festival vendor preparation task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # Extract all structured info in one pass
    extraction = await evaluator.extract(
        prompt=prompt_extract_vendor_preparation(),
        template_class=VendorPreparationExtraction,
        extraction_name="vendor_preparation"
    )

    # Build rubric tree sections
    await build_festival_basics(evaluator, root, extraction)
    await build_financial_compliance(evaluator, root, extraction)
    await build_logistics(evaluator, root, extraction)
    await build_safety_equipment(evaluator, root, extraction)

    return evaluator.get_summary()