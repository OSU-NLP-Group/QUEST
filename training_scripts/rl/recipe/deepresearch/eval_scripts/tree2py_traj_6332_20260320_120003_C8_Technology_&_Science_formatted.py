import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "telecom_outage_reg_compliance_2026"
TASK_DESCRIPTION = """A major U.S. wireless telecommunications provider experienced a network outage on January 14, 2026, that lasted approximately 10 hours and affected approximately 1.5 million customers due to a Mobile Switching Center (MSC) failure. Based on current FCC regulations and industry standards, determine the following:

1. FCC NORS Reporting: What are the specific reporting requirements and deadlines under the FCC's Network Outage Reporting System (NORS) per 47 CFR 4.9 for this outage? Include the notification deadline, initial report deadline, and final report deadline measured from the time of discovery.
2. 911 Special Facility Notification: If this outage potentially affected 911 special facilities, what are the specific notification requirements and timelines under 47 CFR 4.9(h)? Include the initial notification deadline, required communication methods, and first follow-up deadline.
3. Customer Compensation: Based on practices established by major U.S. carriers in 2026, what is a typical compensation amount offered to affected customers for a 10-hour outage? Provide a specific example from a major carrier's compensation policy announced in January 2026.
4. Backup Power Compliance: What are the FCC's backup power requirements under 47 CFR 9.20 that wireless providers must meet as of the date of this outage (January 2026)? Specify the minimum hours of standby backup power that must be offered to customers.
5. Tower Infrastructure Standards: According to the TIA-222 standard, what are the minimum inspection intervals for telecommunications towers, and how do these intervals differ between guyed towers and self-supporting structures?

For each compliance area, provide the specific regulatory citation (CFR section) or industry standard, the numeric thresholds or deadlines, and a reference URL that supports the requirement.
"""


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class NORSInfo(BaseModel):
    citation: Optional[str] = None
    thresholds_text: Optional[str] = None
    notification_deadline: Optional[str] = None
    initial_report_deadline: Optional[str] = None
    final_report_deadline: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class Special911Info(BaseModel):
    citation: Optional[str] = None
    initial_notification_deadline: Optional[str] = None
    required_comm_methods: Optional[str] = None
    first_followup_deadline: Optional[str] = None
    material_elements_list: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


class CompensationInfo(BaseModel):
    typical_amount: Optional[str] = None
    example_carrier: Optional[str] = None
    example_amount: Optional[str] = None
    example_description: Optional[str] = None
    example_urls: List[str] = Field(default_factory=list)


class BackupPowerInfo(BaseModel):
    citation: Optional[str] = None
    min_standby_option_8h: Optional[str] = None
    min_standby_option_24h: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class TIA222Info(BaseModel):
    citation: Optional[str] = None
    intervals_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ComplianceExtraction(BaseModel):
    nors: Optional[NORSInfo] = None
    special911: Optional[Special911Info] = None
    compensation: Optional[CompensationInfo] = None
    backup: Optional[BackupPowerInfo] = None
    tower: Optional[TIA222Info] = None


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_compliance() -> str:
    return """
Extract the required regulatory and industry-standard details exactly as stated in the answer. Return a single JSON object with the following nested structure and fields. If a field is missing in the answer, set it to null (or an empty list for URL lists).

- nors:
  - citation: The FCC regulatory citation text for NORS (e.g., "47 CFR 4.9(e)" or "47 C.F.R. § 4.9").
  - thresholds_text: The answer's phrasing that explains NORS reportability thresholds (e.g., duration ≥30 minutes AND (≥900,000 user-minutes OR ≥30,000 users)).
  - notification_deadline: The stated NORS notification deadline from discovery (e.g., "within 120 minutes", "within 2 hours").
  - initial_report_deadline: The stated NORS initial report deadline from discovery (e.g., "within 72 hours").
  - final_report_deadline: The stated NORS final report deadline from discovery (e.g., "within 30 days").
  - reference_urls: All URLs cited in the answer that support NORS requirements.

- special911:
  - citation: The FCC regulatory citation for special facility/911 notifications (e.g., "47 CFR 4.9(h)").
  - initial_notification_deadline: The stated initial notification deadline (e.g., "within 30 minutes of discovery").
  - required_comm_methods: The answer's wording for required communication methods (should indicate BOTH telephone AND electronic means).
  - first_followup_deadline: The stated first follow-up deadline (e.g., "within 2 hours of initial contact").
  - material_elements_list: The list of 'material information elements' the answer enumerates as required by 47 CFR 4.9(h)(2). Extract as an array of strings, preserving order.
  - reference_urls: All URLs cited in the answer that support 911 special facility notification requirements.

- compensation:
  - typical_amount: The typical compensation amount stated for a ~10-hour outage in 2026 (e.g., "$5 credit", "one-day service credit").
  - example_carrier: The major carrier name for the January 2026 example (e.g., "AT&T", "Verizon", "T-Mobile").
  - example_amount: The example amount offered (e.g., "$5", "one day of service credit").
  - example_description: Short text describing the example policy or announcement as stated in the answer.
  - example_urls: All URLs cited for the January 2026 example.

- backup:
  - citation: The FCC regulatory citation for backup power (e.g., "47 CFR 9.20").
  - min_standby_option_8h: The answer's statement that at least one 8-hour standby power option must be offered (if provided).
  - min_standby_option_24h: The answer's statement that at least one 24-hour standby power option must be offered (if provided).
  - reference_urls: All URLs cited that support the backup power requirements.

- tower:
  - citation: The industry standard citation text identifying TIA-222 (e.g., "TIA-222-H", "TIA/EIA-222", "ANSI/TIA-222-I").
  - intervals_text: The answer's statement of minimum inspection intervals by structure type (e.g., "guyed every 3 years; self-supporting every 5 years").
  - reference_urls: All URLs cited that support the inspection interval requirements.

Special rules:
- Return only URLs that are explicitly present in the answer. If none are present for a section, return an empty list for that section’s URLs.
- Do not invent any values; if an item is not stated, set it to null (or [] for lists).
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _contains_citation(text: Optional[str], section_hint: str) -> bool:
    if not text:
        return False
    t = text.lower()
    # Basic robust checks for citations like "47 CFR 4.9" or "47 C.F.R. § 4.9(h)"
    if section_hint == "4.9":
        return ("cfr" in t and "4.9" in t) or ("§" in t and "4.9" in t)
    if section_hint == "4.9(h)":
        # accept forms like 4.9(h) or 4.9 (h)
        return ("cfr" in t and "4.9" in t and "h" in t) or ("§" in t and "4.9" in t and "h" in t)
    if section_hint == "9.20":
        return ("cfr" in t and "9.20" in t) or ("§" in t and "9.20" in t)
    if section_hint == "TIA-222":
        return "tia-222" in t or "tia/eia-222" in t or "ansi/tia-222" in t or re.search(r"\btia\s*[-/]?\s*222\b", t) is not None
    return False


def _has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_nors(evaluator: Evaluator, parent, nors: Optional[NORSInfo]) -> None:
    nors_node = evaluator.add_parallel(
        id="NORS_Reporting_Requirements",
        desc="FCC NORS reporting requirements for the outage under 47 CFR 4.9, including reportability and all required deadlines from time of discovery.",
        parent=parent,
        critical=True,
    )

    citation_ok = _contains_citation(nors.citation if nors else None, "4.9")
    evaluator.add_custom_node(
        result=citation_ok,
        id="NORS_Citation_Provided",
        desc="Answer provides the applicable FCC regulatory citation for NORS reporting requirements (47 CFR 4.9 / relevant subsections).",
        parent=nors_node,
        critical=True,
    )

    # Ensure we have at least one URL for NORS section
    nors_refurl_node = evaluator.add_custom_node(
        result=_has_any_url(nors.reference_urls if nors else []),
        id="NORS_Reference_URL",
        desc="Answer includes at least one supporting reference URL for NORS requirements.",
        parent=nors_node,
        critical=True,
    )

    # Reportability determination (logical check using task context + answer content)
    rep_node = evaluator.add_leaf(
        id="Outage_Reportability_Determination",
        desc="Answer correctly states that the outage is NORS-reportable by meeting: duration ≥30 minutes AND (user impact ≥900,000 user-minutes OR ≥30,000 users affected), per 47 CFR 4.9 thresholds.",
        parent=nors_node,
        critical=True,
    )
    thresholds_text = nors.thresholds_text if nors and nors.thresholds_text else ""
    reportability_claim = (
        "The answer explicitly determines that the described Jan 14, 2026 outage is reportable in FCC NORS and "
        "explains that NORS reportability requires duration of at least 30 minutes AND either ≥900,000 user‑minutes "
        "or ≥30,000 users affected. Given the scenario (≈10 hours; ≈1.5 million customers), the determination is correct."
    )
    await evaluator.verify(
        claim=reportability_claim,
        node=rep_node,
        additional_instruction=(
            "Judge based on the answer text and task context. Accept paraphrases such as '2 hours = 120 minutes' and "
            "thresholds described in equivalent wording. The key is that the answer both states the correct thresholds "
            "and concludes the outage is NORS‑reportable."
        ),
    )

    # Deadlines verified against URLs
    notif_node = evaluator.add_leaf(
        id="NORS_Notification_Deadline",
        desc="Answer provides the NORS notification deadline measured from discovery (within 120 minutes for wireless providers per 47 CFR 4.9(e)(1)).",
        parent=nors_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9 (including subsection (e)(1) for wireless providers), a NORS outage notification must be filed within 120 minutes (2 hours) of discovery.",
        node=notif_node,
        sources=(nors.reference_urls if nors else []),
        extra_prerequisites=[nors_refurl_node],
        additional_instruction="Verify that the rule states notification is due within 120 minutes from time of discovery (for wireless/CMRS providers).",
    )

    init_node = evaluator.add_leaf(
        id="NORS_Initial_Report_Deadline",
        desc="Answer provides the NORS initial report deadline measured from discovery (within 72 hours per 47 CFR 4.9(e)(4)).",
        parent=nors_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9 (including subsection (e)(4)), the initial NORS report must be filed within 72 hours of discovery.",
        node=init_node,
        sources=(nors.reference_urls if nors else []),
        extra_prerequisites=[nors_refurl_node],
        additional_instruction="Confirm the initial report deadline is 72 hours from discovery (accept equivalent clear wording).",
    )

    final_node = evaluator.add_leaf(
        id="NORS_Final_Report_Deadline",
        desc="Answer provides the NORS final report deadline measured from discovery (within 30 days per 47 CFR 4.9(e)(4)).",
        parent=nors_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9 (including subsection (e)(4)), the final NORS report is due within 30 days (measured from discovery or as the section specifies).",
        node=final_node,
        sources=(nors.reference_urls if nors else []),
        extra_prerequisites=[nors_refurl_node],
        additional_instruction="Allow minor phrasing differences; focus on the 30-day final report timeline specified in §4.9.",
    )


async def verify_911(evaluator: Evaluator, parent, info: Optional[Special911Info]) -> None:
    sec_node = evaluator.add_parallel(
        id="Special_Facility_911_Notification",
        desc="If the outage potentially affected 911 special facilities, provide notification requirements and timelines under 47 CFR 4.9(h).",
        parent=parent,
        critical=True,
    )

    citation_ok = _contains_citation(info.citation if info else None, "4.9(h)")
    evaluator.add_custom_node(
        result=citation_ok,
        id="911_Citation_Provided",
        desc="Answer provides the applicable FCC regulatory citation for special facility/911 notification (47 CFR 4.9(h) / relevant subsections).",
        parent=sec_node,
        critical=True,
    )

    ref_node = evaluator.add_custom_node(
        result=_has_any_url(info.reference_urls if info else []),
        id="911_Reference_URL",
        desc="Answer includes at least one supporting reference URL for 911 special facility notification requirements.",
        parent=sec_node,
        critical=True,
    )

    init_deadline = evaluator.add_leaf(
        id="911_Initial_Notification_Deadline",
        desc="Answer provides the initial notification deadline measured from discovery (within 30 minutes per 47 CFR 4.9(h)(4)).",
        parent=sec_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9(h)(4), initial special-facility (911) outage notification must be made within 30 minutes of discovery.",
        node=init_deadline,
        sources=(info.reference_urls if info else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Look for 'within 30 minutes' measured from time of discovery; accept equivalent phrasing.",
    )

    comm_methods = evaluator.add_leaf(
        id="911_Required_Communication_Methods",
        desc="Answer states the required communication methods (both telephone AND electronic means) per 47 CFR 4.9(h)(3).",
        parent=sec_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9(h)(3), the provider must contact the 911 special facility by both telephone and an electronic means (e.g., email).",
        node=comm_methods,
        sources=(info.reference_urls if info else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Verify that both telephone and electronic (such as email) communication are required (not just one).",
    )

    followup = evaluator.add_leaf(
        id="911_First_Followup_Deadline",
        desc="Answer provides the first follow-up notification deadline (within 2 hours of initial contact) per 47 CFR 4.9(h)(5).",
        parent=sec_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Per 47 CFR 4.9(h)(5), the first follow-up to the 911 special facility is required within 2 hours of the initial contact.",
        node=followup,
        sources=(info.reference_urls if info else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm a 2-hour follow-up window after the initial notification/contact.",
    )

    materials = evaluator.add_leaf(
        id="911_Material_Information_Elements",
        desc="Answer states that special facility notifications must include the material information elements required by 47 CFR 4.9(h)(2), and enumerates 10 required elements (consistent with the constraint).",
        parent=sec_node,
        critical=True,
    )
    # Check the answer text context for enumeration sufficiency
    await evaluator.verify(
        claim="The answer lists at least 10 distinct 'material information elements' required by 47 CFR 4.9(h)(2) for 911 special-facility notifications.",
        node=materials,
        additional_instruction=(
            "Judge based on the answer text: a correct response should explicitly enumerate about ten items for §4.9(h)(2). "
            "Allow equivalent naming/ordering, but require approximately ten distinct required elements."
        ),
    )


async def verify_compensation(evaluator: Evaluator, parent, comp: Optional[CompensationInfo]) -> None:
    comp_node = evaluator.add_parallel(
        id="Customer_Compensation",
        desc="Typical customer compensation practice for a ~10-hour outage in 2026, with a specific major-carrier example announced in January 2026.",
        parent=parent,
        critical=True,
    )

    # Typical amount (presence and concreteness judged from answer)
    typical_leaf = evaluator.add_leaf(
        id="Typical_Compensation_Amount",
        desc="Answer provides a typical compensation amount for affected customers for a ~10-hour outage (numeric/explicit amount).",
        parent=comp_node,
        critical=True,
    )
    typical_amount = (comp.typical_amount if comp and comp.typical_amount else "").strip()
    await evaluator.verify(
        claim=f"The answer provides a concrete typical compensation amount for a ~10-hour outage in 2026 (e.g., a dollar credit or a day-of-service); here it states: '{typical_amount}'.",
        node=typical_leaf,
        additional_instruction="Consider this correct if the typical amount is explicit (e.g., '$5 credit' or 'one day of service credit'). Vague language without a number or explicit benefit should be judged incorrect.",
    )

    # Example from January 2026 verified against URLs
    example_leaf = evaluator.add_leaf(
        id="January_2026_Major_Carrier_Example",
        desc="Answer provides one specific example from a major carrier’s compensation policy/announcement in January 2026, including the carrier name and the compensation amount.",
        parent=comp_node,
        critical=True,
    )
    ex_carrier = (comp.example_carrier if comp and comp.example_carrier else "").strip()
    ex_amount = (comp.example_amount if comp and comp.example_amount else "").strip()
    await evaluator.verify(
        claim=f"In January 2026, {ex_carrier} publicly announced compensation of {ex_amount} (or equivalent) for affected customers due to an outage.",
        node=example_leaf,
        sources=(comp.example_urls if comp else []),
        additional_instruction="Verify that the URL(s) substantiate a January 2026 announcement by a major U.S. carrier and clearly state the compensation amount/benefit.",
    )

    # Reference URL presence for the example
    evaluator.add_custom_node(
        result=_has_any_url(comp.example_urls if comp else []),
        id="Compensation_Reference_URL",
        desc="Answer includes at least one supporting reference URL for the compensation example/policy announcement.",
        parent=comp_node,
        critical=True,
    )


async def verify_backup_power(evaluator: Evaluator, parent, bp: Optional[BackupPowerInfo]) -> None:
    bp_node = evaluator.add_parallel(
        id="Backup_Power_Compliance",
        desc="FCC backup power requirements under 47 CFR 9.20 applicable as of January 2026, including minimum standby hours that must be offered.",
        parent=parent,
        critical=True,
    )

    citation_ok = _contains_citation(bp.citation if bp else None, "9.20")
    evaluator.add_custom_node(
        result=citation_ok,
        id="Backup_Power_Citation_Provided",
        desc="Answer provides the applicable FCC regulatory citation for backup power requirements (47 CFR 9.20 / relevant subsections).",
        parent=bp_node,
        critical=True,
    )

    ref_node = evaluator.add_custom_node(
        result=_has_any_url(bp.reference_urls if bp else []),
        id="Backup_Power_Reference_URL",
        desc="Answer includes at least one supporting reference URL for the backup power requirements.",
        parent=bp_node,
        critical=True,
    )

    # 8-hour option verification
    eight_leaf = evaluator.add_leaf(
        id="Backup_Power_8_Hour_Option",
        desc="Answer specifies the requirement to offer at least one backup power option with a minimum of 8 hours standby power per 47 CFR 9.20(a)(1).",
        parent=bp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR 9.20 requires providers to offer at least one backup power option that provides a minimum of 8 hours of standby power.",
        node=eight_leaf,
        sources=(bp.reference_urls if bp else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Confirm the rule language (accept equivalent phrasing) about an 8-hour standby option being offered to customers.",
    )

    # 24-hour option verification
    twentyfour_leaf = evaluator.add_leaf(
        id="Backup_Power_24_Hour_Option",
        desc="Answer specifies the requirement (as applicable in Jan 2026) to offer at least one backup power option with a minimum of 24 hours standby power per 47 CFR 9.20(a)(2).",
        parent=bp_node,
        critical=True,
    )
    await evaluator.verify(
        claim="47 CFR 9.20 requires providers to offer at least one backup power option that provides a minimum of 24 hours of standby power.",
        node=twentyfour_leaf,
        sources=(bp.reference_urls if bp else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Verify the existence of a 24-hour standby backup power option requirement in §9.20 (accept equivalent wording).",
    )


async def verify_tower(evaluator: Evaluator, parent, tower: Optional[TIA222Info]) -> None:
    tw_node = evaluator.add_parallel(
        id="Tower_Infrastructure_Standards",
        desc="TIA-222 tower infrastructure inspection intervals, including differences between guyed and self-supporting towers.",
        parent=parent,
        critical=True,
    )

    citation_ok = _contains_citation(tower.citation if tower else None, "TIA-222")
    evaluator.add_custom_node(
        result=citation_ok,
        id="TIA222_Standard_Citation_Provided",
        desc="Answer identifies the relevant industry standard as TIA-222 (citation/standard identification).",
        parent=tw_node,
        critical=True,
    )

    ref_node = evaluator.add_custom_node(
        result=_has_any_url(tower.reference_urls if tower else []),
        id="Tower_Inspection_Reference_URL",
        desc="Answer includes at least one supporting reference URL for the TIA-222 inspection interval requirement.",
        parent=tw_node,
        critical=True,
    )

    intervals_leaf = evaluator.add_leaf(
        id="Tower_Inspection_Intervals_By_Type",
        desc="Answer provides the minimum inspection intervals and distinguishes between guyed towers vs self-supporting structures (guyed: every 3 years minimum; self-supporting: every 5 years minimum).",
        parent=tw_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Under the TIA-222 standard (or authoritative guidance referencing it), minimum inspection intervals are: guyed towers at least every 3 years and self-supporting towers at least every 5 years.",
        node=intervals_leaf,
        sources=(tower.reference_urls if tower else []),
        extra_prerequisites=[ref_node],
        additional_instruction="Verify explicit mention of inspection frequencies differentiating guyed (3 yrs) vs self-supporting (5 yrs); credible secondary sources that quote TIA-222 are acceptable.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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

    # Create top-level critical node to mirror rubric root
    top_node = evaluator.add_parallel(
        id="Telecommunications_Outage_Regulatory_Compliance",
        desc="Evaluate whether the answer provides all requested regulatory/industry-standard requirements, numeric deadlines/thresholds, citations/standards, and supporting URLs for the specified outage scenario.",
        parent=root,
        critical=True,
    )

    # Extract structured information
    extracted: ComplianceExtraction = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction",
    )

    # Build verification subtrees
    await verify_nors(evaluator, top_node, extracted.nors if extracted else None)
    await verify_911(evaluator, top_node, extracted.special911 if extracted else None)
    await verify_compensation(evaluator, top_node, extracted.compensation if extracted else None)
    await verify_backup_power(evaluator, top_node, extracted.backup if extracted else None)
    await verify_tower(evaluator, top_node, extracted.tower if extracted else None)

    # Return evaluation summary
    return evaluator.get_summary()