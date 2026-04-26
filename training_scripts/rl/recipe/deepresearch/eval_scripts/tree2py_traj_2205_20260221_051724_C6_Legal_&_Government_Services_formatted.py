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
TASK_ID = "la_title44_manual"
TASK_DESCRIPTION = """
A Louisiana-based legal aid organization is developing a comprehensive training manual for paralegals who will assist clients with filing public records requests under Louisiana Revised Statutes Title 44. The manual must provide a complete, step-by-step procedural guide covering the entire process from initial request filing through final resolution.

The manual must document the following specific requirements:

1. Requester Eligibility: The minimum age requirement for individuals who may file public records requests under Louisiana law

2. Custodian Definition: The complete legal definition of "custodian" under LSA-R.S. 44:1, including all categories of individuals who may serve in this role

3. Record Classification and Exemptions:
   - The statutory definition of "public records" including all types of materials and formats covered
   - The two specific categories of documentary materials that are statutorily exempted from public records disclosure (one related to security systems, one related to school buildings)

4. Response Timeline Requirements:
   - The custodian's obligation when a public record is immediately available at the time of request
   - The deadline by which the custodian must respond when a record is not immediately available or when the custodian determines the requested material is not a public record
   - The precise method for calculating this response deadline, specifically identifying all three types of days that are excluded from the count

5. Custodian Mandatory Duties and Prohibitions:
   - The custodian's duty regarding segregation of public and non-public records
   - The explicit prohibition on custodian inquiries regarding the requester's purpose

6. Written Notification Requirement: When written notification is required and what triggers this obligation

For each requirement, the manual must provide:
- The specific legal standard or procedural requirement
- The applicable Louisiana Revised Statutes citation (LSA-R.S. Title 44)
- An official reference URL to the Louisiana State Legislature website where the statute can be verified
"""

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class RequirementTriple(BaseModel):
    """Generic triplet for one requirement item."""
    legal_standard: Optional[str] = None
    citation: Optional[str] = None
    legislature_url: Optional[str] = None


class ResponseDeadlinePart(BaseModel):
    """Specialized fields for the response deadline requirement."""
    three_day_standard: Optional[str] = None
    excluded_days_standard: Optional[str] = None
    citation: Optional[str] = None
    legislature_url: Optional[str] = None


class ManualExtraction(BaseModel):
    """All required fields extracted from the answer/manual."""
    # End-to-end procedure
    procedure_steps: List[str] = Field(default_factory=list)

    # 1. Requester Eligibility
    requester_eligibility: RequirementTriple = RequirementTriple()

    # 2. Custodian Definition (must be 44:1)
    custodian_definition: RequirementTriple = RequirementTriple()

    # 3. Record classification & exemptions
    public_records_definition: RequirementTriple = RequirementTriple()
    security_exemption: RequirementTriple = RequirementTriple()
    school_exemption: RequirementTriple = RequirementTriple()

    # 4. Response timeline requirements
    immediate_availability: RequirementTriple = RequirementTriple()
    response_deadline: ResponseDeadlinePart = ResponseDeadlinePart()

    # 5. Custodian mandatory duties & prohibitions
    segregation_duty: RequirementTriple = RequirementTriple()
    no_purpose_inquiry: RequirementTriple = RequirementTriple()

    # 6. Written notification requirement
    written_notification: RequirementTriple = RequirementTriple()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_manual_requirements() -> str:
    return """
    Extract the following structured information from the manual/answer. Return exactly the requested JSON fields.

    GENERAL SOURCE RULES:
    - For any legislature_url field, extract only official Louisiana State Legislature URLs under the domain "legis.la.gov".
    - Use only URLs explicitly present in the answer. Do not invent URLs.
    - If multiple legislature URLs are given for a single item, return the most directly relevant one to the cited statute.
    - If any requested field is missing in the answer, set it to null (or an empty list for arrays).

    FIELDS TO EXTRACT:

    0) End-to-End Procedure:
       - procedure_steps: array of strings, each briefly naming a step in sequence (from initial request filing through final resolution). If no steps are provided, return an empty array.

    1) Requester Eligibility:
       - requester_eligibility.legal_standard: the minimum age rule stated (e.g., "any person of the age of majority", "any person/no minimum age", etc.)
       - requester_eligibility.citation: the Title 44 statute citation for the eligibility rule (e.g., "LSA-R.S. 44:31")
       - requester_eligibility.legislature_url: official legis.la.gov URL that verifies the cited rule

    2) Custodian Definition (LSA-R.S. 44:1):
       - custodian_definition.legal_standard: the full definition of "custodian" (e.g., public official/head of public body with custody/control or authorized representative)
       - custodian_definition.citation: the Title 44 citation (must reference 44:1)
       - custodian_definition.legislature_url: official legis.la.gov URL to Title 44:1 verifying the definition

    3) Record Classification and Exemptions:
       3a) Public Records Definition:
           - public_records_definition.legal_standard: what materials/formats are included (including electronic records)
           - public_records_definition.citation: Title 44 citation for the definition
           - public_records_definition.legislature_url: official legis.la.gov URL verifying the definition
       3b) Security Systems Exemption:
           - security_exemption.legal_standard: the security-systems-related exemption text
           - security_exemption.citation: Title 44 citation
           - security_exemption.legislature_url: official legis.la.gov URL verifying the exemption
       3c) School Buildings Exemption:
           - school_exemption.legal_standard: the school-building blueprints/floor-plans exemption text
           - school_exemption.citation: Title 44 citation
           - school_exemption.legislature_url: official legis.la.gov URL verifying the exemption

    4) Response Timeline Requirements:
       4a) Immediate Availability:
           - immediate_availability.legal_standard: obligation when a record is immediately available at time of request
           - immediate_availability.citation: Title 44 citation
           - immediate_availability.legislature_url: official legis.la.gov URL verifying the rule
       4b) Response Deadline (Not Immediately Available OR Not a Public Record):
           - response_deadline.three_day_standard: the 3-day response rule wording for the specified situations
           - response_deadline.excluded_days_standard: the method for computing the deadline, explicitly naming the three types of excluded days (Saturdays, Sundays, legal public holidays)
           - response_deadline.citation: Title 44 citation
           - response_deadline.legislature_url: official legis.la.gov URL verifying the rule

    5) Custodian Mandatory Duties and Prohibitions:
       5a) Segregation Duty:
           - segregation_duty.legal_standard: duty to segregate public from non-public records
           - segregation_duty.citation: Title 44 citation
           - segregation_duty.legislature_url: official legis.la.gov URL verifying the duty
       5b) No Purpose Inquiry:
           - no_purpose_inquiry.legal_standard: prohibition on asking requester purpose
           - no_purpose_inquiry.citation: Title 44 citation
           - no_purpose_inquiry.legislature_url: official legis.la.gov URL verifying the prohibition

    6) Written Notification Requirement:
       - written_notification.legal_standard: when written notification is required and the trigger(s)
       - written_notification.citation: Title 44 citation
       - written_notification.legislature_url: official legis.la.gov URL verifying the requirement
    """


# --------------------------------------------------------------------------- #
# Helper functions for building verification subtrees                         #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _has_url(s: Optional[str]) -> bool:
    return _has_text(s)


async def _verify_standard_citation_url_triple(
    evaluator: Evaluator,
    parent_node,
    *,
    existence_id: str,
    existence_desc: str,
    legal_leaf_id: str,
    legal_leaf_desc: str,
    citation_leaf_id: str,
    citation_leaf_desc: str,
    url_leaf_id: str,
    url_leaf_desc: str,
    triple: RequirementTriple,
    enforce_citation_contains: Optional[str] = None,
    topic_hint: Optional[str] = None
) -> None:
    """
    Generic builder for one requirement group that has: legal_standard, citation, legislature_url.
    Adds a critical existence check gating the rest, then 3 critical verification leaves.
    """
    # Existence (critical sibling to gate subsequent leaves)
    exists = _has_text(triple.legal_standard) and _has_text(triple.citation) and _has_url(triple.legislature_url)
    evaluator.add_custom_node(
        result=exists,
        id=existence_id,
        desc=existence_desc,
        parent=parent_node,
        critical=True
    )

    # Legal standard → verify against the official statute page
    legal_node = evaluator.add_leaf(
        id=legal_leaf_id,
        desc=legal_leaf_desc,
        parent=parent_node,
        critical=True
    )
    legal_claim = f"The statute supports the following rule for {topic_hint or 'this item'}: {triple.legal_standard or ''}"
    await evaluator.verify(
        claim=legal_claim,
        node=legal_node,
        sources=triple.legislature_url,
        additional_instruction=(
            "Determine whether the quoted legal standard is explicitly supported by the provided Louisiana Legislature page. "
            "Be lenient to phrasing differences but require the substance to match the statute text. "
            "If the provided standard is missing, vague, or contradicted by the statute, mark as not supported."
        )
    )

    # Citation → check the page corresponds to the cited Title 44 section
    citation_node = evaluator.add_leaf(
        id=citation_leaf_id,
        desc=citation_leaf_desc,
        parent=parent_node,
        critical=True
    )
    citation_claim = f"The provided statute page corresponds to the citation: {triple.citation or ''} (LSA-R.S. Title 44)."
    add_ins = "Confirm the page is the cited Title 44 section or a subsection thereof; minor formatting differences are acceptable."
    if enforce_citation_contains:
        add_ins += f" Additionally, ensure the cited section includes '{enforce_citation_contains}'."
    await evaluator.verify(
        claim=citation_claim,
        node=citation_node,
        sources=triple.legislature_url,
        additional_instruction=add_ins
    )

    # Legislature URL → verify official domain/page
    url_node = evaluator.add_leaf(
        id=url_leaf_id,
        desc=url_leaf_desc,
        parent=parent_node,
        critical=True
    )
    url_claim = (
        f"The provided URL is an official Louisiana State Legislature page (legis.la.gov) "
        f"and it presents the statute relevant to {topic_hint or 'this item'}"
        + (f", including {triple.citation}." if _has_text(triple.citation) else ".")
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=triple.legislature_url,
        additional_instruction=(
            "Confirm the page is hosted on legis.la.gov and corresponds to Title 44 content relevant to the item. "
            "If the domain is not legis.la.gov or the page is unrelated, mark as not supported."
        )
    )


async def _verify_response_deadline_group(
    evaluator: Evaluator,
    parent_node,
    resp: ResponseDeadlinePart
) -> None:
    """
    Builds the subtree for response deadline + excluded-days calculation.
    """
    # Existence gate (critical)
    exists = _has_text(resp.three_day_standard) and _has_text(resp.excluded_days_standard) and _has_text(resp.citation) and _has_url(resp.legislature_url)
    evaluator.add_custom_node(
        result=exists,
        id="Response_Deadline_Fields_Present",
        desc="Required fields present for response deadline (3-day rule, excluded-days method, citation, official URL).",
        parent=parent_node,
        critical=True
    )

    # Three-day rule
    three_node = evaluator.add_leaf(
        id="Three_Day_Deadline_Legal_Standard",
        desc="States the 3-day response deadline rule for the specified situations as provided in the constraints.",
        parent=parent_node,
        critical=True
    )
    three_claim = f"The statute provides the following 3-day response rule: {resp.three_day_standard or ''}"
    await evaluator.verify(
        claim=three_claim,
        node=three_node,
        sources=resp.legislature_url,
        additional_instruction=(
            "Verify the page explicitly establishes a three-day response deadline in the situations described "
            "(record not immediately available, or determination that the requested material is not a public record)."
        )
    )

    # Excluded days calculation
    excluded_node = evaluator.add_leaf(
        id="Excluded_Days_Calculation",
        desc="States the excluded-days calculation method as provided in the constraints (exclusive of Saturdays, Sundays, and legal public holidays).",
        parent=parent_node,
        critical=True
    )
    excluded_claim = f"The statute establishes that the response deadline is calculated exclusive of these days: {resp.excluded_days_standard or ''}"
    await evaluator.verify(
        claim=excluded_claim,
        node=excluded_node,
        sources=resp.legislature_url,
        additional_instruction=(
            "Confirm that the calculation expressly excludes all three: Saturdays, Sundays, and legal public holidays. "
            "If any one of these is missing or contradicted, mark as not supported."
        )
    )

    # Citation
    citation_node = evaluator.add_leaf(
        id="Response_Deadline_Title44_Citation",
        desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the response deadline and excluded-days calculation rule.",
        parent=parent_node,
        critical=True
    )
    citation_claim = f"The provided page corresponds to the citation: {resp.citation or ''} (LSA-R.S. Title 44)."
    await evaluator.verify(
        claim=citation_claim,
        node=citation_node,
        sources=resp.legislature_url,
        additional_instruction="Confirm the page matches the cited Title 44 section/subsection for the 3-day rule and excluded-days calculation."
    )

    # Legislature URL
    url_node = evaluator.add_leaf(
        id="Response_Deadline_Legislature_URL",
        desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited response deadline and excluded-days calculation rule.",
        parent=parent_node,
        critical=True
    )
    url_claim = "The provided URL is an official Louisiana Legislature (legis.la.gov) page presenting the cited response deadline and excluded-days rule."
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=resp.legislature_url,
        additional_instruction="Confirm official domain and relevant statute content."
    )


# --------------------------------------------------------------------------- #
# Main verification construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ManualExtraction) -> None:
    """
    Build the entire verification tree according to the rubric, and run all checks.
    """
    # Top-level critical node
    doc_node = evaluator.add_parallel(
        id="Louisiana_Public_Records_Process_Documentation",
        desc="Manual addresses all specified Louisiana Title 44 public-records-request requirements. For each enumerated requirement (1–6), includes: (i) the legal/procedural standard, (ii) a Title 44 statute citation, and (iii) an official Louisiana State Legislature URL verifying the statute.",
        parent=evaluator.root,
        critical=True
    )

    # 0) End-to-End step-by-step procedure (leaf)
    step_leaf = evaluator.add_leaf(
        id="End_to_End_Step_by_Step_Procedure",
        desc="Provides a complete step-by-step procedural guide covering the entire process from initial request filing through final resolution.",
        parent=doc_node,
        critical=True
    )
    steps_summary = "; ".join(extracted.procedure_steps[:12]) if extracted.procedure_steps else ""
    step_claim = (
        "The answer includes a complete, sequential step-by-step procedural guide covering the full lifecycle "
        "from initial filing of a public records request through final resolution (including custodian actions, timing, exemptions, written notice, and resolution/appeal)."
    )
    await evaluator.verify(
        claim=step_claim,
        node=step_leaf,
        additional_instruction=(
            "Pass only if the answer lays out steps across the entire process start-to-finish. "
            "Minor omissions are acceptable only if the overall flow is clearly complete; "
            "fail if the guide is fragmentary or does not reach final resolution."
        )
    )

    # 1) Requester Eligibility
    elig_node = evaluator.add_parallel(
        id="Requester_Eligibility",
        desc="Minimum age requirement to file a public records request under Louisiana law.",
        parent=doc_node,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, elig_node,
        existence_id="Requester_Eligibility_Fields_Present",
        existence_desc="Required fields present for requester eligibility (legal standard, citation, official URL).",
        legal_leaf_id="Eligibility_Legal_Standard",
        legal_leaf_desc="States the minimum age requirement for requesters (as specified in the constraints).",
        citation_leaf_id="Eligibility_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the requester eligibility/age rule.",
        url_leaf_id="Eligibility_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL that verifies the cited requester eligibility/age rule.",
        triple=extracted.requester_eligibility,
        topic_hint="requester minimum age requirement"
    )

    # 2) Custodian Definition (44:1)
    cust_node = evaluator.add_parallel(
        id="Custodian_Definition",
        desc="Definition of “custodian” under LSA-R.S. 44:1, including all categories of individuals who may serve in this role.",
        parent=doc_node,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, cust_node,
        existence_id="Custodian_Definition_Fields_Present",
        existence_desc="Required fields present for custodian definition (legal standard, citation, official URL).",
        legal_leaf_id="Custodian_Legal_Definition",
        legal_leaf_desc="Correctly defines custodian consistent with the constraints (public official/head of public body with custody/control OR specifically authorized representative).",
        citation_leaf_id="Custodian_Citation_44_1",
        citation_leaf_desc="Cites LSA-R.S. 44:1 for the custodian definition (as required by the question).",
        url_leaf_id="Custodian_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL for LSA-R.S. 44:1 verifying the custodian definition.",
        triple=extracted.custodian_definition,
        enforce_citation_contains="44:1",
        topic_hint="custodian definition under 44:1"
    )

    # 3) Record Classification and Exemptions
    rec_main = evaluator.add_parallel(
        id="Record_Classification_and_Exemptions",
        desc="Definition of “public records” and the two specified statutory exemption categories (security-systems related; school-buildings related).",
        parent=doc_node,
        critical=True
    )

    # 3a) Public Records Definition
    pr_node = evaluator.add_parallel(
        id="Public_Records_Definition",
        desc="Statutory definition of public records including covered material types/formats (including electronic records).",
        parent=rec_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, pr_node,
        existence_id="Public_Records_Definition_Fields_Present",
        existence_desc="Required fields present for public-records definition (legal standard, citation, official URL).",
        legal_leaf_id="Public_Records_Legal_Standard",
        legal_leaf_desc="Accurately states what materials/formats are included within the statutory definition of public records (as specified in the constraints).",
        citation_leaf_id="Public_Records_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the public-records definition.",
        url_leaf_id="Public_Records_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited public-records definition.",
        triple=extracted.public_records_definition,
        topic_hint="public records definition (including electronic formats)"
    )

    # 3b) Security Systems Exemption
    sec_node = evaluator.add_parallel(
        id="Security_Systems_Exemption",
        desc="Exemption category related to security systems.",
        parent=rec_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, sec_node,
        existence_id="Security_Exemption_Fields_Present",
        existence_desc="Required fields present for security-systems exemption (legal standard, citation, official URL).",
        legal_leaf_id="Security_Exemption_Legal_Standard",
        legal_leaf_desc="Accurately states the security-systems-related documentary-materials exemption as specified in the constraints.",
        citation_leaf_id="Security_Exemption_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the security-systems exemption.",
        url_leaf_id="Security_Exemption_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited security-systems exemption.",
        triple=extracted.security_exemption,
        topic_hint="security systems exemption"
    )

    # 3c) School Buildings Exemption
    sch_node = evaluator.add_parallel(
        id="School_Buildings_Exemption",
        desc="Exemption category related to school buildings.",
        parent=rec_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, sch_node,
        existence_id="School_Exemption_Fields_Present",
        existence_desc="Required fields present for school-building exemption (legal standard, citation, official URL).",
        legal_leaf_id="School_Exemption_Legal_Standard",
        legal_leaf_desc="Accurately states the public-school blueprints/floor-plans exemption as specified in the constraints.",
        citation_leaf_id="School_Exemption_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the school-building exemption.",
        url_leaf_id="School_Exemption_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited school-building exemption.",
        triple=extracted.school_exemption,
        topic_hint="school building blueprints/floor plans exemption"
    )

    # 4) Response Timeline Requirements
    resp_main = evaluator.add_parallel(
        id="Response_Timeline_Requirements",
        desc="Custodian obligations for immediate availability, response deadline when not immediately available or not a public record, and method for calculating the deadline (excluded days).",
        parent=doc_node,
        critical=True
    )

    # 4a) Immediate Availability
    imm_node = evaluator.add_parallel(
        id="Immediate_Availability",
        desc="Obligation when record is immediately available at the time of request.",
        parent=resp_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, imm_node,
        existence_id="Immediate_Availability_Fields_Present",
        existence_desc="Required fields present for immediate-availability rule (legal standard, citation, official URL).",
        legal_leaf_id="Immediate_Availability_Legal_Standard",
        legal_leaf_desc="States the immediate-availability rule as specified in the constraints.",
        citation_leaf_id="Immediate_Availability_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the immediate-availability rule.",
        url_leaf_id="Immediate_Availability_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited immediate-availability rule.",
        triple=extracted.immediate_availability,
        topic_hint="immediate availability obligation"
    )

    # 4b) Response Deadline group
    resp_deadline_node = evaluator.add_parallel(
        id="Response_Deadline_When_Not_Immediately_Available_Or_Not_Public_Record",
        desc="Deadline by which the custodian must respond in the two specified situations and how to compute it (excluded days). (This node evaluates timing/calculation only; the separate Written_Notification_Requirement node evaluates whether notice must be written.)",
        parent=resp_main,
        critical=True
    )
    await _verify_response_deadline_group(evaluator, resp_deadline_node, extracted.response_deadline)

    # 5) Custodian Mandatory Duties and Prohibitions
    duties_main = evaluator.add_parallel(
        id="Custodian_Mandatory_Duties_and_Prohibitions",
        desc="Segregation duty and prohibition on asking requester purpose.",
        parent=doc_node,
        critical=True
    )

    # 5a) Segregation
    seg_node = evaluator.add_parallel(
        id="Segregation",
        desc="Custodian duty regarding segregation of public and non-public records.",
        parent=duties_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, seg_node,
        existence_id="Segregation_Fields_Present",
        existence_desc="Required fields present for segregation duty (legal standard, citation, official URL).",
        legal_leaf_id="Segregation_Legal_Standard",
        legal_leaf_desc="States the segregation duty as specified in the constraints (duty to segregate public records from non-public records).",
        citation_leaf_id="Segregation_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the segregation duty.",
        url_leaf_id="Segregation_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited segregation duty.",
        triple=extracted.segregation_duty,
        topic_hint="segregation of public and non-public records"
    )

    # 5b) No Purpose Inquiry
    nopi_node = evaluator.add_parallel(
        id="No_Purpose_Inquiry",
        desc="Prohibition on custodian inquiries regarding requester purpose.",
        parent=duties_main,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, nopi_node,
        existence_id="No_Purpose_Inquiry_Fields_Present",
        existence_desc="Required fields present for no-purpose-inquiry prohibition (legal standard, citation, official URL).",
        legal_leaf_id="No_Purpose_Inquiry_Legal_Standard",
        legal_leaf_desc="States the no-purpose-inquiry prohibition as specified in the constraints.",
        citation_leaf_id="No_Purpose_Inquiry_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the no-purpose-inquiry prohibition.",
        url_leaf_id="No_Purpose_Inquiry_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited no-purpose-inquiry prohibition.",
        triple=extracted.no_purpose_inquiry,
        topic_hint="prohibition on asking requester purpose"
    )

    # 6) Written Notification Requirement
    wn_node = evaluator.add_parallel(
        id="Written_Notification_Requirement",
        desc="When written notification is required and what triggers this obligation. (This node evaluates the written-notice trigger(s) only and does not re-evaluate the 3-day deadline, to avoid duplication with Response_Timeline_Requirements.)",
        parent=doc_node,
        critical=True
    )
    await _verify_standard_citation_url_triple(
        evaluator, wn_node,
        existence_id="Written_Notification_Fields_Present",
        existence_desc="Required fields present for written-notification requirement (legal standard, citation, official URL).",
        legal_leaf_id="Written_Notification_Legal_Standard",
        legal_leaf_desc="States that written notification is required, and identifies the trigger condition(s) specified in the question/constraints (e.g., determination that the requested material is not a public record; and any other written-notice trigger the question explicitly requires).",
        citation_leaf_id="Written_Notification_Title44_Citation",
        citation_leaf_desc="Provides an applicable Louisiana Revised Statutes Title 44 citation for the written-notification requirement.",
        url_leaf_id="Written_Notification_Legislature_URL",
        url_leaf_desc="Provides an official Louisiana State Legislature (legis.la.gov) URL verifying the cited written-notification requirement.",
        triple=extracted.written_notification,
        topic_hint="written notification triggers"
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
    Evaluate an answer for the Louisiana Title 44 public records manual task.
    Returns a standard evaluation summary dict.
    """
    # Initialize evaluator with a parallel root (we'll add a critical top-level node under it)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract all needed structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_manual_requirements(),
        template_class=ManualExtraction,
        extraction_name="manual_requirements",
    )

    # Record simple custom info for debugging
    evaluator.add_custom_info(
        info={
            "procedure_steps_count": len(extracted.procedure_steps),
            "has_all_groups": all([
                extracted.requester_eligibility is not None,
                extracted.custodian_definition is not None,
                extracted.public_records_definition is not None,
                extracted.security_exemption is not None,
                extracted.school_exemption is not None,
                extracted.immediate_availability is not None,
                extracted.response_deadline is not None,
                extracted.segregation_duty is not None,
                extracted.no_purpose_inquiry is not None,
                extracted.written_notification is not None,
            ])
        },
        info_type="extraction_stats",
        info_name="extraction_overview"
    )

    # Build tree and run verifications
    await build_verification_tree(evaluator, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()