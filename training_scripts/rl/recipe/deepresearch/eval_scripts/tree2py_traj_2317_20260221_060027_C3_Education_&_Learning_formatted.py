import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "mtsu_title_iv_eligibility"
TASK_DESCRIPTION = (
    "Verify whether Middle Tennessee State University is eligible to participate in Title IV federal student aid programs by confirming: "
    "(1) its institutional type classification under Title IV regulations, "
    "(2) its state legal authorization status in Tennessee, "
    "(3) its primary institutional accrediting agency from the U.S. Department of Education's recognized list, "
    "(4) its student admission policy regarding high school diploma requirements, and "
    "(5) whether it has any additional compliance requirements. "
    "Provide the official U.S. Department of Education URL that lists recognized institutional accrediting agencies as supporting evidence for the accreditation verification."
)


# ----------------------------
# Extraction Models
# ----------------------------
class TitleIVTypeInfo(BaseModel):
    institutional_type: Optional[str] = None
    institutional_type_sources: List[str] = Field(default_factory=list)


class StateAuthorizationInfo(BaseModel):
    authorized_by_name_sources: List[str] = Field(default_factory=list)
    complaint_process_sources: List[str] = Field(default_factory=list)


class AccreditationInfo(BaseModel):
    primary_agency: Optional[str] = None
    agency_sources: List[str] = Field(default_factory=list)
    usde_recognized_agency_list_url: Optional[str] = None
    extra_usde_urls: List[str] = Field(default_factory=list)


class AdmissionPolicyInfo(BaseModel):
    policy_statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComplianceInfo(BaseModel):
    two_year_rule_statement: Optional[str] = None
    two_year_rule_sources: List[str] = Field(default_factory=list)
    ppa_statement: Optional[str] = None
    ppa_sources: List[str] = Field(default_factory=list)
    bankruptcy_statement: Optional[str] = None
    bankruptcy_sources: List[str] = Field(default_factory=list)
    title_iv_criminal_statement: Optional[str] = None
    title_iv_criminal_sources: List[str] = Field(default_factory=list)


class EligibilityExtraction(BaseModel):
    institution_name: Optional[str] = None
    title_iv_type: TitleIVTypeInfo = Field(default_factory=TitleIVTypeInfo)
    state_authorization: StateAuthorizationInfo = Field(default_factory=StateAuthorizationInfo)
    accreditation: AccreditationInfo = Field(default_factory=AccreditationInfo)
    admission_policy: AdmissionPolicyInfo = Field(default_factory=AdmissionPolicyInfo)
    additional_compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)


# ----------------------------
# Extraction Prompt
# ----------------------------
def prompt_extract_eligibility() -> str:
    return (
        "Extract structured information about Middle Tennessee State University's Title IV eligibility from the answer.\n"
        "Return a JSON object with the following fields and sub-objects. Only extract information explicitly present in the answer; do not invent.\n"
        "1) institution_name: The institution's name as stated (e.g., 'Middle Tennessee State University').\n"
        "2) title_iv_type: {\n"
        "   institutional_type: The institution's classification under Title IV types as stated in the answer. Accept exact phrasing used, e.g., 'Institution of Higher Education (public)', 'Proprietary Institution of Higher Education', or 'Postsecondary Vocational Institution'. If not explicitly labeled as a Title IV type, use the institution nature phrase provided (e.g., 'public university') from the answer.\n"
        "   institutional_type_sources: All URLs cited that support this classification or the institution nature.\n"
        "}\n"
        "3) state_authorization: {\n"
        "   authorized_by_name_sources: All URLs cited that show Tennessee legally authorizes the institution by name to provide postsecondary education (e.g., Tennessee statute, THEC/TBR/Board of Trustees or state authorization page).\n"
        "   complaint_process_sources: All URLs cited that show Tennessee has a student complaint process concerning the institution (e.g., THEC consumer complaint page or SARA process applicable to Tennessee).\n"
        "}\n"
        "4) accreditation: {\n"
        "   primary_agency: The primary institutional accrediting agency named in the answer (e.g., 'Southern Association of Colleges and Schools Commission on Colleges').\n"
        "   agency_sources: All URLs cited that show the institution is accredited by that agency (e.g., institution accreditation page or the agency page listing the institution).\n"
        "   usde_recognized_agency_list_url: The single official U.S. Department of Education URL that lists recognized INSTITUTIONAL accrediting agencies (on an ed.gov domain). If not provided, return null.\n"
        "   extra_usde_urls: Any other USDE accreditation-related URLs explicitly cited in the answer (array, may be empty).\n"
        "}\n"
        "5) admission_policy: {\n"
        "   policy_statement: The statement in the answer about regular student admission requiring a high school diploma or recognized equivalent (or being beyond compulsory attendance age).\n"
        "   sources: All URLs cited for the institution's admission policy.\n"
        "}\n"
        "6) additional_compliance: {\n"
        "   two_year_rule_statement: The statement addressing the two-year rule clause (either that the institution is not proprietary/vocational, or that it has been authorized and providing the same instruction for two consecutive years).\n"
        "   two_year_rule_sources: All URLs cited for the two-year rule.\n"
        "   ppa_statement: The statement that the institution has entered into a Program Participation Agreement (PPA) signed by the president/CEO/chancellor.\n"
        "   ppa_sources: All URLs cited for the PPA.\n"
        "   bankruptcy_statement: The statement that the institution has not filed for relief in bankruptcy and does not have an order for bankruptcy.\n"
        "   bankruptcy_sources: All URLs cited for bankruptcy status.\n"
        "   title_iv_criminal_statement: The statement that the institution has not pled guilty or been found guilty of crimes involving Title IV funds.\n"
        "   title_iv_criminal_sources: All URLs cited for Title IV criminal findings.\n"
        "}\n"
        "Special URL rules: Extract only valid URLs explicitly present in the answer. Include protocol. If a URL is missing or not present in the answer, use null or empty array accordingly."
    )


# ----------------------------
# Helpers
# ----------------------------
def _inst_name(extracted: EligibilityExtraction) -> str:
    return (extracted.institution_name or "Middle Tennessee State University").strip()


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


# ----------------------------
# Verification Builder
# ----------------------------
async def build_verification_tree(evaluator: Evaluator, root, ex: EligibilityExtraction) -> None:
    inst = _inst_name(ex)

    # Top-level critical node (parallel aggregation of all required criteria)
    top = evaluator.add_parallel(
        id="Title_IV_Eligibility_Verification",
        desc="Verify that the institution meets all requirements to be eligible for Title IV federal student aid program participation per the provided constraints.",
        parent=root,
        critical=True
    )

    # 1) Institutional Type Classification (leaf, critical)
    type_node = evaluator.add_leaf(
        id="Institutional_Type_Classification",
        desc="Verify the institution is classified as one of the three eligible types under Title IV: Institution of Higher Education (public or other nonprofit), Proprietary Institution of Higher Education (private for-profit), or Postsecondary Vocational Institution (public or private nonprofit).",
        parent=top,
        critical=True
    )
    inst_type = ex.title_iv_type.institutional_type or ""
    type_sources = _unique_urls(ex.title_iv_type.institutional_type_sources)
    type_claim = (
        f"{inst} is classified under Title IV eligible types as '{inst_type}', which should correspond to one of: "
        "Institution of Higher Education (public or other nonprofit), Proprietary Institution of Higher Education (private for-profit), "
        "or Postsecondary Vocational Institution (public or private nonprofit). The provided sources support this classification."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_node,
        sources=type_sources,
        additional_instruction=(
            "Map common institutional descriptions to Title IV types. For example, a 'public university' or 'state university' "
            "awarding associate/bachelor's degrees aligns with 'Institution of Higher Education (public or other nonprofit)'. "
            "Confirm that the sources explicitly identify the institution's nature (public/nonprofit vs proprietary/for-profit vs vocational)."
        ),
    )

    # 2) State Legal Authorization (parallel group, critical)
    state_group = evaluator.add_parallel(
        id="State_Legal_Authorization",
        desc="Verify the institution meets state-authorization requirements in Tennessee.",
        parent=top,
        critical=True
    )

    # 2a) Authorized By State By Name (leaf, critical)
    auth_node = evaluator.add_leaf(
        id="Authorized_By_State_By_Name",
        desc="Verify the institution is legally authorized by name by the state to provide postsecondary education programs in that state.",
        parent=state_group,
        critical=True
    )
    auth_sources = _unique_urls(ex.state_authorization.authorized_by_name_sources)
    auth_claim = (
        f"{inst} is legally authorized by the State of Tennessee, by name, to provide postsecondary education programs in Tennessee."
    )
    await evaluator.verify(
        claim=auth_claim,
        node=auth_node,
        sources=auth_sources,
        additional_instruction=(
            "Accept evidence such as Tennessee statutes, THEC/Board pages, or the institution's official state authorization page "
            "explicitly indicating legal authorization by name in Tennessee."
        ),
    )

    # 2b) State Complaint Process (leaf, critical)
    complaint_node = evaluator.add_leaf(
        id="State_Complaint_Process",
        desc="Verify the state has a process to review and act on complaints concerning the institution.",
        parent=state_group,
        critical=True
    )
    complaint_sources = _unique_urls(ex.state_authorization.complaint_process_sources)
    complaint_claim = (
        "The State of Tennessee has a formal process to review and act on student complaints concerning the institution."
    )
    await evaluator.verify(
        claim=complaint_claim,
        node=complaint_node,
        sources=complaint_sources,
        additional_instruction=(
            "Evidence may include THEC consumer complaint pages, Tennessee SARA complaint information, or other official state pages that outline the complaint process applicable to institutions."
        ),
    )

    # 3) Accreditation Verification (parallel group, critical)
    accred_group = evaluator.add_parallel(
        id="Accreditation_Verification",
        desc="Verify the institution satisfies Title IV accreditation requirements.",
        parent=top,
        critical=True
    )
    primary_agency = (ex.accreditation.primary_agency or "").strip()
    agency_sources = _unique_urls(ex.accreditation.agency_sources)
    usde_list_url = ex.accreditation.usde_recognized_agency_list_url or ""
    usde_extra_urls = _unique_urls(ex.accreditation.extra_usde_urls)

    # 3a) Accredited By Recognized Agency (leaf, critical)
    accred_leaf = evaluator.add_leaf(
        id="Accredited_By_Recognized_Agency",
        desc="Verify the institution is accredited by a nationally recognized accrediting agency or association that appears on the U.S. Department of Education's official list of recognized institutional accrediting agencies.",
        parent=accred_group,
        critical=True
    )
    accred_claim = (
        f"{inst} is accredited by {primary_agency}, and {primary_agency} appears on the official U.S. Department of Education recognized list of INSTITUTIONAL accrediting agencies."
    )
    accred_sources = _unique_urls(agency_sources + ([usde_list_url] if usde_list_url else []) + usde_extra_urls)
    await evaluator.verify(
        claim=accred_claim,
        node=accred_leaf,
        sources=accred_sources,
        additional_instruction=(
            "Confirm two parts: (1) the agency accredits the institution (via the institution's accreditation page or the agency's listing), "
            "and (2) the agency is listed on an official USDE page of recognized INSTITUTIONAL accrediting agencies. "
            "Programmatic-only agencies do not satisfy Title IV institutional accreditation requirements."
        ),
    )

    # 3b) Primary Accrediting Agency Designation (leaf, critical)
    primary_designation_leaf = evaluator.add_leaf(
        id="Primary_Accrediting_Agency_Designation",
        desc="Verify the institution designates one accrediting agency as its primary accrediting agency.",
        parent=accred_group,
        critical=True
    )
    designation_claim = f"{inst} designates {primary_agency} as its primary institutional accrediting agency."
    await evaluator.verify(
        claim=designation_claim,
        node=primary_designation_leaf,
        sources=agency_sources,
        additional_instruction=(
            "Look for the institution's accreditation page or official disclosures indicating a single institutional accreditor (e.g., 'accredited by SACSCOC'). "
            "Wording that clearly identifies one institutional accreditor suffices."
        ),
    )

    # 3c) USDE Recognized Agency List URL (existence check + leaf, critical)
    # Existence check for URL presence (critical)
    usde_url_exists = evaluator.add_custom_node(
        result=bool(usde_list_url.strip()),
        id="USDE_Recognized_Agency_List_URL_Provided",
        desc="Official USDE recognized institutional accrediting agencies list URL is provided in the answer.",
        parent=accred_group,
        critical=True
    )
    # Verification that the provided URL is indeed the USDE page listing recognized INSTITUTIONAL accrediting agencies
    usde_list_leaf = evaluator.add_leaf(
        id="USDE_Recognized_Agency_List_URL",
        desc="Provide the official U.S. Department of Education URL that lists recognized institutional accrediting agencies as supporting evidence.",
        parent=accred_group,
        critical=True
    )
    usde_list_claim = (
        "This page explicitly lists recognized INSTITUTIONAL accrediting agencies recognized by the U.S. Department of Education."
    )
    await evaluator.verify(
        claim=usde_list_claim,
        node=usde_list_leaf,
        sources=usde_list_url if usde_list_url else None,
        extra_prerequisites=[usde_url_exists],
        additional_instruction=(
            "Verify that the URL is on an ed.gov domain and the page enumerates recognized INSTITUTIONAL accrediting agencies "
            "(e.g., HLC, MSCHE, SACSCOC, WSCUC, NECHE, NWCCU). Pages for programmatic agencies alone do not satisfy this requirement."
        ),
    )

    # 4) Student Admission Policy (leaf, critical)
    admission_leaf = evaluator.add_leaf(
        id="Student_Admission_Policy",
        desc="Verify the institution admits as regular students only individuals who have a high school diploma or its recognized equivalent, or who are beyond the age of compulsory school attendance in the state where the institution is located.",
        parent=top,
        critical=True
    )
    adm_sources = _unique_urls(ex.admission_policy.sources)
    admission_claim = (
        f"{inst} admits as regular students only individuals with a high school diploma or recognized equivalent (e.g., GED), "
        "or who are beyond Tennessee's age of compulsory school attendance."
    )
    await evaluator.verify(
        claim=admission_claim,
        node=admission_leaf,
        sources=adm_sources,
        additional_instruction=(
            "Check official undergraduate admissions pages or policies. Statements requiring a high school diploma or GED for regular admission suffice."
        ),
    )

    # 5) Additional Compliance Requirements (parallel group, critical)
    compliance_group = evaluator.add_parallel(
        id="Additional_Compliance_Requirements",
        desc="Verify additional Title IV compliance requirements specified in the constraints.",
        parent=top,
        critical=True
    )

    # 5a) Two-Year Rule Compliance (leaf, critical)
    two_year_leaf = evaluator.add_leaf(
        id="Two_Year_Rule_Compliance",
        desc="Verify either (a) the institution is not classified as a proprietary institution or postsecondary vocational institution, OR (b) if it is so classified, it has been legally authorized to provide (and has continuously been providing) the same postsecondary instruction for at least two consecutive years prior to Title IV participation.",
        parent=compliance_group,
        critical=True
    )
    type_lower = (inst_type or "").lower()
    if ("proprietary" in type_lower) or ("vocational" in type_lower):
        two_year_sources = _unique_urls(ex.additional_compliance.two_year_rule_sources)
        two_year_claim = (
            f"If classified as '{inst_type}', {inst} has been legally authorized and continuously providing the same postsecondary instruction "
            "for at least two consecutive years prior to Title IV participation."
        )
        two_year_instruction = (
            "Look for evidence (e.g., institutional history, authorization timeline, program continuity) that explicitly demonstrates at least two consecutive years of the same instruction before Title IV participation."
        )
        await evaluator.verify(
            claim=two_year_claim,
            node=two_year_leaf,
            sources=two_year_sources,
            additional_instruction=two_year_instruction,
        )
    else:
        # Clause (a): not proprietary/vocational
        two_year_sources = _unique_urls(ex.title_iv_type.institutional_type_sources)
        two_year_claim = (
            f"{inst} is not a proprietary institution and not a postsecondary vocational institution; therefore it satisfies the two-year rule via clause (a)."
        )
        two_year_instruction = (
            "Confirm that the institution is a public university or other nonprofit institution of higher education according to the sources."
        )
        await evaluator.verify(
            claim=two_year_claim,
            node=two_year_leaf,
            sources=two_year_sources,
            additional_instruction=two_year_instruction,
        )

    # 5b) Program Participation Agreement (leaf, critical)
    ppa_leaf = evaluator.add_leaf(
        id="Program_Participation_Agreement",
        desc="Verify the institution has entered into a Program Participation Agreement (PPA) and that it is signed by the institution's president, chief executive officer, or chancellor.",
        parent=compliance_group,
        critical=True
    )
    ppa_sources = _unique_urls(ex.additional_compliance.ppa_sources)
    ppa_claim = (
        f"{inst} has entered into a Title IV Program Participation Agreement (PPA), signed by the institution's president, chief executive officer, or chancellor."
    )
    await evaluator.verify(
        claim=ppa_claim,
        node=ppa_leaf,
        sources=ppa_sources,
        additional_instruction=(
            "Accept authoritative federal sources (e.g., U.S. Department of Education/Federal Student Aid documentation) that indicate a signed PPA. "
            "Institution disclosures or state/federal records explicitly confirming a signed PPA are acceptable. If sources do not establish a PPA, mark not supported."
        ),
    )

    # 5c) No Bankruptcy (leaf, critical)
    bankruptcy_leaf = evaluator.add_leaf(
        id="No_Bankruptcy",
        desc="Verify the institution has not filed for relief in bankruptcy and does not have an order for bankruptcy entered against it.",
        parent=compliance_group,
        critical=True
    )
    bankruptcy_sources = _unique_urls(ex.additional_compliance.bankruptcy_sources)
    bankruptcy_claim = (
        f"{inst} has not filed for relief in bankruptcy and does not have an order for bankruptcy entered against it."
    )
    await evaluator.verify(
        claim=bankruptcy_claim,
        node=bankruptcy_leaf,
        sources=bankruptcy_sources,
        additional_instruction=(
            "Verify via credible sources (official notices, court or government records, authoritative news or institutional disclosures) that there is no bankruptcy filing or order against the institution."
        ),
    )

    # 5d) No Title IV Criminal Findings (leaf, critical)
    criminal_leaf = evaluator.add_leaf(
        id="No_Title_IV_Criminal_Findings",
        desc="Verify the institution has not pled guilty or been found guilty of crimes involving Title IV funds.",
        parent=compliance_group,
        critical=True
    )
    criminal_sources = _unique_urls(ex.additional_compliance.title_iv_criminal_sources)
    criminal_claim = (
        f"{inst} has not pled guilty or been found guilty of crimes involving Title IV funds."
    )
    await evaluator.verify(
        claim=criminal_claim,
        node=criminal_leaf,
        sources=criminal_sources,
        additional_instruction=(
            "Check authoritative sources (U.S. Department of Education enforcement actions, DOJ press releases, significant credible news) for any criminal findings involving Title IV funds. If none exist per sources, the claim is supported."
        ),
    )


# ----------------------------
# Main evaluation entry
# ----------------------------
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=EligibilityExtraction,
        extraction_name="title_iv_eligibility_extraction"
    )

    # Add some context info for debugging and transparency
    evaluator.add_custom_info(
        info={
            "institution_name": extracted.institution_name,
            "institutional_type": extracted.title_iv_type.institutional_type,
            "primary_accrediting_agency": extracted.accreditation.primary_agency,
            "usde_recognized_agency_list_url": extracted.accreditation.usde_recognized_agency_list_url
        },
        info_type="extraction_summary",
        info_name="key_fields_summary"
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()