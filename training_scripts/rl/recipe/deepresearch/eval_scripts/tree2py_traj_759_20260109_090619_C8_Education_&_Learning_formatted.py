import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# =============================== #
# Task constants                  #
# =============================== #
TASK_ID = "online_ba_bs_business_admin_single_program"
TASK_DESCRIPTION = (
    "Identify one regionally accredited online Bachelor of Science or Bachelor of Arts degree program in Business "
    "Administration offered by a U.S. university that meets ALL of the following requirements: (1) The institution "
    "holds regional accreditation from one of the seven CHEA-recognized regional accrediting organizations, verifiable "
    "through the CHEA database or U.S. Department of Education DAPIP database; (2) The business program holds specialized "
    "accreditation from either AACSB International or ACBSP; (3) Per-credit tuition cost is publicly documented and falls "
    "between $300-$400 per credit for standard undergraduate rates; (4) The total estimated program cost for 120 credits "
    "is documented and does not exceed $50,000 before financial aid; (5) The program accepts federal financial aid (FAFSA); "
    "(6) Career counseling or career services are explicitly available to online students; (7) Academic advising services are "
    "explicitly provided for online students; (8) Faculty credentials (terminal degrees in field) are documented or described; "
    "(9) The institution accepts at least 60 transfer credits from other accredited institutions; (10) Program completion "
    "timeline information is provided for full-time or part-time study; (11) The program offers asynchronous course options "
    "(courses not requiring real-time attendance); (12) Technical support availability is documented for online students; "
    "(13) Admission requirements are clearly stated with specific GPA or prerequisite requirements (or explicitly stated as "
    "having no minimum requirements); (14) The program can be completed 100% online without any required on-campus attendance. "
    "Provide the institution name, specific program name, and reference URLs documenting each of the 14 requirements."
)


# =============================== #
# Extraction models               #
# =============================== #
class ProgramEvidence(BaseModel):
    claim: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    # Identification
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_page_url: Optional[str] = None

    # Program Basics (from the question)
    basics_us: ProgramEvidence = Field(default_factory=ProgramEvidence)
    basics_degree: ProgramEvidence = Field(default_factory=ProgramEvidence)

    # Requirements 1-14 evidences (+ optional value fields for 3 and 4)
    req1_regional_accreditation: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req2_specialized_accreditation: ProgramEvidence = Field(default_factory=ProgramEvidence)

    req3_per_credit_tuition: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req3_per_credit_amount: Optional[str] = None  # e.g., "$350" or "350"

    req4_total_cost_120: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req4_total_cost_value: Optional[str] = None  # e.g., "$45,000" or "45000"

    req5_fafsa: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req6_career_services: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req7_academic_advising: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req8_faculty_credentials: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req9_transfer_credits: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req10_timeline: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req11_asynchronous: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req12_tech_support: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req13_admission_requirements: ProgramEvidence = Field(default_factory=ProgramEvidence)
    req14_fully_online: ProgramEvidence = Field(default_factory=ProgramEvidence)


# =============================== #
# Extraction prompt               #
# =============================== #
def prompt_extract_program() -> str:
    return """
    Extract exactly one target program and all associated evidence from the answer. If multiple programs are mentioned, extract only the FIRST one and ignore the others.

    REQUIRED FIELDS:
    - institution_name: The university/institution name (string)
    - program_name: The specific degree program name (string)
    - program_page_url: A main program overview URL if present (string or null)

    PROGRAM BASICS EVIDENCE:
    - basics_us: { claim, urls[] }
      • claim: A short statement from the answer asserting the institution is a U.S. university.
      • urls: URLs the answer provided to support that (prefer official institution pages).
    - basics_degree: { claim, urls[] }
      • claim: A short statement asserting the program is a BA or BS in Business Administration.
      • urls: URLs supporting the BA/BS in Business Administration from official program pages/catalogs.

    NUMBERED REQUIREMENTS EVIDENCE (1–14):
    For each req, extract:
      • claim: Short statement as presented (include concrete values if present, e.g., $350/credit or $45,000 total).
      • urls: All URLs the answer cites for that specific requirement. 
    Fields to fill:
      - req1_regional_accreditation: { claim, urls[] }
      - req2_specialized_accreditation: { claim, urls[] }
      - req3_per_credit_tuition: { claim, urls[] }
      - req3_per_credit_amount: per-credit amount (string or null), as presented (e.g., "350" or "$350")
      - req4_total_cost_120: { claim, urls[] }
      - req4_total_cost_value: total for 120 credits (string or null), as presented (e.g., "45000" or "$45,000")
      - req5_fafsa: { claim, urls[] }
      - req6_career_services: { claim, urls[] }
      - req7_academic_advising: { claim, urls[] }
      - req8_faculty_credentials: { claim, urls[] }
      - req9_transfer_credits: { claim, urls[] }
      - req10_timeline: { claim, urls[] }
      - req11_asynchronous: { claim, urls[] }
      - req12_tech_support: { claim, urls[] }
      - req13_admission_requirements: { claim, urls[] }
      - req14_fully_online: { claim, urls[] }

    RULES:
    - All URLs must be explicitly present in the answer; do not invent URLs.
    - If a field is missing in the answer, set it to null (string fields) or [] (for urls arrays).
    - Prefer official sources:
      • Institution program/catalog (.edu) pages.
      • Accreditation verification: CHEA (chea.org) or U.S. Dept. of Education DAPIP (ed.gov with 'dapip' in path) for regional; AACSB (aacsb.edu) or ACBSP (acbsp.org / acbspsearch.org) for specialized business accreditation.
    - Do not summarize from your own knowledge; extract only what the answer already provides.
    """


# =============================== #
# Helper utilities                #
# =============================== #
ACCREDITATION_DIR_DOMAINS = {
    "chea.org",
    "ed.gov",            # Require 'dapip' keyword in path for DAPIP later
    "aacsb.edu",
    "acbsp.org",
    "acbspsearch.org",
}

REGIONAL_ACCREDITORS_HINT = (
    "Recognized regional accreditors include: MSCHE, NECHE, HLC, NWCCU, SACSCOC, WSCUC, and ACCJC."
)

def _domain_from_url(u: str) -> str:
    try:
        netloc = urlparse(u if u.startswith("http") else f"http://{u}").netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        return netloc
    except Exception:
        return ""


def _is_allowed_source(url: str) -> bool:
    """
    Allowed if:
    - Official institution website (any .edu domain), OR
    - Official accreditation directories (chea.org, ed.gov for DAPIP, aacsb.edu, acbsp.org/acbspsearch.org)
    """
    d = _domain_from_url(url)
    if not d:
        return False
    if d.endswith(".edu"):
        return True
    base = d.split(".")[-2] + "." + d.split(".")[-1] if "." in d else d
    if base in ACCREDITATION_DIR_DOMAINS:
        return True
    # Also allow some regional accreditor sites (informational, but not required for req1)
    regional_accreditor_domains = {
        "hlcommission.org",
        "nwccu.org",
        "sacscoc.org",
        "neche.org",
        "wscuc.org",
        "msche.org",
        "accjc.org",
    }
    if d.endswith(tuple(regional_accreditor_domains)):
        return True
    return False


def _has_chea_or_dapip(urls: List[str]) -> bool:
    """Specific to requirement #1: must include CHEA or DAPIP (ed.gov with 'dapip' in path)."""
    for u in urls:
        d = _domain_from_url(u)
        path = urlparse(u if u.startswith("http") else f"http://{u}").path.lower()
        if d.endswith("chea.org"):
            return True
        if d.endswith("ed.gov") and "dapip" in path:
            return True
    return False


def _has_aacsb_or_acbsp(urls: List[str]) -> bool:
    """Specific to requirement #2: must include AACSB or ACBSP directory evidence."""
    for u in urls:
        d = _domain_from_url(u)
        if d.endswith("aacsb.edu") or d.endswith("acbsp.org") or d.endswith("acbspsearch.org"):
            return True
    return False


def _validate_documentation_urls(ex: ProgramExtraction) -> Tuple[bool, Dict[str, Any]]:
    """
    Check that for each required criterion (program basics + the 14 requirements), at least one URL is provided,
    and that URLs are from allowed sources. Additionally enforce req1 and req2 stricter allowed-source conditions.
    """
    criteria_to_urls: Dict[str, List[str]] = {
        "basics_us": ex.basics_us.urls or [],
        "basics_degree": ex.basics_degree.urls or [],
        "req1": ex.req1_regional_accreditation.urls or [],
        "req2": ex.req2_specialized_accreditation.urls or [],
        "req3": ex.req3_per_credit_tuition.urls or [],
        "req4": ex.req4_total_cost_120.urls or [],
        "req5": ex.req5_fafsa.urls or [],
        "req6": ex.req6_career_services.urls or [],
        "req7": ex.req7_academic_advising.urls or [],
        "req8": ex.req8_faculty_credentials.urls or [],
        "req9": ex.req9_transfer_credits.urls or [],
        "req10": ex.req10_timeline.urls or [],
        "req11": ex.req11_asynchronous.urls or [],
        "req12": ex.req12_tech_support.urls or [],
        "req13": ex.req13_admission_requirements.urls or [],
        "req14": ex.req14_fully_online.urls or [],
    }

    missing_criteria = [k for k, v in criteria_to_urls.items() if len([s for s in v if isinstance(s, str) and s.strip()]) == 0]

    invalid_urls: Dict[str, List[str]] = {}
    for k, urls in criteria_to_urls.items():
        bad = [u for u in urls if not _is_allowed_source(u)]
        if bad:
            invalid_urls[k] = bad

    # Special stricter checks for req1 and req2
    req1_ok_source = _has_chea_or_dapip(criteria_to_urls["req1"])
    req2_ok_source = _has_aacsb_or_acbsp(criteria_to_urls["req2"])

    all_good = (len(missing_criteria) == 0) and (len(invalid_urls) == 0) and req1_ok_source and req2_ok_source

    debug_info = {
        "missing_criteria": missing_criteria,
        "invalid_urls": invalid_urls,
        "req1_has_chea_or_dapip": req1_ok_source,
        "req2_has_aacsb_or_acbsp": req2_ok_source,
        "total_criteria_checked": len(criteria_to_urls),
    }
    return all_good, debug_info


def _nonempty_list(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


# =============================== #
# Verification helpers            #
# =============================== #
async def _add_and_verify_leaf(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    critical: bool,
    add_ins: str,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources if sources else None,
        additional_instruction=add_ins,
    )


def _fallback_claim(value: Optional[str], default_text: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default_text


# =============================== #
# Main evaluation routine         #
# =============================== #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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
        default_model=model,
    )

    # Extract structured program info from the answer
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build tree: Qualifying_Program (critical root child)
    qual_node = evaluator.add_parallel(
        id="Qualifying_Program",
        desc="Identify exactly one qualifying program and provide required identifying info + documentation",
        parent=root,
        critical=True,
    )

    # 1) Response Identification (critical)
    resp_id = evaluator.add_parallel(
        id="Response_Identification",
        desc="Response includes the required identifying fields for the selected program",
        parent=qual_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(extracted.institution_name and extracted.institution_name.strip()),
        id="Institution_Name_Provided",
        desc="Institution (university) name is provided",
        parent=resp_id,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(extracted.program_name and extracted.program_name.strip()),
        id="Program_Name_Provided",
        desc="Specific degree program name is provided",
        parent=resp_id,
        critical=True,
    )

    # 2) Program Basics From Question (critical)
    basics = evaluator.add_parallel(
        id="Program_Basics_From_Question",
        desc="Program matches the basic program type requested in the proposed question",
        parent=qual_node,
        critical=True,
    )
    # 2.a US University
    us_uni_claim = _fallback_claim(
        extracted.basics_us.claim,
        f"{extracted.institution_name or 'The institution'} is a U.S. university/institution.",
    )
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=basics,
        node_id="US_University",
        desc="The program is offered by a U.S. university/institution",
        claim=us_uni_claim,
        sources=_nonempty_list(extracted.basics_us.urls),
        critical=True,
        add_ins="Verify that the institution is located in the United States (address/contact/about page on *.edu is acceptable).",
    )
    # 2.b BA/BS in Business Administration
    degree_claim = _fallback_claim(
        extracted.basics_degree.claim,
        "The program is a Bachelor of Arts (BA) or Bachelor of Science (BS) in Business Administration.",
    )
    await _add_and_verify_leaf(
        evaluator=evaluator,
        parent=basics,
        node_id="BA_or_BS_in_Business_Administration",
        desc="The program is a Bachelor of Arts (BA) or Bachelor of Science (BS) in Business Administration",
        claim=degree_claim,
        sources=_nonempty_list(extracted.basics_degree.urls) or _nonempty_list([extracted.program_page_url] if extracted.program_page_url else []),
        critical=True,
        add_ins="Confirm the page shows the degree is either BA or BS and the major/discipline is Business Administration (or equivalent phrasing like 'Business Administration (BS)').",
    )

    # 3) Documentation URLs Provided and Allowed Sources (critical)
    doc_ok, doc_debug = _validate_documentation_urls(extracted)
    evaluator.add_custom_info(info=doc_debug, info_type="diagnostics", info_name="documentation_urls_check")
    evaluator.add_custom_node(
        result=doc_ok,
        id="Documentation_URLs_Provided_And_Allowed_Sources",
        desc="For each required criterion (program basics + the 14 numbered requirements), at least one publicly accessible URL is provided, and URLs come from the institution’s official website and/or official accreditation verification sources (CHEA/DAPIP and/or official specialized-accreditor directories)",
        parent=qual_node,
        critical=True,
    )

    # 4) Numbered Requirements 1–14 (critical)
    reqs_node = evaluator.add_parallel(
        id="Numbered_Requirements_1_to_14",
        desc="Meets each of the 14 explicitly listed requirements",
        parent=qual_node,
        critical=True,
    )

    # Requirement 1: Regional accreditation (CHEA or DAPIP evidence)
    req1_claim = _fallback_claim(
        extracted.req1_regional_accreditation.claim,
        f"{extracted.institution_name or 'The institution'} holds regional accreditation recognized by CHEA, verifiable via CHEA or U.S. Dept. of Education DAPIP.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "1_Regional_Accreditation_Verifiable",
        "Institution holds regional accreditation verifiable via CHEA or U.S. Dept. of Education DAPIP",
        req1_claim,
        _nonempty_list(extracted.req1_regional_accreditation.urls),
        True,
        f"Use the CHEA directory (chea.org) or DAPIP (ed.gov with 'dapip' path) to confirm accreditation. {REGIONAL_ACCREDITORS_HINT}",
    )

    # Requirement 2: Specialized business accreditation (AACSB or ACBSP)
    req2_claim = _fallback_claim(
        extracted.req2_specialized_accreditation.claim,
        "The business program (or the business school housing it) holds specialized accreditation from AACSB or ACBSP.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "2_Specialized_Business_Accreditation",
        "Business program holds specialized accreditation from AACSB or ACBSP",
        req2_claim,
        _nonempty_list(extracted.req2_specialized_accreditation.urls),
        True,
        "Verify in AACSB (aacsb.edu) or ACBSP (acbsp.org / acbspsearch.org) directories. School-level business accreditation covering the bachelor's program is acceptable.",
    )

    # Requirement 3: Per-credit $300-$400 and documented
    per_credit_text = extracted.req3_per_credit_amount or ""
    req3_default = "Standard undergraduate per-credit tuition is publicly documented and is between $300 and $400."
    req3_claim = _fallback_claim(
        extracted.req3_per_credit_tuition.claim,
        req3_default if not per_credit_text else f"Standard undergraduate per-credit tuition is {per_credit_text}, which is between $300 and $400."
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "3_Per_Credit_Tuition_300_to_400",
        "Standard undergraduate per-credit tuition is publicly documented on the institution’s website and is between $300 and $400 per credit",
        req3_claim,
        _nonempty_list(extracted.req3_per_credit_tuition.urls),
        True,
        "Confirm the page shows an undergraduate online per-credit tuition in the $300–$400 range (inclusive). If multiple rates exist, use the standard online undergraduate rate.",
    )

    # Requirement 4: Total cost for 120 credits ≤ $50,000 and documented
    total_cost_text = extracted.req4_total_cost_value or ""
    req4_default = "The total estimated program cost for 120 credits is documented and does not exceed $50,000."
    req4_claim = _fallback_claim(
        extracted.req4_total_cost_120.claim,
        req4_default if not total_cost_text else f"The total estimated program cost for 120 credits is documented as {total_cost_text}, which does not exceed $50,000."
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "4_Total_Cost_120_Credits_At_Most_50000",
        "Total estimated program cost for 120 credits is documented and does not exceed $50,000 before financial aid/scholarships",
        req4_claim,
        _nonempty_list(extracted.req4_total_cost_120.urls),
        True,
        "Verify that an official page documents a total cost for a 120-credit bachelor's program and that it is ≤ $50,000. Pages that only give per-credit without computing a 120-credit total do not satisfy this requirement.",
    )

    # Requirement 5: FAFSA accepted
    req5_claim = _fallback_claim(
        extracted.req5_fafsa.claim,
        "The institution accepts federal financial aid (FAFSA).",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "5_FAFSA_Accepted",
        "Program/institution accepts federal financial aid (FAFSA)",
        req5_claim,
        _nonempty_list(extracted.req5_fafsa.urls),
        True,
        "Look for official financial aid pages stating FAFSA participation or Title IV eligibility.",
    )

    # Requirement 6: Career services for online students
    req6_claim = _fallback_claim(
        extracted.req6_career_services.claim,
        "Career services are explicitly available to online students.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "6_Career_Services_For_Online_Students",
        "Career counseling/services are explicitly available to online students",
        req6_claim,
        _nonempty_list(extracted.req6_career_services.urls),
        True,
        "The page should explicitly indicate availability to online students; if only on-campus-only services are shown, do not pass.",
    )

    # Requirement 7: Academic advising for online students
    req7_claim = _fallback_claim(
        extracted.req7_academic_advising.claim,
        "Academic advising services are explicitly provided for online students.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "7_Academic_Advising_For_Online_Students",
        "Academic advising services are explicitly provided for online students",
        req7_claim,
        _nonempty_list(extracted.req7_academic_advising.urls),
        True,
        "Confirm that online students have access to academic advising (explicit mention of online students or all students including online).",
    )

    # Requirement 8: Faculty credentials documented
    req8_claim = _fallback_claim(
        extracted.req8_faculty_credentials.claim,
        "Faculty credentials (e.g., terminal degrees in the field and/or professional qualifications) are documented or described.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "8_Faculty_Credentials_Documented",
        "Faculty credentials (e.g., terminal degrees in field and/or professional qualifications) are documented or described",
        req8_claim,
        _nonempty_list(extracted.req8_faculty_credentials.urls),
        True,
        "Verify faculty profiles or program/school pages that document degrees (e.g., PhD/DBA) or equivalent credentials.",
    )

    # Requirement 9: Accept at least 60 transfer credits
    req9_claim = _fallback_claim(
        extracted.req9_transfer_credits.claim,
        "The institution accepts at least 60 transfer credits from other accredited institutions.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "9_Transfer_Credits_At_Least_60_From_Regionally_Accredited",
        "Institution accepts at least 60 transfer credits from other regionally accredited institutions and the numeric transfer-credit policy is publicly documented",
        req9_claim,
        _nonempty_list(extracted.req9_transfer_credits.urls),
        True,
        "Look for transfer policies indicating a maximum accepted transfer credit count of ≥ 60 toward the bachelor's degree.",
    )

    # Requirement 10: Program timeline provided
    req10_claim = _fallback_claim(
        extracted.req10_timeline.claim,
        "Program completion timeline information is provided for full-time and/or part-time study.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "10_Program_Timeline_Provided",
        "Program completion timeline information is provided for full-time and/or part-time study",
        req10_claim,
        _nonempty_list(extracted.req10_timeline.urls),
        True,
        "Confirm that an official page provides time-to-completion or pacing details for full-time or part-time paths.",
    )

    # Requirement 11: Asynchronous options
    req11_claim = _fallback_claim(
        extracted.req11_asynchronous.claim,
        "The program offers asynchronous course options (no required real-time attendance).",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "11_Asynchronous_Options",
        "Program offers asynchronous course options (no required real-time attendance)",
        req11_claim,
        _nonempty_list(extracted.req11_asynchronous.urls),
        True,
        "Verify that asynchronous course delivery is offered; if only synchronous/live attendance is required, do not pass.",
    )

    # Requirement 12: Technical support documented
    req12_claim = _fallback_claim(
        extracted.req12_tech_support.claim,
        "Technical support availability for online students is documented.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "12_Technical_Support_Documented",
        "Technical support availability for online students is documented (e.g., hours and/or access methods)",
        req12_claim,
        _nonempty_list(extracted.req12_tech_support.urls),
        True,
        "Confirm IT/tech help resources specific to online students (service hours, contact methods, helpdesk portal).",
    )

    # Requirement 13: Admission requirements
    req13_claim = _fallback_claim(
        extracted.req13_admission_requirements.claim,
        "Admission requirements are clearly stated and include specific GPA/prerequisites or explicitly state there is no minimum requirement.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "13_Admission_Requirements_Specific_Or_None",
        "Admission requirements are clearly stated and include specific GPA/prerequisites or explicitly state there is no minimum requirement",
        req13_claim,
        _nonempty_list(extracted.req13_admission_requirements.urls),
        True,
        "Check undergraduate admissions/program pages for a stated minimum GPA/prerequisites or an explicit statement that there is no minimum.",
    )

    # Requirement 14: 100% online, no on-campus
    req14_claim = _fallback_claim(
        extracted.req14_fully_online.claim,
        "The program can be completed 100% online with no required on-campus attendance.",
    )
    await _add_and_verify_leaf(
        evaluator, reqs_node, "14_100_Percent_Online_No_On_Campus",
        "Program can be completed 100% online with no required on-campus attendance/residency",
        req14_claim,
        _nonempty_list(extracted.req14_fully_online.urls),
        True,
        "Confirm that there are no required on-campus residencies, labs, or proctored sessions necessitating physical presence.",
    )

    # Final summary
    return evaluator.get_summary()