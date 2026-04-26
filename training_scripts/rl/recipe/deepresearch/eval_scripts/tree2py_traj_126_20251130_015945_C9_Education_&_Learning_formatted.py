import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "uc_phd_programs_eval"
TASK_DESCRIPTION = (
    "You are an international student planning to apply for fully-funded PhD programs in Computer Science or related Engineering fields at public universities in California. "
    "Identify exactly 3 universities from the University of California system that offer PhD programs in Computer Science, Computer Engineering, or Electrical Engineering and Computer Sciences. "
    "For each of the 3 universities you identify, provide the following comprehensive information: "
    "(1) The exact name of the PhD program, "
    "(2) The minimum undergraduate GPA requirement stated for admission (on a 4.0 scale), "
    "(3) Whether the program currently requires GRE scores for admission, "
    "(4) The exact number of letters of recommendation required for the application, "
    "(5) Whether international applicants must demonstrate English proficiency through TOEFL or IELTS, "
    "(6) Whether the program guarantees full funding including both stipend and tuition coverage for admitted PhD students, "
    "(7) Whether unofficial transcripts are acceptable for initial application review, "
    "(8) Whether a statement of purpose is required as part of the application, "
    "(9) The minimum GPA required to maintain eligibility for graduate assistantships during the program, "
    "(10) Whether teaching assistantships are available as a potential funding source, "
    "(11) Whether the program provides tuition waivers for doctoral students, and "
    "(12) The approximate normative time to degree completion for the PhD program. "
    "All information must be current and verifiable from official university or department websites."
)


# ---------------------------- Data Models ---------------------------------- #
class UniversityProgramInfo(BaseModel):
    campus_name: Optional[str] = None
    program_name: Optional[str] = None
    program_level: Optional[str] = None  # e.g., "PhD", "Doctor of Philosophy"
    program_field: Optional[str] = None  # e.g., "Computer Science", "EECS", "Computer Engineering"
    min_undergrad_gpa: Optional[str] = None  # e.g., "3.0 on 4.0 scale"
    gre_requirement_status: Optional[str] = None  # e.g., "required", "optional", "not accepted", "not required"
    letters_of_recommendation_count: Optional[str] = None  # keep as string for flexibility
    english_proficiency_requirement: Optional[str] = None  # e.g., "required", "not required", "waiver possible"
    full_funding_guarantee: Optional[str] = None  # e.g., "guaranteed", "not guaranteed", "varies"
    unofficial_transcripts_policy: Optional[str] = None  # e.g., "acceptable", "not acceptable"
    statement_of_purpose_requirement: Optional[str] = None  # e.g., "required", "optional"
    assistantship_eligibility_min_gpa: Optional[str] = None  # e.g., "3.0"
    teaching_assistantships_availability: Optional[str] = None  # e.g., "available", "not available"
    tuition_waiver_provision: Optional[str] = None  # e.g., "provided", "not provided"
    normative_time_to_degree: Optional[str] = None  # e.g., "5-6 years"
    official_urls: List[str] = Field(default_factory=list)  # official university/department URLs to verify


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityProgramInfo] = Field(default_factory=list)


# ------------------------- Extraction Prompts ------------------------------ #
def prompt_extract_universities() -> str:
    return (
        "Extract from the answer all University of California (UC) campuses and their PhD program details that were listed. "
        "Only include universities explicitly mentioned. For each university entry, extract the following fields:\n"
        "- campus_name: The campus name (e.g., 'University of California, Berkeley' or 'UC Berkeley').\n"
        "- program_name: The exact name of the PhD program.\n"
        "- program_level: The program level string as stated (e.g., 'PhD', 'Doctor of Philosophy').\n"
        "- program_field: The program field (e.g., 'Computer Science', 'Computer Engineering', 'Electrical Engineering and Computer Sciences').\n"
        "- min_undergrad_gpa: The minimum undergraduate GPA requirement stated for admission, on a 4.0 scale.\n"
        "- gre_requirement_status: The stated GRE status for the program (e.g., 'required', 'not required', 'optional', 'not accepted').\n"
        "- letters_of_recommendation_count: The exact number of letters of recommendation required.\n"
        "- english_proficiency_requirement: Whether international applicants must demonstrate English proficiency via TOEFL or IELTS (e.g., 'required', 'not required', 'waiver possible').\n"
        "- full_funding_guarantee: Whether the program guarantees full funding including stipend and tuition coverage (e.g., 'guaranteed', 'not guaranteed', 'varies').\n"
        "- unofficial_transcripts_policy: Whether unofficial transcripts are acceptable for initial application review (e.g., 'acceptable', 'not acceptable').\n"
        "- statement_of_purpose_requirement: Whether a statement of purpose is required (e.g., 'required', 'optional').\n"
        "- assistantship_eligibility_min_gpa: The minimum GPA to maintain eligibility for graduate assistantships during the program.\n"
        "- teaching_assistantships_availability: Whether teaching assistantships are available as a funding source (e.g., 'available', 'not available').\n"
        "- tuition_waiver_provision: Whether tuition waivers are provided for doctoral students (e.g., 'provided', 'not provided', 'depends').\n"
        "- normative_time_to_degree: The approximate normative time to PhD degree completion.\n"
        "- official_urls: A list of official university or department website URLs that were provided in the answer and can be used to verify these claims. "
        "Extract only URLs that are explicitly present in the answer; do not invent URLs.\n\n"
        "Return the result as an object with a 'universities' array containing one object per campus with these fields. "
        "If a field is not present in the answer, set it to null. If no official URLs are given for an entry, return an empty list for official_urls."
    )


# --------------------------- Helper Functions ------------------------------ #
UC_DOMAINS = {
    "berkeley.edu",
    "ucla.edu",
    "ucsd.edu",
    "uci.edu",
    "ucdavis.edu",
    "ucsb.edu",
    "ucr.edu",
    "ucsc.edu",
    "ucsf.edu",
    "ucmerced.edu",
    "ucop.edu",
}


def _hostname(url: str) -> Optional[str]:
    try:
        parsed = urlparse(url)
        return parsed.hostname
    except Exception:
        return None


def is_official_uc_url(url: str) -> bool:
    host = _hostname(url or "")
    if not host:
        return False
    return any(host.endswith(dom) for dom in UC_DOMAINS)


def dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def normalize_yes_no_flag(text: Optional[str]) -> Optional[bool]:
    if not text:
        return None
    t = text.strip().lower()
    positives = {"yes", "required", "must", "needed", "mandatory", "guaranteed", "available", "provided", "acceptable", "true"}
    negatives = {"no", "not required", "optional", "not accepted", "not guaranteed", "not available", "not provided", "unacceptable", "false"}
    if t in positives:
        return True
    if t in negatives:
        return False
    # try fuzzy
    if any(word in t for word in ["required", "must", "mandatory", "guarantee", "guaranteed"]):
        return True
    if any(word in t for word in ["optional", "not required", "not accepted", "no guarantee", "no guaranteed"]):
        return False
    return None


def first_three_unique_by_campus(items: List[UniversityProgramInfo]) -> List[UniversityProgramInfo]:
    unique = []
    seen = set()
    for it in items:
        name = (it.campus_name or "").strip()
        key = name.lower()
        if key and key not in seen:
            unique.append(it)
            seen.add(key)
        if len(unique) == 3:
            break
    return unique


# ------------------------ Verification Subtree ----------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityProgramInfo,
    idx: int,
) -> None:
    uni_node = evaluator.add_parallel(
        id=f"university_{idx + 1}",
        desc=f"University #{idx + 1} and its PhD program information",
        parent=parent_node,
        critical=False,
    )

    # Prepare sources
    sources = dedupe_urls(uni.official_urls)

    # 1) UC campus status (critical)
    uc_node = evaluator.add_leaf(
        id=f"u{idx + 1}_uc_campus_status",
        desc=f"University #{idx + 1} is a University of California (UC) system campus (public institution in California).",
        parent=uni_node,
        critical=True,
    )
    campus_name = uni.campus_name or ""
    uc_claim = (
        f"The campus '{campus_name}' is a campus of the University of California (a public university system in California)."
        if campus_name.strip()
        else "This program's official pages belong to a UC campus in California."
    )
    await evaluator.verify(
        claim=uc_claim,
        node=uc_node,
        sources=sources,
        additional_instruction="Verify from the official pages that this campus is part of the University of California system and is a public institution in California.",
    )

    # 2) Official sources provided (critical custom check)
    official_sources_node = evaluator.add_custom_node(
        result=bool(sources) and any(is_official_uc_url(u) for u in sources),
        id=f"u{idx + 1}_official_sources_provided",
        desc=f"Provides official university/department website URL(s) sufficient to verify the listed claims for University #{idx + 1}.",
        parent=uni_node,
        critical=True,
    )

    # 3) Program name provided (critical custom check)
    program_name_exists = bool(uni.program_name and uni.program_name.strip())
    evaluator.add_custom_node(
        result=program_name_exists,
        id=f"u{idx + 1}_program_name",
        desc=f"Provides the exact name of the PhD program.",
        parent=uni_node,
        critical=True,
    )

    # 4) Program is PhD-level (critical, verify on sources)
    phd_node = evaluator.add_leaf(
        id=f"u{idx + 1}_program_is_phd_level",
        desc="Confirms the program is PhD-level.",
        parent=uni_node,
        critical=True,
    )
    phd_claim = (
        f"The program '{uni.program_name}' is a PhD-level program."
        if uni.program_name
        else "The program is a PhD-level program."
    )
    await evaluator.verify(
        claim=phd_claim,
        node=phd_node,
        sources=sources,
        additional_instruction="Confirm that the program is explicitly labeled as PhD/Doctor of Philosophy on the official pages.",
        extra_prerequisites=[official_sources_node],
    )

    # 5) Program field in scope (critical)
    field_node = evaluator.add_leaf(
        id=f"u{idx + 1}_program_field_in_scope",
        desc="Confirms the program is in Computer Science, Computer Engineering, or Electrical Engineering and Computer Sciences.",
        parent=uni_node,
        critical=True,
    )
    field_str = uni.program_field or ""
    field_claim = (
        f"The program field '{field_str}' is within Computer Science, Computer Engineering, or Electrical Engineering and Computer Sciences."
        if field_str.strip()
        else "This program is within Computer Science or Computer Engineering or Electrical Engineering and Computer Sciences."
    )
    await evaluator.verify(
        claim=field_claim,
        node=field_node,
        sources=sources,
        additional_instruction="Verify the academic field of the program matches Computer Science, Computer Engineering, or Electrical Engineering & Computer Sciences.",
        extra_prerequisites=[official_sources_node],
    )

    # 6) Minimum undergrad GPA (critical)
    gpa_node = evaluator.add_leaf(
        id=f"u{idx + 1}_min_undergrad_gpa",
        desc="States the minimum undergraduate GPA requirement for admission on a 4.0 scale.",
        parent=uni_node,
        critical=True,
    )
    gpa_val = uni.min_undergrad_gpa or ""
    gpa_claim = (
        f"The minimum undergraduate GPA requirement for admission is {gpa_val} on a 4.0 scale."
        if gpa_val.strip()
        else "The minimum undergraduate GPA requirement for admission is specified on a 4.0 scale."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_node,
        sources=sources,
        additional_instruction="Check admissions pages to confirm the minimum GPA requirement stated on a 4.0 scale.",
        extra_prerequisites=[official_sources_node],
    )

    # 7) GRE requirement status (critical)
    gre_node = evaluator.add_leaf(
        id=f"u{idx + 1}_gre_requirement_status",
        desc="Clearly states GRE status (required/optional/not accepted/not required).",
        parent=uni_node,
        critical=True,
    )
    gre_status = (uni.gre_requirement_status or "").strip()
    gre_claim = (
        f"The GRE requirement status for this program is '{gre_status}'."
        if gre_status
        else "The program explicitly states its GRE requirement status (required/optional/not accepted/not required)."
    )
    await evaluator.verify(
        claim=gre_claim,
        node=gre_node,
        sources=sources,
        additional_instruction="Verify whether GRE is required, optional, not accepted, or not required, as stated on official pages.",
        extra_prerequisites=[official_sources_node],
    )

    # 8) Letters of recommendation count (critical)
    lor_node = evaluator.add_leaf(
        id=f"u{idx + 1}_letters_of_recommendation_count",
        desc="States the exact number of letters of recommendation required.",
        parent=uni_node,
        critical=True,
    )
    lor_count = (uni.letters_of_recommendation_count or "").strip()
    lor_claim = (
        f"Exactly {lor_count} letters of recommendation are required for the application."
        if lor_count
        else "The application requires an exact number of letters of recommendation (the number is specified on official pages)."
    )
    await evaluator.verify(
        claim=lor_claim,
        node=lor_node,
        sources=sources,
        additional_instruction="Locate the application requirements to confirm the exact number of recommendation letters.",
        extra_prerequisites=[official_sources_node],
    )

    # 9) English proficiency via TOEFL/IELTS (critical)
    eng_node = evaluator.add_leaf(
        id=f"u{idx + 1}_english_proficiency_requirement",
        desc="States whether international applicants must demonstrate English proficiency via TOEFL or IELTS.",
        parent=uni_node,
        critical=True,
    )
    eng_flag = normalize_yes_no_flag(uni.english_proficiency_requirement)
    if eng_flag is True:
        eng_claim = "International applicants must demonstrate English proficiency via TOEFL or IELTS."
    elif eng_flag is False:
        eng_claim = "International applicants are not required to submit TOEFL or IELTS for English proficiency."
    else:
        eng_claim = (
            f"English proficiency requirement: {uni.english_proficiency_requirement or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=eng_claim,
        node=eng_node,
        sources=sources,
        additional_instruction="Check graduate admissions or department pages for TOEFL/IELTS requirements for international applicants.",
        extra_prerequisites=[official_sources_node],
    )

    # 10) Full funding guarantee (critical)
    funding_node = evaluator.add_leaf(
        id=f"u{idx + 1}_full_funding_guarantee",
        desc="States whether the program guarantees full funding and explicitly addresses both stipend and tuition coverage (or clearly states there is no guarantee).",
        parent=uni_node,
        critical=True,
    )
    funding_flag = normalize_yes_no_flag(uni.full_funding_guarantee)
    if funding_flag is True:
        funding_claim = "The program guarantees full funding including both stipend and tuition coverage for admitted PhD students."
    elif funding_flag is False:
        funding_claim = "The program does not guarantee full funding for all admitted PhD students."
    else:
        funding_claim = (
            f"Funding guarantee status for admitted PhD students: {uni.full_funding_guarantee or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=funding_claim,
        node=funding_node,
        sources=sources,
        additional_instruction="Look for explicit statements regarding guaranteed funding, stipend, and tuition coverage for PhD students.",
        extra_prerequisites=[official_sources_node],
    )

    # 11) Unofficial transcripts acceptable (critical)
    unofficial_node = evaluator.add_leaf(
        id=f"u{idx + 1}_unofficial_transcripts_policy",
        desc="States whether unofficial transcripts are acceptable for initial application review.",
        parent=uni_node,
        critical=True,
    )
    unofficial_flag = normalize_yes_no_flag(uni.unofficial_transcripts_policy)
    if unofficial_flag is True:
        unofficial_claim = "Unofficial transcripts are acceptable for initial application review."
    elif unofficial_flag is False:
        unofficial_claim = "Unofficial transcripts are not acceptable for initial application review."
    else:
        unofficial_claim = (
            f"Unofficial transcripts policy: {uni.unofficial_transcripts_policy or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=unofficial_claim,
        node=unofficial_node,
        sources=sources,
        additional_instruction="Check the application materials section to confirm policy on unofficial transcripts for initial review.",
        extra_prerequisites=[official_sources_node],
    )

    # 12) Statement of purpose required (critical)
    sop_node = evaluator.add_leaf(
        id=f"u{idx + 1}_statement_of_purpose_requirement",
        desc="States whether a statement of purpose is required as part of the application.",
        parent=uni_node,
        critical=True,
    )
    sop_flag = normalize_yes_no_flag(uni.statement_of_purpose_requirement)
    if sop_flag is True:
        sop_claim = "A statement of purpose is required as part of the application."
    elif sop_flag is False:
        sop_claim = "A statement of purpose is not required as part of the application."
    else:
        sop_claim = (
            f"Statement of purpose requirement: {uni.statement_of_purpose_requirement or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=sop_claim,
        node=sop_node,
        sources=sources,
        additional_instruction="Verify the required application documents, specifically whether a statement of purpose is required.",
        extra_prerequisites=[official_sources_node],
    )

    # 13) Assistantship eligibility minimum GPA (critical)
    elig_node = evaluator.add_leaf(
        id=f"u{idx + 1}_assistantship_eligibility_min_gpa",
        desc="States the minimum GPA required to maintain eligibility for graduate assistantships during the program.",
        parent=uni_node,
        critical=True,
    )
    elig_gpa = (uni.assistantship_eligibility_min_gpa or "").strip()
    elig_claim = (
        f"To maintain eligibility for graduate assistantships, a minimum GPA of {elig_gpa} is required."
        if elig_gpa
        else "The minimum GPA required to maintain eligibility for graduate assistantships is stated on official pages."
    )
    await evaluator.verify(
        claim=elig_claim,
        node=elig_node,
        sources=sources,
        additional_instruction="Check graduate division or department policy pages for assistantship eligibility GPA minimum.",
        extra_prerequisites=[official_sources_node],
    )

    # 14) Teaching assistantships availability (critical)
    ta_node = evaluator.add_leaf(
        id=f"u{idx + 1}_teaching_assistantships_availability",
        desc="States whether teaching assistantships are available as a potential funding source.",
        parent=uni_node,
        critical=True,
    )
    ta_flag = normalize_yes_no_flag(uni.teaching_assistantships_availability)
    if ta_flag is True:
        ta_claim = "Teaching assistantships are available as a potential funding source."
    elif ta_flag is False:
        ta_claim = "Teaching assistantships are not available as a potential funding source."
    else:
        ta_claim = (
            f"Teaching assistantship availability: {uni.teaching_assistantships_availability or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=ta_claim,
        node=ta_node,
        sources=sources,
        additional_instruction="Verify whether TA positions are available to PhD students as funding.",
        extra_prerequisites=[official_sources_node],
    )

    # 15) Tuition waiver provision (critical)
    tuition_node = evaluator.add_leaf(
        id=f"u{idx + 1}_tuition_waiver_provision",
        desc="States whether tuition waivers are provided for doctoral students (or clearly states not provided/depends).",
        parent=uni_node,
        critical=True,
    )
    tuition_flag = normalize_yes_no_flag(uni.tuition_waiver_provision)
    if tuition_flag is True:
        tuition_claim = "Tuition waivers are provided for doctoral students."
    elif tuition_flag is False:
        tuition_claim = "Tuition waivers are not provided for doctoral students."
    else:
        tuition_claim = (
            f"Tuition waiver provision: {uni.tuition_waiver_provision or 'stated on official pages'}."
        )
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_node,
        sources=sources,
        additional_instruction="Check funding or policy pages for explicit tuition waiver information for doctoral students.",
        extra_prerequisites=[official_sources_node],
    )

    # 16) Normative time to degree (critical)
    time_node = evaluator.add_leaf(
        id=f"u{idx + 1}_normative_time_to_degree",
        desc="Provides the approximate normative time to PhD degree completion.",
        parent=uni_node,
        critical=True,
    )
    time_val = (uni.normative_time_to_degree or "").strip()
    time_claim = (
        f"The normative time to PhD degree completion is approximately {time_val}."
        if time_val
        else "The normative time to PhD degree completion is stated on official pages."
    )
    await evaluator.verify(
        claim=time_claim,
        node=time_node,
        sources=sources,
        additional_instruction="Locate the normative time to degree or typical time to completion information for the PhD program.",
        extra_prerequisites=[official_sources_node],
    )


# ----------------------------- Main Entry ---------------------------------- #
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="uc_phd_programs_extraction",
    )

    # Evaluate critical: exactly three distinct universities (no more, no fewer)
    total_entries = len(extracted.universities)
    distinct_names = [u.campus_name for u in extracted.universities if (u.campus_name or "").strip()]
    distinct_set = set(n.strip().lower() for n in distinct_names if n)
    exactly_three = (total_entries == 3) and (len(distinct_set) == 3)

    evaluator.add_custom_node(
        result=exactly_three,
        id="exactly_three_universities",
        desc="Response identifies exactly 3 distinct universities (no more, no fewer).",
        parent=root,
        critical=True,
    )

    # Filter first 3 unique universities for downstream verification
    first_three = first_three_unique_by_campus(extracted.universities)
    evaluator.add_custom_info(
        info={
            "total_extracted_entries": total_entries,
            "distinct_campus_count": len(distinct_set),
            "distinct_campuses": list(distinct_set),
            "verified_universities_count": len(first_three),
        },
        info_type="extraction_stats",
    )

    # Ensure exactly 3 nodes exist; pad with empty placeholders if fewer
    while len(first_three) < 3:
        first_three.append(UniversityProgramInfo())

    # Build university verification subtrees
    for i in range(3):
        await verify_university(evaluator, root, first_three[i], i)

    return evaluator.get_summary()