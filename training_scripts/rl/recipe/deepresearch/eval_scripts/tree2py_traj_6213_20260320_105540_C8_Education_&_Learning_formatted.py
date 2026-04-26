import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_undergrad_requirements_two_institutions"
TASK_DESCRIPTION = """
Identify two undergraduate institutions in the United States that meet ALL of the following admission and program requirements for the 2024-2025 or 2025-2026 application cycle:

1. Accept either the Common Application or Coalition Application for undergraduate admissions
2. Require exactly two teacher recommendation letters from core academic subjects (mathematics, sciences, languages, or social studies)
3. Require one counselor or school official recommendation letter
4. Require a Secondary School Report with official transcript
5. Require SAT or ACT scores, OR have a clearly defined test-optional or test-flexible policy that accepts alternative standardized tests (such as AP, IB, or other exams)
6. Require a personal essay (such as the Common App essay)
7. Require at least one school-specific supplemental essay beyond the personal statement
8. Offer an Early Decision or Early Action application option with a deadline in October or November
9. Require the CSS Profile for institutional financial aid consideration
10. Have a policy requiring first-year undergraduate students to live on campus (with exceptions only for students commuting from parent/guardian homes)
11. Require a minimum of 120 credit hours for a bachelor's degree
12. Require new students to attend a mandatory orientation program

For each institution you identify, provide:
- The full official name of the institution
- Reference URL(s) from the institution's official admissions website that confirm the requirements
- A brief explanation of how the institution meets each of the twelve requirements
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class InstitutionPolicy(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

    # Extracted text snippets (strings preferred for robustness)
    application_platform: Optional[str] = None           # Common App / Coalition App acceptance
    teacher_recommendations: Optional[str] = None        # Exactly two teacher recs from core academic subjects
    counselor_recommendation: Optional[str] = None       # One counselor/school official recommendation
    transcript_requirement: Optional[str] = None         # Secondary School Report with official transcript
    testing_policy: Optional[str] = None                 # SAT/ACT required or test-optional/flexible with AP/IB alternatives
    personal_essay: Optional[str] = None                 # Personal/Common App essay requirement
    supplemental_essays: Optional[str] = None            # At least one school-specific supplemental essay
    early_option: Optional[str] = None                   # ED/EA with Oct/Nov deadline
    css_profile: Optional[str] = None                    # CSS Profile required for institutional aid
    housing_requirement: Optional[str] = None            # First-year on-campus requirement with commuter exception
    credit_hours: Optional[str] = None                   # Minimum 120 credit hours for bachelor's
    orientation: Optional[str] = None                    # Mandatory new-student orientation
    cycle_covered: Optional[str] = None                  # 2024-2025 or 2025-2026 if explicitly mentioned


class InstitutionsExtraction(BaseModel):
    institutions: List[InstitutionPolicy] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_institutions() -> str:
    return """
Extract up to two U.S. undergraduate institutions as presented in the answer. For each institution, extract:

- name: Full official name of the institution (as written in the answer)
- sources: A list of all URLs explicitly cited in the answer that serve as official university pages relevant to confirming the requirements (e.g., admissions, financial aid, housing, registrar/catalog, orientation). Include every relevant URL shown in the answer. Do not invent URLs.
- application_platform: The extracted sentence/phrase that indicates the institution accepts the Common Application or the Coalition Application.
- teacher_recommendations: The extracted sentence/phrase about teacher recommendations; look for “two teacher recommendations” or “two academic teacher evaluations” from core subjects.
- counselor_recommendation: The extracted sentence/phrase indicating one counselor or school official recommendation is required.
- transcript_requirement: The extracted sentence/phrase indicating a Secondary School Report (or Counselor Report) with official transcript is required.
- testing_policy: The extracted sentence/phrase indicating standardized testing policy: either SAT/ACT required OR a clearly defined test-optional/test-flexible policy that accepts alternative standardized tests such as AP/IB.
- personal_essay: The extracted sentence/phrase indicating a personal essay (e.g., Common App personal essay) is required.
- supplemental_essays: The extracted sentence/phrase indicating at least one school-specific supplemental essay is required beyond the personal statement.
- early_option: The extracted sentence/phrase indicating Early Decision or Early Action is offered and has an October or November deadline.
- css_profile: The extracted sentence/phrase indicating the CSS Profile is required for institutional financial aid consideration.
- housing_requirement: The extracted sentence/phrase indicating first-year students must live on campus with exceptions only for local commuters living with parents/guardians.
- credit_hours: The extracted sentence/phrase indicating minimum credits for a bachelor's degree (e.g., 120 credit hours).
- orientation: The extracted sentence/phrase indicating new students must attend a mandatory orientation program.
- cycle_covered: If the answer mentions specific cycles (2024-2025 or 2025-2026), extract that text, otherwise null.

Rules:
- Extract EXACTLY what appears in the answer. Do not infer or add information not in the answer.
- If something is missing in the answer, set that field to null.
- Return a JSON object with a single key 'institutions' which is an array of at most two InstitutionPolicy objects in the same order as presented in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def first_nonempty_name(name: Optional[str], fallback: str) -> str:
    if name and name.strip():
        return name.strip()
    return fallback


async def _add_and_verify_claim(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    url_ref_node,
    critical: bool = True,
    add_ins: Optional[str] = None,
):
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
        additional_instruction=add_ins or "None",
        extra_prerequisites=[url_ref_node] if url_ref_node else None,
    )


# --------------------------------------------------------------------------- #
# Verification for one institution                                            #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    inst: InstitutionPolicy,
    inst_index: int,
) -> None:
    inst_num = inst_index + 1
    inst_node = evaluator.add_parallel(
        id=f"Institution_{inst_num}",
        desc=f"{'First' if inst_num == 1 else 'Second'} institution meets all specified requirements",
        parent=parent_node,
        critical=False,
    )

    name = first_nonempty_name(inst.name, f"Institution #{inst_num}")
    sources = inst.sources or []

    # URL reference presence (critical)
    url_ref_node = evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"Inst{inst_num}_URL_Reference",
        desc="Official admissions website URL provided to verify requirements",
        parent=inst_node,
        critical=True
    )

    # 1) Common App or Coalition App
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Application_Platform",
        desc="Institution accepts Common Application or Coalition Application",
        claim=f"According to the cited official webpages, {name} accepts either the Common Application or the Coalition Application for first-year undergraduate admissions.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept equivalent phrasing such as 'Common App', 'Coalition for College', or 'Coalition Application'. If both are accepted, it still satisfies the requirement. Evidence should clearly indicate acceptance for undergraduate first-year applicants for the 2024–2025 or 2025–2026 cycle."
    )

    # 2) Exactly two teacher recommendations from core subjects
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Teacher_Recommendations",
        desc="Institution requires exactly two teacher recommendation letters from core academic subjects",
        claim=f"{name} requires exactly two teacher recommendations from core/academic subjects for first-year applicants.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Core academic subjects include mathematics, sciences, English/language arts, social studies/history, and world/world/foreign languages. Equivalent phrasing like 'two academic teacher evaluations' should be accepted. Policies permitting optional extra recommendations still count if two academic teacher recommendations are required."
    )

    # 3) One counselor recommendation
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Counselor_Recommendation",
        desc="Institution requires one counselor or school official recommendation",
        claim=f"{name} requires one counselor (school counselor/school official) recommendation for first-year applicants.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept terms like 'Counselor Recommendation', 'School Report recommendation', or similar."
    )

    # 4) Secondary School Report with official transcript
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Transcript_Requirements",
        desc="Institution requires Secondary School Report with official transcript",
        claim=f"{name} requires a Secondary School Report (or Counselor/School Report) that includes an official transcript for first-year applicants.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept equivalents like 'Secondary School Report', 'School Report', and language clearly indicating an official high school transcript is required."
    )

    # 5) Testing policy (SAT/ACT required OR defined test-optional/flexible accepting AP/IB alternatives)
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Testing_Policy",
        desc="Institution requires SAT/ACT or has clearly defined test-optional/test-flexible policy",
        claim=f"{name} either requires SAT or ACT scores, OR has a clearly defined test-optional or test-flexible policy that accepts alternative standardized tests (such as AP or IB exams).",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Pass if the page clearly states SAT/ACT are required OR if it clearly states a test-optional/test-flexible policy AND explicitly accepts alternative standardized tests (e.g., AP or IB) in lieu of SAT/ACT. The policy should apply to the 2024–2025 or 2025–2026 cycle."
    )

    # 6) Personal essay required
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Personal_Essay",
        desc="Institution requires personal essay as part of application",
        claim=f"{name} requires a personal essay (such as the Common App personal essay) as part of the application.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept if the institution requires the Common App personal essay or an equivalent personal statement for first-year applicants."
    )

    # 7) School-specific supplemental essays required
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Supplemental_Essays",
        desc="Institution requires at least one school-specific supplemental essay",
        claim=f"{name} requires at least one school-specific supplemental essay or short written response(s) beyond the main personal essay.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept equivalents like 'writing supplement', 'supplemental questions', or 'short answers' that are required in addition to the main personal essay."
    )

    # 8) ED/EA with October or November deadline
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Early_Application",
        desc="Institution offers Early Decision or Early Action with November or earlier deadline",
        claim=f"{name} offers either Early Decision or Early Action with at least one application deadline that falls in October or November.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept ED1 or EA deadlines in October or November (e.g., Oct 15, Nov 1, Nov 15). If multiple rounds exist, any round in Oct/Nov qualifies."
    )

    # 9) CSS Profile required for institutional aid
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_CSS_Profile",
        desc="Institution requires CSS Profile for institutional financial aid",
        claim=f"{name} requires the CSS Profile for consideration of institutional need-based financial aid.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Pass if the institution expressly requires the CSS Profile for institutional (school) aid. FAFSA alone is not sufficient to pass this check."
    )

    # 10) First-year on-campus housing requirement (commuter exception)
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Housing_Requirement",
        desc="Institution requires first-year students to live on campus",
        claim=f"{name} requires first-year undergraduate students to live on campus, with exceptions only for students commuting from a parent or guardian's home.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept if the policy is generally mandatory for first-years and exceptions are limited to living with parents/guardians nearby (commuter status) or similarly constrained exceptions."
    )

    # 11) Minimum 120 credit hours for bachelor’s degree
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Credit_Requirements",
        desc="Institution requires minimum 120 credit hours for bachelor's degree",
        claim=f"{name} requires a minimum of 120 credit hours to earn a bachelor's degree.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept language like 'minimum of 120 credits/credit hours/semester hours for a bachelor's'. If some programs exceed 120, it still passes if the general minimum is 120."
    )

    # 12) Mandatory new-student orientation
    await _add_and_verify_claim(
        evaluator, inst_node,
        node_id=f"Inst{inst_num}_Orientation",
        desc="Institution requires new students to attend orientation program",
        claim=f"{name} requires new first-year students to attend a mandatory orientation program.",
        sources=sources,
        url_ref_node=url_ref_node,
        add_ins="Accept 'mandatory', 'required', or equivalent language indicating attendance is required for new students."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator (root as non-critical parallel to allow partial credit if only one institution qualifies)
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

    # Extraction: institutions and their cited URLs + snippets
    extracted = await evaluator.extract(
        prompt=prompt_extract_institutions(),
        template_class=InstitutionsExtraction,
        extraction_name="institutions_extraction",
    )

    # Normalize to exactly two institutions (pad with empty if fewer)
    institutions: List[InstitutionPolicy] = list(extracted.institutions or [])
    if len(institutions) < 2:
        institutions = institutions + [InstitutionPolicy() for _ in range(2 - len(institutions))]
    else:
        institutions = institutions[:2]

    # Add custom info on extraction counts
    evaluator.add_custom_info(
        info={
            "extracted_institution_count": len(extracted.institutions or []),
            "used_institution_count": 2,
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Build two institution subtrees
    for idx in range(2):
        try:
            await verify_institution(evaluator, root, institutions[idx], idx)
        except Exception as e:
            # If an unexpected error occurs for one institution, record a failed custom node to avoid breaking the whole eval
            evaluator.add_custom_node(
                result=False,
                id=f"Institution_{idx+1}_unexpected_error",
                desc=f"Unexpected error while verifying Institution #{idx+1}: {str(e)}",
                parent=root,
                critical=False
            )

    # Return final structured summary
    return evaluator.get_summary()