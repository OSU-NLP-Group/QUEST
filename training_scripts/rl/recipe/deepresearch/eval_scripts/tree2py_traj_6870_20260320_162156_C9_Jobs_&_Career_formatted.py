import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_k12_athletic_admin_pathway"
TASK_DESCRIPTION = """A professional with a bachelor's degree in kinesiology from an accredited university is planning a long-term career transition into K-12 athletic administration in California. Their goal is to ultimately become a high school athletic director. They need to chart a complete career pathway that includes: (1) obtaining initial teaching credentials, (2) advancing to a clear teaching credential, (3) obtaining high school coaching certifications, (4) meeting all continuing professional development requirements for credential maintenance, and (5) ultimately qualifying for an athletic director position.

For this career pathway, identify and provide detailed information about:

A. Initial Teaching Credential Requirements: What are the specific educational prerequisites, program completions, and examinations required to obtain a California Preliminary Teaching Credential, including the specific names of required tests and program approval standards?

B. Clear Credential Advancement: What is the required duration and type of program needed to advance from a Preliminary to Clear Teaching Credential in California, and what verification of teaching performance is necessary?

C. Continuing Professional Education: What are the specific hour requirements for maintaining a California teaching credential, including both annual professional development requirements and five-year renewal cycles? What specialized training areas are mandatory?

D. High School Coaching Certification: What specific coaching education courses (with course provider names), safety certifications, and minimum experience requirements are necessary to become certified as a high school coach? Reference state athletic association requirements where applicable.

E. Athletic Director Qualifications: What advanced degree (specify degree level and field of study), minimum years of administrative/coaching leadership experience, and regulatory compliance knowledge areas are required for a high school athletic director position?

For each component (A through E), provide:
- Specific names of required credentials, courses, or certifications
- Numerical requirements (years, hours, credits) where applicable
- Relevant governing body or accreditation standards
- At least one authoritative reference URL that documents these requirements
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ComponentA(BaseModel):
    authoritative_urls: List[str] = Field(default_factory=list)


class ComponentB(BaseModel):
    authoritative_urls: List[str] = Field(default_factory=list)


class ComponentC(BaseModel):
    annual_pd_statement: Optional[str] = None
    annual_pd_hours_number: Optional[str] = None
    authoritative_urls: List[str] = Field(default_factory=list)


class ComponentD(BaseModel):
    state_association_names: List[str] = Field(default_factory=list)
    authoritative_urls: List[str] = Field(default_factory=list)


class ComponentE(BaseModel):
    authoritative_urls: List[str] = Field(default_factory=list)


class CareerPathwayExtraction(BaseModel):
    global_regionally_accredited_statement: Optional[str] = None
    component_a: Optional[ComponentA] = None
    component_b: Optional[ComponentB] = None
    component_c: Optional[ComponentC] = None
    component_d: Optional[ComponentD] = None
    component_e: Optional[ComponentE] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_pathway() -> str:
    return """
    Extract structured information from the answer for each component (A–E). Return JSON matching the schema below. Follow these rules:
    - Only extract URLs that are explicitly present in the answer text. Do not infer or fabricate any URL.
    - Include URLs even if they appear as markdown links; extract the actual URL targets.
    - If a field is not mentioned, return null (for single value) or an empty list (for arrays).
    - Capture the exact text snippet if the answer explicitly states that degrees/credentials must be from "regionally accredited" institutions.

    JSON schema to produce:
    {
      "global_regionally_accredited_statement": string | null,  // the exact sentence/phrase in the answer if it explicitly states regionally accredited requirement; else null
      "component_a": {
        "authoritative_urls": string[]                          // authoritative URLs the answer cites for California Preliminary Credential requirements (CTC or equivalent)
      },
      "component_b": {
        "authoritative_urls": string[]                          // URLs cited for Preliminary->Clear advancement (CTC/induction)
      },
      "component_c": {
        "annual_pd_statement": string | null,                   // exact statement the answer makes about annual PD requirement (e.g., "no fixed annual requirement" or a number)
        "annual_pd_hours_number": string | null,                // if the answer claims a specific annual hours figure, extract just that number or short phrase (e.g., "20", "25–30"); else null
        "authoritative_urls": string[]                          // URLs cited for PD/renewal requirements
      },
      "component_d": {
        "state_association_names": string[],                    // any state HS athletic association named (e.g., "CIF", "California Interscholastic Federation")
        "authoritative_urls": string[]                          // URLs cited for high school coaching certification/safety
      },
      "component_e": {
        "authoritative_urls": string[]                          // URLs cited for athletic director qualifications
      }
    }

    Notes:
    - Keep URLs exactly as shown in the answer (prepend http:// if missing).
    - Do not add any URL that is not in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _urls_or_empty(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _contains_no_fixed_language(text: Optional[str]) -> bool:
    if not text:
        return False
    s = text.lower()
    tokens = ["no fixed", "no state", "not fixed", "no annual", "no statewide", "no set number"]
    return any(t in s for t in tokens)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_global_regionally_accredited_check(evaluator: Evaluator, parent) -> None:
    node = evaluator.add_leaf(
        id="Global_Regional_Accreditation",
        desc="States that all degrees and credentials referenced must be from regionally accredited institutions (per constraints).",
        parent=parent,
        critical=True,
    )
    claim = "The answer explicitly states that all degrees and credentials referenced must be from regionally accredited institutions."
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge this ONLY by looking at the answer text. Pass if such a statement appears clearly."
    )


async def verify_component_a(evaluator: Evaluator, parent, extraction: CareerPathwayExtraction) -> None:
    comp = extraction.component_a or ComponentA()
    urls = _urls_or_empty(comp.authoritative_urls)

    a_node = evaluator.add_parallel(
        id="A_Initial_Teaching_Credential_Requirements",
        desc="California Preliminary Teaching Credential requirements are provided, including prerequisites/exams/program requirements, and at least one authoritative URL.",
        parent=parent,
        critical=True,
    )

    # Authoritative URL existence
    a_url_exists = evaluator.add_custom_node(
        result=len(urls) > 0,
        id="A_Authoritative_URL",
        desc="Provides at least one authoritative reference URL documenting A’s requirements.",
        parent=a_node,
        critical=True,
    )

    # All other checks (critical) – will auto-skip if the above fails because it's a critical sibling
    checks = []

    n1 = evaluator.add_leaf(
        id="A_Bachelors_Degree_Prereq",
        desc="States that a bachelor’s degree is required to qualify for California teaching credentials (per constraints).",
        parent=a_node,
        critical=True,
    )
    c1 = (
        "A bachelor's degree is required to qualify for a California Preliminary Teaching Credential."
    )
    checks.append((c1, urls, n1, "Look for explicit mention that a BA/BS or higher degree is required."))

    n2 = evaluator.add_leaf(
        id="A_Commission_Approved_Teacher_Prep",
        desc="States completion of a Commission-approved teacher preparation program is required for initial teaching credentials (per constraints).",
        parent=a_node,
        critical=True,
    )
    c2 = (
        "Completion of a Commission-approved teacher preparation program is required for a California Preliminary Teaching Credential."
    )
    checks.append((c2, urls, n2, "The page should say the teacher preparation program must be Commission-approved (CTC-approved)."))

    n3 = evaluator.add_leaf(
        id="A_CTC_Oversight_Body",
        desc="Identifies the California Commission on Teacher Credentialing (CTC) as the body that oversees educator preparation programs and credential standards (per constraints).",
        parent=a_node,
        critical=True,
    )
    c3 = (
        "In California, educator preparation programs and credential standards are overseen by the California Commission on Teacher Credentialing (CTC)."
    )
    checks.append((c3, urls, n3, "Confirm the webpage identifies CTC as the official oversight body."))

    n4 = evaluator.add_leaf(
        id="A_Basic_Skills_Exam",
        desc="Names the basic skills requirement as CBEST or an approved alternative (per constraints).",
        parent=a_node,
        critical=True,
    )
    c4 = (
        "To obtain a California Preliminary Teaching Credential, the basic skills requirement must be satisfied (e.g., by passing CBEST or using an approved alternative)."
    )
    checks.append((c4, urls, n4, "Look for 'CBEST' or explicit reference to the 'basic skills requirement' and acceptable alternatives."))

    n5 = evaluator.add_leaf(
        id="A_Subject_Matter_Competence",
        desc="Names the subject matter competence pathway as CSET or an approved subject matter program (per constraints).",
        parent=a_node,
        critical=True,
    )
    c5 = (
        "Subject matter competence must be demonstrated either by passing CSET exams or completing an approved subject matter program."
    )
    checks.append((c5, urls, n5, "Look for either 'CSET' or 'approved subject matter program' as valid paths."))

    await evaluator.batch_verify(checks)


async def verify_component_b(evaluator: Evaluator, parent, extraction: CareerPathwayExtraction) -> None:
    comp = extraction.component_b or ComponentB()
    urls = _urls_or_empty(comp.authoritative_urls)

    b_node = evaluator.add_parallel(
        id="B_Clear_Credential_Advancement",
        desc="Advancement from Preliminary to Clear credential is described with required program, duration, timing, performance verification, and at least one authoritative URL.",
        parent=parent,
        critical=True,
    )

    # Authoritative URL existence
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="B_Authoritative_URL",
        desc="Provides at least one authoritative reference URL documenting B’s requirements.",
        parent=b_node,
        critical=True,
    )

    checks = []

    n1 = evaluator.add_leaf(
        id="B_Induction_Program_Required",
        desc="States that a state-approved induction program is required to advance from Preliminary to Clear (per constraints).",
        parent=b_node,
        critical=True,
    )
    c1 = "Advancement from a Preliminary to a Clear Teaching Credential requires completion of a state-approved teacher induction program."
    checks.append((c1, urls, n1, "Verify the page explicitly states induction is required for the Clear credential."))

    n2 = evaluator.add_leaf(
        id="B_Induction_Duration",
        desc="States the induction program duration is two years (per constraints).",
        parent=b_node,
        critical=True,
    )
    c2 = "California's teacher induction program typically spans two academic years."
    checks.append((c2, urls, n2, "Look for 'two years' as the standard induction duration."))

    n3 = evaluator.add_leaf(
        id="B_Enrollment_Timing",
        desc="States induction enrollment must occur within the first year of teaching under a preliminary credential (per constraints).",
        parent=b_node,
        critical=True,
    )
    c3 = "Enrollment in the induction program must occur within the teacher’s first year of teaching under a Preliminary credential."
    checks.append((c3, urls, n3, "Confirm the page indicates first-year enrollment for induction."))

    n4 = evaluator.add_leaf(
        id="B_Teaching_Performance_Verification",
        desc="States that satisfactory teaching performance must be demonstrated/verified during the induction period (per constraints).",
        parent=b_node,
        critical=True,
    )
    c4 = "Advancing to the Clear credential requires verification of satisfactory teaching performance during induction."
    checks.append((c4, urls, n4, "Look for language about performance verification/assessment during induction."))

    await evaluator.batch_verify(checks)


async def verify_component_c(evaluator: Evaluator, parent, extraction: CareerPathwayExtraction) -> None:
    comp = extraction.component_c or ComponentC()
    urls = _urls_or_empty(comp.authoritative_urls)

    c_node = evaluator.add_parallel(
        id="C_Continuing_Professional_Education",
        desc="Continuing professional development / credential maintenance requirements are provided with annual and five-year cycle requirements, mandatory training areas, and at least one authoritative URL.",
        parent=parent,
        critical=True,
    )

    # Authoritative URL existence
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="C_Authoritative_URL",
        desc="Provides at least one authoritative reference URL documenting C’s requirements.",
        parent=c_node,
        critical=True,
    )

    checks = []

    # Annual PD hours stated: dynamic claim based on the answer's statement
    n1 = evaluator.add_leaf(
        id="C_Annual_PD_Hours_Stated",
        desc="States a specific annual professional development hour requirement, OR explicitly states that no fixed annual hour requirement exists and explains what governs ongoing PD instead (as requested in the question).",
        parent=c_node,
        critical=True,
    )
    if _contains_no_fixed_language(comp.annual_pd_statement):
        c1 = ("At the state level, California does not mandate a fixed number of annual professional development hours "
              "to maintain a teaching credential; ongoing PD is typically governed by local district policies, induction/individual plans, or employer requirements.")
        add_ins1 = "Pass if the source indicates no fixed statewide annual PD hour requirement for credential maintenance and/or points to district/local governance."
    elif comp.annual_pd_hours_number and comp.annual_pd_hours_number.strip():
        c1 = f"California requires {comp.annual_pd_hours_number.strip()} hours of professional development annually for teachers to maintain credentials."
        add_ins1 = "Verify the specific annual hours stated on the source matches the quantity the answer provided."
    else:
        # Fallback: require that the answer clarifies annual PD expectations; verify using sources if possible.
        c1 = ("California's credential maintenance includes ongoing professional development expectations; sources must clarify either a specific annual hours requirement or that there is no fixed statewide annual requirement.")
        add_ins1 = "Pass if the source clarifies the annual PD expectation as either a fixed number or states that there is no fixed statewide requirement."
    checks.append((c1, urls, n1, add_ins1))

    n2 = evaluator.add_leaf(
        id="C_Five_Year_CPE_Hours",
        desc="States the five-year CPE total as 150 hours every five years for renewal (per constraints).",
        parent=c_node,
        critical=True,
    )
    c2 = "California requires 150 hours of professional growth/continuing education every five years to renew a teaching credential."
    checks.append((c2, urls, n2, "Look for explicit mention of '150 hours' in a five-year renewal period."))

    n3 = evaluator.add_leaf(
        id="C_Renewal_Cycle_Explained",
        desc="Explains the five-year renewal cycle concept/process (as requested in the question).",
        parent=c_node,
        critical=True,
    )
    c3 = "California teaching credentials renew on a five-year cycle."
    checks.append((c3, urls, n3, "Confirm the source describes a five-year renewal cycle and basic process."))

    n4 = evaluator.add_leaf(
        id="C_Mandatory_Training_Disabilities",
        desc="Includes mandatory training in teaching students with disabilities (per constraints).",
        parent=c_node,
        critical=True,
    )
    c4 = "Mandatory preparation/competencies include training related to teaching students with disabilities for California educators."
    checks.append((c4, urls, n4, "Pass if the source identifies required preparation/training addressing students with disabilities."))

    await evaluator.batch_verify(checks)


async def verify_component_d(evaluator: Evaluator, parent, extraction: CareerPathwayExtraction) -> None:
    comp = extraction.component_d or ComponentD()
    urls = _urls_or_empty(comp.authoritative_urls)

    d_node = evaluator.add_parallel(
        id="D_High_School_Coaching_Certification",
        desc="High school coaching certification pathway includes required courses/providers, safety certifications, experience requirements, state association reference where applicable, and at least one authoritative URL.",
        parent=parent,
        critical=True,
    )

    # Authoritative URL existence
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="D_Authoritative_URL",
        desc="Provides at least one authoritative reference URL documenting D’s requirements.",
        parent=d_node,
        critical=True,
    )

    checks = []

    n1 = evaluator.add_leaf(
        id="D_Coaching_Education_Course",
        desc="Names the required coaching education course as NFHS Fundamentals of Coaching or ASEP Coaching Principles (per constraints).",
        parent=d_node,
        critical=True,
    )
    c1 = ("High school coaches must complete a formal coaching education course such as NFHS 'Fundamentals of Coaching' "
           "or ASEP 'Coaching Principles' (or an equivalent accepted by the state association).")
    checks.append((c1, urls, n1, "Look for NFHS or ASEP course names required for high school coaches."))

    n2 = evaluator.add_leaf(
        id="D_Sports_First_Aid_Course",
        desc="Names required Sports First Aid training through ASEP or PREPARE (per constraints).",
        parent=d_node,
        critical=True,
    )
    c2 = ("Coaches must complete Sports First Aid training (e.g., ASEP 'Sport First Aid' or 'PREPARE' or equivalent NFHS safety/first aid coursework).")
    checks.append((c2, urls, n2, "Pass if the source requires Sports First Aid via ASEP/PREPARE or equivalent NFHS safety/first aid course."))

    n3 = evaluator.add_leaf(
        id="D_CPR_AED_Annual",
        desc="States active CPR and AED certification must be maintained annually from an approved provider (per constraints).",
        parent=d_node,
        critical=True,
    )
    c3 = "Coaches must maintain active CPR and AED certification, renewed annually, from an approved provider."
    checks.append((c3, urls, n3, "Confirm the source requires CPR/AED certification and indicates active/renewal cadence (typically annual)."))

    n4 = evaluator.add_leaf(
        id="D_Sport_Specific_Courses",
        desc="States sport-specific NFHS coaching courses are required for each sport coached (per constraints).",
        parent=d_node,
        critical=True,
    )
    c4 = "Sport-specific NFHS coaching courses (or rules clinics) are required for each sport coached."
    checks.append((c4, urls, n4, "Pass if the source requires sport-specific courses/clinics for each sport."))

    n5 = evaluator.add_leaf(
        id="D_Min_Experience",
        desc="States a minimum experience expectation of typically 2–5 years of coaching experience at high school or collegiate level (per constraints).",
        parent=d_node,
        critical=True,
    )
    c5 = "A minimum of roughly 2–5 years of coaching experience at the high school or collegiate level is typically required or preferred."
    checks.append((c5, urls, n5, "Use authoritative job/association pages indicating experience expectations."))

    # State association referenced – check in the answer itself
    n6 = evaluator.add_leaf(
        id="D_State_Association_Referenced",
        desc="References relevant state athletic association requirements where applicable (as requested in the question).",
        parent=d_node,
        critical=True,
    )
    c6 = "The answer references a relevant state high school athletic association (e.g., California Interscholastic Federation/CIF) for coaching requirements."
    # Simple verify (no external URL) because this checks whether the answer referenced the association
    await evaluator.verify(
        claim=c6,
        node=n6,
        additional_instruction="Judge by the answer text: pass if it mentions a relevant state HS athletic association such as CIF."
    )

    # Batch verify the rest (excluding the association reference already sent)
    await evaluator.batch_verify([t for t in checks])


async def verify_component_e(evaluator: Evaluator, parent, extraction: CareerPathwayExtraction) -> None:
    comp = extraction.component_e or ComponentE()
    urls = _urls_or_empty(comp.authoritative_urls)

    e_node = evaluator.add_parallel(
        id="E_Athletic_Director_Qualifications",
        desc="Athletic director qualification requirements include advanced degree level/field, experience, compliance knowledge areas, and at least one authoritative URL.",
        parent=parent,
        critical=True,
    )

    # Authoritative URL existence
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="E_Authoritative_URL",
        desc="Provides at least one authoritative reference URL documenting E’s requirements.",
        parent=e_node,
        critical=True,
    )

    checks = []

    n1 = evaluator.add_leaf(
        id="E_Advanced_Degree",
        desc="Specifies the advanced degree expectation as a master’s degree and acceptable fields (sports management/sports administration/physical education/education administration/related) and notes bachelor’s minimum / master’s preferred as applicable (per constraints).",
        parent=e_node,
        critical=True,
    )
    c1 = ("High school athletic director roles commonly require at least a bachelor's degree with a master's degree preferred "
           "in fields such as sports management, sports administration, physical education, or educational administration.")
    checks.append((c1, urls, n1, "Verify the job/association page indicates master's preferred/required in related fields."))

    n2 = evaluator.add_leaf(
        id="E_Experience_Years",
        desc="States minimum 3–5 years of coaching experience at high school or collegiate level (per constraints).",
        parent=e_node,
        critical=True,
    )
    c2 = "Athletic director positions typically require a minimum of 3–5 years of relevant coaching or athletic administrative experience."
    checks.append((c2, urls, n2, "Look for explicit experience minimums in job/association sources."))

    n3 = evaluator.add_leaf(
        id="E_Leadership_Roles",
        desc="States demonstrated leadership experience in roles such as head coach, assistant athletic director, or department coordinator (per constraints).",
        parent=e_node,
        critical=True,
    )
    c3 = "Demonstrated leadership experience (e.g., head coach, assistant athletic director, or department/program coordinator) is expected for athletic directors."
    checks.append((c3, urls, n3, "Verify job/association pages indicate leadership roles as qualifications."))

    n4 = evaluator.add_leaf(
        id="E_Compliance_NCAA",
        desc="Includes knowledge of NCAA rules and regulations where applicable (per constraints).",
        parent=e_node,
        critical=True,
    )
    c4 = "Knowledge of NCAA rules and regulations is required or strongly preferred for athletic directors."
    checks.append((c4, urls, n4, "Confirm mention of NCAA compliance knowledge in the source."))

    n5 = evaluator.add_leaf(
        id="E_Compliance_State_Association",
        desc="Includes understanding of state high school athletic association rules/requirements (per constraints).",
        parent=e_node,
        critical=True,
    )
    c5 = "Understanding of state high school athletic association rules/requirements is required for athletic directors."
    checks.append((c5, urls, n5, "Look for references to state HS association compliance (e.g., CIF)."))

    n6 = evaluator.add_leaf(
        id="E_Compliance_TitleIX",
        desc="Includes Title IX compliance/gender equity knowledge (per constraints).",
        parent=e_node,
        critical=True,
    )
    c6 = "Knowledge of Title IX compliance and gender equity is required for athletic directors."
    checks.append((c6, urls, n6, "Verify the page mentions Title IX/gender equity knowledge."))

    await evaluator.batch_verify(checks)


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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_pathway(),
        template_class=CareerPathwayExtraction,
        extraction_name="career_pathway_extraction",
    )

    # Build top-level critical node that aggregates everything
    top = evaluator.add_parallel(
        id="Career_Pathway_Requirements",
        desc="Validate that the response provides the complete A–E career pathway and, for each component, includes specific names, numerical requirements where applicable, relevant governing bodies/standards, and at least one authoritative reference URL.",
        parent=root,
        critical=True,
    )

    # 1) Global accreditation statement (in-answer presence)
    await add_global_regionally_accredited_check(evaluator, top)

    # 2) Component A
    await verify_component_a(evaluator, top, extraction)

    # 3) Component B
    await verify_component_b(evaluator, top, extraction)

    # 4) Component C
    await verify_component_c(evaluator, top, extraction)

    # 5) Component D
    await verify_component_d(evaluator, top, extraction)

    # 6) Component E
    await verify_component_e(evaluator, top, extraction)

    # Return summary
    return evaluator.get_summary()