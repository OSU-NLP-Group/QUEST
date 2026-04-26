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
TASK_ID = "four_public_universities_big12_acc_criteria"
TASK_DESCRIPTION = """
Identify four (4) public universities in the United States that currently satisfy ALL of the following criteria: 
(1) The university is a member of either the Big 12 Conference or the Atlantic Coast Conference for NCAA Division I athletics; 
(2) The university has an AACSB-accredited business school or college; 
(3) The university has at least one ABET-accredited undergraduate engineering program; 
(4) The university's total student enrollment (undergraduate plus graduate combined) is at least 20,000 students; 
(5) Graduate students comprise at least 15% of the university's total enrollment; 
(6) Out-of-state undergraduate students comprise at least 30% of the total undergraduate enrollment; 
(7) The university offers differentiated tuition rates for in-state versus out-of-state undergraduate students; 
(8) The in-state undergraduate tuition (excluding mandatory fees) is less than $20,000 per academic year for full-time enrollment; 
(9) The out-of-state undergraduate tuition is at least twice (2.0×) the in-state undergraduate tuition rate. 
For each university, provide its name, current athletic conference affiliation, confirmation of AACSB and ABET accreditation status, current enrollment statistics including total enrollment and graduate enrollment percentage, out-of-state undergraduate enrollment percentage, and current academic year tuition rates for both in-state and out-of-state undergraduate students.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    name: Optional[str] = None
    athletic_conference: Optional[str] = None  # e.g., "Big 12 Conference" or "Atlantic Coast Conference (ACC)"
    aacsb_accredited: Optional[str] = None     # e.g., "yes", "AACSB accredited", or details
    abet_accredited: Optional[str] = None      # e.g., "yes", "ABET accredited program(s)", or details
    total_enrollment: Optional[str] = None     # e.g., "25,000", "about 30,000"
    graduate_enrollment_percent: Optional[str] = None  # e.g., "18%", "0.18", "around 20%"
    out_of_state_undergrad_percent: Optional[str] = None  # e.g., "35%", "0.35"
    in_state_tuition: Optional[str] = None      # e.g., "$12,500", "USD 18,000 (tuition only)"
    out_of_state_tuition: Optional[str] = None  # e.g., "$32,000"
    differentiated_tuition: Optional[str] = None  # e.g., "yes", "different in-state vs out-of-state"
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer for this university


class FourUniversitiesExtraction(BaseModel):
    universities: List[UniversityExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) universities as presented in the answer text. For each university, extract the following fields exactly as stated in the answer:
    - name: the full university name
    - athletic_conference: the named NCAA conference (e.g., "Big 12 Conference", "Atlantic Coast Conference", "ACC")
    - aacsb_accredited: any statement indicating AACSB accreditation of the business school/college
    - abet_accredited: any statement indicating at least one ABET-accredited undergraduate engineering program
    - total_enrollment: the total (undergraduate + graduate) student enrollment figure or phrase
    - graduate_enrollment_percent: the graduate student percentage of total enrollment (include % sign if present)
    - out_of_state_undergrad_percent: the out-of-state undergraduate percentage of undergraduate enrollment (include % sign if present)
    - in_state_tuition: the in-state undergraduate tuition rate per academic year (tuition only, excluding fees if that is what the answer claims)
    - out_of_state_tuition: the out-of-state undergraduate tuition rate per academic year
    - differentiated_tuition: a brief phrase indicating whether in-state vs out-of-state rates differ (e.g., "yes", "different", etc.)
    - sources: a list of all URLs in the answer that are associated with this university (include conference/athletics pages, AACSB directory pages, ABET listings, institutional fact books/dashboards, and tuition pages as cited in the answer)
    
    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - Preserve number formatting and currency symbols in string form (e.g., "$18,500", "27%").
    - For percentages, include the percent sign if present in the answer.
    - For tuition, prefer the tuition-only figures when the answer states "excluding fees".
    - For sources, include every actual URL tied to that specific university in the answer text. Do not invent URLs. Extract valid URLs only.
    - If any field is missing, set it to null (except 'sources' which should be an empty list if none are cited).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for u in urls or []:
        if not u:
            continue
        u_stripped = u.strip()
        if u_stripped and u_stripped not in seen:
            seen.add(u_stripped)
            cleaned.append(u_stripped)
    return cleaned


def _safe_name(name: Optional[str], idx: int) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else f"University #{idx + 1}"


def _normalize_conference_str(conf: Optional[str]) -> Optional[str]:
    if not conf:
        return None
    c = conf.lower()
    if "big 12" in c:
        return "Big 12 Conference"
    if "atlantic coast" in c or "acc" in c:
        return "Atlantic Coast Conference"
    return conf


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityExtraction,
    uni_index: int,
) -> None:
    """
    Build the verification subtree for a single university and run verifications.
    All verification leaf nodes are critical per the rubric. A gating "has name & sources" node is added to enforce source-grounding.
    """

    # Create a parallel node grouping all checks for this university
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_index+1}",
        desc=f"University #{uni_index + 1} verification - meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Gate: Ensure minimal provenance exists (name + at least one URL)
    urls = _dedup_urls(uni.sources or [])
    has_minimal = (uni.name is not None and uni.name.strip() != "") and (len(urls) > 0)
    evaluator.add_custom_node(
        result=has_minimal,
        id=f"u{uni_index}_has_name_sources",
        desc=f"University #{uni_index + 1}: Name and at least one cited source URL are provided",
        parent=uni_node,
        critical=True
    )

    name_for_claims = _safe_name(uni.name, uni_index)

    # 1) Public Institution in the U.S.
    node_public = evaluator.add_leaf(
        id=f"u{uni_index}_public_institution",
        desc="University is a public institution in the United States",
        parent=uni_node,
        critical=True
    )
    claim_public = f"{name_for_claims} is a public university in the United States."
    await evaluator.verify(
        claim=claim_public,
        node=node_public,
        sources=urls,
        additional_instruction="Verify that the institution is publicly funded and recognized as a public university in the U.S. Use official or authoritative sources cited in the answer."
    )

    # 2) Conference Membership: Big 12 or ACC
    node_conf = evaluator.add_leaf(
        id=f"u{uni_index}_conference_membership",
        desc="Currently a member of Big 12 Conference or Atlantic Coast Conference",
        parent=uni_node,
        critical=True
    )
    normalized_conf = _normalize_conference_str(uni.athletic_conference)
    if normalized_conf in {"Big 12 Conference", "Atlantic Coast Conference"}:
        claim_conf = f"{name_for_claims} is currently a member of the {normalized_conf} in NCAA Division I athletics."
    else:
        claim_conf = f"{name_for_claims} is currently a member of either the Big 12 Conference or the Atlantic Coast Conference (ACC) in NCAA Division I athletics."
    await evaluator.verify(
        claim=claim_conf,
        node=node_conf,
        sources=urls,
        additional_instruction="Confirm CURRENT conference membership. Accept either 'Big 12 Conference' or 'Atlantic Coast Conference (ACC)'."
    )

    # 3) AACSB accreditation
    node_aacsb = evaluator.add_leaf(
        id=f"u{uni_index}_aacsb_accreditation",
        desc="Business school holds current AACSB accreditation",
        parent=uni_node,
        critical=True
    )
    claim_aacsb = f"{name_for_claims}'s business school/college holds current AACSB accreditation."
    await evaluator.verify(
        claim=claim_aacsb,
        node=node_aacsb,
        sources=urls,
        additional_instruction="Look for AACSB directory/listing or the school's accreditation statement on cited sources. 'AACSB accredited' must be explicit."
    )

    # 4) ABET accreditation (at least one undergraduate engineering program)
    node_abet = evaluator.add_leaf(
        id=f"u{uni_index}_abet_accreditation",
        desc="At least one ABET-accredited undergraduate engineering program exists",
        parent=uni_node,
        critical=True
    )
    claim_abet = f"{name_for_claims} has at least one ABET-accredited undergraduate engineering program."
    await evaluator.verify(
        claim=claim_abet,
        node=node_abet,
        sources=urls,
        additional_instruction="Use ABET Accredited Program Search or official program pages as cited. The accreditation must explicitly be ABET at the bachelor's level."
    )

    # 5) Total enrollment >= 20,000
    node_enroll = evaluator.add_leaf(
        id=f"u{uni_index}_total_enrollment_minimum",
        desc="Total enrollment (undergraduate plus graduate) is at least 20,000 students",
        parent=uni_node,
        critical=True
    )
    if uni.total_enrollment:
        claim_enroll = f"{name_for_claims} has total student enrollment of {uni.total_enrollment}, which is at least 20,000."
    else:
        claim_enroll = f"{name_for_claims} has total student enrollment of at least 20,000."
    await evaluator.verify(
        claim=claim_enroll,
        node=node_enroll,
        sources=urls,
        additional_instruction="Verify the most recent total enrollment figure (UG + Grad combined) reported on the cited sources is ≥ 20,000."
    )

    # 6) Graduate students >= 15% of total enrollment
    node_grad = evaluator.add_leaf(
        id=f"u{uni_index}_graduate_enrollment_percentage",
        desc="Graduate students comprise at least 15% of total enrollment",
        parent=uni_node,
        critical=True
    )
    if uni.graduate_enrollment_percent:
        claim_grad = f"Graduate students comprise at least 15% of total enrollment at {name_for_claims}; the answer reports {uni.graduate_enrollment_percent}."
    else:
        claim_grad = f"Graduate students comprise at least 15% of total enrollment at {name_for_claims}."
    await evaluator.verify(
        claim=claim_grad,
        node=node_grad,
        sources=urls,
        additional_instruction="Verify a graduate share ≥ 15% based on cited institutional stats (fact book/dashboards)."
    )

    # 7) Out-of-state undergrads >= 30% of undergraduate enrollment
    node_oos = evaluator.add_leaf(
        id=f"u{uni_index}_out_of_state_percentage",
        desc="Out-of-state undergraduates comprise at least 30% of undergraduate enrollment",
        parent=uni_node,
        critical=True
    )
    if uni.out_of_state_undergrad_percent:
        claim_oos = f"Out-of-state undergraduates comprise at least 30% of undergraduate enrollment at {name_for_claims}; the answer reports {uni.out_of_state_undergrad_percent}."
    else:
        claim_oos = f"Out-of-state undergraduates comprise at least 30% of undergraduate enrollment at {name_for_claims}."
    await evaluator.verify(
        claim=claim_oos,
        node=node_oos,
        sources=urls,
        additional_instruction="Confirm that out-of-state UG share is ≥ 30% using cited sources (fact books, common data sets, or official dashboards)."
    )

    # 8) Differentiated tuition exists (in-state vs out-of-state)
    node_diff = evaluator.add_leaf(
        id=f"u{uni_index}_differentiated_tuition_exists",
        desc="University offers different tuition rates for in-state versus out-of-state undergraduate students",
        parent=uni_node,
        critical=True
    )
    if uni.in_state_tuition and uni.out_of_state_tuition:
        claim_diff = f"{name_for_claims} publishes different undergraduate tuition rates for in-state ({uni.in_state_tuition}) and out-of-state ({uni.out_of_state_tuition}) students."
    else:
        claim_diff = f"{name_for_claims} offers different undergraduate tuition rates for in-state vs out-of-state students."
    await evaluator.verify(
        claim=claim_diff,
        node=node_diff,
        sources=urls,
        additional_instruction="Look for tuition pages showing distinct in-state and out-of-state undergraduate tuition rates."
    )

    # 9) In-state tuition < $20,000 per academic year (excluding fees)
    node_instate = evaluator.add_leaf(
        id=f"u{uni_index}_in_state_tuition_threshold",
        desc="In-state undergraduate tuition (excluding fees) is less than $20,000 per academic year",
        parent=uni_node,
        critical=True
    )
    if uni.in_state_tuition:
        claim_instate = f"The in-state undergraduate tuition at {name_for_claims} is {uni.in_state_tuition}, which is less than $20,000 per academic year (tuition only, excluding mandatory fees)."
    else:
        claim_instate = f"The in-state undergraduate tuition at {name_for_claims} is less than $20,000 per academic year (tuition only, excluding mandatory fees)."
    await evaluator.verify(
        claim=claim_instate,
        node=node_instate,
        sources=urls,
        additional_instruction="Verify using cited tuition pages. Focus on 'tuition only' (excluding fees). If the page only shows tuition+fees and not tuition-only, do NOT consider it supported."
    )

    # 10) Out-of-state tuition >= 2.0 × in-state tuition
    node_oos_mult = evaluator.add_leaf(
        id=f"u{uni_index}_out_of_state_tuition_multiplier",
        desc="Out-of-state undergraduate tuition is at least twice (2.0×) the in-state rate",
        parent=uni_node,
        critical=True
    )
    if uni.in_state_tuition and uni.out_of_state_tuition:
        claim_oos_mult = f"At {name_for_claims}, the out-of-state undergraduate tuition ({uni.out_of_state_tuition}) is at least 2.0 times the in-state tuition ({uni.in_state_tuition})."
    else:
        claim_oos_mult = f"At {name_for_claims}, the out-of-state undergraduate tuition is at least 2.0 times the in-state tuition rate."
    await evaluator.verify(
        claim=claim_oos_mult,
        node=node_oos_mult,
        sources=urls,
        additional_instruction="Use the cited tuition pages. Confirm numerically that out-of-state tuition ≥ 2.0 × in-state tuition."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the 'four public universities meeting all specified criteria' task.
    """
    # Initialize evaluator with a parallel root (partial credit across universities)
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

    # Extract universities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=FourUniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Ensure we process exactly 4 items (pad with empty if fewer)
    universities: List[UniversityExtraction] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityExtraction())

    # Build a parallel node for the overall task (non-critical, allows partial across items)
    task_node = evaluator.add_parallel(
        id="Task_Identify_Four_Universities",
        desc="Identify four public universities meeting all specified criteria",
        parent=root,
        critical=False
    )

    # For each university, build its verification sub-tree and run checks
    for i in range(4):
        await verify_university(
            evaluator=evaluator,
            parent_node=task_node,
            uni=universities[i],
            uni_index=i
        )

    # Return the evaluation summary
    return evaluator.get_summary()