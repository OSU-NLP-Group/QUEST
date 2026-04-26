import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ivy_president_2026"
TASK_DESCRIPTION = """
Identify the person who was appointed as president of an Ivy League university in January 2026, and who previously served as chancellor of a Big Ten Conference university. This person must hold three graduate degrees from three different institutions: specifically, a degree from Harvard University, a degree from Yale University, and a degree from Massachusetts Institute of Technology (MIT). Among these credentials, the person must hold both a Juris Doctor (JD) degree and a Doctor of Philosophy (PhD) degree. Provide the person's full name.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Degree(BaseModel):
    institution: Optional[str] = None
    degree_type: Optional[str] = None  # e.g., JD, PhD, MA, MS, MBA, MPP, etc.
    degree_full_name: Optional[str] = None  # e.g., "Juris Doctor (JD)" or "Doctor of Philosophy (Ph.D.) in Economics"
    field_or_program: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CandidateExtraction(BaseModel):
    full_name: Optional[str] = None

    ivy_university: Optional[str] = None
    appointment_month: Optional[str] = None  # Expected: "January"
    appointment_year: Optional[str] = None   # Expected: "2026"
    appointment_sources: List[str] = Field(default_factory=list)

    previous_chancellor_university: Optional[str] = None
    chancellor_sources: List[str] = Field(default_factory=list)
    big_ten_membership_sources: List[str] = Field(default_factory=list)  # If the answer cites Big Ten membership

    degrees: List[Degree] = Field(default_factory=list)

    extra_sources: List[str] = Field(default_factory=list)  # Other general sources cited in the answer (bio pages, etc.)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_candidate() -> str:
    return """
    From the provided answer text, extract the single candidate (person) the answer identifies along with all cited evidence.

    Return a JSON object with the following fields:
    - full_name: The person's full name as written in the answer (at least first and last name).
    - ivy_university: The Ivy League university where the person was appointed president (as stated in the answer).
    - appointment_month: The month of the presidential appointment announcement (e.g., "January") if stated.
    - appointment_year: The year of the presidential appointment announcement (4-digit string, e.g., "2026") if stated.
    - appointment_sources: A list of all URLs in the answer that specifically support the presidential appointment at the Ivy League university (news releases, official announcements, credible press).
    - previous_chancellor_university: The university where the person previously served as "chancellor" (as stated).
    - chancellor_sources: A list of all URLs that support the person's prior chancellor role.
    - big_ten_membership_sources: A list of URLs (if any) cited in the answer that explicitly indicate the prior chancellor university is a member of the Big Ten Conference.
    - degrees: An array of objects; each object must have:
        • institution: The awarding institution exactly as stated (e.g., "Harvard University", "Harvard Law School", "Yale University", "Massachusetts Institute of Technology (MIT)").
        • degree_type: The compact degree label if present (e.g., "JD", "PhD", "MA", "MS", "MBA", "MPP").
        • degree_full_name: The full degree name if present (e.g., "Juris Doctor (JD)", "Doctor of Philosophy (Ph.D.) in Economics").
        • field_or_program: The field, program, or school if stated (e.g., "Economics", "Harvard Law School").
        • year: The year if provided (as a string).
        • sources: A list of URLs from the answer that support that specific degree.
    - extra_sources: Any other URLs cited in the answer relevant to the person’s bio (e.g., official biography) that may mention multiple degrees or roles.

    STRICT RULES:
    1) Extract ONLY what is explicitly present in the answer text. Do not invent or infer any values.
    2) For any missing field, return null (for single values) or an empty list (for arrays).
    3) For sources, extract only full URLs explicitly mentioned in the answer (including within markdown links).
    4) Keep institution names as written; do not normalize (e.g., "MIT" vs "Massachusetts Institute of Technology").
    5) If multiple candidates are mentioned, extract only the one that the answer ultimately identifies as satisfying all constraints (the final answer).

    Your output must strictly conform to the specified JSON schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _is_likely_full_name(name: Optional[str]) -> bool:
    if not name:
        return False
    parts = [p for p in name.strip().split() if p.strip()]
    if len(parts) < 2:
        return False
    # Disallow single-letter initials for both main tokens
    long_tokens = [p for p in parts if len(p.replace(".", "")) >= 2]
    return len(long_tokens) >= 2


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        u2 = (u or "").strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def _collect_sources(*url_groups: List[str]) -> List[str]:
    combined: List[str] = []
    for group in url_groups:
        combined.extend(group or [])
    return _dedupe_urls(combined)


def _institution_matches(label: Optional[str], target: str) -> bool:
    s = _norm(label)
    t = _norm(target)
    if not s:
        return False
    if t == "harvard university":
        return ("harvard" in s)
    if t == "yale university":
        return ("yale" in s)
    if t in ("massachusetts institute of technology", "mit"):
        return ("massachusetts institute of technology" in s) or (s == "mit") or (" mit" in f" {s} ") or ("(mit)" in s)
    return t in s


def _degree_type_matches(deg: Degree, syns: List[str]) -> bool:
    candidates = [
        _norm(deg.degree_type),
        _norm(deg.degree_full_name),
    ]
    for c in candidates:
        if not c:
            continue
        for syn in syns:
            if _norm(syn) in c:
                return True
    return False


def _gather_degree_sources_by_institution(degrees: List[Degree], inst_target: str, fallback: List[str]) -> Tuple[bool, List[str]]:
    matched = [d for d in degrees if _institution_matches(d.institution, inst_target)]
    has_match = len(matched) > 0
    srcs = _collect_sources(*[d.sources for d in matched])
    if not srcs:
        # allow fallback to extra biography/source links if any
        srcs = _dedupe_urls(list(fallback))
    return has_match, srcs


def _gather_degree_sources_by_type(degrees: List[Degree], type_synonyms: List[str], fallback: List[str]) -> Tuple[bool, List[str]]:
    matched = [d for d in degrees if _degree_type_matches(d, type_synonyms)]
    has_match = len(matched) > 0
    srcs = _collect_sources(*[d.sources for d in matched])
    if not srcs:
        srcs = _dedupe_urls(list(fallback))
    return has_match, srcs


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_person_tree(evaluator: Evaluator, parent_node, ex: CandidateExtraction) -> None:
    # High-level sequential, critical node (maps to JSON "Person_Identification")
    person_node = evaluator.add_sequential(
        id="Person_Identification",
        desc="Identify the person who satisfies all stated constraints and provide their full name.",
        parent=parent_node,
        critical=True
    )

    # 1) Constraint satisfaction (parallel, all critical)
    constraints_node = evaluator.add_parallel(
        id="Candidate_Satisfies_All_Constraints",
        desc="The identified person satisfies all constraints in the prompt (appointment, prior role, and education requirements).",
        parent=person_node,
        critical=True
    )

    # 1.a) Ivy League presidency appointment
    # Add a critical presence-of-sources check to enforce web-grounding
    ivy_sources_present = evaluator.add_custom_node(
        result=bool(ex.appointment_sources),
        id="Ivy_Appointment_Sources_Present",
        desc="Appointment sources are provided for Ivy League presidency claim.",
        parent=constraints_node,
        critical=True
    )

    ivy_leaf = evaluator.add_leaf(
        id="Ivy_League_Presidency_Appointment",
        desc="Person was appointed as president of an Ivy League university.",
        parent=constraints_node,
        critical=True
    )
    ivy_claim_person = ex.full_name or "the identified person"
    ivy_claim_univ = ex.ivy_university or "the specified university"
    ivy_claim = f"{ivy_claim_person} was appointed as president of {ivy_claim_univ}."
    await evaluator.verify(
        claim=ivy_claim,
        node=ivy_leaf,
        sources=_dedupe_urls(ex.appointment_sources),
        additional_instruction="Verify that the cited page(s) explicitly state the person was appointed/named/selected as the university's president (overall university president, not a sub-unit). Accept phrasing like 'named as next president' or 'appointed president'."
    )

    # 1.b) Appointment announcement occurred in January 2026
    jan_leaf = evaluator.add_leaf(
        id="Appointment_Announcement_In_January_2026",
        desc="The presidential appointment announcement occurred in January 2026.",
        parent=constraints_node,
        critical=True
    )
    jan_claim = f"The appointment of {ivy_claim_person} as president of {ivy_claim_univ} was announced in January 2026."
    await evaluator.verify(
        claim=jan_claim,
        node=jan_leaf,
        sources=_dedupe_urls(ex.appointment_sources),
        additional_instruction="Check the announcement or news post date. The announcement must be in January 2026."
    )

    # 1.c) Previous chancellor at a Big Ten Conference university
    ch_sources_present = evaluator.add_custom_node(
        result=bool(ex.chancellor_sources or ex.big_ten_membership_sources),
        id="Chancellor_Sources_Present",
        desc="Sources are provided for prior chancellor role (and Big Ten membership where applicable).",
        parent=constraints_node,
        critical=True
    )
    ch_leaf = evaluator.add_leaf(
        id="Previous_Chancellor_At_Big_Ten_University",
        desc="Person previously served as chancellor at a Big Ten Conference member university.",
        parent=constraints_node,
        critical=True
    )
    ch_univ = ex.previous_chancellor_university or "the specified university"
    ch_claim = f"{ivy_claim_person} previously served as chancellor at {ch_univ}, which is a member of the Big Ten Conference."
    ch_sources = _collect_sources(_dedupe_urls(ex.chancellor_sources), _dedupe_urls(ex.big_ten_membership_sources))
    await evaluator.verify(
        claim=ch_claim,
        node=ch_leaf,
        sources=ch_sources,
        additional_instruction="It is acceptable if one source confirms the chancellor role and another confirms Big Ten membership. Together, they must support the overall claim."
    )

    # 1.d) Graduate degree requirements (critical, parallel)
    degrees_node = evaluator.add_parallel(
        id="Graduate_Degree_Requirements",
        desc="Person meets the stated graduate-degree requirements (institutions and degree types).",
        parent=constraints_node,
        critical=True
    )

    # Enforce that there are some degree sources available (bio pages or degree-specific links)
    degree_any_sources = _collect_sources(
        *[d.sources for d in ex.degrees],
        _dedupe_urls(ex.extra_sources)
    )
    degree_sources_present = evaluator.add_custom_node(
        result=bool(degree_any_sources),
        id="Degree_Sources_Present",
        desc="At least one source is provided to support degree credentials.",
        parent=degrees_node,
        critical=True
    )

    # Institution checks (Harvard, Yale, MIT) — all critical
    inst_checks = evaluator.add_parallel(
        id="Three_Graduate_Degrees_From_Harvard_Yale_MIT",
        desc="Person holds three graduate degrees from three different institutions: Harvard University, Yale University, and MIT.",
        parent=degrees_node,
        critical=True
    )

    # Harvard
    harvard_match, harvard_srcs = _gather_degree_sources_by_institution(
        ex.degrees, "Harvard University", _dedupe_urls(ex.extra_sources)
    )
    harvard_srcs_present = evaluator.add_custom_node(
        result=harvard_match and bool(harvard_srcs),
        id="Harvard_Degree_Sources_Present",
        desc="Sources support a graduate degree from Harvard University.",
        parent=inst_checks,
        critical=True
    )
    harvard_leaf = evaluator.add_leaf(
        id="Graduate_Degree_From_Harvard",
        desc="The person holds a graduate degree from Harvard University.",
        parent=inst_checks,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ivy_claim_person} holds a graduate degree from Harvard University.",
        node=harvard_leaf,
        sources=harvard_srcs,
        additional_instruction="Graduate degree includes any post‑baccalaureate degree (e.g., JD, PhD, MA, MS, MBA, MPP). Pages that specify 'Harvard Law School' or other Harvard graduate schools count as Harvard University."
    )

    # Yale
    yale_match, yale_srcs = _gather_degree_sources_by_institution(
        ex.degrees, "Yale University", _dedupe_urls(ex.extra_sources)
    )
    yale_srcs_present = evaluator.add_custom_node(
        result=yale_match and bool(yale_srcs),
        id="Yale_Degree_Sources_Present",
        desc="Sources support a graduate degree from Yale University.",
        parent=inst_checks,
        critical=True
    )
    yale_leaf = evaluator.add_leaf(
        id="Graduate_Degree_From_Yale",
        desc="The person holds a graduate degree from Yale University.",
        parent=inst_checks,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ivy_claim_person} holds a graduate degree from Yale University.",
        node=yale_leaf,
        sources=yale_srcs,
        additional_instruction="Graduate degree includes any post‑baccalaureate degree (e.g., JD, PhD, MA, MS, MBA, MPP)."
    )

    # MIT
    mit_match, mit_srcs = _gather_degree_sources_by_institution(
        ex.degrees, "Massachusetts Institute of Technology", _dedupe_urls(ex.extra_sources)
    )
    # Try a second pass using "MIT" if no match yet
    if not mit_match:
        mit_match, mit_srcs = _gather_degree_sources_by_institution(
            ex.degrees, "MIT", _dedupe_urls(ex.extra_sources)
        )
    mit_srcs_present = evaluator.add_custom_node(
        result=mit_match and bool(mit_srcs),
        id="MIT_Degree_Sources_Present",
        desc="Sources support a graduate degree from MIT.",
        parent=inst_checks,
        critical=True
    )
    mit_leaf = evaluator.add_leaf(
        id="Graduate_Degree_From_MIT",
        desc="The person holds a graduate degree from Massachusetts Institute of Technology (MIT).",
        parent=inst_checks,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ivy_claim_person} holds a graduate degree from the Massachusetts Institute of Technology (MIT).",
        node=mit_leaf,
        sources=mit_srcs,
        additional_instruction="Graduate degree includes any post‑baccalaureate degree. Accept references to 'MIT' or spelled‑out 'Massachusetts Institute of Technology'."
    )

    # Degree type checks (JD and PhD) — all critical
    type_checks = evaluator.add_parallel(
        id="JD_And_PhD_Included_Among_Those_Credentials",
        desc="Among those stated credentials/graduate degrees, the person holds both a JD and a PhD.",
        parent=degrees_node,
        critical=True
    )

    # JD
    jd_match, jd_srcs = _gather_degree_sources_by_type(
        ex.degrees,
        ["JD", "J.D.", "Juris Doctor"],
        _dedupe_urls(ex.extra_sources)
    )
    jd_srcs_present = evaluator.add_custom_node(
        result=jd_match and bool(jd_srcs),
        id="JD_Degree_Sources_Present",
        desc="Sources support that the person holds a Juris Doctor (JD) degree.",
        parent=type_checks,
        critical=True
    )
    jd_leaf = evaluator.add_leaf(
        id="Holds_JD_Degree",
        desc="The person holds a Juris Doctor (JD) degree.",
        parent=type_checks,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ivy_claim_person} holds a Juris Doctor (JD) degree.",
        node=jd_leaf,
        sources=jd_srcs,
        additional_instruction="The evidence should clearly show the person earned a JD (law degree)."
    )

    # PhD
    phd_match, phd_srcs = _gather_degree_sources_by_type(
        ex.degrees,
        ["PhD", "Ph.D.", "Doctor of Philosophy"],
        _dedupe_urls(ex.extra_sources)
    )
    phd_srcs_present = evaluator.add_custom_node(
        result=phd_match and bool(phd_srcs),
        id="PhD_Degree_Sources_Present",
        desc="Sources support that the person holds a Doctor of Philosophy (PhD) degree.",
        parent=type_checks,
        critical=True
    )
    phd_leaf = evaluator.add_leaf(
        id="Holds_PhD_Degree",
        desc="The person holds a Doctor of Philosophy (PhD) degree.",
        parent=type_checks,
        critical=True
    )
    await evaluator.verify(
        claim=f"{ivy_claim_person} holds a Doctor of Philosophy (PhD) degree.",
        node=phd_leaf,
        sources=phd_srcs,
        additional_instruction="The evidence should clearly indicate a PhD/Ph.D./Doctor of Philosophy degree."
    )

    # 2) The response provides the full name (critical)
    name_leaf = evaluator.add_custom_node(
        result=_is_likely_full_name(ex.full_name),
        id="Response_Provides_Full_Name",
        desc="The response provides the person’s full name (not only a partial name or initials).",
        parent=person_node,
        critical=True
    )

    # Record some structured info for debugging/traceability (non-scoring)
    evaluator.add_custom_info(
        info={
            "extracted_full_name": ex.full_name,
            "extracted_ivy_university": ex.ivy_university,
            "extracted_appointment_month": ex.appointment_month,
            "extracted_appointment_year": ex.appointment_year,
            "extracted_previous_chancellor_university": ex.previous_chancellor_university,
            "degree_count": len(ex.degrees),
            "has_jd": jd_match,
            "has_phd": phd_match,
            "has_harvard_degree": harvard_match,
            "has_yale_degree": yale_match,
            "has_mit_degree": mit_match
        },
        info_type="extraction_debug",
        info_name="extraction_debug"
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator; main logic is under the critical sequential child
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

    # Extract structured candidate info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_candidate(),
        template_class=CandidateExtraction,
        extraction_name="candidate_extraction"
    )

    # Build verification tree and run checks
    await build_and_verify_person_tree(evaluator, root, extracted)

    # Return evaluator summary
    return evaluator.get_summary()