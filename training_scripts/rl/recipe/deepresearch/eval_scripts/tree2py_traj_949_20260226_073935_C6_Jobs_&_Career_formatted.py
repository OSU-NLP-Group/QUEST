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
TASK_ID = "ucla_uw_columbia_person_identification"
TASK_DESCRIPTION = (
    "Who served as Dean of UCLA School of Law from August 2015 until June 2022, then became the 30th Chancellor of "
    "the University of Wisconsin-Madison starting on August 4, 2022 (after being announced for the position on May 16, 2022), "
    "and was subsequently appointed as the 21st President of Columbia University (announced on January 25, 2026, with an effective date of July 1, 2026)? "
    "Provide the person's full name."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PersonExtraction(BaseModel):
    # Core identification
    full_name: Optional[str] = None

    # Generic catch-all list of every URL explicitly present in the answer
    urls_all: List[str] = Field(default_factory=list)

    # Role/date/designation-specific URL buckets (only include URLs explicitly mentioned in the answer)
    urls_ucla_dean_dates: List[str] = Field(default_factory=list)                 # Dean of UCLA Law Aug 2015 – June 2022
    urls_uw_start: List[str] = Field(default_factory=list)                        # UW–Madison Chancellor start Aug 4, 2022
    urls_uw_announcement: List[str] = Field(default_factory=list)                 # UW–Madison Chancellor announced May 16, 2022
    urls_uw_designation_30th: List[str] = Field(default_factory=list)             # 30th Chancellor designation

    urls_columbia_effective: List[str] = Field(default_factory=list)              # Columbia President effective July 1, 2026
    urls_columbia_announcement: List[str] = Field(default_factory=list)           # Columbia President announced Jan 25, 2026
    urls_columbia_designation_21st: List[str] = Field(default_factory=list)       # 21st President designation

    # Institution-type constraint URLs
    urls_ucla_institution_type: List[str] = Field(default_factory=list)           # UCLA is public research; law school context
    urls_uw_institution_type: List[str] = Field(default_factory=list)             # UW–Madison flagship Big Ten public research
    urls_columbia_institution_type: List[str] = Field(default_factory=list)       # Columbia is a private Ivy League institution

    # Background
    urls_legal_scholar: List[str] = Field(default_factory=list)                   # Person is a legal scholar


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_person_info() -> str:
    return """
    Extract the answer to the following identification task from the provided answer text.

    You must return a single JSON object with these fields:
    - full_name: The single person's full name exactly as presented in the answer. If there are multiple names, extract the one the answer asserts as the final/selected person. If none, return null.
    - urls_all: An array of ALL URLs explicitly present in the answer (including Markdown links). Include only valid URLs. If protocol is missing, prepend "http://".
    - urls_ucla_dean_dates: URLs explicitly cited that support that the person served as Dean of UCLA School of Law from August 2015 until June 2022.
    - urls_uw_start: URLs explicitly cited that support that the person became Chancellor of the University of Wisconsin–Madison with a start date of August 4, 2022.
    - urls_uw_announcement: URLs explicitly cited that support that the person was announced for the UW–Madison Chancellor role on May 16, 2022.
    - urls_uw_designation_30th: URLs explicitly cited that support the designation as the 30th Chancellor of UW–Madison.
    - urls_columbia_effective: URLs explicitly cited that support that the Columbia University presidency is effective July 1, 2026.
    - urls_columbia_announcement: URLs explicitly cited that support that the Columbia presidency was announced on January 25, 2026.
    - urls_columbia_designation_21st: URLs explicitly cited that support the designation as the 21st President of Columbia University.
    - urls_ucla_institution_type: URLs explicitly cited that support that UCLA is a public research university and that UCLA School of Law is the law school of UCLA (i.e., a public research university law school).
    - urls_uw_institution_type: URLs explicitly cited that support that the University of Wisconsin–Madison is a flagship Big Ten public research university.
    - urls_columbia_institution_type: URLs explicitly cited that support that Columbia University is a private Ivy League institution.
    - urls_legal_scholar: URLs explicitly cited that support that the person has a background as a legal scholar (e.g., law professor/legal academic).

    IMPORTANT:
    - Include only URLs that are explicitly present in the answer text. Do not invent URLs.
    - If the answer provides URLs but does not label them by category, you may keep those URLs in urls_all and leave category-specific lists empty.
    - Do not guess or infer URLs that are not explicitly given in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_sources(*lists: Optional[List[str]]) -> List[str]:
    """Merge multiple URL lists while preserving order and removing duplicates."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def has_full_name(name: Optional[str]) -> bool:
    """Basic sanity check that a full name is provided (heuristic)."""
    if not name:
        return False
    # Require at least two tokens (e.g., first and last), allow middle initials
    parts = [p for p in name.strip().split() if p]
    return len(parts) >= 2


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_constraints(
    evaluator: Evaluator,
    parent_node,
    extraction: PersonExtraction,
) -> None:
    """
    Build the 'SatisfiesAllConstraints' parallel node and verify each critical constraint as leaf nodes.
    Uses verify_by_urls with evidence extracted from the answer.
    """
    person = extraction.full_name or ""

    constraints_node = evaluator.add_parallel(
        id="SatisfiesAllConstraints",
        desc="The named person satisfies all role, timing, designation, institution-type, and background constraints from the question/constraints.",
        parent=parent_node,
        critical=True,
    )

    # Prepare leaves
    leaves_and_claims: List[tuple] = []

    # 1) Dean role & dates at UCLA Law
    node_dean = evaluator.add_leaf(
        id="DeanRole_UCLALaw_Dates",
        desc="Person served as Dean of UCLA School of Law from August 2015 until June 2022.",
        parent=constraints_node,
        critical=True,
    )
    claim_dean = f"{person} served as Dean of UCLA School of Law from August 2015 until June 2022."
    src_dean = merge_sources(extraction.urls_ucla_dean_dates, extraction.urls_all)
    ins_dean = "Verify role title (Dean of UCLA School of Law) and the time span August 2015 through June 2022; allow minor phrasing such as '2015–2022' or 'June 2022'."

    leaves_and_claims.append((claim_dean, src_dean, node_dean, ins_dean))

    # 2) UW–Madison Chancellor start date
    node_uw_start = evaluator.add_leaf(
        id="ChancellorRole_UWMadison_StartDate",
        desc="Person became Chancellor of the University of Wisconsin–Madison with a start date of August 4, 2022.",
        parent=constraints_node,
        critical=True,
    )
    claim_uw_start = f"{person} became Chancellor of the University of Wisconsin–Madison effective August 4, 2022."
    src_uw_start = merge_sources(extraction.urls_uw_start, extraction.urls_all)
    ins_uw_start = "Confirm that the start/assumed office date is August 4, 2022. Allow phrasing like 'assumed office on Aug. 4, 2022'."

    leaves_and_claims.append((claim_uw_start, src_uw_start, node_uw_start, ins_uw_start))

    # 3) UW–Madison Chancellor announcement date
    node_uw_ann = evaluator.add_leaf(
        id="ChancellorRole_AnnouncementDate",
        desc="Person was announced for the Chancellor of the University of Wisconsin–Madison position on May 16, 2022.",
        parent=constraints_node,
        critical=True,
    )
    claim_uw_ann = f"It was announced on May 16, 2022 that {person} would become the Chancellor of the University of Wisconsin–Madison."
    src_uw_ann = merge_sources(extraction.urls_uw_announcement, extraction.urls_all)
    ins_uw_ann = "Check press release/news or page metadata that explicitly indicates the announcement date May 16, 2022."

    leaves_and_claims.append((claim_uw_ann, src_uw_ann, node_uw_ann, ins_uw_ann))

    # 4) Columbia President effective date
    node_col_eff = evaluator.add_leaf(
        id="PresidentRole_Columbia_EffectiveDate",
        desc="Person was appointed as President of Columbia University with an effective date of July 1, 2026.",
        parent=constraints_node,
        critical=True,
    )
    claim_col_eff = f"{person} was appointed as President of Columbia University with an effective date of July 1, 2026."
    src_col_eff = merge_sources(extraction.urls_columbia_effective, extraction.urls_all)
    ins_col_eff = "Verify that the effective (assumes office) date for the Columbia presidency is July 1, 2026."

    leaves_and_claims.append((claim_col_eff, src_col_eff, node_col_eff, ins_col_eff))

    # 5) Columbia President announcement date
    node_col_ann = evaluator.add_leaf(
        id="PresidentRole_AnnouncementDate",
        desc="Person's appointment as President of Columbia University was announced on January 25, 2026.",
        parent=constraints_node,
        critical=True,
    )
    claim_col_ann = f"The appointment of {person} as President of Columbia University was announced on January 25, 2026."
    src_col_ann = merge_sources(extraction.urls_columbia_announcement, extraction.urls_all)
    ins_col_ann = "Check official announcement/news page date and confirm January 25, 2026."

    leaves_and_claims.append((claim_col_ann, src_col_ann, node_col_ann, ins_col_ann))

    # 6) Institution type for UCLA Law (public research university law school)
    node_ucla_inst = evaluator.add_leaf(
        id="InstitutionType_DeanRole_PublicResearchLawSchool",
        desc="The deanship institution (UCLA School of Law / UCLA) is a public research university law school, satisfying the stated institution-type constraint for the first role.",
        parent=constraints_node,
        critical=True,
    )
    claim_ucla_inst = "UCLA School of Law is the law school of the University of California, Los Angeles (UCLA), which is a public research university."
    src_ucla_inst = merge_sources(extraction.urls_ucla_institution_type, extraction.urls_all)
    ins_ucla_inst = "Verify UCLA is a public research university and UCLA School of Law is the law school of UCLA."

    leaves_and_claims.append((claim_ucla_inst, src_ucla_inst, node_ucla_inst, ins_ucla_inst))

    # 7) Institution type for UW–Madison (flagship Big Ten public research university)
    node_uw_inst = evaluator.add_leaf(
        id="InstitutionType_ChancellorRole_FlagshipBigTenPublicResearch",
        desc="The chancellor institution (University of Wisconsin–Madison) is a flagship Big Ten public research university, satisfying the stated institution-type constraint for the second role.",
        parent=constraints_node,
        critical=True,
    )
    claim_uw_inst = "The University of Wisconsin–Madison is a flagship Big Ten public research university."
    src_uw_inst = merge_sources(extraction.urls_uw_institution_type, extraction.urls_all)
    ins_uw_inst = "Confirm UW–Madison is in the Big Ten, is a public research university, and is widely regarded as the state's flagship campus."

    leaves_and_claims.append((claim_uw_inst, src_uw_inst, node_uw_inst, ins_uw_inst))

    # 8) Institution type for Columbia (private Ivy League)
    node_col_inst = evaluator.add_leaf(
        id="InstitutionType_PresidentRole_PrivateIvyLeague",
        desc="The presidency institution (Columbia University) is a private Ivy League institution, satisfying the stated institution-type constraint for the third role.",
        parent=constraints_node,
        critical=True,
    )
    claim_col_inst = "Columbia University is a private Ivy League institution."
    src_col_inst = merge_sources(extraction.urls_columbia_institution_type, extraction.urls_all)
    ins_col_inst = "Confirm that Columbia is a private university and a member of the Ivy League."

    leaves_and_claims.append((claim_col_inst, src_col_inst, node_col_inst, ins_col_inst))

    # 9) 30th Chancellor designation
    node_uw_30th = evaluator.add_leaf(
        id="Designation_30thChancellor",
        desc="Person is designated as the 30th Chancellor of the University of Wisconsin–Madison.",
        parent=constraints_node,
        critical=True,
    )
    claim_uw_30th = f"{person} is the 30th Chancellor of the University of Wisconsin–Madison."
    src_uw_30th = merge_sources(extraction.urls_uw_designation_30th, extraction.urls_all)
    ins_uw_30th = "Look for explicit ordinal '30th' tied to Chancellor of UW–Madison for this person."

    leaves_and_claims.append((claim_uw_30th, src_uw_30th, node_uw_30th, ins_uw_30th))

    # 10) 21st President designation at Columbia
    node_col_21st = evaluator.add_leaf(
        id="Designation_21stPresident",
        desc="Person is designated as the 21st President of Columbia University.",
        parent=constraints_node,
        critical=True,
    )
    claim_col_21st = f"{person} is the 21st President of Columbia University."
    src_col_21st = merge_sources(extraction.urls_columbia_designation_21st, extraction.urls_all)
    ins_col_21st = "Look for explicit ordinal '21st' referring to President of Columbia University for this person."

    leaves_and_claims.append((claim_col_21st, src_col_21st, node_col_21st, ins_col_21st))

    # 11) Legal scholar background
    node_legal = evaluator.add_leaf(
        id="LegalScholarBackground",
        desc="Person has a background as a legal scholar.",
        parent=constraints_node,
        critical=True,
    )
    claim_legal = f"{person} is a legal scholar (e.g., a legal academic or law professor)."
    src_legal = merge_sources(extraction.urls_legal_scholar, extraction.urls_all)
    ins_legal = "Verify the page describes the person as a legal scholar/academic in law (e.g., professor of law)."

    leaves_and_claims.append((claim_legal, src_legal, node_legal, ins_legal))

    # Execute verifications (parallel-friendly). Each leaf uses its own sources and instruction.
    await evaluator.batch_verify(
        [(c, s, n, ins) for (c, s, n, ins) in leaves_and_claims]
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
    """
    Evaluate an answer for identifying the specific person across roles/timelines/designations/institution types.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Single top-level dimension
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

    # Extract structured info from the answer
    extracted: PersonExtraction = await evaluator.extract(
        prompt=prompt_extract_person_info(),
        template_class=PersonExtraction,
        extraction_name="person_extraction",
    )

    # Build the tree as per rubric
    person_node = evaluator.add_sequential(
        id="PersonIdentification",
        desc="Evaluate whether the response provides the full name of the person who satisfies all stated role, timing, designation, institution-type, and background constraints.",
        parent=root,
        critical=True,
    )

    # FullNameProvided (critical)
    full_name_ok = has_full_name(extracted.full_name)
    evaluator.add_custom_node(
        result=full_name_ok,
        id="FullNameProvided",
        desc="Response provides a single person's full name.",
        parent=person_node,
        critical=True,
    )

    # Constraints (critical, parallel). Even if we add verifications now, the system will auto-skip them if FullNameProvided failed.
    await build_and_verify_constraints(
        evaluator=evaluator,
        parent_node=person_node,
        extraction=extracted,
    )

    # Return evaluation summary
    return evaluator.get_summary()