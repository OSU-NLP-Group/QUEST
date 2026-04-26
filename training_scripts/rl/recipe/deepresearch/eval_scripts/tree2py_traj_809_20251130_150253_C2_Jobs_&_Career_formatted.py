import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "andrew_aurich_career_path"
TASK_DESCRIPTION = (
    "Andrew Aurich is a college football head coach who took his current position in 2024. "
    "Prior to becoming a head coach, he progressed through various assistant coaching roles including serving as an offensive coordinator at one institution before moving to another program where he coached position-specific roles. "
    "Provide the following information about Andrew Aurich's career path: "
    "(1) At which institution did Andrew Aurich serve as offensive coordinator, and in what year did he hold this role? "
    "(2) Immediately before being appointed as a head coach, at which institution did Andrew Aurich work, what specific coaching position did he hold there, and in what year did he serve in this role before his head coaching appointment? "
    "(3) At which institution was Andrew Aurich appointed as head coach, and on what date was this appointment officially announced? "
    "For each piece of information, provide a reference URL that supports your answer."
)

# Ground truth / constraints for verification
GT = {
    "oc_institution": "Princeton",
    "oc_year": "2019",
    "pre_hc_institution": "Rutgers",
    "pre_hc_role": "Tight Ends Coach",
    "pre_hc_year": "2023",
    "hc_institution": "Harvard",
    "hc_announcement_date": "February 12, 2024",
}

# --------------------------------------------------------------------------- #
# Pydantic models for extraction                                              #
# --------------------------------------------------------------------------- #
class OffensiveCoordinatorInfo(BaseModel):
    institution: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PreHeadCoachInfo(BaseModel):
    institution: Optional[str] = None
    role: Optional[str] = None
    year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HeadCoachAppointmentInfo(BaseModel):
    institution: Optional[str] = None
    announcement_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AndrewAurichCareerExtraction(BaseModel):
    offensive_coordinator: Optional[OffensiveCoordinatorInfo] = None
    pre_head_coach: Optional[PreHeadCoachInfo] = None
    head_coach_appointment: Optional[HeadCoachAppointmentInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_aurich_career() -> str:
    return """
Extract from the answer the specific facts requested about Andrew Aurich’s career path. Return a single JSON object with the following top-level keys and subfields:

offensive_coordinator:
  - institution: the institution where the answer says he served as offensive coordinator
  - year: the year the answer says he served as offensive coordinator
  - sources: an array of URL strings that the answer cites to support this offensive coordinator role

pre_head_coach:
  - institution: the institution where the answer says he worked immediately before becoming a head coach
  - role: the specific position (e.g., "tight ends coach") he held immediately before becoming a head coach
  - year: the year the answer says he served in that immediately pre–head coach position
  - sources: an array of URL strings that the answer cites to support this immediately pre–head coach role and year

head_coach_appointment:
  - institution: the institution where the answer says he was appointed as head coach
  - announcement_date: the official announcement date of his head coach appointment as stated in the answer (keep the exact string or formatting used in the answer)
  - sources: an array of URL strings that the answer cites to support the appointment and the announcement date

Rules:
- Extract exactly what the answer explicitly states. Do not invent or infer missing details.
- For URLs, return only valid URL strings that are explicitly present in the answer (including those inside markdown links).
- If a subfield is not explicitly provided in the answer, set it to null (for strings) or [] (for sources).
- Keep dates and years as strings exactly as written in the answer (e.g., "February 12, 2024", "Feb. 12, 2024", "2019 season").
    """.strip()


# --------------------------------------------------------------------------- #
# Helper for URL-based verification                                           #
# --------------------------------------------------------------------------- #
async def _verify_with_urls_or_fail(
    evaluator: Evaluator,
    claim: str,
    node,
    sources: Optional[List[str]],
    additional_instruction: str
) -> bool:
    valid_sources = [u for u in (sources or []) if isinstance(u, str) and u.strip()]
    if not valid_sources:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=valid_sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_offensive_coordinator_checks(
    evaluator: Evaluator,
    parent_node,
    oc: Optional[OffensiveCoordinatorInfo]
) -> None:
    section = evaluator.add_parallel(
        id="offensive_coordinator_role",
        desc="Offensive coordinator institution/year must match constraints, with supporting URL.",
        parent=parent_node,
        critical=True
    )

    inst_val = (oc.institution if oc else None) or ""
    year_val = (oc.year if oc else None) or ""
    srcs = (oc.sources if oc else []) or []

    # Institution must be Princeton
    oc_inst_node = evaluator.add_leaf(
        id="oc_institution_correct",
        desc="States that Andrew Aurich served as offensive coordinator at Princeton.",
        parent=section,
        critical=True
    )
    oc_inst_claim = (
        f"The institution named in the answer for Andrew Aurich's offensive coordinator role "
        f"is Princeton (allow variants like 'Princeton University' or 'Princeton Tigers'). "
        f"The extracted institution value is '{inst_val}'. These refer to the same institution."
    )
    await evaluator.verify(
        claim=oc_inst_claim,
        node=oc_inst_node,
        additional_instruction="Judge based on the answer text; allow reasonable name variants and short forms referring to Princeton University."
    )

    # Year must be 2019
    oc_year_node = evaluator.add_leaf(
        id="oc_year_correct",
        desc="States that Andrew Aurich served as offensive coordinator in 2019.",
        parent=section,
        critical=True
    )
    oc_year_claim = (
        f"The year identified in the answer for his offensive coordinator role is 2019. "
        f"The extracted value is '{year_val}'. Treat '2019 season' or formats like '2018–19' that clearly include year 2019 as acceptable."
    )
    await evaluator.verify(
        claim=oc_year_claim,
        node=oc_year_node,
        additional_instruction="Allow minor formatting variants (e.g., '2019 season', '’19', '2018–19' if it explicitly covers 2019)."
    )

    # URL must support Princeton + 2019 offensive coordinator
    oc_ref_node = evaluator.add_leaf(
        id="oc_reference_url",
        desc="Provides a valid URL that supports the Princeton + 2019 offensive coordinator claim.",
        parent=section,
        critical=True
    )
    oc_ref_claim = "Andrew Aurich served as offensive coordinator at Princeton in 2019."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=oc_ref_claim,
        node=oc_ref_node,
        sources=srcs,
        additional_instruction=(
            "Find explicit support that Andrew Aurich held the offensive coordinator role at Princeton in 2019. "
            "Allow reasonable variants like 'offensive coordinator/quarterbacks coach' or 'co-offensive coordinator' as long as 'offensive coordinator' is clearly indicated."
        )
    )


async def build_pre_head_coach_checks(
    evaluator: Evaluator,
    parent_node,
    pre: Optional[PreHeadCoachInfo]
) -> None:
    section = evaluator.add_parallel(
        id="immediate_pre_head_coach_role",
        desc="Immediately pre–head coach institution/role/year must match constraints, with supporting URL.",
        parent=parent_node,
        critical=True
    )

    inst_val = (pre.institution if pre else None) or ""
    role_val = (pre.role if pre else None) or ""
    year_val = (pre.year if pre else None) or ""
    srcs = (pre.sources if pre else []) or []

    # Institution must be Rutgers
    pre_inst_node = evaluator.add_leaf(
        id="pre_hc_institution_correct",
        desc="States that immediately before becoming a head coach, Aurich worked at Rutgers.",
        parent=section,
        critical=True
    )
    pre_inst_claim = (
        f"The institution named in the answer for his immediately pre–head coach role is Rutgers "
        f"(allow variants like 'Rutgers University' or 'Rutgers Scarlet Knights'). "
        f"The extracted institution value is '{inst_val}'. These refer to the same institution."
    )
    await evaluator.verify(
        claim=pre_inst_claim,
        node=pre_inst_node,
        additional_instruction="Judge based on the answer text; allow common Rutgers variants (Rutgers University, Rutgers Scarlet Knights)."
    )

    # Role must be tight ends coach
    pre_role_node = evaluator.add_leaf(
        id="pre_hc_specific_role_correct",
        desc="States that his immediately pre–head coach position was tight ends coach.",
        parent=section,
        critical=True
    )
    pre_role_claim = (
        f"The specific position immediately before becoming head coach is tight ends coach. "
        f"The extracted role value is '{role_val}'. Accept equivalent phrasing like 'TE coach' or 'tight ends' coach."
    )
    await evaluator.verify(
        claim=pre_role_claim,
        node=pre_role_node,
        additional_instruction="Consider 'tight ends coach', 'TE coach', 'coach – tight ends', or similar variants as equivalent."
    )

    # Year must be 2023
    pre_year_node = evaluator.add_leaf(
        id="pre_hc_year_correct",
        desc="States that he served in that immediately pre–head coach position in 2023.",
        parent=section,
        critical=True
    )
    pre_year_claim = (
        f"The year of his immediately pre–head coach role is 2023. "
        f"The extracted value is '{year_val}'. Treat '2023 season' or similar as acceptable."
    )
    await evaluator.verify(
        claim=pre_year_claim,
        node=pre_year_node,
        additional_instruction="Allow formats like '2023 season' or other minor variations that clearly indicate 2023."
    )

    # URL must support Rutgers + TE coach + 2023
    pre_ref_node = evaluator.add_leaf(
        id="pre_hc_reference_url",
        desc="Provides a valid URL that supports the Rutgers + tight ends coach + 2023 claim.",
        parent=section,
        critical=True
    )
    pre_ref_claim = "In 2023, Andrew Aurich served as the tight ends coach at Rutgers University."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=pre_ref_claim,
        node=pre_ref_node,
        sources=srcs,
        additional_instruction="Verify within the page that in 2023 he was Rutgers' tight ends coach (allow 'TE coach' or equivalent phrasing)."
    )


async def build_head_coach_checks(
    evaluator: Evaluator,
    parent_node,
    hc: Optional[HeadCoachAppointmentInfo]
) -> None:
    section = evaluator.add_parallel(
        id="head_coach_appointment",
        desc="Head coach institution/announcement date must match constraints, with supporting URL.",
        parent=parent_node,
        critical=True
    )

    inst_val = (hc.institution if hc else None) or ""
    date_val = (hc.announcement_date if hc else None) or ""
    srcs = (hc.sources if hc else []) or []

    # Institution must be Harvard
    hc_inst_node = evaluator.add_leaf(
        id="hc_institution_correct",
        desc="States that Aurich was appointed head coach at Harvard.",
        parent=section,
        critical=True
    )
    hc_inst_claim = (
        f"The institution named in the answer for his head coach appointment is Harvard "
        f"(allow variants like 'Harvard University' or 'Harvard Crimson'). "
        f"The extracted institution value is '{inst_val}'. These refer to the same institution."
    )
    await evaluator.verify(
        claim=hc_inst_claim,
        node=hc_inst_node,
        additional_instruction="Judge based on the answer text; allow common Harvard variants (Harvard University, Harvard Crimson)."
    )

    # Announcement date must be February 12, 2024
    hc_date_node = evaluator.add_leaf(
        id="hc_announcement_date_correct",
        desc="States that the official announcement date was February 12, 2024.",
        parent=section,
        critical=True
    )
    hc_date_claim = (
        f"The official announcement date of Andrew Aurich's head coach appointment was February 12, 2024. "
        f"The extracted date value is '{date_val}'. Treat variants like 'Feb. 12, 2024' or '2/12/2024' as equivalent."
    )
    await evaluator.verify(
        claim=hc_date_claim,
        node=hc_date_node,
        additional_instruction="Allow common date formatting variants that unambiguously indicate February 12, 2024."
    )

    # URL must support Harvard + Feb 12, 2024 appointment announcement
    hc_ref_node = evaluator.add_leaf(
        id="hc_reference_url",
        desc="Provides a valid URL that supports the Harvard + February 12, 2024 appointment announcement claim.",
        parent=section,
        critical=True
    )
    hc_ref_claim = "Andrew Aurich was appointed head coach at Harvard University, and the appointment was officially announced on February 12, 2024."
    await _verify_with_urls_or_fail(
        evaluator,
        claim=hc_ref_claim,
        node=hc_ref_node,
        sources=srcs,
        additional_instruction="Confirm that the page explicitly states Harvard appointed Andrew Aurich as head coach and that the announcement date is February 12, 2024."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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

    # Add a critical wrapper node matching the rubric's named section
    career_node = evaluator.add_parallel(
        id="andrew_aurich_career_path",
        desc="Verify the required career-path details and that each section includes a supporting URL, matching all constraint-specified facts.",
        parent=root,
        critical=True
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_aurich_career(),
        template_class=AndrewAurichCareerExtraction,
        extraction_name="aurich_career_extraction"
    )

    # Record ground truth constraints for transparency
    evaluator.add_ground_truth(
        {
            "offensive_coordinator_expected": {
                "institution": GT["oc_institution"],
                "year": GT["oc_year"],
            },
            "pre_head_coach_expected": {
                "institution": GT["pre_hc_institution"],
                "role": GT["pre_hc_role"],
                "year": GT["pre_hc_year"],
            },
            "head_coach_expected": {
                "institution": GT["hc_institution"],
                "announcement_date": GT["hc_announcement_date"],
            },
        },
        gt_type="expected_facts"
    )

    # Build verifications according to rubric tree
    await build_offensive_coordinator_checks(
        evaluator,
        career_node,
        extracted.offensive_coordinator if extracted else None
    )
    await build_pre_head_coach_checks(
        evaluator,
        career_node,
        extracted.pre_head_coach if extracted else None
    )
    await build_head_coach_checks(
        evaluator,
        career_node,
        extracted.head_coach_appointment if extracted else None
    )

    return evaluator.get_summary()