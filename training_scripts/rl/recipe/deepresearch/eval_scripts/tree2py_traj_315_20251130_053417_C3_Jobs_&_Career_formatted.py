import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "unc_oc_qualification_2025"
TASK_DESCRIPTION = (
    "Does the current offensive coordinator at the University of North Carolina (as of November 2025) "
    "meet the typical qualifications for a Power-Five FBS head football coaching position? Your analysis must "
    "address the following criteria: (1) Identify the current offensive coordinator at UNC and verify their current "
    "role and responsibilities; (2) Verify that the coach holds at least a bachelor's degree (the minimum educational "
    "requirement for FBS head coaching positions); (3) Document the coach's total years of coaching experience from "
    "the start of their coaching career; (4) Determine whether the coach's total coaching experience meets or exceeds "
    "the 16.9-year average for first-time FBS head coaches; and (5) Analyze the coach's career progression to determine "
    "if it follows one of the common pathways to Power-Five head coaching positions: prior Group of Five head coach "
    "(33.8%), Power-Five coordinator (18.2%), Power-Five head coach (14.3%), or internal promotion (14.3%). Provide a "
    "complete assessment with supporting evidence and reference URLs for each criterion."
)

AS_OF_TIMEFRAME = "November 2025"
THRESHOLD_YEARS = 16.9

# Canonical pathway labels and synonyms
CANONICAL_PATHWAYS = {
    "g5_head_coach": ["group of five head coach", "g5 head coach", "group-of-five head coach", "g5 hc"],
    "p5_coordinator": ["power five coordinator", "p5 coordinator", "power-five coordinator", "power 5 coordinator"],
    "p5_head_coach": ["power five head coach", "p5 head coach", "power-five head coach", "power 5 head coach", "former p5 head coach"],
    "internal_promotion": ["internal promotion", "promoted from within", "internal hire", "internal elevation", "internal"],
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachQualificationExtraction(BaseModel):
    # Step 1: Identity & role/responsibilities
    oc_name: Optional[str] = None
    role_summary: Optional[str] = None
    role_urls: List[str] = Field(default_factory=list)

    # Step 2: Education
    degree_summary: Optional[str] = None  # e.g., "B.S. in ...", "Bachelor's degree", "Master's", etc.
    education_urls: List[str] = Field(default_factory=list)

    # Step 2: Experience (timeline and total years)
    start_year: Optional[str] = None  # e.g., "2008" or "2008 season"
    total_years_through_nov2025: Optional[str] = None  # e.g., "17", "approximately 18", "17.5"
    experience_urls: List[str] = Field(default_factory=list)

    # Step 2: Threshold average figure source
    avg_16_9_source_urls: List[str] = Field(default_factory=list)

    # Step 2: Career pathway
    pathway_label: Optional[str] = None  # agent-provided mapped label (e.g., "P5 coordinator")
    pathway_summary: Optional[str] = None  # brief explanation text of why that pathway applies
    pathway_urls: List[str] = Field(default_factory=list)

    # Step 2: Overall determination
    overall_determination: Optional[str] = None  # e.g., "meets typical qualifications" or "does not meet"
    overall_summary: Optional[str] = None  # explanation tying to criteria


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_qualification() -> str:
    return f"""
Extract the following fields exactly as they are presented in the answer. Do not invent information.

1) Identity and Role at UNC (as of {AS_OF_TIMEFRAME}):
   - oc_name: The current offensive coordinator's full name at the University of North Carolina (UNC) as of {AS_OF_TIMEFRAME}.
   - role_summary: A brief description of the OC's role and responsibilities at UNC (more than just the title; e.g., coordinating the offense, play-calling, game planning).
   - role_urls: All URLs the answer cites to support the OC identity and role/responsibilities at UNC.

2) Education:
   - degree_summary: The highest degree or a statement confirming at least a bachelor's degree (e.g., "B.S.", "Bachelor of Arts", "BA/BS", "Master's", "Doctorate", etc.). If the answer is ambiguous, extract the best summary provided in the answer.
   - education_urls: All URLs cited to support the degree claim.

3) Coaching Experience:
   - start_year: The start year (or approximate start date) of the coach's coaching career as stated or implied in the answer (e.g., "2008", "2008 season").
   - total_years_through_nov2025: The total years of coaching experience THROUGH {AS_OF_TIMEFRAME} (as stated or calculated by the answer). Extract the numeric or textual representation EXACTLY as shown in the answer (e.g., "17", "approximately 18", "17.5", "17-18").
   - experience_urls: All URLs cited in the answer that document the career timeline to compute or verify total years.

4) 16.9-year Average Source:
   - avg_16_9_source_urls: The URL(s) cited by the answer to support the figure "16.9-year average for first-time FBS head coaches".

5) Career Pathway:
   - pathway_label: The category (as presented by the answer) that best fits the career pathway toward Power-Five head coach. Allowed categories include:
       • "Group of Five head coach" (G5 head coach)
       • "Power-Five coordinator" (P5 coordinator)
       • "Power-Five head coach" (P5 head coach)
       • "internal promotion"
       • or explicitly indicate none fit (e.g., "none" or "no common pathway fits").
   - pathway_summary: Brief reasoning from the answer explaining why this classification applies (or why none fit).
   - pathway_urls: URL(s) cited to support the career history used for the pathway determination.

6) Overall Determination:
   - overall_determination: The final conclusion stated by the answer about whether the coach meets the typical qualifications for a Power-Five FBS head coaching position (e.g., "meets typical qualifications" or "does not meet").
   - overall_summary: A brief justification tying the conclusion to the evaluated criteria (education, total experience, threshold comparison, and career pathway).

Rules:
- If any field is missing from the answer, set it to null (or an empty list for URLs).
- For all URL fields, extract the actual URLs exactly as shown (including protocol). If none are given, return an empty list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_years_to_float(text: Optional[str]) -> Optional[float]:
    """
    Parse a textual representation of years into a float if possible.
    Examples:
        "17" -> 17.0
        "17.5" -> 17.5
        "approximately 18" -> 18.0
        "17-18" -> 17.0 (first number)
        "about 17+" -> 17.0
    """
    if not text:
        return None
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def normalize_pathway_label(label: Optional[str]) -> Optional[str]:
    """
    Normalize a free-text pathway label into one of the canonical pathway keys:
        - "g5_head_coach"
        - "p5_coordinator"
        - "p5_head_coach"
        - "internal_promotion"
      or return "none" if the label explicitly states no common pathway fits,
      or None if it cannot be determined.
    """
    if not label:
        return None
    s = label.strip().lower()

    # None/No-fit handling
    if any(k in s for k in ["none", "no common pathway", "does not fit", "no fit"]):
        return "none"

    # Check canonical pathways
    for key, synonyms in CANONICAL_PATHWAYS.items():
        for syn in synonyms:
            if syn in s:
                return key

    # Heuristics: detect patterns like "p5 coordinator", "g5 head coach" even if not exact synonyms listed
    if ("p5" in s or "power five" in s or "power-five" in s or "power 5" in s) and "coordinator" in s:
        return "p5_coordinator"
    if ("p5" in s or "power five" in s or "power-five" in s or "power 5" in s) and ("head coach" in s or "hc" in s):
        return "p5_head_coach"
    if ("g5" in s or "group of five" in s or "group-of-five" in s) and ("head coach" in s or "hc" in s):
        return "g5_head_coach"
    if "internal" in s and ("promotion" in s or "hire" in s or "elevation" in s):
        return "internal_promotion"

    return None


def canonical_label_to_human(label: Optional[str]) -> str:
    mapping = {
        "g5_head_coach": "prior Group of Five head coach",
        "p5_coordinator": "Power-Five coordinator",
        "p5_head_coach": "Power-Five head coach",
        "internal_promotion": "internal promotion",
        "none": "none of the listed pathways fit",
    }
    if label in mapping:
        return mapping[label]
    return "unknown"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_step1_identity_and_role(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    """
    Step 1: Identify the current UNC OC as of Nov 2025 and verify role/responsibilities with URLs.
    """
    step1_node = evaluator.add_parallel(
        id="Step1_Identify_Current_OC_And_Role",
        desc="Identify the current offensive coordinator at UNC as of November 2025 and verify their role/responsibilities, with supporting reference URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: OC identity as of Nov 2025
    oc_identity_leaf = evaluator.add_leaf(
        id="OC_Identity_AsOf_Nov2025",
        desc="Correctly names the UNC offensive coordinator as of November 2025 (time-bounded identification).",
        parent=step1_node,
        critical=True,
    )
    oc_name = data.oc_name or ""
    identity_claim = f"As of {AS_OF_TIMEFRAME}, the current offensive coordinator of the University of North Carolina (UNC) football program is {oc_name}."
    await evaluator.verify(
        claim=identity_claim,
        node=oc_identity_leaf,
        sources=data.role_urls,
        additional_instruction=(
            f"Verify the identity explicitly as of {AS_OF_TIMEFRAME}. Prefer official UNC Athletics pages, official bios, "
            "or reputable, time-relevant news/press releases. If the source is older, ensure it still implies this status by "
            f"{AS_OF_TIMEFRAME} (e.g., ongoing role)."
        ),
    )

    # Leaf: Role & responsibilities verified (beyond just title)
    role_resp_leaf = evaluator.add_leaf(
        id="Role_And_Responsibilities_Verified",
        desc="Provides a verifiable description of the coach's OC role/responsibilities at UNC (not just the title).",
        parent=step1_node,
        critical=True,
    )
    role_summary = (data.role_summary or "").strip()
    role_claim = (
        f"The sources support that {oc_name} serves as UNC's offensive coordinator as of {AS_OF_TIMEFRAME}, "
        f"and that their responsibilities include items mentioned in the answer such as: {role_summary if role_summary else 'coordinating the offense (e.g., game planning, play-calling).'}"
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_resp_leaf,
        sources=data.role_urls,
        additional_instruction=(
            "Do not accept claims that only restate the title. Look for responsibilities typically associated with an OC "
            "(e.g., coordinating the offense, play-calling, game planning, overseeing offensive staff), and ensure the "
            "source(s) substantiate at least one responsibility described in the answer."
        ),
    )

    # Leaf: Role evidence URLs exist (custom check)
    role_urls_exist = evaluator.add_custom_node(
        result=bool(data.role_urls),
        id="Role_Evidence_URLs",
        desc="Includes supporting reference URL(s) for the identity/role/responsibilities claim(s).",
        parent=step1_node,
        critical=True,
    )
    # role_urls_exist is a custom node with binary result assigned


async def verify_step2_education(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    edu_node = evaluator.add_parallel(
        id="Education_Bachelors_Minimum",
        desc="Verify the coach holds at least a bachelor's degree, with supporting reference URL(s).",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Bachelor's (or higher) verified by sources
    bachelors_leaf = evaluator.add_leaf(
        id="Bachelors_Verified",
        desc="States and supports that the coach has at least a bachelor's degree.",
        parent=edu_node,
        critical=True,
    )
    # We phrase claim independent from exact degree text, sources must show bachelor's or higher.
    bachelors_claim = (
        "The cited sources support that the coach holds at least a bachelor's degree (e.g., BA, BS, B.S., B.A.) "
        "or a higher degree."
    )
    await evaluator.verify(
        claim=bachelors_claim,
        node=bachelors_leaf,
        sources=data.education_urls,
        additional_instruction=(
            "Search the provided pages for educational background details. Accept common synonyms for bachelor's "
            "(BA, BS, B.A., B.S.) and higher-level degrees (Master's, Doctorate) as satisfying 'at least a bachelor's.'"
        ),
    )

    # Leaf: Education evidence URLs exist (custom)
    evaluator.add_custom_node(
        result=bool(data.education_urls),
        id="Education_Evidence_URLs",
        desc="Includes supporting reference URL(s) for the bachelor's-degree verification.",
        parent=edu_node,
        critical=True,
    )


async def verify_step2_experience(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    exp_node = evaluator.add_parallel(
        id="Total_Coaching_Experience_Documented",
        desc="Document the coach's total years of coaching experience from the start of their coaching career through Nov 2025, supported by references.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Start of coaching career identified
    start_leaf = evaluator.add_leaf(
        id="Start_Of_Coaching_Career_Identified",
        desc="Identifies the start date/year of the coach's coaching career (basis for the calculation).",
        parent=exp_node,
        critical=True,
    )
    start_year_text = data.start_year or ""
    start_claim = f"The coach began their coaching career in or around the year '{start_year_text}'."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=data.experience_urls,
        additional_instruction=(
            "Use bios, timelines, or credible articles. If the exact year is slightly ambiguous, allow reasonable "
            "phrasing like 'around 2008' if supported by the timeline."
        ),
    )

    # Leaf: Total years through Nov 2025 stated/calculated and checkable
    total_years_leaf = evaluator.add_leaf(
        id="Total_Years_Calculated_Through_Nov2025",
        desc="Calculates or clearly states total coaching years up to Nov 2025 in a way that can be checked from the provided timeline.",
        parent=exp_node,
        critical=True,
    )
    total_years_text = data.total_years_through_nov2025 or ""
    total_years_claim = (
        f"As of {AS_OF_TIMEFRAME}, the coach's total coaching experience is about '{total_years_text}' years, "
        "and this is checkable from the cited career timeline."
    )
    await evaluator.verify(
        claim=total_years_claim,
        node=total_years_leaf,
        sources=data.experience_urls,
        additional_instruction=(
            "If the exact total is not explicitly on the page, verify that the timeline on the cited pages allows "
            "a reasonable calculation that matches the stated total years (allowing rounding)."
        ),
    )

    # Leaf: Experience timeline evidence URLs exist (custom)
    evaluator.add_custom_node(
        result=bool(data.experience_urls),
        id="Experience_Timeline_Evidence_URLs",
        desc="Includes supporting reference URL(s) documenting the career timeline used to compute total years.",
        parent=exp_node,
        critical=True,
    )


async def verify_step2_threshold_compare(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    cmp_node = evaluator.add_parallel(
        id="Compare_To_16_9_Year_Average",
        desc="Determine whether the coach's total experience meets or exceeds the 16.9-year average for first-time FBS head coaches, and cite the source of the 16.9-year figure.",
        parent=parent_node,
        critical=True,
    )

    # Compute a parsed numeric years for custom info and possible reasoning
    parsed_years = parse_years_to_float(data.total_years_through_nov2025)
    computed_meets = (parsed_years is not None) and (parsed_years >= THRESHOLD_YEARS)

    evaluator.add_custom_info(
        info={
            "parsed_total_years": parsed_years,
            "threshold_years": THRESHOLD_YEARS,
            "meets_or_exceeds": computed_meets,
            "raw_total_years_text": data.total_years_through_nov2025,
        },
        info_type="computed_metrics",
        info_name="experience_threshold_computation",
    )

    # Leaf: Threshold comparison result (does the answer explicitly state >= 16.9?)
    # We'll verify from the answer text using simple verification (no sources required here).
    cmp_leaf = evaluator.add_leaf(
        id="Threshold_Comparison_Result",
        desc="Explicitly states whether total coaching experience is >= 16.9 years.",
        parent=cmp_node,
        critical=True,
    )
    cmp_claim = (
        "The answer explicitly states whether the coach's total coaching experience is at least 16.9 years, "
        "clearly indicating '>= 16.9' or an equivalent conclusion (e.g., 'above the 16.9-year average')."
    )
    await evaluator.verify(
        claim=cmp_claim,
        node=cmp_leaf,
        additional_instruction=(
            "Examine the answer text (not external sources) and determine if it clearly concludes whether the total "
            "experience meets or exceeds the 16.9-year average."
        ),
    )

    # Leaf: Provide a source URL supporting the 16.9-year average figure
    avg_src_leaf = evaluator.add_leaf(
        id="Average_16_9_Source_URL",
        desc="Provides a reference URL supporting the 16.9-year average figure.",
        parent=cmp_node,
        critical=True,
    )
    avg_claim = (
        "A reliable provided source states that the average coaching experience for first-time FBS head coaches is 16.9 years."
    )
    await evaluator.verify(
        claim=avg_claim,
        node=avg_src_leaf,
        sources=data.avg_16_9_source_urls,
        additional_instruction=(
            "Confirm that the provided URL(s) explicitly support the numeric figure 16.9 years as an average for first-time "
            "FBS head coaches. If the URL is irrelevant or does not contain that number, fail."
        ),
    )


async def verify_step2_pathway(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    path_node = evaluator.add_parallel(
        id="Career_Pathway_Analysis",
        desc="Analyze whether the coach's career progression fits one of the listed common pathways (prior G5 head coach, P5 coordinator, P5 head coach, internal promotion) and support the analysis with evidence/URLs.",
        parent=parent_node,
        critical=True,
    )

    normalized = normalize_pathway_label(data.pathway_label)
    # Leaf: pathway mapped (custom check that it's in listed categories or explicitly 'none')
    mapped_ok = normalized in {"g5_head_coach", "p5_coordinator", "p5_head_coach", "internal_promotion", "none"}
    evaluator.add_custom_node(
        result=mapped_ok,
        id="Pathway_Mapped_To_Listed_Categories",
        desc="Maps the coach's career progression to at least one of the specified pathways (or explicitly concludes none fit), grounded in career history.",
        parent=path_node,
        critical=True,
    )

    # Leaf: pathway evidence URLs (verify that the sources support the classification or the 'none fit' conclusion)
    path_evidence_leaf = evaluator.add_leaf(
        id="Pathway_Evidence_URLs",
        desc="Includes supporting reference URL(s) for the career history/pathway determination.",
        parent=path_node,
        critical=True,
    )
    human_label = canonical_label_to_human(normalized)
    path_summary = (data.pathway_summary or "").strip()
    pathway_claim = (
        f"The provided sources support the pathway determination for the coach as: {human_label}. "
        f"Reasoning provided in the answer: {path_summary if path_summary else 'The answer cites career history supporting this classification.'}"
    )
    await evaluator.verify(
        claim=pathway_claim,
        node=path_evidence_leaf,
        sources=data.pathway_urls,
        additional_instruction=(
            "Check the cited career history (roles, levels, and transitions). "
            "Confirm that it supports the stated classification (e.g., P5 coordinator, G5 head coach, P5 head coach, "
            "or internal promotion). If the answer claims none fit, verify that the history indeed does not match any listed category."
        ),
    )


async def verify_step2_overall(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    # Overall determination leaf (critical)
    overall_leaf = evaluator.add_leaf(
        id="Overall_Determination",
        desc="Provides a clear final assessment answering whether the coach meets typical qualifications, explicitly tying the conclusion to the evaluated criteria.",
        parent=parent_node,
        critical=True,
    )
    # Let the judge check the answer text for a clear final conclusion and linkage to criteria.
    overall_claim = (
        "The answer provides a clear final assessment on whether the coach meets typical qualifications for a Power-Five "
        "FBS head coaching position, and explicitly ties the conclusion to the evaluated criteria: education (at least bachelor's), "
        "total coaching experience, the >=16.9-year threshold comparison, and the career pathway analysis."
    )
    await evaluator.verify(
        claim=overall_claim,
        node=overall_leaf,
        additional_instruction=(
            "Look for an explicit 'meets' or 'does not meet' conclusion and references to each criterion. The answer should "
            "link its conclusion to the evidence evaluated, not just a bare conclusion."
        ),
    )


async def verify_step2_qualifications(
    evaluator: Evaluator,
    parent_node,
    data: CoachQualificationExtraction,
) -> None:
    step2_node = evaluator.add_parallel(
        id="Step2_Qualifications_Assessment",
        desc="Evaluate the coach against the listed qualification criteria (education, total experience, threshold comparison, pathway analysis), each with supporting evidence/URLs, and provide an overall determination.",
        parent=parent_node,
        critical=True,
    )

    # Education
    await verify_step2_education(evaluator, step2_node, data)
    # Experience
    await verify_step2_experience(evaluator, step2_node, data)
    # Threshold compare
    await verify_step2_threshold_compare(evaluator, step2_node, data)
    # Pathway analysis
    await verify_step2_pathway(evaluator, step2_node, data)
    # Overall determination
    await verify_step2_overall(evaluator, step2_node, data)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the UNC OC qualification (as of Nov 2025) task and return a structured result dictionary.
    """
    evaluator = Evaluator()
    # Initialize with a generic root (non-critical by framework design). We'll add a critical sequential node under it.
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

    # Extract structured information from the answer
    extracted: CoachQualificationExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_qualification(),
        template_class=CoachQualificationExtraction,
        extraction_name="coach_qualification_extraction",
    )

    # Record GT-like constants for transparency
    evaluator.add_custom_info(
        info={
            "as_of_timeframe": AS_OF_TIMEFRAME,
            "threshold_years": THRESHOLD_YEARS,
            "allowed_pathways": [
                "prior Group of Five head coach",
                "Power-Five coordinator",
                "Power-Five head coach",
                "internal promotion",
                "or explicitly: none fit",
            ],
        },
        info_type="task_constants",
        info_name="evaluation_parameters",
    )

    # Root assessment node (critical + sequential)
    root_assess = evaluator.add_sequential(
        id="Root_Assessment",
        desc="Assess whether UNC's current offensive coordinator (as of Nov 2025) meets typical qualifications for a Power-Five FBS head football coaching position, addressing all required criteria with supporting evidence/URLs.",
        parent=root,
        critical=True,
    )

    # Step 1
    await verify_step1_identity_and_role(evaluator, root_assess, extracted)

    # Step 2
    await verify_step2_qualifications(evaluator, root_assess, extracted)

    # Finalize and return
    return evaluator.get_summary()