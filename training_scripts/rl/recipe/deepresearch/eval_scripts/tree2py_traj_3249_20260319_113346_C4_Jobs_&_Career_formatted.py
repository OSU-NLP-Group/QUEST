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
TASK_ID = "pa_level_ii_certification"
TASK_DESCRIPTION = """
What are the complete requirements for a Pennsylvania teacher to advance from a Level I Instructional Certificate to a Level II (Permanent) Instructional Certificate? Your answer should include: (1) the minimum teaching experience required, (2) the maximum timeline before the Level I certificate expires, (3) the post-baccalaureate credit requirements, (4) any required program completion, (5) clarification on whether the timeline is based on calendar years or service years, (6) the specific credit level requirements, (7) the recommended application timing, and (8) the permanent status distinction of the Level II certificate.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementField(BaseModel):
    """One requirement as stated in the answer, with the URLs the answer cited for it."""
    content: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PARequirementsExtraction(BaseModel):
    """Structured extraction of all eight requirement items."""
    min_teaching_experience: Optional[RequirementField] = None
    max_service_timeline: Optional[RequirementField] = None
    post_baccalaureate_credits: Optional[RequirementField] = None
    induction_program: Optional[RequirementField] = None
    service_year_definition: Optional[RequirementField] = None
    credit_type_specification: Optional[RequirementField] = None
    application_timing: Optional[RequirementField] = None
    permanent_status: Optional[RequirementField] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_pa_requirements() -> str:
    return """
    Extract, exactly as stated in the answer, the eight Pennsylvania Level II certification requirement items below.
    For EACH item, return:
      - content: the specific statement as written in the answer (paraphrase minimally only if needed for clarity)
      - sources: a list of URL(s) explicitly cited in the answer that support this specific item.
                 Only include URLs actually present in the answer. If none are provided, return an empty list.

    Extract the following fields (use these exact JSON field names):
      1) min_teaching_experience
         - Example target phrasing: "minimum of 3 years of successful/satisfactory teaching experience in Pennsylvania"
      2) max_service_timeline
         - Example target phrasing: "Level I is valid for 6 years of service; must complete Level II before the 6 years expire"
      3) post_baccalaureate_credits
         - Example target phrasing: "24 post-baccalaureate semester credits earned after the initial bachelor's degree"
      4) induction_program
         - Example target phrasing: "completion of a PDE-approved induction program"
      5) service_year_definition
         - Example target phrasing: "the 6-year window is measured in service years (actual years of employment), not calendar years"
      6) credit_type_specification
         - Example target phrasing: "the 24 credits can be undergraduate- or graduate-level"
      7) application_timing
         - Example target phrasing: "apply after at least 3 years but before completing 6 years of service"
      8) permanent_status
         - Example target phrasing: "Instructional II is a permanent certificate (unlike temporary Level I)"

    Rules:
    - Do NOT invent URLs. Only include URLs explicitly present in the answer text.
    - If the answer merges multiple items into one sentence, still extract the most relevant snippet for each field separately.
    - If an item is missing in the answer, set its 'content' to null and 'sources' to an empty list.
    - Keep numeric values (e.g., 3 years, 6 years, 24 credits) exactly as the answer states (e.g., numerals vs words).
    """


# --------------------------------------------------------------------------- #
# Helper: add a requirement verification subtree                              #
# --------------------------------------------------------------------------- #
async def add_requirement_check(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    node_desc: str,
    extracted: Optional[RequirementField],
    normative_claim: str,
    critical_main: bool,
    add_ins: Optional[str] = None
) -> None:
    """
    Build a two-leaf subtree for one requirement:
      - A critical existence/sourcing check (the answer stated it and cited at least one URL)
      - A critical verification that the normative claim is supported by the cited URLs
    """
    # Parent node for this requirement
    req_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical_main
    )

    # Leaf 1: The answer states this AND cites at least one URL (critical gate)
    stated_and_sourced = evaluator.add_custom_node(
        result=bool(extracted and extracted.content and extracted.content.strip()) and bool(extracted and extracted.sources),
        id=f"{node_id}_stated_with_source",
        desc="This requirement is explicitly stated in the answer and has at least one cited URL",
        parent=req_node,
        critical=True
    )

    # Leaf 2: The normative requirement is supported by the cited URLs (critical)
    supported_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc="The cited source(s) support this requirement as written in official guidance",
        parent=req_node,
        critical=True
    )

    await evaluator.verify(
        claim=normative_claim,
        node=supported_leaf,
        sources=(extracted.sources if extracted else []),
        additional_instruction=(add_ins or "None")
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for Pennsylvania Level II (Permanent) Instructional Certification requirements.
    """
    # Initialize evaluator (root non-critical to allow partial credit for non-critical items)
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

    # Extract structured requirement statements and cited URLs from the answer
    extracted: PARequirementsExtraction = await evaluator.extract(
        prompt=prompt_extract_pa_requirements(),
        template_class=PARequirementsExtraction,
        extraction_name="pa_level_ii_requirements"
    )

    # Ground truth style normative claims (for LLM verification against cited URLs)
    gt = {
        "min_teaching_experience": "To convert a Pennsylvania Level I Instructional Certificate to a Level II (Permanent) Instructional Certificate, the educator must have completed at least three (3) years of satisfactory/successful teaching service in Pennsylvania while holding a Level I certificate.",
        "max_service_timeline": "A Pennsylvania Level I Instructional Certificate is valid for a maximum of six (6) years of service. An educator must complete the requirements and apply to convert to Level II before completing six service years, or the Level I certificate will expire.",
        "post_baccalaureate_credits": "Conversion to Pennsylvania Instructional II requires twenty-four (24) post-baccalaureate semester credits that were earned after the conferral of the initial bachelor's degree.",
        "induction_program": "Completion of a Pennsylvania Department of Education (PDE)-approved induction program is required to obtain the Instructional II certificate.",
        "service_year_definition": "The six-year validity window for Pennsylvania Instructional I is measured in years of service (i.e., years actually employed in a Pennsylvania school while holding Level I), not in calendar years.",
        "credit_type_specification": "The required 24 post-baccalaureate credits may be taken at either the undergraduate or graduate level (i.e., undergraduate- or graduate-level coursework is accepted if earned after the bachelor's degree).",
        "application_timing": "Educators should apply for Instructional II after completing at least three service years but before reaching six service years; do not wait until the Level I service window expires.",
        "permanent_status": "The Pennsylvania Instructional II certificate is permanent (it does not expire), unlike the temporary Instructional I certificate."
    }
    evaluator.add_ground_truth({"expected_normative_claims": gt}, gt_type="normative_requirements")

    # Build verification subtrees for each requirement
    # Critical items (per rubric): first four
    await add_requirement_check(
        evaluator, root,
        node_id="minimum_teaching_experience",
        node_desc="Minimum teaching experience requirement: at least 3 years of successful/satisfactory PA teaching",
        extracted=getattr(extracted, "min_teaching_experience", None),
        normative_claim=gt["min_teaching_experience"],
        critical_main=True,
        add_ins="Focus on whether Pennsylvania requires at least 3 years of satisfactory/successful teaching service in PA while holding Level I. Allow synonyms like 'satisfactory'/'successful.'"
    )

    await add_requirement_check(
        evaluator, root,
        node_id="maximum_service_timeline",
        node_desc="Maximum Level I timeline: valid for six (6) years of service; must convert before 6 years expire",
        extracted=getattr(extracted, "max_service_timeline", None),
        normative_claim=gt["max_service_timeline"],
        critical_main=True,
        add_ins="Confirm that the window is 6 service years (not calendar years) and that conversion must occur before completing six service years."
    )

    await add_requirement_check(
        evaluator, root,
        node_id="post_baccalaureate_credit_requirement",
        node_desc="Post-baccalaureate credits: 24 semester credits earned after the initial bachelor's degree",
        extracted=getattr(extracted, "post_baccalaureate_credits", None),
        normative_claim=gt["post_baccalaureate_credits"],
        critical_main=True,
        add_ins="Look for '24' and 'post-baccalaureate' and 'after the bachelor's degree' language on the cited page(s)."
    )

    await add_requirement_check(
        evaluator, root,
        node_id="induction_program_requirement",
        node_desc="Induction program: completion of a PDE-approved induction program",
        extracted=getattr(extracted, "induction_program", None),
        normative_claim=gt["induction_program"],
        critical_main=True,
        add_ins="The page should indicate that a PDE-approved (teacher) induction program is required for Level II."
    )

    # Non-critical items (per rubric): remaining four
    await add_requirement_check(
        evaluator, root,
        node_id="service_year_definition",
        node_desc="Service year definition: 6-year validity counted by years of service, not calendar years",
        extracted=getattr(extracted, "service_year_definition", None),
        normative_claim=gt["service_year_definition"],
        critical_main=False,
        add_ins="Check for wording that clarifies 'service years' (employment years) vs calendar years. Minor wording differences are acceptable."
    )

    await add_requirement_check(
        evaluator, root,
        node_id="credit_type_specification",
        node_desc="Credit type specification: the 24 credits may be undergraduate or graduate level",
        extracted=getattr(extracted, "credit_type_specification", None),
        normative_claim=gt["credit_type_specification"],
        critical_main=False,
        add_ins="Confirm the page states the 24 post-bacc credits can be undergraduate- or graduate-level coursework (earned after bachelor's)."
    )

    await add_requirement_check(
        evaluator, root,
        node_id="application_timing",
        node_desc="Recommended application timing: apply after ≥3 years but before 6 service years",
        extracted=getattr(extracted, "application_timing", None),
        normative_claim=gt["application_timing"],
        critical_main=False,
        add_ins="Look for guidance to apply after three service years but before the sixth service year ends."
    )

    await add_requirement_check(
        evaluator, root,
        node_id="permanent_status",
        node_desc="Permanent status distinction: Instructional II is a permanent certificate",
        extracted=getattr(extracted, "permanent_status", None),
        normative_claim=gt["permanent_status"],
        critical_main=False,
        add_ins="Confirm that Instructional II is described as 'permanent' or 'does not expire' (distinct from temporary Level I)."
    )

    # Return standardized summary
    return evaluator.get_summary()