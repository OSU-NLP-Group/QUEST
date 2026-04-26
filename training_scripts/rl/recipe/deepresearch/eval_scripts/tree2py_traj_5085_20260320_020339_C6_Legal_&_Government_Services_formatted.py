import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "epa_ca_water_rule_min_days"
TASK_DESCRIPTION = (
    "The U.S. Environmental Protection Agency (EPA) is developing a significant new environmental regulation to address "
    "water quality standards in the state of California. The EPA plans to use a federal advisory committee to provide "
    "expert recommendations before proceeding with formal rulemaking. Calculate the absolute minimum number of calendar "
    "days required from the initial publication of the advisory committee meeting notice in the Federal Register to the "
    "date when the final rule becomes legally effective. Your calculation must account for all mandatory federal "
    "procedural requirements, including: Federal Advisory Committee Act (FACA) meeting notice requirements, "
    "Administrative Procedure Act (APA) notice-and-comment rulemaking procedures, Office of Information and Regulatory "
    "Affairs (OIRA) regulatory review under Executive Order 12866 for significant rules, and final rule publication and "
    "effectiveness requirements. For each mandatory waiting period you identify, provide: (1) The specific number of "
    "minimum calendar days required by law, (2) The legal authority (statute, executive order, or regulation) that "
    "establishes this requirement, (3) A reference URL that supports your answer. Then, calculate and state the total "
    "minimum number of calendar days for the complete process, explaining how you arrived at this total. Assume: the "
    "advisory committee meeting occurs immediately after the minimum notice period expires; the Notice of Proposed "
    "Rulemaking (NPRM) is published immediately after the advisory committee meeting; the comment period ends and OIRA "
    "review begins immediately after the minimum comment period; OIRA completes its review exactly at the 90-day "
    "maximum; the final rule is published immediately after OIRA approval; and all processes follow the minimum legal "
    "requirements with no additional delays."
)

# Ground-truth expectations used for summary context (not hard enforcement)
GT_INFO = {
    "expected_periods": {
        "FACA_notice_days": 15,
        "APA_comment_days": 30,         # As per rubric requirement
        "OIRA_review_days": 90,
        "APA_effective_delay_days": 30
    },
    "expected_minimum_total_days": 165  # 15 + 30 + 90 + 30
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TimelineExtraction(BaseModel):
    # FACA advisory committee meeting notice
    faca_notice_days: Optional[str] = None
    faca_citation_urls: List[str] = Field(default_factory=list)
    faca_fr_publication_stated: Optional[bool] = None
    faca_fr_publication_urls: List[str] = Field(default_factory=list)

    # APA notice-and-comment NPRM stage
    apa_comment_days: Optional[str] = None
    apa_citation_urls: List[str] = Field(default_factory=list)
    nprm_fr_publication_stated: Optional[bool] = None
    nprm_fr_publication_urls: List[str] = Field(default_factory=list)

    # OIRA review under EO 12866
    oira_review_days: Optional[str] = None
    oira_citation_urls: List[str] = Field(default_factory=list)
    significant_rule_stated: Optional[bool] = None
    significant_rule_urls: List[str] = Field(default_factory=list)

    # Final rule publication + effective date delay
    final_effective_delay_days: Optional[str] = None
    apa_effective_citation_urls: List[str] = Field(default_factory=list)
    final_fr_publication_stated: Optional[bool] = None
    final_fr_publication_urls: List[str] = Field(default_factory=list)

    # Total minimum timeline in the answer
    total_minimum_days: Optional[str] = None
    total_explanation: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_timeline() -> str:
    return """
Extract the rulemaking timing details mentioned in the answer. You must strictly follow these rules:
- Extract ONLY what is explicitly stated in the answer.
- For each requirement, extract numeric day counts as plain strings without units if possible (e.g., "15", "30", "90"). If the answer uses phrases like "15 days" or "15 calendar days", you may still return "15".
- For each legal authority, extract every URL the answer cites to support that requirement. Return an empty array if none are provided.
- For each Federal Register publication requirement, extract whether the answer explicitly states the publication requirement (true/false) and any URLs it cites for that statement.

Fields to extract:
1) FACA advisory committee meeting notice:
   - faca_notice_days: the day count stated for the minimum advance notice (e.g., "15")
   - faca_citation_urls: array of URLs supporting the FACA notice requirement (e.g., 41 C.F.R. §102-3.150, 5 U.S.C. §1009, GSA FACA guidance)
   - faca_fr_publication_stated: true if the answer explicitly states that the notice must be published in the Federal Register; otherwise false or null
   - faca_fr_publication_urls: URLs specifically supporting the Federal Register publication requirement for FACA notices

2) APA notice-and-comment (NPRM) period:
   - apa_comment_days: the minimum public comment period stated (e.g., "30")
   - apa_citation_urls: URLs supporting the stated APA comment period requirement (e.g., 5 U.S.C. §553)
   - nprm_fr_publication_stated: true if the answer states NPRMs must be published in the Federal Register
   - nprm_fr_publication_urls: URLs supporting that NPRMs must be published in the Federal Register

3) OIRA review under Executive Order 12866:
   - oira_review_days: the OIRA review duration stated (e.g., "90")
   - oira_citation_urls: URLs supporting the EO 12866 OIRA review timeline
   - significant_rule_stated: true if the answer says OIRA review applies to "significant" regulatory actions
   - significant_rule_urls: URLs supporting that OIRA review requirement applies to significant rules

4) Final rule publication and effective date:
   - final_effective_delay_days: the minimum delay between final rule publication and its effective date stated (e.g., "30")
   - apa_effective_citation_urls: URLs supporting the APA effective date requirement (e.g., 5 U.S.C. §553(d))
   - final_fr_publication_stated: true if the answer says final rules must be published in the Federal Register
   - final_fr_publication_urls: URLs supporting the Federal Register publication requirement for final rules

5) Total minimum timeline:
   - total_minimum_days: the total minimum number of calendar days stated in the answer for the entire process (e.g., "165")
   - total_explanation: the answer's textual explanation of how the total is calculated

If any field is not present in the answer, set it to null (for scalars) or [] (for URL arrays).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _coalesce_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                u = u.strip()
            if u and (u not in seen):
                seen.add(u)
                merged.append(u)
    return merged


def _num_str_is(value: Optional[str], target: int) -> bool:
    if not value:
        return False
    try:
        # Extract digits from common textual expressions
        stripped = "".join(ch for ch in value if ch.isdigit())
        if stripped == "":
            return False
        return int(stripped) == target
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Section builders                                                            #
# --------------------------------------------------------------------------- #
async def build_faca_section(evaluator: Evaluator, parent, data: TimelineExtraction):
    # Container: Advisory_Committee_Notice_Period
    faca_main = evaluator.add_parallel(
        id="Advisory_Committee_Notice_Period",
        desc="Verifies correct identification of FACA's 15-day advance notice requirement for federal advisory committee meetings published in the Federal Register",
        parent=parent,
        critical=False
    )

    # Sub: FACA_15Day_Requirement
    faca_15 = evaluator.add_parallel(
        id="FACA_15Day_Requirement",
        desc="Confirms the solution identifies that FACA requires 15 calendar days advance notice in the Federal Register for advisory committee meetings",
        parent=faca_main,
        critical=False
    )

    # Leaf: Notice_Period_Value (simple verification from answer)
    node_val = evaluator.add_leaf(
        id="Notice_Period_Value",
        desc="The solution specifies 15 calendar days as the FACA advance notice period",
        parent=faca_15,
        critical=True
    )
    await evaluator.verify(
        claim="The solution specifies the FACA advance meeting notice period is 15 calendar days.",
        node=node_val,
        additional_instruction="Judge only from the answer text. Accept reasonable wording variants like '15 days', '15 calendar days', or '15-day notice'."
    )

    # Leaf: FACA_Legal_Citation (verify by URLs)
    node_cite = evaluator.add_leaf(
        id="FACA_Legal_Citation",
        desc="The solution provides a valid reference URL citing 5 U.S.C. §1009 or 41 C.F.R. §102-3.150 for the FACA notice requirement",
        parent=faca_15,
        critical=True
    )
    faca_support_urls = _coalesce_urls(data.faca_citation_urls)
    await evaluator.verify(
        claim="The provided source(s) explicitly state that FACA/GSA regulations require at least 15 calendar days advance notice of advisory committee meetings in the Federal Register (e.g., 41 C.F.R. § 102-3.150 or 5 U.S.C. § 1009).",
        node=node_cite,
        sources=faca_support_urls if faca_support_urls else None,
        additional_instruction="Mark as not supported if no URLs are provided or if none of the pages explicitly state a 15 calendar day Federal Register notice requirement for advisory committee meetings."
    )

    # Sub: Federal_Register_Publication
    faca_pub = evaluator.add_parallel(
        id="Federal_Register_Publication",
        desc="Confirms the solution identifies that FACA meeting notices must be published in the Federal Register",
        parent=faca_main,
        critical=False
    )

    # Leaf: Publication_Venue
    node_pub = evaluator.add_leaf(
        id="Publication_Venue",
        desc="The solution states that FACA notices must appear in the Federal Register",
        parent=faca_pub,
        critical=True
    )
    faca_pub_urls = _coalesce_urls(data.faca_fr_publication_urls, data.faca_citation_urls)
    await evaluator.verify(
        claim="The provided source(s) indicate that advisory committee meeting notices under FACA must be published in the Federal Register.",
        node=node_pub,
        sources=faca_pub_urls if faca_pub_urls else None,
        additional_instruction="Support requires an explicit Federal Register publication requirement for FACA meeting notices. If no source is provided, mark as not supported."
    )


async def build_apa_notice_comment_section(evaluator: Evaluator, parent, data: TimelineExtraction):
    # Container: Notice_And_Comment_Period
    apa_main = evaluator.add_parallel(
        id="Notice_And_Comment_Period",
        desc="Verifies correct identification of the APA's 30-day minimum public comment period for proposed rules",
        parent=parent,
        critical=False
    )

    # Sub: APA_Comment_Period_Minimum
    apa_min = evaluator.add_parallel(
        id="APA_Comment_Period_Minimum",
        desc="Confirms the solution identifies that the Administrative Procedure Act requires at least 30 days for public comment after NPRM publication",
        parent=apa_main,
        critical=False
    )

    # Leaf: Comment_Period_Value
    node_val = evaluator.add_leaf(
        id="Comment_Period_Value",
        desc="The solution specifies 30 days as the minimum comment period required by the APA",
        parent=apa_min,
        critical=True
    )
    await evaluator.verify(
        claim="The solution specifies that the APA requires at least a 30-day public comment period after NPRM publication.",
        node=node_val,
        additional_instruction="Judge only from the answer text. Accept clear statements indicating a minimum 30-day APA comment period."
    )

    # Leaf: APA_Legal_Citation
    node_cite = evaluator.add_leaf(
        id="APA_Legal_Citation",
        desc="The solution provides a valid reference URL citing 5 U.S.C. § 553 for the comment period requirement",
        parent=apa_min,
        critical=True
    )
    apa_urls = _coalesce_urls(data.apa_citation_urls)
    await evaluator.verify(
        claim="At least one provided source explicitly ties a minimum 30-day public comment period to the Administrative Procedure Act (5 U.S.C. § 553).",
        node=node_cite,
        sources=apa_urls if apa_urls else None,
        additional_instruction="Mark as not supported if no URLs are provided or if none explicitly support a 30-day minimum APA comment period."
    )

    # Sub: NPRM_Federal_Register_Publication
    nprm_pub = evaluator.add_parallel(
        id="NPRM_Federal_Register_Publication",
        desc="Confirms the solution identifies that proposed rules must be published in the Federal Register",
        parent=apa_main,
        critical=False
    )

    # Leaf: NPRM_Publication_Requirement
    node_nprm_pub = evaluator.add_leaf(
        id="NPRM_Publication_Requirement",
        desc="The solution states that NPRMs must be published in the Federal Register under the APA",
        parent=nprm_pub,
        critical=True
    )
    nprm_pub_urls = _coalesce_urls(data.nprm_fr_publication_urls, data.apa_citation_urls)
    await evaluator.verify(
        claim="The provided source(s) indicate that notices of proposed rulemaking (NPRMs) must be published in the Federal Register under the Administrative Procedure Act.",
        node=node_nprm_pub,
        sources=nprm_pub_urls if nprm_pub_urls else None,
        additional_instruction="Support requires an explicit Federal Register publication requirement for NPRMs under APA (e.g., 5 U.S.C. § 553(b)). If no source is provided, mark as not supported."
    )


async def build_oira_section(evaluator: Evaluator, parent, data: TimelineExtraction):
    # Container: OIRA_Review_Period
    oira_main = evaluator.add_parallel(
        id="OIRA_Review_Period",
        desc="Verifies correct identification of OIRA's 90-day maximum review timeline for significant regulatory actions under Executive Order 12866",
        parent=parent,
        critical=False  # keep non-critical to allow non-critical child per framework constraint
    )

    # Sub: OIRA_90Day_Review_Maximum
    oira_90 = evaluator.add_parallel(
        id="OIRA_90Day_Review_Maximum",
        desc="Confirms the solution identifies that OIRA has up to 90 calendar days to review significant rules under EO 12866",
        parent=oira_main,
        critical=False
    )

    # Leaf: Review_Period_Value
    node_val = evaluator.add_leaf(
        id="Review_Period_Value",
        desc="The solution specifies 90 calendar days as the OIRA review period for significant rules",
        parent=oira_90,
        critical=True
    )
    await evaluator.verify(
        claim="The solution specifies that OIRA review under Executive Order 12866 lasts up to 90 calendar days.",
        node=node_val,
        additional_instruction="Judge from the answer text; accept reasonable phrasings like 'up to 90 days' or '90-day maximum'."
    )

    # Leaf: EO_12866_Citation
    node_cite = evaluator.add_leaf(
        id="EO_12866_Citation",
        desc="The solution provides a valid reference URL citing Executive Order 12866 for the OIRA review timeline",
        parent=oira_90,
        critical=True
    )
    oira_urls = _coalesce_urls(data.oira_citation_urls)
    await evaluator.verify(
        claim="At least one provided source explicitly cites Executive Order 12866 and states that OIRA review is up to 90 days (or 90-day maximum).",
        node=node_cite,
        sources=oira_urls if oira_urls else None,
        additional_instruction="Mark as not supported if no URLs are provided or if none explicitly support the 90-day review timeline under EO 12866."
    )

    # Sub: Significant_Rule_Classification (non-critical)
    signif = evaluator.add_parallel(
        id="Significant_Rule_Classification",
        desc="Confirms the solution recognizes that the OIRA review requirement applies to significant regulatory actions",
        parent=oira_main,
        critical=False
    )

    # Leaf: Significant_Rule_Identification (non-critical)
    node_signif = evaluator.add_leaf(
        id="Significant_Rule_Identification",
        desc="The solution notes that OIRA review applies specifically to significant rules as defined by EO 12866",
        parent=signif,
        critical=False
    )
    signif_urls = _coalesce_urls(data.significant_rule_urls, data.oira_citation_urls)
    await evaluator.verify(
        claim="The answer acknowledges that OIRA review under EO 12866 applies to 'significant' regulatory actions (as defined in the order).",
        node=node_signif,
        sources=signif_urls if signif_urls else None,
        additional_instruction="You may confirm either from the answer text or from the provided EO 12866 sources that the review requirement targets significant regulatory actions."
    )


async def build_final_rule_effective_section(evaluator: Evaluator, parent, data: TimelineExtraction):
    # Container: Final_Rule_Effective_Date_Delay
    final_main = evaluator.add_parallel(
        id="Final_Rule_Effective_Date_Delay",
        desc="Verifies correct identification of the APA's 30-day minimum delay between final rule publication and effectiveness",
        parent=parent,
        critical=False
    )

    # Sub: APA_30Day_Effective_Delay
    eff_30 = evaluator.add_parallel(
        id="APA_30Day_Effective_Delay",
        desc="Confirms the solution identifies that the APA requires final rules to have an effective date at least 30 days after Federal Register publication",
        parent=final_main,
        critical=False
    )

    # Leaf: Effective_Delay_Value
    node_val = evaluator.add_leaf(
        id="Effective_Delay_Value",
        desc="The solution specifies 30 days as the minimum delay between final rule publication and effectiveness",
        parent=eff_30,
        critical=True
    )
    await evaluator.verify(
        claim="The solution specifies that a final rule’s effective date must be at least 30 days after its Federal Register publication.",
        node=node_val,
        additional_instruction="Judge from the answer text. Accept formulations like '30-day delayed effective date' or '30 days after publication' (ignoring exceptions)."
    )

    # Leaf: APA_Effective_Date_Citation
    node_cite = evaluator.add_leaf(
        id="APA_Effective_Date_Citation",
        desc="The solution provides a valid reference URL citing 5 U.S.C. § 553(d) for the effective date delay requirement",
        parent=eff_30,
        critical=True
    )
    apa_eff_urls = _coalesce_urls(data.apa_effective_citation_urls)
    await evaluator.verify(
        claim="At least one provided source explicitly cites 5 U.S.C. § 553(d) and states that a rule is generally effective no less than 30 days after publication (subject to exceptions).",
        node=node_cite,
        sources=apa_eff_urls if apa_eff_urls else None,
        additional_instruction="Mark as not supported if no URLs are provided or if none explicitly support the 30-day effective-date delay under 5 U.S.C. § 553(d)."
    )

    # Sub: Final_Rule_Federal_Register_Publication
    final_pub = evaluator.add_parallel(
        id="Final_Rule_Federal_Register_Publication",
        desc="Confirms the solution identifies that final rules must be published in the Federal Register",
        parent=final_main,
        critical=False
    )

    # Leaf: Final_Rule_Publication_Requirement
    node_pub = evaluator.add_leaf(
        id="Final_Rule_Publication_Requirement",
        desc="The solution states that final rules must be published in the Federal Register under the APA",
        parent=final_pub,
        critical=True
    )
    final_pub_urls = _coalesce_urls(data.final_fr_publication_urls, data.apa_effective_citation_urls)
    await evaluator.verify(
        claim="The provided source(s) indicate that final rules must be published in the Federal Register (under APA/publication requirements).",
        node=node_pub,
        sources=final_pub_urls if final_pub_urls else None,
        additional_instruction="Support requires an explicit Federal Register publication requirement for final rules (e.g., 5 U.S.C. § 552(a)(1)(D) or related authoritative sources). If no source is provided, mark as not supported."
    )


async def build_total_timeline_section(evaluator: Evaluator, parent, data: TimelineExtraction):
    # Container: Total_Minimum_Timeline_Calculation (sequential)
    total_main = evaluator.add_sequential(
        id="Total_Minimum_Timeline_Calculation",
        desc="Verifies that the solution correctly sums all mandatory waiting periods to calculate the total minimum timeline",
        parent=parent,
        critical=False
    )

    # Sub: All_Periods_Included (parallel)
    included = evaluator.add_parallel(
        id="All_Periods_Included",
        desc="Confirms the solution includes all four mandatory waiting periods in the calculation: FACA notice (15 days), public comment (30 days), OIRA review (90 days), and effective date delay (30 days)",
        parent=total_main,
        critical=False
    )

    # Leaf: FACA_Period_Included
    node_faca_inc = evaluator.add_leaf(
        id="FACA_Period_Included",
        desc="The calculation includes the 15-day FACA notice period",
        parent=included,
        critical=True
    )
    await evaluator.verify(
        claim="The solution's total calculation includes the 15-day FACA advisory committee meeting notice period.",
        node=node_faca_inc,
        additional_instruction="Judge from the answer text; inclusion may be stated explicitly or implied by summing the 15-day period."
    )

    # Leaf: Comment_Period_Included
    node_cmt_inc = evaluator.add_leaf(
        id="Comment_Period_Included",
        desc="The calculation includes the 30-day comment period",
        parent=included,
        critical=True
    )
    await evaluator.verify(
        claim="The solution's total calculation includes a 30-day public comment period.",
        node=node_cmt_inc,
        additional_instruction="Judge from the answer text; inclusion may be explicit or implied by the sum."
    )

    # Leaf: OIRA_Period_Included
    node_oira_inc = evaluator.add_leaf(
        id="OIRA_Period_Included",
        desc="The calculation includes the 90-day OIRA review period",
        parent=included,
        critical=True
    )
    await evaluator.verify(
        claim="The solution's total calculation includes the 90-day OIRA review period under EO 12866.",
        node=node_oira_inc,
        additional_instruction="Judge from the answer text; inclusion may be explicit or implied by the sum."
    )

    # Leaf: Effective_Delay_Included
    node_eff_inc = evaluator.add_leaf(
        id="Effective_Delay_Included",
        desc="The calculation includes the 30-day effective date delay",
        parent=included,
        critical=True
    )
    await evaluator.verify(
        claim="The solution's total calculation includes the 30-day delay between final rule publication and its effective date.",
        node=node_eff_inc,
        additional_instruction="Judge from the answer text; inclusion may be explicit or implied by the sum."
    )

    # Sub: Correct_Total_Minimum
    correct_total = evaluator.add_parallel(
        id="Correct_Total_Minimum",
        desc="Confirms the solution calculates the correct minimum total of at least 165 calendar days (15+30+90+30)",
        parent=total_main,
        critical=False
    )

    # Leaf: Minimum_165_Days
    node_min_165 = evaluator.add_leaf(
        id="Minimum_165_Days",
        desc="The solution states that the minimum total timeline is at least 165 calendar days",
        parent=correct_total,
        critical=True
    )
    await evaluator.verify(
        claim="The solution states a total minimum timeline of at least 165 calendar days.",
        node=node_min_165,
        additional_instruction="Accept values equal to or greater than 165 days. Do not penalize if the answer explicitly states 'at least 165 days' or a higher minimum."
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
    # Initialize evaluator (root is a non-critical parallel aggregator)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_timeline(),
        template_class=TimelineExtraction,
        extraction_name="timeline_extraction"
    )

    # Record ground-truth expectations for transparency in the summary
    evaluator.add_ground_truth(GT_INFO, gt_type="expected_requirements")

    # Top-level rubric container (kept non-critical to satisfy framework constraints with mixed children)
    top = evaluator.add_parallel(
        id="Federal_Rulemaking_Timeline_Compliance",
        desc="Evaluates whether the calculated minimum timeline correctly accounts for all mandatory procedural waiting periods required by federal law",
        parent=root,
        critical=False
    )

    # Build sections
    await build_faca_section(evaluator, top, extracted)
    await build_apa_notice_comment_section(evaluator, top, extracted)
    await build_oira_section(evaluator, top, extracted)
    await build_final_rule_effective_section(evaluator, top, extracted)
    await build_total_timeline_section(evaluator, top, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()