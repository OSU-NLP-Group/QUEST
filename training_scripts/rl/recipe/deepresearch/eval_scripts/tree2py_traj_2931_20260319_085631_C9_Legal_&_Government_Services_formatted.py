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
TASK_ID = "scotus_multistate_amicus_2025_2026_window_FebApr"
TASK_DESCRIPTION = """
Identify three Supreme Court cases from the 2025-2026 term in which multistate coalitions of state attorneys general filed amicus curiae briefs at the merits stage, where oral arguments were scheduled between February 1 and April 30, 2026. For each case, provide the following information: 
(1) Case Identification: The full case name, Supreme Court docket number, and a link to the official Supreme Court docket page or case information. 
(2) Legal Issue: The primary legal question or constitutional issue being addressed by the Court. 
(3) Oral Argument Date: The specific date (or date range) when oral arguments were scheduled or occurred. 
(4) Coalition Details: The minimum number of state attorneys general who joined the multistate amicus brief (must be at least 15), a complete list of all participating states or jurisdictions, the name and state of the lead attorney general who coordinated the coalition, evidence of coordination (such as a joint press release or coalition announcement), and a link to the lead attorney general's press release or official announcement about the brief. 
(5) Filing Information: The exact date the multistate amicus brief was filed with the Supreme Court, which party the brief supports (petitioner, respondent, or neither party), confirmation that the brief was filed at the merits stage (after certiorari was granted), not during the petition stage, and a link to the actual amicus brief or the Supreme Court's filing record showing the brief. 
(6) Federal Position (if available): The position taken by the U.S. Solicitor General in the case and whether it supports, opposes, or is neutral to the state attorneys general coalition's position. 
All cases must involve substantive legal issues related to federal policy, constitutional interpretation, or interstate matters, rather than purely procedural disputes.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CaseIdentification(BaseModel):
    case_name: Optional[str] = None
    docket_number: Optional[str] = None
    docket_url: Optional[str] = None
    term: Optional[str] = None  # e.g., "October Term 2025" or "2025-2026"


class CaseTemporal(BaseModel):
    oral_argument_date: Optional[str] = None  # allow date or range as a string


class CaseIssue(BaseModel):
    question_presented: Optional[str] = None
    issue_category: Optional[str] = None  # e.g., "constitutional", "federal policy", "interstate", etc.


class CoalitionInfo(BaseModel):
    min_states_count: Optional[str] = None  # keep as string for robustness (e.g., "20+" or "at least 19")
    state_list: List[str] = Field(default_factory=list)
    lead_ag_name: Optional[str] = None
    lead_ag_state: Optional[str] = None
    coordination_evidence: Optional[str] = None
    lead_ag_url: Optional[str] = None
    partisan_alignment: Optional[str] = None  # optional, e.g., "bipartisan", "predominantly Democratic"
    regions_represented: List[str] = Field(default_factory=list)


class FilingInfo(BaseModel):
    stage: Optional[str] = None  # expect values like "merits", "petition"
    brief_type: Optional[str] = None  # expect "amicus curiae"
    filing_date: Optional[str] = None
    supported_party: Optional[str] = None  # "petitioner", "respondent", or "neither"
    brief_urls: List[str] = Field(default_factory=list)


class FederalPosition(BaseModel):
    sg_position: Optional[str] = None  # "support", "oppose", "neutral" (or textual description)
    sg_brief_url: Optional[str] = None


class CaseInfo(BaseModel):
    identification: CaseIdentification = Field(default_factory=CaseIdentification)
    temporal: CaseTemporal = Field(default_factory=CaseTemporal)
    issue: CaseIssue = Field(default_factory=CaseIssue)
    coalition: CoalitionInfo = Field(default_factory=CoalitionInfo)
    filing: FilingInfo = Field(default_factory=FilingInfo)
    federal: FederalPosition = Field(default_factory=FederalPosition)


class CasesExtraction(BaseModel):
    cases: List[CaseInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cases() -> str:
    return """
    Extract up to the first three qualifying Supreme Court cases from the provided answer. 
    A qualifying case must be from the 2025-2026 Supreme Court term, include a multistate coalition of at least 15 state attorneys general filing an amicus curiae brief at the merits stage, and have oral argument scheduled or occurred between February 1 and April 30, 2026.

    For each case, return a JSON object with the following nested structure:

    {
      "cases": [
        {
          "identification": {
            "case_name": string or null,
            "docket_number": string or null,
            "docket_url": string or null,   // Prefer official Supreme Court docket/case page on supremecourt.gov
            "term": string or null          // e.g., "October Term 2025" or "2025-2026"
          },
          "temporal": {
            "oral_argument_date": string or null // specific date or date range in free text
          },
          "issue": {
            "question_presented": string or null, // short paraphrase ok
            "issue_category": string or null      // choose from: "constitutional", "federal policy", "interstate", or "other/substantive"; avoid "procedural"
          },
          "coalition": {
            "min_states_count": string or null,   // keep as text, e.g., "20", "20+", or "at least 19"
            "state_list": [strings],              // all participating states/jurisdictions as named in the answer
            "lead_ag_name": string or null,
            "lead_ag_state": string or null,
            "coordination_evidence": string or null, // brief note e.g., "press release announcing coalition"
            "lead_ag_url": string or null,        // URL to the lead AG press release/announcement
            "partisan_alignment": string or null, // optional descriptor like "bipartisan"
            "regions_represented": [strings]      // optional list of U.S. regions represented; free text allowed
          },
          "filing": {
            "stage": string or null,             // expect "merits" if qualifying
            "brief_type": string or null,        // expect "amicus curiae"
            "filing_date": string or null,       // exact date string if provided
            "supported_party": string or null,   // "petitioner", "respondent", or "neither"
            "brief_urls": [strings]              // URLs to the brief (PDF) and/or Supreme Court docket filing record
          },
          "federal": {
            "sg_position": string or null,       // U.S. Solicitor General's position if mentioned; e.g., "support", "oppose", "neutral"
            "sg_brief_url": string or null       // URL to SG/DOJ brief or statement if available
          }
        }
      ]
    }

    Rules:
    - Extract strictly from the provided answer text; do not invent URLs or data.
    - If any field is missing in the answer, set it to null (or an empty list for arrays).
    - Normalize state names sensibly (e.g., "District of Columbia" or "D.C." are acceptable).
    - The docket_url should be the official Supreme Court site if present (supremecourt.gov). If multiple links are provided, choose the official SCOTUS link for docket_url and put others (e.g., brief PDFs) into filing.brief_urls.
    - Return at most three cases in the "cases" array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_nonempty(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for u in urls:
        if u and isinstance(u, str):
            u = u.strip()
            if u and u not in seen:
                seen.add(u)
                result.append(u)
    return result


def _combine_sources(*many_lists: List[str]) -> List[str]:
    combo: List[str] = []
    for lst in many_lists:
        combo.extend(lst or [])
    return _unique_nonempty(combo)


# --------------------------------------------------------------------------- #
# Verification logic per case                                                 #
# --------------------------------------------------------------------------- #
async def verify_single_case(evaluator: Evaluator, parent_node, case: CaseInfo, idx: int) -> None:
    """
    Build the verification subtree and run checks for a single case.
    Follows the rubric tree with slight adjustments to satisfy framework constraints:
    - 'coalition_characteristics' is set to non-critical to allow a non-critical child ('coalition_composition').
    """
    case_num = idx + 1
    id_prefix = f"case_{case_num}"

    # Top-level case node (parallel, non-critical to allow partial credit across cases)
    case_node = evaluator.add_parallel(
        id=f"{id_prefix}",
        desc=f"{['First','Second','Third'][idx]} qualifying Supreme Court case with multistate AG amicus brief",
        parent=parent_node,
        critical=False
    )

    # 1) Basic information and identification (critical group)
    basic_info = evaluator.add_parallel(
        id=f"{id_prefix}_basic_information",
        desc="Basic case information and identification",
        parent=case_node,
        critical=True
    )

    # 1.a) Proper case identification (critical)
    case_ident = evaluator.add_parallel(
        id=f"{id_prefix}_case_identification",
        desc="Proper case identification",
        parent=basic_info,
        critical=True
    )

    # Leaf: case name + docket number match docket page
    node_case_name = evaluator.add_leaf(
        id=f"{id_prefix}_case_name",
        desc="Correct case name and docket number provided",
        parent=case_ident,
        critical=True
    )
    ci = case.identification
    docket_sources = _unique_nonempty([ci.docket_url])
    claim_case_name = (
        f"The official Supreme Court docket page shows that the case caption is '{ci.case_name}' "
        f"and the docket number is '{ci.docket_number}'."
    )
    await evaluator.verify(
        claim=claim_case_name,
        node=node_case_name,
        sources=docket_sources or None,
        additional_instruction="Use the official SCOTUS docket/case page to confirm both the caption and docket number. "
                               "If either value is blank, 'null', or not matching, mark Incorrect. Allow minor formatting variations."
    )

    # Leaf: docket URL is an official SCOTUS page
    node_case_url = evaluator.add_leaf(
        id=f"{id_prefix}_case_citation_url",
        desc="URL reference to Supreme Court docket or official case page",
        parent=case_ident,
        critical=True
    )
    claim_case_url = (
        f"This URL is the official Supreme Court docket or case information page for docket '{ci.docket_number}' "
        f"and case '{ci.case_name}'."
    )
    await evaluator.verify(
        claim=claim_case_url,
        node=node_case_url,
        sources=docket_sources or None,
        additional_instruction="Verify that the page is on supremecourt.gov and corresponds to this case. "
                               "If the URL is not official or is unrelated, mark Incorrect."
    )

    # 1.b) Temporal requirements (critical)
    temporal = evaluator.add_parallel(
        id=f"{id_prefix}_temporal_requirements",
        desc="Case meets temporal constraints",
        parent=basic_info,
        critical=True
    )

    # Leaf: correct term (2025-2026)
    node_term = evaluator.add_leaf(
        id=f"{id_prefix}_correct_term",
        desc="Case is from the 2025-2026 Supreme Court term (October 2025 - October 2026)",
        parent=temporal,
        critical=True
    )
    claim_term = "This case belongs to the Supreme Court October Term 2025 (the 2025-2026 term)."
    await evaluator.verify(
        claim=claim_term,
        node=node_term,
        sources=docket_sources or None,
        additional_instruction="Confirm via the official Supreme Court docket/case page. Accept 'OT 2025' or equivalent phrasing."
    )

    # Leaf: oral argument in window Feb 1 - Apr 30, 2026
    node_arg_window = evaluator.add_leaf(
        id=f"{id_prefix}_oral_argument_window",
        desc="Oral argument scheduled or occurred between February 1 and April 30, 2026",
        parent=temporal,
        critical=True
    )
    oral_date = case.temporal.oral_argument_date or ""
    claim_arg_window = (
        f"The oral argument date for this case is '{oral_date}', which falls between February 1, 2026 and April 30, 2026 (inclusive)."
    )
    await evaluator.verify(
        claim=claim_arg_window,
        node=node_arg_window,
        sources=docket_sources or None,
        additional_instruction="Use the SCOTUS docket/case page to check the oral argument date (scheduled or occurred). "
                               "If the extracted date is blank or outside the window, mark Incorrect."
    )

    # 1.c) Legal issue (critical)
    legal_issue = evaluator.add_parallel(
        id=f"{id_prefix}_legal_issue",
        desc="Primary legal issue identification",
        parent=basic_info,
        critical=True
    )

    # Leaf: question presented
    node_qp = evaluator.add_leaf(
        id=f"{id_prefix}_question_presented",
        desc="Clear statement of the primary legal question or issue before the Court",
        parent=legal_issue,
        critical=True
    )
    qp = case.issue.question_presented or ""
    claim_qp = f"The primary legal issue or question presented for this case is: \"{qp}\"."
    await evaluator.verify(
        claim=claim_qp,
        node=node_qp,
        sources=docket_sources or None,
        additional_instruction="Confirm that this statement reasonably matches the question presented or core legal issue on the official SCOTUS page. "
                               "Paraphrase is acceptable if equivalent. If blank, mark Incorrect."
    )

    # Leaf: issue category is substantive (federal policy/constitutional/interstate)
    node_issue_cat = evaluator.add_leaf(
        id=f"{id_prefix}_issue_category",
        desc="Issue involves federal policy, constitutional interpretation, or interstate matters (not procedural)",
        parent=legal_issue,
        critical=True
    )
    issue_cat = case.issue.issue_category or ""
    claim_issue_cat = (
        f"Based on the official case materials, the issue is substantive (e.g., federal policy, constitutional interpretation, or interstate), "
        f"and not purely procedural. The extracted category is '{issue_cat}'."
    )
    await evaluator.verify(
        claim=claim_issue_cat,
        node=node_issue_cat,
        sources=docket_sources or None,
        additional_instruction="Use the SCOTUS docket/case description to judge substance vs. procedure. "
                               "If clearly procedural only or category is blank, mark Incorrect."
    )

    # 2) Coalition characteristics (Set to NON-CRITICAL at parent to allow a non-critical sub-branch as per framework constraints)
    coalition = evaluator.add_parallel(
        id=f"{id_prefix}_coalition_characteristics",
        desc="Multistate attorney general coalition details",
        parent=case_node,
        critical=False  # adjusted from JSON critical to satisfy critical-children consistency
    )

    # 2.a) Coalition size requirements (critical within this group)
    coalition_size = evaluator.add_parallel(
        id=f"{id_prefix}_coalition_size_requirements",
        desc="Coalition size meets threshold",
        parent=coalition,
        critical=True
    )
    # Leaf: at least 15 AGs
    node_min_states = evaluator.add_leaf(
        id=f"{id_prefix}_minimum_states",
        desc="Coalition includes at least 15 state attorneys general as signatories",
        parent=coalition_size,
        critical=True
    )
    min_states_str = case.coalition.min_states_count or ""
    sources_size = _combine_sources([case.coalition.lead_ag_url] if case.coalition.lead_ag_url else [],
                                    case.filing.brief_urls)
    claim_min_states = (
        "At least 15 state attorneys general participated in a multistate amicus brief in this case "
        f"(the answer indicates '{min_states_str}' and/or provides a state list)."
    )
    await evaluator.verify(
        claim=claim_min_states,
        node=node_min_states,
        sources=sources_size or None,
        additional_instruction="Use the lead AG press release and/or the amicus brief to confirm the coalition size. "
                               "If fewer than 15 or unclear, mark Incorrect."
    )

    # Leaf: full list of participating states
    node_state_list = evaluator.add_leaf(
        id=f"{id_prefix}_state_list",
        desc="Complete list of participating states or jurisdictions provided",
        parent=coalition_size,
        critical=True
    )
    state_list = case.coalition.state_list or []
    claim_state_list = (
        f"The participating states/jurisdictions for this coalition include exactly or at least these: {state_list}."
    )
    await evaluator.verify(
        claim=claim_state_list,
        node=node_state_list,
        sources=sources_size or None,
        additional_instruction="Match the listed states/jurisdictions against the press release and/or brief signatories. "
                               "Minor naming variations (e.g., 'D.C.' vs 'District of Columbia') are acceptable."
    )

    # 2.b) Coalition leadership (critical within this group)
    coalition_lead = evaluator.add_parallel(
        id=f"{id_prefix}_coalition_leadership",
        desc="Coalition coordination and leadership",
        parent=coalition,
        critical=True
    )
    # Leaf: lead AG identification
    node_lead_ag = evaluator.add_leaf(
        id=f"{id_prefix}_lead_attorney_general",
        desc="Identification of the lead attorney general(s) who coordinated the brief",
        parent=coalition_lead,
        critical=True
    )
    lead_name = case.coalition.lead_ag_name or ""
    lead_state = case.coalition.lead_ag_state or ""
    claim_lead_ag = f"The coalition was coordinated/led by {lead_name}, Attorney General of {lead_state}."
    await evaluator.verify(
        claim=claim_lead_ag,
        node=node_lead_ag,
        sources=_unique_nonempty([case.coalition.lead_ag_url]) or sources_size or None,
        additional_instruction="Confirm leadership/coordination per the press release; co-leadership is acceptable if consistent. "
                               "If blank, mark Incorrect."
    )

    # Leaf: coordination evidence
    node_coord_evidence = evaluator.add_leaf(
        id=f"{id_prefix}_coordination_evidence",
        desc="Evidence of coordination (joint press release, named lead state, or identical filing)",
        parent=coalition_lead,
        critical=True
    )
    coord_ev = case.coalition.coordination_evidence or "official coalition press release/announcement"
    claim_coord_ev = (
        f"There is evidence of coalition coordination for this case, such as a joint press release or named lead state (e.g., '{coord_ev}')."
    )
    await evaluator.verify(
        claim=claim_coord_ev,
        node=node_coord_evidence,
        sources=_unique_nonempty([case.coalition.lead_ag_url]) or sources_size or None,
        additional_instruction="Check for explicit coalition language (e.g., 'led by', 'joined by X states', 'multistate coalition')."
    )

    # Leaf: lead AG URL validity
    node_lead_ag_url = evaluator.add_leaf(
        id=f"{id_prefix}_lead_ag_url",
        desc="URL reference to lead AG's press release or coalition announcement",
        parent=coalition_lead,
        critical=True
    )
    claim_lead_ag_url = (
        f"This URL is an official press release or coalition announcement by the Attorney General of {lead_state} ({lead_name}) about this brief."
    )
    await evaluator.verify(
        claim=claim_lead_ag_url,
        node=node_lead_ag_url,
        sources=_unique_nonempty([case.coalition.lead_ag_url]) or None,
        additional_instruction="Verify this is an official AG website or equivalent official channel for the lead/coordination announcement."
    )

    # 2.c) Coalition composition (NON-CRITICAL)
    coalition_comp = evaluator.add_parallel(
        id=f"{id_prefix}_coalition_composition",
        desc="Coalition composition characteristics",
        parent=coalition,
        critical=False
    )
    # Leaf: partisan alignment (non-critical)
    node_partisan = evaluator.add_leaf(
        id=f"{id_prefix}_partisan_alignment",
        desc="Partisan composition identified (bipartisan, predominantly Democratic, or predominantly Republican)",
        parent=coalition_comp,
        critical=False
    )
    partisan = case.coalition.partisan_alignment or ""
    claim_partisan = f"The coalition's partisan composition can be characterized as '{partisan}'."
    await evaluator.verify(
        claim=claim_partisan,
        node=node_partisan,
        sources=sources_size or None,
        additional_instruction="If the sources support a reasonable inference (e.g., based on states/AGs), mark Correct; otherwise mark Incorrect."
    )

    # Leaf: regional diversity (non-critical)
    node_regions = evaluator.add_leaf(
        id=f"{id_prefix}_regional_diversity",
        desc="Coalition includes states from at least three different U.S. regions",
        parent=coalition_comp,
        critical=False
    )
    claim_regions = (
        f"From the participating states list {state_list}, the coalition includes states from at least three different U.S. regions."
    )
    await evaluator.verify(
        claim=claim_regions,
        node=node_regions,
        sources=sources_size or None,
        additional_instruction="Use the listed states on the page(s) to infer regions (e.g., U.S. Census regions). If insufficient diversity is evident, mark Incorrect."
    )

    # 3) Filing compliance (critical)
    filing_grp = evaluator.add_parallel(
        id=f"{id_prefix}_filing_compliance",
        desc="Amicus brief filing information and compliance",
        parent=case_node,
        critical=True
    )

    # 3.a) Filing stage verification (critical)
    filing_stage_grp = evaluator.add_parallel(
        id=f"{id_prefix}_filing_stage_verification",
        desc="Verification of filing stage",
        parent=filing_grp,
        critical=True
    )

    # Leaf: merits stage (not petition stage)
    node_merits = evaluator.add_leaf(
        id=f"{id_prefix}_merits_stage",
        desc="Brief was filed at the merits stage (after certiorari was granted, not during petition stage)",
        parent=filing_stage_grp,
        critical=True
    )
    stage = case.filing.stage or ""
    claim_merits = (
        "The state AG amicus brief in this case was filed at the merits stage (after certiorari was granted), "
        "not at the petition stage."
    )
    await evaluator.verify(
        claim=claim_merits,
        node=node_merits,
        sources=_combine_sources(docket_sources, case.filing.brief_urls) or None,
        additional_instruction="Check the SCOTUS docket entries and brief header/cover to confirm 'merits' timing (e.g., 'on the merits')."
    )

    # Leaf: brief type is amicus curiae
    node_brief_type = evaluator.add_leaf(
        id=f"{id_prefix}_brief_type",
        desc="Brief is identified as amicus curiae brief (not party brief)",
        parent=filing_stage_grp,
        critical=True
    )
    brief_type = case.filing.brief_type or ""
    claim_brief_type = f"The filing is an amicus curiae brief by state attorneys general (extracted type: '{brief_type}')."
    await evaluator.verify(
        claim=claim_brief_type,
        node=node_brief_type,
        sources=_combine_sources(docket_sources, case.filing.brief_urls) or None,
        additional_instruction="Confirm the document title/cover and docket entry indicate 'amicus curiae' and that it's from state AGs."
    )

    # 3.b) Filing details (critical)
    filing_details = evaluator.add_parallel(
        id=f"{id_prefix}_filing_details",
        desc="Specific filing information",
        parent=filing_grp,
        critical=True
    )

    # Leaf: filing date
    node_filing_date = evaluator.add_leaf(
        id=f"{id_prefix}_filing_date",
        desc="Specific date when the multistate amicus brief was filed with the Court",
        parent=filing_details,
        critical=True
    )
    filing_date = case.filing.filing_date or ""
    claim_filing_date = f"The multistate AG amicus brief was filed on {filing_date}."
    await evaluator.verify(
        claim=claim_filing_date,
        node=node_filing_date,
        sources=_combine_sources(docket_sources, case.filing.brief_urls) or None,
        additional_instruction="Verify the filing date from the SCOTUS docket entry and/or the brief cover. If the value is blank, mark Incorrect."
    )

    # Leaf: supported party
    node_supported = evaluator.add_leaf(
        id=f"{id_prefix}_supported_party",
        desc="Identification of which party the brief supports (petitioner, respondent, or neither party)",
        parent=filing_details,
        critical=True
    )
    supported_party = case.filing.supported_party or ""
    claim_supported = f"The amicus brief supports the '{supported_party}'."
    await evaluator.verify(
        claim=claim_supported,
        node=node_supported,
        sources=_combine_sources(case.filing.brief_urls, docket_sources, [case.coalition.lead_ag_url] if case.coalition.lead_ag_url else []) or None,
        additional_instruction="Confirm from the brief cover or press release whether it supports petitioner/respondent/neither. If blank, mark Incorrect."
    )

    # Leaf: brief URL(s) validity
    node_brief_url = evaluator.add_leaf(
        id=f"{id_prefix}_brief_url",
        desc="URL reference to the actual amicus brief or Supreme Court filing record",
        parent=filing_details,
        critical=True
    )
    claim_brief_url = "At least one of the provided URLs points to the amicus brief (PDF) or the Supreme Court's filing record for the brief in this case."
    await evaluator.verify(
        claim=claim_brief_url,
        node=node_brief_url,
        sources=case.filing.brief_urls or docket_sources or None,
        additional_instruction="Pass if any link is the brief PDF or the exact SCOTUS filing entry for the brief."
    )

    # 3.c) Accessibility (critical)
    access_grp = evaluator.add_parallel(
        id=f"{id_prefix}_accessibility",
        desc="Brief accessibility verification",
        parent=filing_grp,
        critical=True
    )
    node_public_access = evaluator.add_leaf(
        id=f"{id_prefix}_public_access",
        desc="Brief is publicly accessible through official state AG website, Supreme Court website, or verified legal database",
        parent=access_grp,
        critical=True
    )
    claim_public_access = (
        "The amicus brief is publicly accessible via an official state AG website, the Supreme Court website, or a verified legal database."
    )
    await evaluator.verify(
        claim=claim_public_access,
        node=node_public_access,
        sources=case.filing.brief_urls or docket_sources or None,
        additional_instruction="If the linked page is a paywalled or inaccessible resource without a public copy, mark Incorrect."
    )

    # 4) Federal relationship (non-critical)
    fed_grp = evaluator.add_parallel(
        id=f"{id_prefix}_federal_relationship",
        desc="Federal government position relative to the multistate coalition",
        parent=case_node,
        critical=False
    )
    sg_grp = evaluator.add_parallel(
        id=f"{id_prefix}_solicitor_general",
        desc="U.S. Solicitor General's position",
        parent=fed_grp,
        critical=False
    )
    # Leaf: SG alignment (non-critical)
    node_sg_align = evaluator.add_leaf(
        id=f"{id_prefix}_sg_alignment",
        desc="Solicitor General's position identified as supporting, opposing, or neutral to state AGs' position",
        parent=sg_grp,
        critical=False
    )
    sg_pos = case.federal.sg_position or ""
    claim_sg_align = f"The U.S. Solicitor General's position in this case is '{sg_pos}', characterized relative to the state AG coalition."
    await evaluator.verify(
        claim=claim_sg_align,
        node=node_sg_align,
        sources=_unique_nonempty([case.federal.sg_brief_url]) or None,
        additional_instruction="Confirm via SG/DOJ brief or official page. If not available in sources or blank, mark Incorrect."
    )

    # Leaf: SG brief URL (non-critical)
    node_sg_url = evaluator.add_leaf(
        id=f"{id_prefix}_sg_brief_url",
        desc="URL reference to Solicitor General's brief or DOJ position statement",
        parent=sg_grp,
        critical=False
    )
    claim_sg_url = "This URL is the U.S. Solicitor General's brief or an official DOJ page stating the position in this case."
    await evaluator.verify(
        claim=claim_sg_url,
        node=node_sg_url,
        sources=_unique_nonempty([case.federal.sg_brief_url]) or None,
        additional_instruction="Verify that the URL is an official DOJ/SG resource relevant to this case."
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
    """
    Evaluate an answer for the Supreme Court multistate AG amicus (2025-2026 term, Feb–Apr 2026 argument window) task.
    """
    # Initialize evaluator (root is parallel aggregation)
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

    # Record ground truth constraints (for transparency)
    evaluator.add_ground_truth({
        "term": "October Term 2025 (2025-2026)",
        "oral_argument_window_inclusive": ["2026-02-01", "2026-04-30"],
        "coalition_min_ag_threshold": 15,
        "required_stage": "merits (not petition)",
        "required_brief_type": "amicus curiae",
        "notes": "Official SCOTUS docket/case page on supremecourt.gov preferred for case identification and temporal checks."
    }, gt_type="constraints")

    # Extract up to 3 cases from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cases(),
        template_class=CasesExtraction,
        extraction_name="cases_extraction"
    )

    # Prepare exactly 3 cases: take first three, pad with empty if fewer
    cases: List[CaseInfo] = list(extracted.cases[:3])
    while len(cases) < 3:
        cases.append(CaseInfo())

    # Build verification subtrees for each case
    await asyncio.gather(*[
        verify_single_case(evaluator, root, case, idx)
        for idx, case in enumerate(cases)
    ])

    # Return the summary including the verification tree
    return evaluator.get_summary()