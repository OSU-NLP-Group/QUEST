import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "federal_rules_2026_window"
TASK_DESCRIPTION = """You are a regulatory compliance officer tracking federal rulemaking activity. Your task is to identify TWO proposed rules from TWO DIFFERENT federal agencies (such as EPA, DOT, HHS, USDA, FDA, DOL, DOE, Treasury, or other executive agencies) that meet ALL of the following criteria:

Selection Criteria for Each Rule:
1. Published in the Federal Register between January 1, 2026 and February 21, 2026 (inclusive)
2. The public comment period is currently open and closes after February 21, 2026
3. The rule is classified as a "significant regulatory action" requiring review by the Office of Information and Regulatory Affairs (OIRA) under Executive Order 12866

For EACH of the two rules, provide:
- A. The complete Federal Register citation including volume number, page number, and publication date (e.g., "91 FR 12345, February 5, 2026")
- B. The docket number (the unique identifier assigned by the agency, e.g., "EPA-HQ-OAR-2024-0505")
- C. The Regulation Identifier Number (RIN) (the unique alphanumeric code, e.g., "2060-AW68")
- D. The exact comment period end date (the deadline by which public comments must be submitted)
- E. The primary method for submitting public comments (typically via Regulations.gov, but may include alternative methods)
- F. A reference URL to either the Federal Register notice or the Regulations.gov docket page

Additionally, for ONE of the two agencies (your choice):
- G. The agency's FOIA office contact information (email address or online submission portal URL for filing Freedom of Information Act requests)
- H. The statutory FOIA response timeline (the number of business days the agency has to respond to FOIA requests under federal law)

Present your findings in a structured format that clearly identifies each rule and all required information.
"""


# ----------------------------- Data Models ---------------------------------- #
class RuleItem(BaseModel):
    title: Optional[str] = None
    agency_name: Optional[str] = None
    agency_support_url: Optional[str] = None  # Any page the answer cites to support agency identification
    fr_citation: Optional[str] = None  # e.g., "91 FR 12345, February 5, 2026"
    fr_volume: Optional[str] = None    # e.g., "91"
    fr_page: Optional[str] = None      # e.g., "12345"
    fr_pub_date: Optional[str] = None  # e.g., "February 5, 2026"
    fr_url: Optional[str] = None       # URL to the Federal Register notice
    docket_number: Optional[str] = None
    docket_url: Optional[str] = None
    rin: Optional[str] = None
    rin_url: Optional[str] = None      # Could be the FR page, Regulations.gov, or reginfo/OIRA page
    comment_end_date: Optional[str] = None  # e.g., "March 10, 2026"
    submission_method: Optional[str] = None # e.g., "Submit via Regulations.gov with docket number"
    submission_url: Optional[str] = None    # Direct submission portal or instructions URL


class FOIAInfo(BaseModel):
    agency_name: Optional[str] = None
    foia_contact_method: Optional[str] = None  # Email address or "Online portal URL"
    foia_contact_url: Optional[str] = None     # URL of the agency FOIA page/portal
    foia_timeline_days: Optional[str] = None   # Expected to be "20 business days" or similar
    foia_timeline_url: Optional[str] = None    # URL confirming the timeline


class RulesAndFOIAExtraction(BaseModel):
    rules: List[RuleItem] = Field(default_factory=list)
    foia: Optional[FOIAInfo] = None


# --------------------------- Extraction Prompt ------------------------------- #
def prompt_extract_rules_and_foia() -> str:
    return """
    Extract up to TWO proposed rules and ONE agency FOIA information block exactly as presented in the answer text.

    For each rule (limit to the first two mentioned), extract the following fields from the answer:
    - title: The rule title or subject (if present)
    - agency_name: The issuing federal executive agency (e.g., EPA, DOT, HHS, USDA, FDA, DOL, DOE, Treasury)
    - agency_support_url: A URL cited that supports the agency identification (if any; can be the Federal Register or Regulations.gov page)
    - fr_citation: The full Federal Register citation as presented (e.g., "91 FR 12345, February 5, 2026")
    - fr_volume: The Federal Register volume number (string, e.g., "91")
    - fr_page: The starting page number (string)
    - fr_pub_date: The publication date (string exactly as shown, e.g., "February 5, 2026")
    - fr_url: A URL to the Federal Register document page
    - docket_number: The agency docket number identifier (e.g., "EPA-HQ-OAR-2024-0505")
    - docket_url: A URL to the Regulations.gov docket page
    - rin: The Regulation Identifier Number (e.g., "2060-AW68")
    - rin_url: A URL that confirms the RIN (e.g., reginfo/OIRA listing, FR page, or Regulations.gov)
    - comment_end_date: The exact comment period end date (string, e.g., "March 10, 2026")
    - submission_method: The primary method for submitting comments (e.g., "Regulations.gov with docket number")
    - submission_url: A URL to the submission portal or instructions (often Regulations.gov)

    Then extract FOIA information for ONE of the two agencies (the answer may specify which; if multiple agencies' FOIA are given, pick the first one):
    - agency_name: The agency for which FOIA info is given
    - foia_contact_method: Either an email address or "Online portal" with URL if applicable (string)
    - foia_contact_url: The URL of the FOIA office contact page or portal
    - foia_timeline_days: The stated statutory response timeline in business days (typically "20 business days")
    - foia_timeline_url: A URL confirming the response timeline

    IMPORTANT:
    - Only extract values explicitly present in the answer. Do not invent any values.
    - Extract actual URLs shown in the answer. If a URL is not included, return null for that field.
    - If a field is not mentioned, set it to null.
    - Ensure all date strings and identifiers are extracted exactly as shown in the answer.
    """


# ------------------------------ Helpers -------------------------------------- #
def non_empty_urls(*urls: Optional[str]) -> List[str]:
    """Return a de-duplicated list of non-empty URLs."""
    seen = set()
    out = []
    for u in urls:
        if u and isinstance(u, str):
            trimmed = u.strip()
            if trimmed and trimmed not in seen:
                out.append(trimmed)
                seen.add(trimmed)
    return out


def pick_primary_url(*urls: Optional[str]) -> Optional[str]:
    """Pick the first non-empty URL candidate."""
    lst = non_empty_urls(*urls)
    return lst[0] if lst else None


# --------------------------- Verification Logic ------------------------------ #
async def verify_rule(
    evaluator: Evaluator,
    rules_parent,
    rule: RuleItem,
    rule_index: int,
    other_agency_name: Optional[str]
) -> None:
    """
    Build verification sub-tree for one rule and run checks in a sensible order to
    maximize meaningful gating.
    """
    idx_label = f"Rule_{rule_index + 1}"

    # Create the rule node (critical: required to meet the task)
    rule_node = evaluator.add_parallel(
        id=idx_label,
        desc=("First proposed rule meeting all specified criteria" if rule_index == 0
              else "Second proposed rule from a different agency meeting all specified criteria"),
        parent=rules_parent,
        critical=True
    )

    # ---------------- Agency Verification ----------------
    agency_node = evaluator.add_parallel(
        id=f"{idx_label}_Agency_Verification",
        desc=("Verify the rule is from a valid federal agency and different from Rule 2"
              if rule_index == 0 else "Verify the rule is from a valid federal agency different from Rule 1"),
        parent=rule_node,
        critical=True
    )

    # Leaf: The rule is from a federal executive agency
    agency_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Federal_Agency",
        desc=("The rule is from a federal executive agency (EPA, DOT, HHS, USDA, FDA, DOL, DOE, Treasury, or similar)"),
        parent=agency_node,
        critical=True
    )
    agency_sources = non_empty_urls(rule.fr_url, rule.docket_url, rule.agency_support_url)
    agency_claim = f"The issuing agency for this rule is '{rule.agency_name}', and it is a U.S. federal executive branch agency."
    await evaluator.verify(
        claim=agency_claim,
        node=agency_leaf,
        sources=agency_sources,
        additional_instruction="Verify the page shows the issuing agency and that it is a federal executive agency (e.g., Department of Transportation, Environmental Protection Agency, etc.). Allow reasonable naming variants (e.g., 'Department of Health and Human Services' vs 'HHS')."
    )

    # Leaf: Agency is different from the other rule's agency (simple logical check)
    uniq_node = evaluator.add_custom_node(
        result=bool(rule.agency_name) and bool(other_agency_name) and (rule.agency_name.strip().lower() != other_agency_name.strip().lower()),
        id=f"{idx_label}_Agency_Uniqueness",
        desc=f"The agency '{rule.agency_name or 'UNKNOWN'}' is different from the other rule's agency '{other_agency_name or 'UNKNOWN'}'",
        parent=agency_node,
        critical=True
    )

    # Leaf: A valid URL reference supporting the agency identification
    agency_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Agency_URL",
        desc="A valid URL reference supporting the agency identification",
        parent=agency_node,
        critical=True
    )
    agency_support = pick_primary_url(rule.agency_support_url, rule.fr_url, rule.docket_url)
    agency_url_claim = f"This webpage confirms the issuing agency is '{rule.agency_name}'."
    await evaluator.verify(
        claim=agency_url_claim,
        node=agency_url_leaf,
        sources=agency_support,
        additional_instruction="Confirm that the page explicitly shows the agency as the issuer or author of the rule/document."
    )

    # ---------------- Publication Criteria ----------------
    criteria_node = evaluator.add_parallel(
        id=f"{idx_label}_Publication_Criteria",
        desc="Verify the rule meets publication date, comment status, and OIRA review requirements",
        parent=rule_node,
        critical=True
    )

    # Existence/gating: at least one URL for criteria checks
    criteria_url_exists = evaluator.add_custom_node(
        result=bool(pick_primary_url(rule.fr_url, rule.docket_url)),
        id=f"{idx_label}_Criteria_URL",
        desc="A valid URL reference supporting the publication and review criteria",
        parent=criteria_node,
        critical=True
    )

    # Leaf: Publication date within window
    pub_date_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Publication_Date",
        desc="The rule was published in the Federal Register between January 1, 2026 and February 21, 2026",
        parent=criteria_node,
        critical=True
    )
    pub_claim = f"The Federal Register publication date for this rule is '{rule.fr_pub_date}', and this date lies between January 1, 2026 and February 21, 2026 (inclusive)."
    await evaluator.verify(
        claim=pub_claim,
        node=pub_date_leaf,
        sources=rule.fr_url or rule.docket_url,
        additional_instruction="Look for the FR publication date on the FederalRegister.gov page. If only the docket page is available, confirm the FR citation date shown there if present."
    )

    # Leaf: Comment status open and closes after Feb 21, 2026
    comment_status_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Comment_Status",
        desc="The comment period is currently open and closes after February 21, 2026",
        parent=criteria_node,
        critical=True
    )
    comment_claim = f"The public comment deadline is '{rule.comment_end_date}', which is after February 21, 2026; therefore the comment period extends beyond that date."
    await evaluator.verify(
        claim=comment_claim,
        node=comment_status_leaf,
        sources=rule.docket_url or rule.fr_url,
        additional_instruction="Confirm the 'Comments Close' or 'Comments Due' date on the docket/FR page; accept the claim if the deadline is later than February 21, 2026."
    )

    # Leaf: OIRA classification significant under EO 12866
    oira_leaf = evaluator.add_leaf(
        id=f"{idx_label}_OIRA_Classification",
        desc="The rule is classified as a significant regulatory action requiring OIRA review under Executive Order 12866",
        parent=criteria_node,
        critical=True
    )
    oira_sources = non_empty_urls(rule.fr_url, rule.rin_url, rule.docket_url)
    oira_claim = "This rule is identified as a 'significant regulatory action' subject to OIRA review under Executive Order 12866."
    await evaluator.verify(
        claim=oira_claim,
        node=oira_leaf,
        sources=oira_sources,
        additional_instruction="Look for explicit statements like 'This action is a significant regulatory action' and references to Executive Order 12866/OIRA review."
    )

    # ---------------- Federal Register Citation ----------------
    fr_node = evaluator.add_parallel(
        id=f"{idx_label}_FR_Citation",
        desc="Complete and properly formatted Federal Register citation",
        parent=rule_node,
        critical=True
    )

    fr_volume_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Volume",
        desc="The correct Federal Register volume number (should be 91 for 2026)" if rule_index == 0 else "The correct Federal Register volume number",
        parent=fr_node,
        critical=True
    )
    vol_claim = f"The Federal Register volume number for this notice is '{rule.fr_volume}'."
    await evaluator.verify(
        claim=vol_claim,
        node=fr_volume_leaf,
        sources=rule.fr_url,
        additional_instruction="Confirm the volume number shown on the FederalRegister.gov document (2026 volume is typically 91)."
    )

    fr_page_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Page",
        desc="The correct starting page number in the Federal Register" if rule_index == 0 else "The correct starting page number",
        parent=fr_node,
        critical=True
    )
    page_claim = f"The Federal Register page number at which this notice begins is '{rule.fr_page}'."
    await evaluator.verify(
        claim=page_claim,
        node=fr_page_leaf,
        sources=rule.fr_url,
        additional_instruction="Use the 'Document Details' or 'Pages' metadata on FederalRegister.gov to confirm the starting page."
    )

    fr_date_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Date",
        desc="The correct publication date matching the specified date range" if rule_index == 0 else "The correct publication date",
        parent=fr_node,
        critical=True
    )
    fr_date_claim = f"The publication date shown on the Federal Register page is '{rule.fr_pub_date}'."
    await evaluator.verify(
        claim=fr_date_claim,
        node=fr_date_leaf,
        sources=rule.fr_url,
        additional_instruction="Check the displayed publication date on the FederalRegister.gov page."
    )

    fr_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_FR_URL",
        desc="A valid URL to the Federal Register document",
        parent=fr_node,
        critical=True
    )
    fr_url_claim = "This URL points to the Federal Register document page for the rule."
    await evaluator.verify(
        claim=fr_url_claim,
        node=fr_url_leaf,
        sources=rule.fr_url,
        additional_instruction="Confirm that the URL is on federalregister.gov and corresponds to a document page (not a generic landing page)."
    )

    # ---------------- Docket ----------------
    docket_node = evaluator.add_parallel(
        id=f"{idx_label}_Docket",
        desc="Valid docket number in proper agency format",
        parent=rule_node,
        critical=True
    )

    docket_num_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Docket_Number",
        desc="The docket number follows the agency's standard format (e.g., EPA-HQ-OAR-2024-0505)" if rule_index == 0 else "The docket number follows the agency's standard format",
        parent=docket_node,
        critical=True
    )
    docket_claim = f"The docket number for this rulemaking is '{rule.docket_number}'."
    await evaluator.verify(
        claim=docket_claim,
        node=docket_num_leaf,
        sources=rule.docket_url or rule.fr_url,
        additional_instruction="Verify the docket number on the Regulations.gov docket page or the FR notice."
    )

    docket_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Docket_URL",
        desc="A valid URL to the docket on Regulations.gov",
        parent=docket_node,
        critical=True
    )
    docket_url_claim = f"This URL is the Regulations.gov docket page for docket '{rule.docket_number}'."
    await evaluator.verify(
        claim=docket_url_claim,
        node=docket_url_leaf,
        sources=rule.docket_url,
        additional_instruction="Confirm the URL is a Regulations.gov docket page that matches the docket identifier."
    )

    # ---------------- RIN ----------------
    rin_node = evaluator.add_parallel(
        id=f"{idx_label}_RIN",
        desc="Valid Regulation Identifier Number assigned to the rulemaking",
        parent=rule_node,
        critical=True
    )

    rin_num_leaf = evaluator.add_leaf(
        id=f"{idx_label}_RIN_Number",
        desc="The RIN follows the standard 4-digit agency code plus 4-character sequence format (e.g., 2060-AW68)" if rule_index == 0 else "The RIN follows the standard format",
        parent=rin_node,
        critical=True
    )
    rin_sources = non_empty_urls(rule.rin_url, rule.fr_url, rule.docket_url)
    rin_claim = f"The Regulation Identifier Number (RIN) for this rule is '{rule.rin}'."
    await evaluator.verify(
        claim=rin_claim,
        node=rin_num_leaf,
        sources=rin_sources,
        additional_instruction="Confirm the RIN on any authoritative page (FR, Regulations.gov, or reginfo/OIRA)."
    )

    rin_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_RIN_URL",
        desc="A valid URL reference confirming the RIN",
        parent=rin_node,
        critical=True
    )
    rin_url_claim = f"This URL confirms the RIN '{rule.rin}' for this rule."
    await evaluator.verify(
        claim=rin_url_claim,
        node=rin_url_leaf,
        sources=rule.rin_url,
        additional_instruction="Confirm the page displays the RIN in association with the same rulemaking."
    )

    # ---------------- Comment Deadline ----------------
    deadline_node = evaluator.add_parallel(
        id=f"{idx_label}_Comment_Deadline",
        desc="The exact date and time when the comment period closes",
        parent=rule_node,
        critical=True
    )

    deadline_date_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Deadline_Date",
        desc="The specific deadline date (must be after February 21, 2026)",
        parent=deadline_node,
        critical=True
    )
    deadline_claim = f"The comment period closes on '{rule.comment_end_date}', which is after February 21, 2026."
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_date_leaf,
        sources=rule.docket_url or rule.fr_url,
        additional_instruction="Verify the 'Comments Close' date shown on the official docket or FR notice is later than Feb 21, 2026."
    )

    deadline_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Deadline_URL",
        desc="A valid URL reference confirming the deadline",
        parent=deadline_node,
        critical=True
    )
    deadline_url_claim = f"This URL confirms the comment deadline '{rule.comment_end_date}' for this rule."
    await evaluator.verify(
        claim=deadline_url_claim,
        node=deadline_url_leaf,
        sources=rule.docket_url or rule.fr_url,
        additional_instruction="The page should clearly display the comments due/close date."
    )

    # ---------------- Submission Method ----------------
    submission_node = evaluator.add_parallel(
        id=f"{idx_label}_Submission_Method",
        desc="The primary method for submitting public comments",
        parent=rule_node,
        critical=True
    )

    submission_desc_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Method_Description",
        desc="Description of how to submit comments (typically via Regulations.gov with docket number)" if rule_index == 0 else "Description of how to submit comments",
        parent=submission_node,
        critical=True
    )
    submission_claim = f"The primary method for submitting comments is '{rule.submission_method}'."
    await evaluator.verify(
        claim=submission_claim,
        node=submission_desc_leaf,
        sources=rule.docket_url or rule.fr_url,
        additional_instruction="Check the 'Submit a comment' or instructions section on the official page; accept common phrasing indicating Regulations.gov submissions."
    )

    submission_url_leaf = evaluator.add_leaf(
        id=f"{idx_label}_Method_URL",
        desc="A valid URL for the submission portal or instructions",
        parent=submission_node,
        critical=True
    )
    submission_url_claim = "This URL is the submission portal or official instructions page for filing public comments on this rule."
    await evaluator.verify(
        claim=submission_url_claim,
        node=submission_url_leaf,
        sources=rule.submission_url,
        additional_instruction="Often this is a Regulations.gov URL; confirm it allows/points to comment submission for this specific rule/docket."
    )


async def verify_foia_block(
    evaluator: Evaluator,
    parent_node,
    foia: FOIAInfo
) -> None:
    """
    Build FOIA verification nodes for one agency and run checks.
    """
    foia_node = evaluator.add_parallel(
        id="Agency_FOIA_Information",
        desc="FOIA office contact information and response timeline for one of the two agencies",
        parent=parent_node,
        critical=True  # Treat FOIA information as required per task instruction
    )

    # FOIA Contact
    contact_node = evaluator.add_parallel(
        id="FOIA_Contact",
        desc="The agency's FOIA office email address or online submission portal",
        parent=foia_node,
        critical=True
    )

    contact_method_leaf = evaluator.add_leaf(
        id="FOIA_Contact_Method",
        desc="Valid email address or URL for FOIA request submission",
        parent=contact_node,
        critical=True
    )
    contact_method_claim = f"The FOIA contact method for '{foia.agency_name}' is '{foia.foia_contact_method}', as indicated by the agency FOIA page."
    await evaluator.verify(
        claim=contact_method_claim,
        node=contact_method_leaf,
        sources=foia.foia_contact_url,
        additional_instruction="Confirm whether the contact method is a valid email address or an online submission portal described on the page."
    )

    contact_url_leaf = evaluator.add_leaf(
        id="FOIA_Contact_URL",
        desc="A valid URL reference to the agency's FOIA page confirming the contact method",
        parent=contact_node,
        critical=True
    )
    contact_url_claim = "This URL is the agency's FOIA page or portal confirming how to submit FOIA requests."
    await evaluator.verify(
        claim=contact_url_claim,
        node=contact_url_leaf,
        sources=foia.foia_contact_url,
        additional_instruction="The page should be an official agency site that provides FOIA submission instructions."
    )

    # FOIA Timeline
    timeline_node = evaluator.add_parallel(
        id="FOIA_Response_Timeline",
        desc="The statutory response timeline for FOIA requests to that agency",
        parent=foia_node,
        critical=True
    )

    timeline_days_leaf = evaluator.add_leaf(
        id="FOIA_Timeline_Days",
        desc="The number of business days the agency has to respond (typically 20 business days under federal law)",
        parent=timeline_node,
        critical=True
    )
    timeline_days_claim = f"The agency states the FOIA response timeline is '{foia.foia_timeline_days}' (typically 20 business days under federal law)."
    await evaluator.verify(
        claim=timeline_days_claim,
        node=timeline_days_leaf,
        sources=foia.foia_timeline_url or foia.foia_contact_url,
        additional_instruction="Confirm the FOIA response timeframe (usually 20 business days) as described on the agency FOIA page or a linked policy page."
    )

    timeline_url_leaf = evaluator.add_leaf(
        id="FOIA_Timeline_URL",
        desc="A valid URL reference confirming the response timeline requirement",
        parent=timeline_node,
        critical=True
    )
    timeline_url_claim = "This URL confirms the agency's FOIA response timeline requirement."
    await evaluator.verify(
        claim=timeline_url_claim,
        node=timeline_url_leaf,
        sources=foia.foia_timeline_url or foia.foia_contact_url,
        additional_instruction="The page should explicitly state the number of business days to respond or link to policy explaining it."
    )


# ------------------------------ Main Entry ----------------------------------- #
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
    Evaluate an answer for the federal rulemaking window task and return a structured summary.
    """
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_rules_and_foia(),
        template_class=RulesAndFOIAExtraction,
        extraction_name="rules_and_foia"
    )

    # Record a small custom info block about the evaluation window
    evaluator.add_custom_info(
        info={"publication_window": "Jan 1, 2026 – Feb 21, 2026", "require_oira_significant": True, "require_two_distinct_agencies": True},
        info_type="constraints",
        info_name="task_constraints"
    )

    # Build top-level task completion node (critical gate)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Complete identification of two proposed federal rules meeting all specified criteria, plus agency FOIA information",
        parent=root,
        critical=True
    )

    # Rules Identification node (critical)
    rules_node = evaluator.add_parallel(
        id="Rules_Identification",
        desc="Identify two proposed rules from two different federal agencies meeting all publication and review criteria",
        parent=task_node,
        critical=True
    )

    # Prepare up to two rules, pad with empty if fewer
    rules_list: List[RuleItem] = list(extraction.rules[:2])
    while len(rules_list) < 2:
        rules_list.append(RuleItem())

    # Determine other-agency names for uniqueness checks
    agency_names = [r.agency_name or "" for r in rules_list]
    other_for_first = agency_names[1] if len(agency_names) > 1 else None
    other_for_second = agency_names[0] if len(agency_names) > 0 else None

    # Verify each rule sub-tree
    await verify_rule(evaluator, rules_node, rules_list[0], 0, other_for_first)
    await verify_rule(evaluator, rules_node, rules_list[1], 1, other_for_second)

    # FOIA block (for one of the agencies)
    foia_info = extraction.foia or FOIAInfo()
    await verify_foia_block(evaluator, task_node, foia_info)

    # Return structured summary
    return evaluator.get_summary()