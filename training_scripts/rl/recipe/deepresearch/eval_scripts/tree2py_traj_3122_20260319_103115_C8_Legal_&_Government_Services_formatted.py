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
TASK_ID = "scotus_ieepa_2026"
TASK_DESCRIPTION = """
In February 2026, the U.S. Supreme Court issued a landmark ruling striking down tariffs that had been imposed under the International Emergency Economic Powers Act (IEEPA). This decision had major implications for importers and led to subsequent refund processes and replacement tariff measures.

Provide a comprehensive analysis that includes the following information:

Supreme Court Case Details:
- The case number
- The full case name
- The date the decision was issued
- The vote count (how many justices voted in favor vs. against)
- The specific statutory authority that was challenged
- Who authored the majority opinion
- Whether this case was consolidated with another case, and if so, provide the other case number and name

Lead Plaintiff Information:
- The city and state where Learning Resources, Inc. is located
- What type of business Learning Resources, Inc. operates
- The name of Learning Resources' CEO
- Identify V.O.S. Selections, Inc. as another plaintiff

State Plaintiffs:
- The total number of states that were plaintiffs in the consolidated case
- The names of at least three states that were plaintiffs (such as Oregon, Arizona, Colorado, Connecticut, Delaware, Illinois, or Maine)

Refund Process Details:
- Which federal court has jurisdiction over tariff refund cases
- The name of the judge who issued the refund order in March 2026
- The date that refund order was issued
- The name of the CBP system that processes refunds (full acronym explanation)
- The electronic payment enrollment program required for importers to receive refunds (full acronym explanation)
- The category of parties eligible to receive refunds (must be importers of record)

Replacement Tariff Measures:
- The specific section and act that provided statutory authority for replacement tariffs imposed after the Supreme Court ruling
- The tariff rate of the replacement measure
- The effective date when the replacement tariff took effect
- The duration (in days) of the replacement tariff
- The expiration date of the replacement tariff

All information must be factually accurate and verifiable through official government sources, court documents, or reputable news reporting from 2026.
""".strip()


# Ground-truth expectations specified by rubric (used to form verification claims)
EXPECTED = {
    "case_number": "24-1287",
    "case_name": "Learning Resources, Inc. v. Trump",
    "decision_date": "February 20, 2026",
    "vote_count": "6-3",
    "statute_challenged": "International Emergency Economic Powers Act (IEEPA)",
    "majority_author": "Chief Justice John Roberts",
    "consol_case_number": "25-250",
    "consol_case_name": "Trump v. V.O.S. Selections, Inc.",
    "lead_location": "Vernon Hills, Illinois",
    "lead_business_type": "educational toy company",
    "lead_ceo": "Rick Woldenberg",
    "states_total_count": "12",
    "refund_court": "U.S. Court of International Trade",
    "refund_judge": "Richard K. Eaton",
    "refund_order_date": "March 4, 2026",
    "cbp_ace_full": "Automated Commercial Environment",
    "cbp_ach_full": "Automated Clearing House",
    "refund_eligibility": "importer of record",
    "replacement_statute": "Section 122 of the Trade Act of 1974",
    "replacement_rate": "10%",
    "replacement_effective_date": "February 24, 2026",
    "replacement_duration_days": "150",
    "replacement_expiration_date": "July 24, 2026",
}

ALLOWED_STATES = {"Oregon", "Arizona", "Colorado", "Connecticut", "Delaware", "Illinois", "Maine"}


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CaseDetails(BaseModel):
    case_number: Optional[str] = None
    case_name: Optional[str] = None
    decision_date: Optional[str] = None
    vote_count: Optional[str] = None
    statute_challenged: Optional[str] = None
    majority_author: Optional[str] = None
    consolidated_case_number: Optional[str] = None
    consolidated_case_name: Optional[str] = None
    case_urls: List[str] = Field(default_factory=list)


class LeadPlaintiff(BaseModel):
    city_state: Optional[str] = None
    business_type: Optional[str] = None
    ceo: Optional[str] = None
    lead_urls: List[str] = Field(default_factory=list)


class VOSInfo(BaseModel):
    is_plaintiff: Optional[bool] = None
    small_businesses_count_statement: Optional[str] = None
    vos_urls: List[str] = Field(default_factory=list)


class StatePlaintiffs(BaseModel):
    total_states_count: Optional[str] = None
    named_states: List[str] = Field(default_factory=list)
    states_urls: List[str] = Field(default_factory=list)


class RefundProcess(BaseModel):
    court_jurisdiction: Optional[str] = None
    judge_name: Optional[str] = None
    refund_order_date: Optional[str] = None
    cbp_ace_full_name: Optional[str] = None
    cbp_ach_full_name: Optional[str] = None
    eligibility_category: Optional[str] = None
    refund_urls: List[str] = Field(default_factory=list)


class ReplacementTariff(BaseModel):
    statutory_section_act: Optional[str] = None
    tariff_rate: Optional[str] = None
    effective_date: Optional[str] = None
    duration_days: Optional[str] = None
    expiration_date: Optional[str] = None
    replacement_urls: List[str] = Field(default_factory=list)


class TariffCaseExtraction(BaseModel):
    case: Optional[CaseDetails] = None
    lead: Optional[LeadPlaintiff] = None
    vos: Optional[VOSInfo] = None
    states: Optional[StatePlaintiffs] = None
    refund: Optional[RefundProcess] = None
    replacement: Optional[ReplacementTariff] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following structured fields from the answer text. Do not invent. If a field is not present in the answer, set it to null (or an empty array for URL lists). Extract only URLs explicitly mentioned in the answer text.

1) case:
   - case_number (string)
   - case_name (string)
   - decision_date (string)
   - vote_count (string)
   - statute_challenged (string)
   - majority_author (string)
   - consolidated_case_number (string)
   - consolidated_case_name (string)
   - case_urls (array of URLs explicitly cited for Supreme Court case details)

2) lead:
   - city_state (string; e.g., "Vernon Hills, Illinois")
   - business_type (string; e.g., "educational toy company")
   - ceo (string)
   - lead_urls (array of URLs cited for Learning Resources details)

3) vos:
   - is_plaintiff (boolean if the answer explicitly says V.O.S. Selections, Inc. is a plaintiff)
   - small_businesses_count_statement (string; e.g., "five small businesses", "5", or sentence mentioning the count)
   - vos_urls (array of URLs cited for V.O.S. Selections, Inc. plaintiff information)

4) states:
   - total_states_count (string or numeric in string form)
   - named_states (array of state names exactly as written in the answer)
   - states_urls (array of URLs cited for state-plaintiff information)

5) refund:
   - court_jurisdiction (string; e.g., "U.S. Court of International Trade")
   - judge_name (string; e.g., "Richard K. Eaton")
   - refund_order_date (string; e.g., "March 4, 2026")
   - cbp_ace_full_name (string; full expansion such as "Automated Commercial Environment")
   - cbp_ach_full_name (string; full expansion such as "Automated Clearing House")
   - eligibility_category (string; e.g., "importer of record")
   - refund_urls (array of URLs cited for refund process information)

6) replacement:
   - statutory_section_act (string; e.g., "Section 122 of the Trade Act of 1974")
   - tariff_rate (string; e.g., "10%")
   - effective_date (string; e.g., "February 24, 2026")
   - duration_days (string; e.g., "150")
   - expiration_date (string; e.g., "July 24, 2026")
   - replacement_urls (array of URLs cited for replacement tariff information)
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(x: Optional[str]) -> bool:
    return bool(x) and bool(str(x).strip())


def _urls_or_none(urls: Optional[List[str]]) -> Optional[List[str]]:
    if not urls:
        return None
    # filter obviously empty strings
    cleaned = [u for u in urls if _nonempty(u)]
    return cleaned if cleaned else None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_case_details_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    case = data.case or CaseDetails()
    case_node = evaluator.add_parallel(
        id="Supreme_Court_Case_Details",
        desc="Supreme Court case details verification",
        parent=parent,
        critical=False,
    )

    # Existence and sources gating (critical siblings)
    evaluator.add_custom_node(
        result=_nonempty(case.case_number) and _nonempty(case.case_name) and _nonempty(case.decision_date)
               and _nonempty(case.vote_count) and _nonempty(case.statute_challenged) and _nonempty(case.majority_author),
        id="Case_Details_Provided",
        desc="Case details are provided in the answer (number, name, date, vote, statute, majority author)",
        parent=case_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(case.case_urls and len(case.case_urls) > 0),
        id="Case_Detail_Sources_Provided",
        desc="At least one case-related source URL is provided",
        parent=case_node,
        critical=True
    )

    sources = _urls_or_none(case.case_urls)

    # Individual critical leaves as specified in the rubric
    # Case number 24-1287
    n_case = evaluator.add_leaf(
        id="Case_Number_24_1287",
        desc="The Supreme Court case number is correctly identified as 24-1287",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Supreme Court docket number for the case is 24-1287.",
        node=n_case,
        sources=sources,
        additional_instruction="Verify on official or reputable sources that the docket number for Learning Resources, Inc. v. Trump is 24-1287."
    )

    # Case name Learning Resources, Inc. v. Trump
    n_name = evaluator.add_leaf(
        id="Case_Name_Learning_Resources_v_Trump",
        desc="The case name is correctly identified as 'Learning Resources, Inc. v. Trump'",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The full case name is 'Learning Resources, Inc. v. Trump' (minor punctuation/casing variations acceptable).",
        node=n_name,
        sources=sources,
        additional_instruction="Allow minor punctuation (e.g., commas, periods) and casing variations."
    )

    # Decision date February 20, 2026
    n_date = evaluator.add_leaf(
        id="Decision_Date_February_20_2026",
        desc="The decision date is correctly identified as February 20, 2026",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The decision was issued on February 20, 2026.",
        node=n_date,
        sources=sources,
        additional_instruction="Check the official opinion or reliable reporting for the release date."
    )

    # Vote count 6-3
    n_vote = evaluator.add_leaf(
        id="Vote_Count_6_to_3",
        desc="The vote count is correctly identified as 6-3",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Supreme Court's vote count in this case was 6-3.",
        node=n_vote,
        sources=sources,
        additional_instruction="Accept minor dash variations (e.g., hyphen/en-dash)."
    )

    # Statutory basis IEEPA
    n_statute = evaluator.add_leaf(
        id="Statutory_Basis_IEEPA",
        desc="The statutory authority challenged (IEEPA - International Emergency Economic Powers Act) is correctly identified",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The challenged statutory authority was the International Emergency Economic Powers Act (IEEPA).",
        node=n_statute,
        sources=sources,
        additional_instruction="The case involved tariffs imposed under IEEPA; verify that IEEPA was the statute at issue."
    )

    # Majority author Roberts
    n_auth = evaluator.add_leaf(
        id="Majority_Opinion_Author_Roberts",
        desc="Chief Justice John Roberts is correctly identified as the author of the majority opinion",
        parent=case_node,
        critical=True
    )
    await evaluator.verify(
        claim="The majority opinion was authored by Chief Justice John Roberts.",
        node=n_auth,
        sources=sources
    )

    # Consolidated case reference group (critical)
    cons_node = evaluator.add_parallel(
        id="Consolidated_Case_Reference",
        desc="The answer references the consolidated case No. 25-250 (Trump v. V.O.S. Selections, Inc.)",
        parent=case_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(case.consolidated_case_number) and _nonempty(case.consolidated_case_name),
        id="Consolidated_Case_Fields_Provided",
        desc="Consolidated case number and name are provided",
        parent=cons_node,
        critical=True
    )
    cons_num = evaluator.add_leaf(
        id="Consolidated_Number_25_250",
        desc="The consolidated case number is correctly identified as 25-250",
        parent=cons_node,
        critical=True
    )
    await evaluator.verify(
        claim="The consolidated case number is 25-250.",
        node=cons_num,
        sources=sources
    )
    cons_name = evaluator.add_leaf(
        id="Consolidated_Name_Trump_v_VOS",
        desc="The consolidated case name is correctly identified as 'Trump v. V.O.S. Selections, Inc.'",
        parent=cons_node,
        critical=True
    )
    await evaluator.verify(
        claim="The consolidated case is titled 'Trump v. V.O.S. Selections, Inc.' (minor punctuation/casing variations acceptable).",
        node=cons_name,
        sources=sources,
        additional_instruction="Allow minor punctuation differences."
    )


async def build_lead_plaintiff_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    lead = data.lead or LeadPlaintiff()
    lead_node = evaluator.add_parallel(
        id="Lead_Plaintiff_Info",
        desc="Lead plaintiff (Learning Resources, Inc.) information verification",
        parent=parent,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(lead.city_state) and _nonempty(lead.business_type) and _nonempty(lead.ceo),
        id="Lead_Plaintiff_Fields_Provided",
        desc="Lead plaintiff fields (location, business type, CEO) are provided",
        parent=lead_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(lead.lead_urls and len(lead.lead_urls) > 0),
        id="Lead_Plaintiff_Sources_Provided",
        desc="At least one source URL is provided for Learning Resources information",
        parent=lead_node,
        critical=True
    )
    sources = _urls_or_none(lead.lead_urls)

    loc = evaluator.add_leaf(
        id="Lead_Plaintiff_Company_Location",
        desc="Learning Resources is correctly identified as being located in Vernon Hills, Illinois",
        parent=lead_node,
        critical=True
    )
    await evaluator.verify(
        claim="Learning Resources, Inc. is located in Vernon Hills, Illinois.",
        node=loc,
        sources=sources,
        additional_instruction="Verify with company/corporate information or reputable sources."
    )

    btype = evaluator.add_leaf(
        id="Lead_Plaintiff_Business_Type",
        desc="Learning Resources is correctly identified as an educational toy company",
        parent=lead_node,
        critical=True
    )
    await evaluator.verify(
        claim="Learning Resources, Inc. operates as an educational toy company.",
        node=btype,
        sources=sources
    )

    ceo = evaluator.add_leaf(
        id="Lead_Plaintiff_CEO",
        desc="Rick Woldenberg is correctly identified as CEO of Learning Resources",
        parent=lead_node,
        critical=True
    )
    await evaluator.verify(
        claim="Rick Woldenberg is the CEO of Learning Resources, Inc.",
        node=ceo,
        sources=sources
    )


async def build_vos_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    vos = data.vos or VOSInfo()
    vos_node = evaluator.add_parallel(
        id="VOS_Selections_Plaintiff_Information",
        desc="Information about V.O.S. Selections as a plaintiff is provided",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(vos.vos_urls and len(vos.vos_urls) > 0),
        id="VOS_Sources_Provided",
        desc="At least one source URL is provided for V.O.S. Selections plaintiff info",
        parent=vos_node,
        critical=True
    )
    sources = _urls_or_none(vos.vos_urls) or _urls_or_none((data.case or CaseDetails()).case_urls)

    vosp = evaluator.add_leaf(
        id="VOS_Selections_As_Plaintiff",
        desc="V.O.S. Selections, Inc. is correctly identified as a plaintiff in the consolidated case",
        parent=vos_node,
        critical=True
    )
    await evaluator.verify(
        claim="V.O.S. Selections, Inc. was a plaintiff in the consolidated case related to the Supreme Court ruling.",
        node=vosp,
        sources=sources
    )

    # Five small businesses identified in V.O.S. case
    five = evaluator.add_leaf(
        id="Five_Small_Businesses_In_VOS_Case",
        desc="Five small businesses are identified as plaintiffs in the V.O.S. Selections case",
        parent=vos_node,
        critical=True
    )
    await evaluator.verify(
        claim="Five small businesses were plaintiffs in the V.O.S. Selections case.",
        node=five,
        sources=sources,
        additional_instruction="Look for explicit mention of five small businesses being plaintiffs."
    )


async def build_state_plaintiffs_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    states = data.states or StatePlaintiffs()
    st_node = evaluator.add_parallel(
        id="State_Plaintiffs_Information",
        desc="Information about state plaintiffs is provided",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(states.total_states_count) and len(states.named_states) >= 1,
        id="States_Fields_Provided",
        desc="State plaintiffs fields (total count and names) are provided in the answer",
        parent=st_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(states.states_urls and len(states.states_urls) > 0),
        id="States_Sources_Provided",
        desc="At least one source URL is provided for state-plaintiff information",
        parent=st_node,
        critical=True
    )
    sources = _urls_or_none(states.states_urls) or _urls_or_none((data.case or CaseDetails()).case_urls)

    # Total count == 12
    total = evaluator.add_leaf(
        id="Twelve_States_Count",
        desc="Twelve states are identified as plaintiffs",
        parent=st_node,
        critical=True
    )
    await evaluator.verify(
        claim="Twelve states were plaintiffs in the consolidated case.",
        node=total,
        sources=sources
    )

    # Named states requirement: at least three among the list
    # Precompute intersection for robustness; also add a gating custom node
    intersection = [s for s in states.named_states if s and s.strip() in ALLOWED_STATES]
    evaluator.add_custom_node(
        result=len(intersection) >= 3,
        id="Named_States_Provided_Check",
        desc="At least three of {Oregon, Arizona, Colorado, Connecticut, Delaware, Illinois, Maine} are named in the answer",
        parent=st_node,
        critical=True
    )
    names_leaf = evaluator.add_leaf(
        id="Named_States_Requirement",
        desc="At least three of the following states are named: Oregon, Arizona, Colorado, Connecticut, Delaware, Illinois, or Maine",
        parent=st_node,
        critical=True
    )
    picked = intersection[:3] if len(intersection) >= 3 else intersection
    list_str = ", ".join(picked) if picked else "Oregon, Arizona, Colorado"
    await evaluator.verify(
        claim=f"The state plaintiffs included at least these three states: {list_str}.",
        node=names_leaf,
        sources=sources,
        additional_instruction="Confirm that each listed state was a named plaintiff in the case."
    )


async def build_refund_process_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    refund = data.refund or RefundProcess()
    ref_node = evaluator.add_parallel(
        id="Refund_Process_Main",
        desc="Refund process details verification",
        parent=parent,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(refund.court_jurisdiction) and _nonempty(refund.judge_name) and _nonempty(refund.refund_order_date)
               and _nonempty(refund.cbp_ace_full_name) and _nonempty(refund.cbp_ach_full_name) and _nonempty(refund.eligibility_category),
        id="Refund_Fields_Provided",
        desc="Refund process fields (jurisdiction, judge, date, ACE, ACH, eligibility) are provided in the answer",
        parent=ref_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(refund.refund_urls and len(refund.refund_urls) > 0),
        id="Refund_Sources_Provided",
        desc="At least one refund-process source URL is provided",
        parent=ref_node,
        critical=True
    )
    sources = _urls_or_none(refund.refund_urls)

    # CIT Jurisdiction
    cit = evaluator.add_leaf(
        id="CIT_Jurisdiction",
        desc="The Court of International Trade is correctly identified as having jurisdiction over tariff refund cases",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="The U.S. Court of International Trade has jurisdiction over tariff refund cases related to this matter.",
        node=cit,
        sources=sources,
        additional_instruction="Look for references to CIT jurisdiction over refund or tariff-related actions."
    )

    # Refund order judge
    j_eaton = evaluator.add_leaf(
        id="Refund_Order_Judge_Eaton",
        desc="Judge Richard K. Eaton is correctly identified as issuing the refund order",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="The refund order was issued by Judge Richard K. Eaton.",
        node=j_eaton,
        sources=sources
    )

    # Refund order date
    r_date = evaluator.add_leaf(
        id="Refund_Order_Date_March_4_2026",
        desc="The refund order date is correctly identified as March 4, 2026",
        parent=ref_node,
        critical=True
    )
    await evaluator.verify(
        claim="The refund order was issued on March 4, 2026.",
        node=r_date,
        sources=sources
    )

    # CBP refund processing details (critical group)
    cbp_node = evaluator.add_parallel(
        id="CBP_Refund_Processing_Details",
        desc="Details about CBP refund processing requirements are provided",
        parent=ref_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(refund.cbp_ace_full_name) and _nonempty(refund.cbp_ach_full_name) and _nonempty(refund.eligibility_category),
        id="CBP_Refund_Fields_Provided",
        desc="CBP refund fields (ACE full name, ACH full name, eligibility) are provided",
        parent=cbp_node,
        critical=True
    )

    ace = evaluator.add_leaf(
        id="ACE_System_Identified",
        desc="The ACE (Automated Commercial Environment) system is identified as the platform for refund processing",
        parent=cbp_node,
        critical=True
    )
    await evaluator.verify(
        claim="Refund processing is handled through ACE (Automated Commercial Environment).",
        node=ace,
        sources=sources
    )

    ach = evaluator.add_leaf(
        id="ACH_Enrollment_Required",
        desc="ACH (Automated Clearing House) enrollment is identified as required for refunds",
        parent=cbp_node,
        critical=True
    )
    await evaluator.verify(
        claim="Importers must enroll in ACH (Automated Clearing House) to receive electronic refund payments.",
        node=ach,
        sources=sources
    )

    elig = evaluator.add_leaf(
        id="Importer_Of_Record_Eligibility",
        desc="Only importers of record are identified as eligible for refunds",
        parent=cbp_node,
        critical=True
    )
    await evaluator.verify(
        claim="Refunds are issued only to the importer of record.",
        node=elig,
        sources=sources
    )


async def build_replacement_tariff_checks(evaluator: Evaluator, parent, data: TariffCaseExtraction):
    repl = data.replacement or ReplacementTariff()
    rep_node = evaluator.add_parallel(
        id="Replacement_Tariff_Main",
        desc="Replacement tariff measures verification",
        parent=parent,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(repl.statutory_section_act) and _nonempty(repl.tariff_rate) and _nonempty(repl.effective_date)
               and _nonempty(repl.duration_days) and _nonempty(repl.expiration_date),
        id="Replacement_Tariff_Fields_Provided",
        desc="Replacement tariff fields (statutory basis, rate, effective date, duration, expiration) are provided in the answer",
        parent=rep_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(repl.replacement_urls and len(repl.replacement_urls) > 0),
        id="Replacement_Tariff_Sources_Provided",
        desc="At least one source URL is provided for replacement tariff information",
        parent=rep_node,
        critical=True
    )
    sources = _urls_or_none(repl.replacement_urls)

    # Statutory authority Section 122 of the Trade Act of 1974
    s122 = evaluator.add_leaf(
        id="Section_122_Replacement_Tariff",
        desc="Section 122 of the Trade Act of 1974 is correctly identified as the statutory basis for replacement tariffs",
        parent=rep_node,
        critical=True
    )
    await evaluator.verify(
        claim="The replacement tariffs were imposed under Section 122 of the Trade Act of 1974.",
        node=s122,
        sources=sources
    )

    # Tariff rate 10%
    rate = evaluator.add_leaf(
        id="Replacement_Tariff_Rate_10_Percent",
        desc="The replacement tariff rate is correctly identified as 10%",
        parent=rep_node,
        critical=True
    )
    await evaluator.verify(
        claim="The replacement tariff rate was 10 percent.",
        node=rate,
        sources=sources,
        additional_instruction="Treat '10 percent' and '10%' as equivalent."
    )

    # Timing details (critical group)
    time_node = evaluator.add_parallel(
        id="Replacement_Tariff_Timing_Details",
        desc="Timing details for the Section 122 replacement tariff are provided",
        parent=rep_node,
        critical=True
    )

    eff = evaluator.add_leaf(
        id="Effective_Date_February_24_2026",
        desc="The effective date is correctly identified as February 24, 2026",
        parent=time_node,
        critical=True
    )
    await evaluator.verify(
        claim="The replacement tariff took effect on February 24, 2026.",
        node=eff,
        sources=sources
    )

    dur = evaluator.add_leaf(
        id="Duration_150_Days",
        desc="The duration is correctly identified as 150 days",
        parent=time_node,
        critical=True
    )
    await evaluator.verify(
        claim="The replacement tariff remained in effect for 150 days.",
        node=dur,
        sources=sources
    )

    exp = evaluator.add_leaf(
        id="Expiration_Date_July_24_2026",
        desc="The expiration date is correctly identified as July 24, 2026",
        parent=time_node,
        critical=True
    )
    await evaluator.verify(
        claim="The replacement tariff expired on July 24, 2026.",
        node=exp,
        sources=sources
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel as per rubric root
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TariffCaseExtraction,
        extraction_name="extracted_fields",
    )

    # Record rubric "ground truth" expectations for transparency
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "allowed_state_examples": sorted(list(ALLOWED_STATES)),
        },
        gt_type="rubric_expectations",
    )

    # Build tree under a main analysis node
    analysis = evaluator.add_parallel(
        id="Supreme_Court_Tariff_Case_Analysis",
        desc="Comprehensive analysis of the Supreme Court tariff ruling, parties involved, refund processes, and replacement measures",
        parent=root,
        critical=False,
    )

    # Add all verification subtrees
    await build_case_details_checks(evaluator, analysis, extracted)
    await build_lead_plaintiff_checks(evaluator, analysis, extracted)
    await build_vos_checks(evaluator, analysis, extracted)
    await build_state_plaintiffs_checks(evaluator, analysis, extracted)
    await build_refund_process_checks(evaluator, analysis, extracted)
    await build_replacement_tariff_checks(evaluator, analysis, extracted)

    return evaluator.get_summary()