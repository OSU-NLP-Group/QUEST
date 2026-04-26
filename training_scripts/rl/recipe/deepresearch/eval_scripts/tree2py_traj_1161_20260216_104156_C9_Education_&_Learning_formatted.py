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
TASK_ID = "tx_grad_hcc_plan"
TASK_DESCRIPTION = """A high school junior at a Texas school district with a Houston Community College (HCC) dual credit partnership is planning their complete graduation pathway to achieve all of the following goals: (1) graduate with the Leander ISD Distinguished with Honors plan, (2) qualify for Texas Top 10% automatic admission to Texas public universities, and (3) participate in HCC dual credit courses that count toward the Texas Core Curriculum.

Create and verify a comprehensive graduation plan that satisfies all requirements for these three goals. Your plan must specify:

- All required courses by subject area to meet Leander ISD Distinguished with Honors requirements (26 total credits)
- Specific course selections for: 4 English credits (English I, II, III, IV), 4 Mathematics credits (Algebra I, Geometry, Algebra II, plus one with Algebra II prerequisite), 4 Science credits (Biology, Chemistry or Physics, plus 2 additional), 4 Social Studies credits (World Geography or World History, U.S. History, U.S. Government 0.5, Economics 0.5, plus 1 additional), 3 LOTE credits (same language), 1 PE credit, 1 Fine Arts credit, and 5 Electives
- Completion of at least one endorsement from Leander ISD's four options (STEM, Business and Industry, Public Service, or Arts and Humanities)
- Documentation of FAFSA or TASFA completion requirement
- Verification that the plan achieves Distinguished Level of Achievement (including Algebra II and endorsement completion) to qualify for Top 10% automatic admission
- Confirmation of HCC dual credit eligibility including: school district partnership with HCC, HCC's SACSCOC accreditation status, grade level requirement (10th grade or higher), and college readiness assessment completion (TSIA2, SAT, or ACT)
- Verification that HCC dual credit courses contribute to the 42-hour Texas Core Curriculum that transfers to all Texas public universities

For each requirement category, provide the URL reference that documents the specific requirement.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # Subject-area courses and URLs
    english_courses: List[str] = Field(default_factory=list)
    english_urls: List[str] = Field(default_factory=list)

    math_courses: List[str] = Field(default_factory=list)
    math_urls: List[str] = Field(default_factory=list)

    science_courses: List[str] = Field(default_factory=list)
    science_urls: List[str] = Field(default_factory=list)

    social_studies_courses: List[str] = Field(default_factory=list)
    social_studies_urls: List[str] = Field(default_factory=list)

    lote_language: Optional[str] = None
    lote_courses: List[str] = Field(default_factory=list)
    lote_urls: List[str] = Field(default_factory=list)

    pe_courses: List[str] = Field(default_factory=list)
    pe_urls: List[str] = Field(default_factory=list)

    fine_arts_courses: List[str] = Field(default_factory=list)
    fine_arts_urls: List[str] = Field(default_factory=list)

    electives_courses: List[str] = Field(default_factory=list)
    electives_urls: List[str] = Field(default_factory=list)

    total_credits: Optional[str] = None
    total_credits_urls: List[str] = Field(default_factory=list)

    # Endorsement
    endorsement_name: Optional[str] = None
    endorsement_courses: List[str] = Field(default_factory=list)
    endorsement_urls: List[str] = Field(default_factory=list)

    # FAFSA/TASFA
    fafsa_tasfa_documentation: Optional[str] = None
    fafsa_tasfa_urls: List[str] = Field(default_factory=list)

    # Distinguished Level of Achievement (DLA)
    dla_statement: Optional[str] = None
    dla_basis_text: Optional[str] = None
    dla_urls: List[str] = Field(default_factory=list)

    # Top 10% automatic admission
    top10_statement: Optional[str] = None
    top10_urls: List[str] = Field(default_factory=list)

    # HCC dual credit eligibility and institution
    district_name: Optional[str] = None
    district_hcc_partnership_urls: List[str] = Field(default_factory=list)
    hcc_sacscoc_urls: List[str] = Field(default_factory=list)
    grade_level_requirement_urls: List[str] = Field(default_factory=list)
    college_readiness_pathway_urls: List[str] = Field(default_factory=list)
    rwm_requirement_urls: List[str] = Field(default_factory=list)

    # Texas Core Curriculum and transfer
    core_42_urls: List[str] = Field(default_factory=list)
    dual_credit_courses: List[str] = Field(default_factory=list)
    dual_credit_core_applicability_urls: List[str] = Field(default_factory=list)
    core_block_transfer_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_graduation_plan() -> str:
    return """
Extract the student's proposed graduation plan details exactly as stated in the answer. Return all fields even if some are empty, using null for unknown single-value fields and [] for missing lists. Only extract URLs that are explicitly present in the answer (plain or Markdown).

Required JSON fields:
- english_courses: list of the 4 English courses named in the plan (e.g., ["English I","English II","English III","English IV"])
- english_urls: list of URLs the answer cites for English requirements

- math_courses: list of the 4 Math courses named (must include Algebra I, Geometry, Algebra II, plus one additional with Algebra II prerequisite if provided)
- math_urls: list of URLs the answer cites for Math requirements

- science_courses: list of the 4 Science courses named (must include Biology; either Chemistry or Physics; plus two additional sciences)
- science_urls: list of URLs the answer cites for Science requirements

- social_studies_courses: list of the courses named for Social Studies (should include World Geography or World History; U.S. History; U.S. Government (0.5); Economics (0.5); plus 1 additional social studies)
- social_studies_urls: list of URLs the answer cites for Social Studies requirements

- lote_language: the language name if specified (e.g., "Spanish", "French"), else null
- lote_courses: list of the three LOTE courses named (3 credits in the same language)
- lote_urls: list of URLs the answer cites for LOTE requirements

- pe_courses: list of PE/athletics courses used to fulfill 1 PE credit (list length may be ≥1 if multiple half-credits)
- pe_urls: list of URLs the answer cites for PE requirement

- fine_arts_courses: list of Fine Arts courses used to fulfill the requirement
- fine_arts_urls: list of URLs the answer cites for Fine Arts requirement

- electives_courses: list of the 5 elective courses named
- electives_urls: list of URLs the answer cites for Electives requirement

- total_credits: the total credits stated in the plan (string as shown, e.g., "26"), or null if not stated
- total_credits_urls: list of URLs the answer cites for the 26-credit total requirement

- endorsement_name: the chosen endorsement (one of: "STEM","Business and Industry","Public Service","Arts and Humanities"), or null
- endorsement_courses: list of endorsement-related courses/credits named in the plan
- endorsement_urls: list of URLs the answer cites for endorsement requirements

- fafsa_tasfa_documentation: snippet or phrase showing FAFSA/TASFA (or opt-out) documentation in the plan, or null
- fafsa_tasfa_urls: list of URLs the answer cites for FAFSA/TASFA (or opt-out) graduation requirement

- dla_statement: sentence/phrase in the answer explicitly verifying the plan achieves Distinguished Level of Achievement, or null
- dla_basis_text: sentence/phrase tying DLA to Algebra II and endorsement completion, or null
- dla_urls: list of URLs the answer cites for DLA requirements

- top10_statement: sentence/phrase verifying Top 10% automatic admission requires earning DLA and that the plan meets it, or null
- top10_urls: list of URLs the answer cites for Texas Top 10% automatic admission requirements

- district_name: the school district named that partners with HCC, or null
- district_hcc_partnership_urls: list of URLs that document the district's HCC dual credit partnership
- hcc_sacscoc_urls: list of URLs that document HCC SACSCOC accreditation
- grade_level_requirement_urls: list of URLs that document minimum grade-level eligibility for dual credit (10th grade or higher)
- college_readiness_pathway_urls: list of URLs that document that TSIA2, SAT, or ACT can demonstrate college readiness
- rwm_requirement_urls: list of URLs that document readiness assessment includes Reading, Writing, and Mathematics

- core_42_urls: list of URLs that document Texas Core Curriculum is 42 SCH
- dual_credit_courses: list of HCC dual credit course(s) identified in the plan
- dual_credit_core_applicability_urls: list of URLs that document that the identified dual credit course(s) count toward the Texas Core Curriculum
- core_block_transfer_urls: list of URLs that document that completing the 42-hour core transfers as a complete block to other Texas public institutions

SPECIAL RULES:
- Only include URLs explicitly present in the answer.
- Preserve course names exactly as written by the answer (do not normalize roman numerals vs. numbers).
- If something is not present, return null or [] accordingly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _list_to_str(items: List[str]) -> str:
    return ", ".join(items) if items else "(none)"

def _has_number_like_26(text: Optional[str]) -> bool:
    if text is None:
        return False
    t = text.strip().lower()
    return "26" in t or "twenty-six" in t or "twenty six" in t

def _counts_imply_26(plan: PlanExtraction) -> bool:
    """
    Fallback heuristic: if per-category counts match the required structure,
    then total credits are very likely 26.
    """
    english_ok = len(plan.english_courses) == 4
    math_ok = len(plan.math_courses) == 4
    science_ok = len(plan.science_courses) == 4
    # Social studies credits equal 4, but list may contain 5 course items due to 0.5 credits each for Gov/Econ.
    social_min_ok = len(plan.social_studies_courses) >= 4
    lote_ok = len(plan.lote_courses) == 3
    pe_ok = len(plan.pe_courses) >= 1  # could be one 1.0 credit or two 0.5 courses
    fine_ok = len(plan.fine_arts_courses) >= 1
    electives_ok = len(plan.electives_courses) == 5
    return english_ok and math_ok and science_ok and social_min_ok and lote_ok and pe_ok and fine_ok and electives_ok


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_english_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="English_Requirements",
        desc="English subject-area requirements.",
        parent=parent,
        critical=True
    )

    # Courses & Credits (simple verify against the answer)
    leaf_courses = evaluator.add_leaf(
        id="English_Courses_And_Credits",
        desc="Plan includes exactly 4 English credits specifically consisting of English I, English II, English III, and English IV.",
        parent=node,
        critical=True
    )
    claim = (
        f"The plan lists exactly four English credits: English I, English II, English III, and English IV. "
        f"The plan's English courses are: {_list_to_str(plan.english_courses)}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge only based on the answer text. Treat 'English 1/2/3/4' as equivalent to 'English I/II/III/IV'. Minor naming variations are acceptable if clearly the same level."
    )

    # URL reference (verify by cited URLs)
    leaf_url = evaluator.add_leaf(
        id="English_URL_Reference",
        desc="Provides a URL documenting the Leander ISD English requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that four English credits (English I, English II, English III, and English IV) are required for the Leander ISD Distinguished with Honors graduation plan."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.english_urls,
        additional_instruction="Accept reasonable official sources (e.g., Leander ISD planning guide or TEA-aligned documents) that explicitly state or clearly imply four English credits as described."
    )


async def verify_math_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Mathematics_Requirements",
        desc="Mathematics subject-area requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="Math_Courses_And_Credits",
        desc="Plan includes exactly 4 math credits: Algebra I, Geometry, Algebra II, plus a 4th math course whose minimum prerequisite is Algebra II.",
        parent=node,
        critical=True
    )
    claim = (
        f"The plan lists exactly four Math credits: Algebra I, Geometry, Algebra II, and a fourth math with Algebra II as a prerequisite. "
        f"The plan's Math courses are: {_list_to_str(plan.math_courses)}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based only on the answer. Treat 'Algebra 1' vs 'Algebra I' similarly. For the 4th math, it should be described as advanced beyond Algebra II or have Algebra II as a prerequisite."
    )

    leaf_url = evaluator.add_leaf(
        id="Math_URL_Reference",
        desc="Provides a URL documenting the Leander ISD mathematics requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires Algebra I, Geometry, Algebra II, plus one advanced mathematics course for which Algebra II is the minimum prerequisite."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.math_urls,
        additional_instruction="The page should clearly state (or imply) the sequence Algebra I, Geometry, Algebra II, and an additional advanced math requiring Algebra II."
    )


async def verify_science_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Science_Requirements",
        desc="Science subject-area requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="Science_Courses_And_Credits",
        desc="Plan includes exactly 4 science credits: Biology; either Chemistry or Physics; plus two additional authorized science courses.",
        parent=node,
        critical=True
    )
    claim = (
        f"The plan lists exactly four Science credits including Biology; either Chemistry or Physics; plus two additional sciences. "
        f"The plan's Science courses are: {_list_to_str(plan.science_courses)}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based only on the answer. Accept minor naming variants (e.g., 'Biology I'). Ensure Chemistry or Physics appears, and two other sciences are present."
    )

    leaf_url = evaluator.add_leaf(
        id="Science_URL_Reference",
        desc="Provides a URL documenting the Leander ISD science requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires Biology; Chemistry or Physics; plus two additional science credits."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.science_urls,
        additional_instruction="The page should explicitly state or clearly imply Biology, either Chemistry or Physics, and two additional sciences for graduation."
    )


async def verify_social_studies_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Social_Studies_Requirements",
        desc="Social studies subject-area requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="Social_Studies_Courses_And_Credits",
        desc="Plan includes exactly 4 social studies credits: World Geography or World History; U.S. History; U.S. Government (0.5); Economics (0.5); plus one additional social studies credit.",
        parent=node,
        critical=True
    )
    claim = (
        f"The plan lists the required Social Studies: World Geography or World History; U.S. History; U.S. Government (0.5); Economics (0.5); and one additional Social Studies credit. "
        f"The plan's Social Studies courses are: {_list_to_str(plan.social_studies_courses)}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based on the answer. Accept minor naming variants. Verify that Gov and Econ appear as half credits and that one additional social studies course is present."
    )

    leaf_url = evaluator.add_leaf(
        id="Social_Studies_URL_Reference",
        desc="Provides a URL documenting the Leander ISD social studies requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that Social Studies requires World Geography or World History, U.S. History, U.S. Government (0.5), Economics (0.5), plus one additional Social Studies credit (totaling 4 credits)."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.social_studies_urls,
        additional_instruction="The page should explicitly list this combination or clearly describe the same requirement."
    )


async def verify_lote_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="LOTE_Requirements",
        desc="Languages Other Than English (LOTE) requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="LOTE_Credits_Same_Language",
        desc="Plan includes exactly 3 LOTE credits in the same language.",
        parent=node,
        critical=True
    )
    claim = (
        f"The plan includes exactly three LOTE credits in the same language ({plan.lote_language or 'unspecified language'}): "
        f"{_list_to_str(plan.lote_courses)}."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based solely on the answer. Confirm there are three credits and they appear to be in the same language (e.g., Spanish I/II/III)."
    )

    leaf_url = evaluator.add_leaf(
        id="LOTE_URL_Reference",
        desc="Provides a URL documenting the Leander ISD LOTE requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires three credits in the same LOTE (language other than English)."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.lote_urls,
        additional_instruction="The page should explicitly state or clearly imply 3 LOTE credits in the same language are required."
    )


async def verify_pe_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="PE_Requirements",
        desc="Physical Education requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="PE_Credit",
        desc="Plan includes exactly 1 Physical Education credit.",
        parent=node,
        critical=True
    )
    claim = f"The plan includes exactly one PE (Physical Education) credit: {_list_to_str(plan.pe_courses)}."
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based on the answer. Allow fulfillment via athletics or multiple 0.5-credit courses adding up to 1.0."
    )

    leaf_url = evaluator.add_leaf(
        id="PE_URL_Reference",
        desc="Provides a URL documenting the Leander ISD PE requirement used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires one PE credit (or approved substitutions)."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.pe_urls,
        additional_instruction="The page should clearly state a 1-credit PE requirement or approved substitutes that equal 1 credit."
    )


async def verify_fine_arts_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Fine_Arts_Requirements",
        desc="Fine Arts requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="Fine_Arts_Credit",
        desc="Plan includes exactly 1 Fine Arts credit.",
        parent=node,
        critical=True
    )
    claim = f"The plan includes exactly one Fine Arts credit: {_list_to_str(plan.fine_arts_courses)}."
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based on the answer. Accept courses such as Art, Band, Choir, Theatre, Dance, etc., as appropriate fine arts."
    )

    leaf_url = evaluator.add_leaf(
        id="Fine_Arts_URL_Reference",
        desc="Provides a URL documenting the Leander ISD Fine Arts requirement used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires one Fine Arts credit."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.fine_arts_urls,
        additional_instruction="The page should clearly state one Fine Arts credit is required."
    )


async def verify_electives_requirements(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Electives_Requirements",
        desc="Electives requirements.",
        parent=parent,
        critical=True
    )

    leaf_courses = evaluator.add_leaf(
        id="Electives_Credit_Count",
        desc="Plan includes exactly 5 elective credits.",
        parent=node,
        critical=True
    )
    claim = f"The plan includes exactly five elective credits: {_list_to_str(plan.electives_courses)}."
    await evaluator.verify(
        claim=claim,
        node=leaf_courses,
        additional_instruction="Judge based on the answer. The plan should clearly show five electives totaling 5 credits."
    )

    leaf_url = evaluator.add_leaf(
        id="Electives_URL_Reference",
        desc="Provides a URL documenting the Leander ISD electives requirement used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that graduation requires five elective credits."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.electives_urls,
        additional_instruction="The page should clearly state five elective credits are required."
    )


async def verify_total_credits(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Total_Credits_Requirement",
        desc="Total credit requirement for Leander ISD Distinguished with Honors.",
        parent=parent,
        critical=True
    )

    # Binary custom check based on explicit total or implied counts
    total_ok = _has_number_like_26(plan.total_credits) or _counts_imply_26(plan)
    evaluator.add_custom_node(
        result=total_ok,
        id="Total_Credits_Equals_26",
        desc="Plan totals exactly 26 credits overall.",
        parent=node,
        critical=True
    )

    leaf_url = evaluator.add_leaf(
        id="Total_Credits_URL_Reference",
        desc="Provides a URL documenting the Leander ISD 26-credit total requirement used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that the Leander ISD Distinguished with Honors plan requires a total of 26 credits."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.total_credits_urls,
        additional_instruction="The page should explicitly state a 26-credit total (or equivalently describe it)."
    )


async def verify_endorsement(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Endorsement_Requirement",
        desc="Endorsement completion requirement.",
        parent=parent,
        critical=True
    )

    allowed_endorsements = {"STEM", "Business and Industry", "Public Service", "Arts and Humanities"}
    has_valid_endorsement = plan.endorsement_name is not None and plan.endorsement_name.strip() in allowed_endorsements

    evaluator.add_custom_node(
        result=has_valid_endorsement,
        id="At_Least_One_Leander_Endorsement",
        desc="Plan includes completion of at least one endorsement from Leander ISD’s offered options (STEM, Business and Industry, Public Service, or Arts and Humanities).",
        parent=node,
        critical=True
    )

    leaf_identified = evaluator.add_leaf(
        id="Chosen_Endorsement_Explicitly_Identified",
        desc="Plan explicitly identifies which endorsement is being pursued/completed.",
        parent=node,
        critical=True
    )
    claim_identified = f"The plan explicitly identifies the endorsement as '{plan.endorsement_name or '(none)'}'."
    await evaluator.verify(
        claim=claim_identified,
        node=leaf_identified,
        additional_instruction="Judge based on the answer. The endorsement should be clearly named."
    )

    leaf_requirements = evaluator.add_leaf(
        id="Endorsement_Specific_Requirements_Met",
        desc="Plan includes the specific course/credit requirements needed to complete the identified endorsement.",
        parent=node,
        critical=True
    )
    claim_req = (
        f"The plan includes specific courses/credits to complete the '{plan.endorsement_name or '(none)'}' endorsement: "
        f"{_list_to_str(plan.endorsement_courses)}."
    )
    await evaluator.verify(
        claim=claim_req,
        node=leaf_requirements,
        additional_instruction="Judge based on the answer. The listed courses/credits should plausibly fulfill the identified endorsement per district guidance."
    )

    leaf_url = evaluator.add_leaf(
        id="Endorsement_URL_Reference",
        desc="Provides a URL documenting the Leander ISD endorsement requirements used.",
        parent=node,
        critical=True
    )
    url_claim = f"The provided source(s) document the requirements for the '{plan.endorsement_name or '(none)'}' endorsement in Leander ISD (or TEA-aligned documents)."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.endorsement_urls,
        additional_instruction="The page should clearly describe endorsement requirements for Leander ISD or TEA-aligned guidance."
    )


async def verify_fafsa_tasfa(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="FAFSA_TASFA_Graduation_Requirement",
        desc="FAFSA/TASFA (or opt-out) graduation documentation requirement.",
        parent=parent,
        critical=True
    )

    leaf_included = evaluator.add_leaf(
        id="FAFSA_TASFA_Or_OptOut_Included",
        desc="Plan documents FAFSA or TASFA completion, or an opt-out form submission, as required for Texas high school graduation.",
        parent=node,
        critical=True
    )
    claim_inc = (
        f"The plan documents FAFSA or TASFA completion (or opt-out) as a graduation requirement. "
        f"Evidence in plan: {plan.fafsa_tasfa_documentation or '(none)'}."
    )
    await evaluator.verify(
        claim=claim_inc,
        node=leaf_included,
        additional_instruction="Judge based on the answer. Look for explicit mention of FAFSA/TASFA completion or an opt-out form as required for graduation."
    )

    leaf_url = evaluator.add_leaf(
        id="FAFSA_TASFA_URL_Reference",
        desc="Provides a URL documenting the Texas FAFSA/TASFA (or opt-out) graduation requirement used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that Texas requires students to complete FAFSA or TASFA (or submit an opt-out form) for high school graduation."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.fafsa_tasfa_urls,
        additional_instruction="The page should clearly state the FAFSA/TASFA (or opt-out) graduation requirement in Texas."
    )


async def verify_dla(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Distinguished_Level_Of_Achievement_Verification",
        desc="Distinguished Level of Achievement (DLA) verification needed for Top 10% eligibility.",
        parent=parent,
        critical=True
    )

    leaf_stmt = evaluator.add_leaf(
        id="DLA_Verification_Statement_Included",
        desc="Answer explicitly verifies the plan achieves Texas Distinguished Level of Achievement.",
        parent=node,
        critical=True
    )
    claim_stmt = f"The plan explicitly states the student will earn the Distinguished Level of Achievement (DLA). Statement: {plan.dla_statement or '(none)'}."
    await evaluator.verify(
        claim=claim_stmt,
        node=leaf_stmt,
        additional_instruction="Judge based on the answer. Look for an explicit DLA achievement statement."
    )

    leaf_basis = evaluator.add_leaf(
        id="DLA_Basis_Includes_AlgebraII_And_Endorsement",
        desc="Answer explicitly ties DLA verification to (at minimum) Algebra II completion and endorsement completion.",
        parent=node,
        critical=True
    )
    claim_basis = (
        "The plan ties Distinguished Level of Achievement to Algebra II completion and endorsement completion. "
        f"Evidence in the plan: {plan.dla_basis_text or '(none)'}."
    )
    await evaluator.verify(
        claim=claim_basis,
        node=leaf_basis,
        additional_instruction="Judge based on the answer. Accept if the text clearly links DLA to Algebra II and endorsement completion."
    )

    leaf_url = evaluator.add_leaf(
        id="DLA_URL_Reference",
        desc="Provides a URL documenting Distinguished Level of Achievement requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that earning the Distinguished Level of Achievement includes completing Algebra II and an endorsement."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.dla_urls,
        additional_instruction="The page should clearly describe DLA requirements and mention Algebra II and endorsement."
    )


async def verify_top10(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Top_10_Percent_Automatic_Admission_Verification",
        desc="Texas Top 10% automatic admission verification.",
        parent=parent,
        critical=True
    )

    leaf_linked = evaluator.add_leaf(
        id="Top10_Eligibility_Linked_To_DLA",
        desc="Answer verifies that Top 10% automatic admission eligibility requires earning Distinguished Level of Achievement and confirms the plan meets that condition.",
        parent=node,
        critical=True
    )
    claim_linked = (
        "The plan verifies that Top 10% automatic admission eligibility requires earning the Distinguished Level of Achievement and confirms the plan meets that condition. "
        f"Plan text: {plan.top10_statement or '(none)'}."
    )
    await evaluator.verify(
        claim=claim_linked,
        node=leaf_linked,
        additional_instruction="Judge based on the answer. Look for linkage between Top 10% automatic admission and DLA, and that the plan meets it."
    )

    leaf_url = evaluator.add_leaf(
        id="Top10_URL_Reference",
        desc="Provides a URL documenting Texas Top 10% automatic admission requirements used.",
        parent=node,
        critical=True
    )
    url_claim = "The provided source(s) document that Texas Top 10% automatic admission requires earning the Distinguished Level of Achievement (or otherwise link DLA to Top 10% eligibility)."
    await evaluator.verify(
        claim=url_claim,
        node=leaf_url,
        sources=plan.top10_urls,
        additional_instruction="The page should describe Top 10% automatic admissions and the DLA requirement linkage."
    )


async def verify_hcc_dual_credit_eligibility(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="HCC_Dual_Credit_Eligibility_And_Institution_Verification",
        desc="HCC dual credit eligibility and institutional requirements.",
        parent=parent,
        critical=True
    )

    # District partnership verified by URL
    leaf_partnership_verified = evaluator.add_leaf(
        id="District_HCC_Partnership_Verified",
        desc="Answer verifies that the relevant school district has an official HCC dual credit partnership (as required by the prompt/constraints).",
        parent=node,
        critical=True
    )
    claim_partner = (
        f"There is an official dual credit partnership between {plan.district_name or 'the district'} and Houston Community College."
    )
    await evaluator.verify(
        claim=claim_partner,
        node=leaf_partnership_verified,
        sources=plan.district_hcc_partnership_urls,
        additional_instruction="The page should be from HCC or the school district and clearly indicate a dual credit partnership."
    )

    # Partnership URL existence/content (also verified by URL)
    leaf_partnership_url = evaluator.add_leaf(
        id="District_HCC_Partnership_URL",
        desc="Provides a URL documenting the school district’s HCC dual credit partnership.",
        parent=node,
        critical=True
    )
    claim_partner_url = "This URL is an official page documenting the school district’s dual credit partnership with HCC."
    await evaluator.verify(
        claim=claim_partner_url,
        node=leaf_partnership_url,
        sources=plan.district_hcc_partnership_urls,
        additional_instruction="Accept official HCC or district pages that clearly describe or list the dual credit partnership."
    )

    # HCC SACSCOC accreditation
    leaf_sacs = evaluator.add_leaf(
        id="HCC_SACSCOC_Accreditation_Verified",
        desc="Answer verifies that Houston Community College is accredited by SACSCOC.",
        parent=node,
        critical=True
    )
    claim_sacs = "Houston Community College is accredited by SACSCOC (Southern Association of Colleges and Schools Commission on Colleges)."
    await evaluator.verify(
        claim=claim_sacs,
        node=leaf_sacs,
        sources=plan.hcc_sacscoc_urls,
        additional_instruction="The page should clearly state HCC is accredited by SACSCOC."
    )

    leaf_sacs_url = evaluator.add_leaf(
        id="HCC_SACSCOC_Accreditation_URL",
        desc="Provides a URL documenting HCC’s SACSCOC accreditation status.",
        parent=node,
        critical=True
    )
    claim_sacs_url = "This URL documents HCC’s SACSCOC accreditation status."
    await evaluator.verify(
        claim=claim_sacs_url,
        node=leaf_sacs_url,
        sources=plan.hcc_sacscoc_urls,
        additional_instruction="Accept HCC accreditation pages or SACSCOC listings showing HCC accreditation."
    )

    # Grade level eligibility (answer-level verification; student is a junior)
    leaf_grade = evaluator.add_leaf(
        id="Grade_Level_Eligibility_Met",
        desc="Answer verifies the student meets the minimum grade-level eligibility for dual credit (10th grade or higher).",
        parent=node,
        critical=True
    )
    claim_grade = "The plan verifies the student is at least in 10th grade (the student is a junior/11th grade), satisfying the minimum grade-level eligibility for HCC dual credit."
    await evaluator.verify(
        claim=claim_grade,
        node=leaf_grade,
        additional_instruction="Judge based on the answer text. A 'junior' is 11th grade and meets a '10th grade or higher' requirement."
    )

    leaf_grade_url = evaluator.add_leaf(
        id="Grade_Level_Eligibility_URL",
        desc="Provides a URL documenting the minimum grade-level eligibility requirement for dual credit (10th grade or higher).",
        parent=node,
        critical=True
    )
    claim_grade_url = "The provided source(s) document that high school students must be in at least 10th grade to be eligible for dual credit."
    await evaluator.verify(
        claim=claim_grade_url,
        node=leaf_grade_url,
        sources=plan.grade_level_requirement_urls,
        additional_instruction="The page should explicitly state the minimum grade level (10th grade or higher) for dual credit participation."
    )

    # College readiness pathway (TSIA2, SAT, ACT)
    leaf_readiness = evaluator.add_leaf(
        id="College_Readiness_Pathway_Verified",
        desc="Answer verifies college readiness can be demonstrated via TSIA2, SAT, or ACT scores for dual credit eligibility.",
        parent=node,
        critical=True
    )
    claim_readiness = "College readiness for dual credit can be demonstrated via TSIA2, SAT, or ACT scores."
    await evaluator.verify(
        claim=claim_readiness,
        node=leaf_readiness,
        sources=plan.college_readiness_pathway_urls,
        additional_instruction="The page should list or clearly indicate TSIA2, SAT, and/or ACT as acceptable readiness metrics for dual credit."
    )

    leaf_readiness_url = evaluator.add_leaf(
        id="College_Readiness_Pathway_URL",
        desc="Provides a URL documenting that TSIA2, SAT, or ACT can be used to demonstrate college readiness for dual credit.",
        parent=node,
        critical=True
    )
    claim_readiness_url = "This URL documents that TSIA2, SAT, or ACT can be used to demonstrate college readiness for dual credit."
    await evaluator.verify(
        claim=claim_readiness_url,
        node=leaf_readiness_url,
        sources=plan.college_readiness_pathway_urls,
        additional_instruction="The page should clearly show TSIA2/SAT/ACT as readiness options."
    )

    # R/W/M assessment
    leaf_rwm = evaluator.add_leaf(
        id="Readiness_Assessed_In_RWM_Verified",
        desc="Answer verifies dual credit readiness includes assessment in Reading, Writing, and Mathematics.",
        parent=node,
        critical=True
    )
    claim_rwm = "Dual credit college readiness is assessed in Reading, Writing, and Mathematics."
    await evaluator.verify(
        claim=claim_rwm,
        node=leaf_rwm,
        sources=plan.rwm_requirement_urls,
        additional_instruction="The page should indicate readiness requirements/benchmarks in Reading, Writing, and Math."
    )

    leaf_rwm_url = evaluator.add_leaf(
        id="Readiness_Assessed_In_RWM_URL",
        desc="Provides a URL documenting the Reading/Writing/Mathematics assessment requirement for dual credit.",
        parent=node,
        critical=True
    )
    claim_rwm_url = "This URL documents that Reading, Writing, and Mathematics are assessed for dual credit readiness."
    await evaluator.verify(
        claim=claim_rwm_url,
        node=leaf_rwm_url,
        sources=plan.rwm_requirement_urls,
        additional_instruction="The page should clearly mention R/W/M readiness assessments for dual credit."
    )


async def verify_texas_core_transfer(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Texas_Core_Curriculum_Transfer_Verification",
        desc="Texas Core Curriculum applicability and transfer verification for HCC dual credit.",
        parent=parent,
        critical=True
    )

    # 42 SCH verified
    leaf_42 = evaluator.add_leaf(
        id="Core_Is_42_SCH_Verified",
        desc="Answer verifies that the Texas Core Curriculum consists of 42 semester credit hours.",
        parent=node,
        critical=True
    )
    claim_42 = "The Texas Core Curriculum consists of 42 semester credit hours."
    await evaluator.verify(
        claim=claim_42,
        node=leaf_42,
        sources=plan.core_42_urls,
        additional_instruction="The page should clearly state that the Texas Core Curriculum is 42 semester credit hours."
    )

    leaf_42_url = evaluator.add_leaf(
        id="Core_42_SCH_URL",
        desc="Provides a URL documenting that the Texas Core Curriculum is 42 semester credit hours.",
        parent=node,
        critical=True
    )
    claim_42_url = "This URL documents that the Texas Core Curriculum is 42 semester credit hours."
    await evaluator.verify(
        claim=claim_42_url,
        node=leaf_42_url,
        sources=plan.core_42_urls,
        additional_instruction="Accept official THECB/college catalog pages that clearly state 42 SCH for TCC."
    )

    # Dual credit course(s) identified
    evaluator.add_custom_node(
        result=len(plan.dual_credit_courses) > 0,
        id="Dual_Credit_Courses_Identified",
        desc="Answer identifies the HCC dual credit course(s) included in the plan.",
        parent=node,
        critical=True
    )

    # Dual credit courses verified as Core applicable
    leaf_core_app = evaluator.add_leaf(
        id="Dual_Credit_Courses_Verified_As_Core_Applicable",
        desc="Answer verifies that the identified dual credit course(s) count toward the Texas Core Curriculum.",
        parent=node,
        critical=True
    )
    claim_core_app = (
        f"The identified HCC dual credit course(s) are part of the Texas Core Curriculum: {_list_to_str(plan.dual_credit_courses)}."
    )
    await evaluator.verify(
        claim=claim_core_app,
        node=leaf_core_app,
        sources=plan.dual_credit_core_applicability_urls,
        additional_instruction="The page(s) should show that the specific courses listed apply to (are included in) the Texas Core Curriculum at HCC."
    )

    leaf_core_app_url = evaluator.add_leaf(
        id="Dual_Credit_Core_Applicability_URL",
        desc="Provides a URL documenting that the identified dual credit course(s) apply toward the Texas Core Curriculum.",
        parent=node,
        critical=True
    )
    claim_core_app_url = "This URL documents that the identified dual credit course(s) apply toward the Texas Core Curriculum."
    await evaluator.verify(
        claim=claim_core_app_url,
        node=leaf_core_app_url,
        sources=plan.dual_credit_core_applicability_urls,
        additional_instruction="Accept official HCC/THECB pages that clearly map the listed courses to the Core Curriculum."
    )

    # Core block transfer guarantee
    leaf_block = evaluator.add_leaf(
        id="Core_Block_Transfer_Guarantee_Verified",
        desc="Answer verifies that completing the 42-hour core at a Texas public institution transfers as a complete block to other Texas public institutions.",
        parent=node,
        critical=True
    )
    claim_block = "Completing the 42-hour Texas Core Curriculum at a Texas public institution transfers as a complete block to other Texas public institutions."
    await evaluator.verify(
        claim=claim_block,
        node=leaf_block,
        sources=plan.core_block_transfer_urls,
        additional_instruction="The page should clearly state the core 'transfers as a block' policy in Texas public institutions."
    )

    leaf_block_url = evaluator.add_leaf(
        id="Core_Block_Transfer_URL",
        desc="Provides a URL documenting Texas Core Curriculum transfer-as-a-block policy.",
        parent=node,
        critical=True
    )
    claim_block_url = "This URL documents that the completed 42-hour Texas Core Curriculum transfers as a complete block to other Texas public institutions."
    await evaluator.verify(
        claim=claim_block_url,
        node=leaf_block_url,
        sources=plan.core_block_transfer_urls,
        additional_instruction="Accept official THECB/university policies clearly stating block transfer of the 42-hour core."
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
    Evaluate a single answer for the comprehensive Leander ISD + HCC dual credit graduation plan task.
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
        default_model=model
    )

    # Extract structured plan details
    plan = await evaluator.extract(
        prompt=prompt_extract_graduation_plan(),
        template_class=PlanExtraction,
        extraction_name="graduation_plan_extraction"
    )

    # Top-level critical node mirroring rubric root
    plan_node = evaluator.add_parallel(
        id="Complete_Graduation_Plan_Verification",
        desc="Verify the proposed graduation plan satisfies Leander ISD Distinguished with Honors requirements (26 credits), Texas Top 10% automatic admission eligibility (via Distinguished Level of Achievement), and HCC dual credit participation/core transfer requirements, with URL references as required.",
        parent=root,
        critical=True
    )

    # Subject-area verifications
    await verify_english_requirements(evaluator, plan_node, plan)
    await verify_math_requirements(evaluator, plan_node, plan)
    await verify_science_requirements(evaluator, plan_node, plan)
    await verify_social_studies_requirements(evaluator, plan_node, plan)
    await verify_lote_requirements(evaluator, plan_node, plan)
    await verify_pe_requirements(evaluator, plan_node, plan)
    await verify_fine_arts_requirements(evaluator, plan_node, plan)
    await verify_electives_requirements(evaluator, plan_node, plan)
    await verify_total_credits(evaluator, plan_node, plan)

    # Endorsement, FAFSA/TASFA, DLA, Top 10%
    await verify_endorsement(evaluator, plan_node, plan)
    await verify_fafsa_tasfa(evaluator, plan_node, plan)
    await verify_dla(evaluator, plan_node, plan)
    await verify_top10(evaluator, plan_node, plan)

    # HCC dual credit eligibility and Core transfer
    await verify_hcc_dual_credit_eligibility(evaluator, plan_node, plan)
    await verify_texas_core_transfer(evaluator, plan_node, plan)

    return evaluator.get_summary()