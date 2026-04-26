import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "career_pathway_tx_hs_to_d1_ad"
TASK_DESCRIPTION = """
For an individual planning a career pathway from becoming a head football coach at a public high school in Texas to eventually becoming an athletic director at a NCAA Division I institution, provide a comprehensive analysis that includes:

1. Texas High School Head Football Coach Requirements:
   - Educational degree(s) required to become a head football coach at a public high school in Texas.
   - Specific teaching credential or certification status required in Texas for public high school head football coaches.
   - Coaching certifications required by the UIL (University Interscholastic League) for high school football coaches in Texas.
   - Additional safety or sport-specific certifications required or recommended.

2. NCAA Division I Athletic Director Requirements:
   - Minimum educational degree required to become an athletic director at a NCAA Division I institution.
   - Level of education typically preferred or commonly required for Division I athletic director positions.
   - Common degree fields or areas of study for athletic directors.
   - Typical minimum number of years of athletics administration experience required for Division I athletic director positions.
   - Types of prior experience or roles expected for Division I athletic director candidates.
   - Knowledge of NCAA rules/regulations required.

3. Career Progression:
   - Recommended graduate degree to pursue for someone with this career goal.
   - Typical intermediate career positions between high school head coach and Division I athletic director.
   - Overall typical timeline/experience duration.

For each requirement identified, provide supporting reference URL(s) from official or authoritative sources.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class Clause(BaseModel):
    """
    A single claim/statement extracted from the answer, with supporting URLs.
    """
    statement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class TexasHSRequirementsExtraction(BaseModel):
    degree: Optional[Clause] = None
    teacher_cert: Optional[Clause] = None
    uil_ccp: Optional[Clause] = None
    tackling_cert: Optional[Clause] = None
    safety_certs: Optional[Clause] = None
    additional_certs: Optional[Clause] = None


class NCAAD1ADRequirementsExtraction(BaseModel):
    minimum_degree: Optional[Clause] = None
    preferred_education: Optional[Clause] = None
    common_degree_fields: Optional[Clause] = None
    min_admin_experience: Optional[Clause] = None
    prior_roles: Optional[Clause] = None
    rules_knowledge: Optional[Clause] = None


class CareerProgressionExtraction(BaseModel):
    recommended_grad_degree: Optional[Clause] = None
    intermediate_positions: Optional[Clause] = None
    total_career_timeline: Optional[Clause] = None


class CareerPathwayAnalysisExtraction(BaseModel):
    texas_hs: Optional[TexasHSRequirementsExtraction] = None
    ncaa_d1_ad: Optional[NCAAD1ADRequirementsExtraction] = None
    career_progression: Optional[CareerProgressionExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_career_pathway() -> str:
    return """
    Extract the specific claims and their supporting URLs from the answer for three sections:
    A) Texas public high school head football coach requirements,
    B) NCAA Division I athletic director requirements, and
    C) Career progression guidance.

    For each item listed below, extract:
    - statement: A concise sentence capturing exactly what the answer claims (quote or minimally paraphrase).
    - source_urls: All URLs provided in the answer that support or substantiate the claim. Include only valid URLs. If no URLs are provided for that claim, return an empty list.

    Return a JSON object with this structure:
    {
      "texas_hs": {
        "degree": {"statement": ..., "source_urls": [...]},
        "teacher_cert": {"statement": ..., "source_urls": [...]},
        "uil_ccp": {"statement": ..., "source_urls": [...]},
        "tackling_cert": {"statement": ..., "source_urls": [...]},
        "safety_certs": {"statement": ..., "source_urls": [...]},
        "additional_certs": {"statement": ..., "source_urls": [...]}
      },
      "ncaa_d1_ad": {
        "minimum_degree": {"statement": ..., "source_urls": [...]},
        "preferred_education": {"statement": ..., "source_urls": [...]},
        "common_degree_fields": {"statement": ..., "source_urls": [...]},
        "min_admin_experience": {"statement": ..., "source_urls": [...]},
        "prior_roles": {"statement": ..., "source_urls": [...]},
        "rules_knowledge": {"statement": ..., "source_urls": [...]}
      },
      "career_progression": {
        "recommended_grad_degree": {"statement": ..., "source_urls": [...]},
        "intermediate_positions": {"statement": ..., "source_urls": [...]},
        "total_career_timeline": {"statement": ..., "source_urls": [...]}
      }
    }

    SPECIAL RULES:
    - Extract only what is explicitly stated in the answer; do not invent information.
    - For URLs: include all URLs cited in the answer for the specific claim. Accept plain URLs or markdown links; extract the actual URL.
    - If any claim is not present in the answer, set its "statement" to null and "source_urls" to an empty list.
    - Prefer authoritative/official sources when present (e.g., UIL, TEA, NCAA, NACDA, official university job postings), but still include all URLs cited in the answer for the claim.

    ITEMS TO EXTRACT:
    TEXAS HS HEAD COACH:
    - degree
    - teacher_cert
    - uil_ccp
    - tackling_cert
    - safety_certs
    - additional_certs

    NCAA D1 ATHLETIC DIRECTOR:
    - minimum_degree
    - preferred_education
    - common_degree_fields
    - min_admin_experience
    - prior_roles
    - rules_knowledge

    CAREER PROGRESSION:
    - recommended_grad_degree
    - intermediate_positions
    - total_career_timeline
    """


# --------------------------------------------------------------------------- #
# Helper: Add clause verification nodes                                       #
# --------------------------------------------------------------------------- #
async def add_clause_checks(
    evaluator: Evaluator,
    parent_node,
    item_id: str,
    item_desc: str,
    clause: Optional[Clause],
    additional_instruction: str,
) -> None:
    """
    For a single rubric item:
    - Create a sequential node (critical) for the item.
    - Add existence check (critical): statement present AND >=1 URL.
    - Add support check via URL verification (critical): claim supported by cited sources.
    """
    seq_node = evaluator.add_sequential(
        id=item_id,
        desc=item_desc,
        parent=parent_node,
        critical=True
    )

    # Existence check: statement present and at least one URL
    has_statement = clause is not None and clause.statement is not None and clause.statement.strip() != ""
    has_urls = clause is not None and bool(clause.source_urls) and len(clause.source_urls) > 0

    evaluator.add_custom_node(
        result=has_statement and has_urls,
        id=f"{item_id}_exists",
        desc=f"{item_desc} - Answer includes a specific statement and ≥1 supporting URL",
        parent=seq_node,
        critical=True
    )

    # Support check: verify claim against provided URLs
    support_leaf = evaluator.add_leaf(
        id=f"{item_id}_supported",
        desc=f"{item_desc} - Statement is supported by cited source URL(s)",
        parent=seq_node,
        critical=True
    )

    claim_text = clause.statement if (clause and clause.statement) else ""
    urls = clause.source_urls if (clause and clause.source_urls) else []

    await evaluator.verify(
        claim=claim_text,
        node=support_leaf,
        sources=urls,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Build subtrees and run verifications                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_texas_hs(
    evaluator: Evaluator,
    parent_node,
    ex: Optional[TexasHSRequirementsExtraction],
) -> None:
    tx_node = evaluator.add_parallel(
        id="Texas_HS_Head_Football_Coach_Requirements",
        desc="Texas public HS head football coach requirements, each supported by authoritative URL(s).",
        parent=parent_node,
        critical=True
    )

    # Degree requirement
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_Degree_Requirement_With_Citation",
        "States the required degree for Texas public HS head football coach and provides supporting URL(s)",
        ex.degree if ex else None,
        additional_instruction="Confirm whether the provided sources explicitly support the stated educational degree requirement for head football coaches at Texas public high schools (often teacher roles requiring at least a bachelor's degree under TEA/ISD policies). Rely only on the cited webpage(s)."
    )

    # Teacher certification requirement
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_Teacher_Certification_With_Citation",
        "States the required Texas teaching credential/certification status and provides supporting URL(s)",
        ex.teacher_cert if ex else None,
        additional_instruction="Verify that the cited sources support the claim about Texas teaching credential/certification requirements for public HS head football coaches (e.g., TEA certification, district policy)."
    )

    # UIL CCP requirement
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_UIL_CCP_With_Citation",
        "States UIL Coaches Certification Program (CCP) is required and provides supporting URL(s)",
        ex.uil_ccp if ex else None,
        additional_instruction="Check if the sources clearly state the UIL Coaches Certification Program (CCP) requirements for Texas high school coaches."
    )

    # Tackling certification
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_Tackling_Certification_With_Citation",
        "States Texas football coaches must have tackling certification and provides supporting URL(s)",
        ex.tackling_cert if ex else None,
        additional_instruction="Verify whether the sources explicitly require a tackling certification (e.g., specified by UIL or state-level directive) for Texas HS football coaches."
    )

    # Safety certifications (CPR, First Aid, concussion training)
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_Safety_Certs_With_Citation",
        "Identifies required safety certifications (CPR, First Aid, concussion training) and provides supporting URL(s)",
        ex.safety_certs if ex else None,
        additional_instruction="Confirm that the sources support the specified safety certification requirements (e.g., CPR, First Aid, concussion training) for Texas HS coaches."
    )

    # Additional certifications addressed (required or recommended)
    await add_clause_checks(
        evaluator,
        tx_node,
        "TX_HS_Additional_Certifications_Addressed_With_Citation",
        "Addresses additional safety or sport-specific certifications (required or recommended) with supporting URL(s)",
        ex.additional_certs if ex else None,
        additional_instruction="Verify whether the sources support any additional mandated or recommended certifications beyond those listed; or support that no additional certifications are mandated if that is claimed."
    )


async def build_and_verify_ncaa_d1_ad(
    evaluator: Evaluator,
    parent_node,
    ex: Optional[NCAAD1ADRequirementsExtraction],
) -> None:
    d1_node = evaluator.add_parallel(
        id="NCAA_D1_Athletic_Director_Requirements",
        desc="NCAA Division I athletic director requirements, each supported by authoritative URL(s).",
        parent=parent_node,
        critical=True
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_Minimum_Degree_With_Citation",
        "States the minimum educational degree required for NCAA D1 AD roles with supporting URL(s)",
        ex.minimum_degree if ex else None,
        additional_instruction="Verify that the sources support the stated minimum educational degree requirement for Division I athletic director roles (e.g., job postings, university HR pages)."
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_Preferred_Education_With_Citation",
        "States the typically preferred/commonly required education level for NCAA D1 AD roles with supporting URL(s)",
        ex.preferred_education if ex else None,
        additional_instruction="Confirm that the sources support the claim regarding typically preferred or commonly required higher education level (e.g., master's) for Division I AD positions."
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_Common_Degree_Fields_With_Citation",
        "Identifies common degree fields/areas of study for athletic directors with supporting URL(s)",
        ex.common_degree_fields if ex else None,
        additional_instruction="Verify that the sources support the listed common degree fields for ADs (e.g., sport management, business, education administration)."
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_Min_Admin_Experience_With_Citation",
        "States the typical minimum years of athletics administration experience required with supporting URL(s)",
        ex.min_admin_experience if ex else None,
        additional_instruction="Confirm that the sources support the typical minimum years of athletics administration experience required for D1 AD roles."
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_Prior_Roles_With_Citation",
        "Describes expected prior experience/roles for D1 AD candidates with supporting URL(s)",
        ex.prior_roles if ex else None,
        additional_instruction="Verify that the sources support the expected prior roles/experience (e.g., Deputy AD, Senior Associate AD, compliance leadership) for D1 AD candidates."
    )

    await add_clause_checks(
        evaluator,
        d1_node,
        "D1_AD_NCAA_Rules_Knowledge_With_Citation",
        "States knowledge of NCAA rules/regulations is required with supporting URL(s)",
        ex.rules_knowledge if ex else None,
        additional_instruction="Confirm that the sources explicitly require knowledge of NCAA rules and regulations for D1 athletic directors."
    )


async def build_and_verify_career_progression(
    evaluator: Evaluator,
    parent_node,
    ex: Optional[CareerProgressionExtraction],
) -> None:
    cp_node = evaluator.add_parallel(
        id="Career_Progression",
        desc="Career progression guidance between the two roles, supported by authoritative URL(s).",
        parent=parent_node,
        critical=True
    )

    await add_clause_checks(
        evaluator,
        cp_node,
        "Recommended_Graduate_Degree_With_Citation",
        "Recommends an appropriate graduate degree with supporting URL(s)",
        ex.recommended_grad_degree if ex else None,
        additional_instruction="Verify that the sources support the recommended graduate degree for progressing toward a D1 athletic director role (e.g., sport management, MBA, M.Ed.)."
    )

    await add_clause_checks(
        evaluator,
        cp_node,
        "Intermediate_Positions_With_Citation",
        "Identifies typical intermediate positions with supporting URL(s)",
        ex.intermediate_positions if ex else None,
        additional_instruction="Confirm that the sources support the listed intermediate positions between HS head coach and NCAA D1 AD."
    )

    await add_clause_checks(
        evaluator,
        cp_node,
        "Total_Career_Timeline_With_Citation",
        "States the overall typical combined timeline/experience duration with supporting URL(s)",
        ex.total_career_timeline if ex else None,
        additional_instruction="Verify that the sources substantiate the stated overall timeline/years of experience typically required to become a D1 AD."
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
    Evaluate an answer for the Texas HS Head Coach → NCAA D1 AD career pathway analysis.
    Builds a critical parallel tree with three critical subtrees and verifies each claimed requirement against cited URLs.
    """
    # Initialize evaluator and root
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

    # Analysis root (critical)
    analysis_root = evaluator.add_parallel(
        id="Career_Pathway_Analysis",
        desc="Provide the requested analysis for (1) Texas public HS head football coach, (2) NCAA D1 athletic director, and (3) career progression, with authoritative supporting URLs for each stated requirement/claim.",
        parent=root,
        critical=True
    )

    # Extract structured claims and URLs
    extraction = await evaluator.extract(
        prompt=prompt_extract_career_pathway(),
        template_class=CareerPathwayAnalysisExtraction,
        extraction_name="career_pathway_extraction"
    )

    # Build and verify each subtree
    await build_and_verify_texas_hs(evaluator, analysis_root, extraction.texas_hs)
    await build_and_verify_ncaa_d1_ad(evaluator, analysis_root, extraction.ncaa_d1_ad)
    await build_and_verify_career_progression(evaluator, analysis_root, extraction.career_progression)

    # Return summary with verification tree
    return evaluator.get_summary()