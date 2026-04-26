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
TASK_ID = "ceo_succession_2026_f500_homebuilder"
TASK_DESCRIPTION = """
In January 2026, a Fortune 500 homebuilding company announced a CEO succession plan. The incoming CEO holds a Master's degree from Cornell University (graduated in 2004) and a Bachelor's degree from Texas A&M University. This executive joined their current company in 2004 and has been with the organization for at least 20 years as of the announcement. Who is this CEO successor, and what is the name of the company?
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CEOSuccessionExtraction(BaseModel):
    # Identifications
    ceo_name: Optional[str] = None
    company_name: Optional[str] = None

    # Announcement timing
    announcement_month: Optional[str] = None  # e.g., "January" or "Jan"
    announcement_year: Optional[str] = None   # e.g., "2026"

    # Education (graduate)
    grad_school: Optional[str] = None         # Expect "Cornell University"
    grad_degree: Optional[str] = None         # e.g., "Master of Engineering", "MBA", "M.S."
    grad_year: Optional[str] = None           # Expect "2004"

    # Education (undergraduate)
    undergrad_school: Optional[str] = None    # Expect "Texas A&M University"
    undergrad_degree: Optional[str] = None    # e.g., "B.S.", "Bachelor of Science", etc.

    # Career history
    joined_year: Optional[str] = None         # Expect "2004"

    # Source URLs by category (as explicitly cited in the answer)
    announcement_sources: List[str] = Field(default_factory=list)
    company_profile_sources: List[str] = Field(default_factory=list)
    education_sources: List[str] = Field(default_factory=list)
    career_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ceo_succession() -> str:
    return """
    Extract the following structured information from the answer about the CEO succession and the executive's credentials. Only extract information explicitly present in the answer.

    Required fields:
    - ceo_name: Full name of the CEO successor (string or null).
    - company_name: Name of the company (string or null).

    Announcement timing:
    - announcement_month: The month of the announcement (e.g., "January", "Jan") if stated (string or null).
    - announcement_year: The year of the announcement (e.g., "2026") if stated (string or null).

    Graduate education (Cornell):
    - grad_school: The graduate institution (expect "Cornell University") if stated (string or null).
    - grad_degree: The master's-level degree (e.g., "Master of Engineering", "MBA", "M.S.") if stated (string or null).
    - grad_year: Graduation year for the Cornell master's degree (expect "2004") if stated (string or null).

    Undergraduate education (Texas A&M):
    - undergrad_school: The undergraduate institution (expect "Texas A&M University") if stated (string or null).
    - undergrad_degree: The bachelor's degree title/abbreviation if stated (string or null).

    Career history:
    - joined_year: The year the executive joined the current company (expect "2004") if stated (string or null).

    Sources (URLs explicitly mentioned in the answer; include only valid URLs):
    - announcement_sources: URLs that specifically discuss the CEO succession announcement.
    - company_profile_sources: URLs that support the company's Fortune 500 status and/or that it is a homebuilder (residential home construction).
    - education_sources: URLs that support the executive's education credentials (Cornell master's, Texas A&M bachelor's, graduation year).
    - career_sources: URLs that support the executive's tenure/join year at the company.

    Notes:
    - If a field is not present in the answer, set it to null (or empty array for sources).
    - For URLs, include full links; if a URL is missing protocol, prepend "http://".
    - Do not invent or infer data or URLs not explicitly present in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists with order preserved and de-duplication."""
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def _choose_sources(extracted: CEOSuccessionExtraction, primary: List[str]) -> List[str]:
    """Choose primary if available; otherwise fall back to any other listed sources."""
    if primary and len(primary) > 0:
        return primary
    # fallback: use any sources available across categories
    return _merge_sources(
        extracted.announcement_sources,
        extracted.company_profile_sources,
        extracted.education_sources,
        extracted.career_sources,
    )


def _safe_name(name: Optional[str], fallback: str) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: CEOSuccessionExtraction):
    # Top-level task node (critical, sequential as per rubric)
    task_node = evaluator.add_sequential(
        id="Complete_CEO_Research_Task",
        desc="Identify the CEO successor and company, and verify all specified credentials",
        parent=evaluator.root,
        critical=True
    )

    # 1) Identify CEO and Company (critical, parallel)
    identify_node = evaluator.add_parallel(
        id="Identify_CEO_And_Company",
        desc="Identify the CEO successor and company based on announcement and company characteristics",
        parent=task_node,
        critical=True
    )

    # 1.a) CEO succession was announced in January 2026 (critical leaf)
    ann_sources = _choose_sources(extracted, extracted.announcement_sources)
    company_for_claim = _safe_name(extracted.company_name, "the company")

    node_announced_jan2026 = evaluator.add_leaf(
        id="CEO_Succession_Announced_January_2026",
        desc="The CEO succession was announced in January 2026",
        parent=identify_node,
        critical=True
    )
    claim_announced_jan2026 = (
        f"A CEO succession at {company_for_claim} was announced in January 2026."
    )
    await evaluator.verify(
        claim=claim_announced_jan2026,
        node=node_announced_jan2026,
        sources=ann_sources,
        additional_instruction=(
            "Verify the press release or reputable news source shows that the CEO succession announcement "
            "occurred in January 2026 (any day in January 2026 is acceptable). Consider the page's date "
            "or explicit wording about the timing."
        ),
    )

    # 1.b) Company is Fortune 500 and a homebuilder (critical leaf)
    profile_sources = _choose_sources(extracted, extracted.company_profile_sources)
    node_f500_homebuilder = evaluator.add_leaf(
        id="Company_Is_Fortune_500_Homebuilder",
        desc="The company is listed in Fortune 500 and operates in the homebuilding industry",
        parent=identify_node,
        critical=True
    )
    claim_f500_homebuilder = (
        f"{company_for_claim} is a Fortune 500 company and operates in the homebuilding (residential home construction) industry."
    )
    await evaluator.verify(
        claim=claim_f500_homebuilder,
        node=node_f500_homebuilder,
        sources=profile_sources,
        additional_instruction=(
            "Confirm that the company appears on a Fortune 500 list (recent year is fine) and that it is described as a homebuilder. "
            "Accept synonyms like 'homebuilding', 'residential construction', 'home construction', or 'homebuilder'."
        ),
    )

    # 1.c) Provide both CEO name and company name (critical, parallel)
    provide_idents_node = evaluator.add_parallel(
        id="Provide_Required_Identifications",
        desc="Provide both the CEO successor's name and the company name",
        parent=identify_node,
        critical=True
    )

    # 1.c.i) CEO name provided (critical existence)
    ceo_name_ok = bool(extracted.ceo_name and extracted.ceo_name.strip())
    evaluator.add_custom_node(
        result=ceo_name_ok,
        id="CEO_Name_Provided",
        desc="The answer provides the full name of the CEO successor",
        parent=provide_idents_node,
        critical=True
    )

    # 1.c.ii) Company name provided (critical existence)
    company_name_ok = bool(extracted.company_name and extracted.company_name.strip())
    evaluator.add_custom_node(
        result=company_name_ok,
        id="Company_Name_Provided",
        desc="The answer provides the name of the company",
        parent=provide_idents_node,
        critical=True
    )

    # 2) Verify all credentials (critical, parallel)
    verify_creds_node = evaluator.add_parallel(
        id="Verify_All_Credentials",
        desc="Verify that the identified executive meets all educational and career requirements",
        parent=task_node,
        critical=True
    )

    # 2.a) Graduate education requirements (critical, parallel)
    grad_node = evaluator.add_parallel(
        id="Graduate_Education_Requirements",
        desc="Verify graduate degree credentials",
        parent=verify_creds_node,
        critical=True
    )
    edu_sources = _choose_sources(extracted, extracted.education_sources)
    ceo_for_claim = _safe_name(extracted.ceo_name, "the executive")

    # 2.a.i) Has Cornell Master's degree (critical leaf)
    node_cornell_masters = evaluator.add_leaf(
        id="Has_Cornell_Masters_Degree",
        desc="The executive holds a Master's degree from Cornell University",
        parent=grad_node,
        critical=True
    )
    claim_cornell_masters = (
        f"{ceo_for_claim} holds a master's-level degree (e.g., M.S., M.Eng., MBA, MPS) from Cornell University."
    )
    await evaluator.verify(
        claim=claim_cornell_masters,
        node=node_cornell_masters,
        sources=edu_sources,
        additional_instruction=(
            "Accept any master's-level degree (M.S., M.Eng., MBA, MPS, etc.) from Cornell University as satisfying "
            "the 'Master's degree from Cornell University' requirement."
        ),
    )

    # 2.a.ii) Graduated from Cornell in 2004 (critical leaf)
    node_cornell_2004 = evaluator.add_leaf(
        id="Graduated_From_Cornell_In_2004",
        desc="The executive graduated from Cornell University in 2004",
        parent=grad_node,
        critical=True
    )
    claim_cornell_2004 = (
        f"{ceo_for_claim} graduated from Cornell University in 2004."
    )
    await evaluator.verify(
        claim=claim_cornell_2004,
        node=node_cornell_2004,
        sources=edu_sources,
        additional_instruction=(
            "Confirm the graduation year is 2004, including equivalents like 'Class of 2004' or a degree notation such as 'M.Eng. (2004)'."
        ),
    )

    # 2.b) Undergraduate education requirement (critical, parallel)
    undergrad_node = evaluator.add_parallel(
        id="Undergraduate_Education_Requirement",
        desc="Verify undergraduate degree credentials",
        parent=verify_creds_node,
        critical=True
    )

    node_texasam_bachelors = evaluator.add_leaf(
        id="Has_Texas_AM_Bachelors_Degree",
        desc="The executive holds a Bachelor's degree from Texas A&M University",
        parent=undergrad_node,
        critical=True
    )
    claim_texasam_bachelors = (
        f"{ceo_for_claim} holds a bachelor's degree from Texas A&M University."
    )
    await evaluator.verify(
        claim=claim_texasam_bachelors,
        node=node_texasam_bachelors,
        sources=edu_sources,
        additional_instruction=(
            "Accept variants such as 'Texas A&M', 'Texas A&M University–College Station'. "
            "Any bachelor's-level degree counts (e.g., B.S., BBA, BA)."
        ),
    )

    # 2.c) Career timeline requirements (critical, parallel)
    career_node = evaluator.add_parallel(
        id="Career_Timeline_Requirements",
        desc="Verify career history at current company",
        parent=verify_creds_node,
        critical=True
    )
    career_sources = _choose_sources(extracted, extracted.career_sources)

    # 2.c.i) Joined current company in 2004 (critical leaf)
    node_joined_2004 = evaluator.add_leaf(
        id="Joined_Current_Company_In_2004",
        desc="The executive joined their current company in 2004",
        parent=career_node,
        critical=True
    )
    claim_joined_2004 = (
        f"{ceo_for_claim} joined {company_for_claim} in 2004."
    )
    await evaluator.verify(
        claim=claim_joined_2004,
        node=node_joined_2004,
        sources=career_sources,
        additional_instruction=(
            "Look for explicit statements such as 'joined in 2004', 'with the company since 2004', or "
            "a career timeline that clearly indicates a 2004 start at the company."
        ),
    )

    # 2.c.ii) Has twenty-plus years tenure as of announcement (critical leaf)
    node_tenure_20_plus = evaluator.add_leaf(
        id="Has_Twenty_Plus_Years_Tenure",
        desc="The executive has been with the company for at least 20 years as of the announcement date",
        parent=career_node,
        critical=True
    )
    claim_tenure_20_plus = (
        f"As of January 2026, {ceo_for_claim} has been with {company_for_claim} for at least 20 years."
    )
    await evaluator.verify(
        claim=claim_tenure_20_plus,
        node=node_tenure_20_plus,
        sources=career_sources if career_sources else ann_sources,
        additional_instruction=(
            "This can be satisfied either by an explicit '20+ years' statement or by inferring from a 2004 join year "
            "in relation to the January 2026 timeframe (i.e., at least 20 full years)."
        ),
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
    Build the verification tree and run all checks for the CEO succession task.
    """
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregator for this evaluation wrapper
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
        prompt=prompt_extract_ceo_succession(),
        template_class=CEOSuccessionExtraction,
        extraction_name="ceo_succession_extraction"
    )

    # Build and execute verification tree
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()