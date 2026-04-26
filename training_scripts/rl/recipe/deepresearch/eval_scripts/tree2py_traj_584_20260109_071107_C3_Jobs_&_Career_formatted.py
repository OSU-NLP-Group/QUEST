import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "young_f500_healthcare_female_ceo"
TASK_DESCRIPTION = (
    "A female executive was appointed as CEO of a Fortune 500 healthcare company between 2020 and 2024. "
    "At the time of her appointment, she was under 45 years old, making her one of the youngest Fortune 500 CEOs. "
    "She holds an MBA from a top-tier business school and had accumulated between 10 to 15 years of professional "
    "experience before her CEO appointment—significantly less than the typical 15-20 years expected for healthcare CEOs. "
    "Prior to joining her current company, she held a C-level or VP-level position (such as Chief Product Officer or equivalent) "
    "at a major healthcare technology or analytics company. Who is this executive, and what year did she earn her MBA degree?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ExecutiveInfo(BaseModel):
    # Core identity
    name: Optional[str] = None

    # Current company & role context
    current_company: Optional[str] = None
    role_title: Optional[str] = None  # e.g., "CEO"
    industry_kind: Optional[str] = None  # e.g., "healthcare", "health insurance"
    appointment_date: Optional[str] = None  # e.g., "March 2022"
    appointment_year: Optional[str] = None  # e.g., "2022"

    # Age context
    age_at_appointment: Optional[str] = None  # e.g., "40", "age 40", "under 45"
    birth_date: Optional[str] = None
    birth_year: Optional[str] = None

    # Education
    mba_school: Optional[str] = None
    mba_year: Optional[str] = None

    # Prior experience/role
    prior_company: Optional[str] = None
    prior_role_title: Optional[str] = None  # e.g., "Chief Product Officer", "SVP"
    prior_company_category: Optional[str] = None  # e.g., "healthcare technology", "analytics"

    # Total experience before CEO appointment (as stated)
    experience_years_prior: Optional[str] = None  # keep as string to handle ranges/approx

    # URL sources cited in the answer
    general_sources: List[str] = Field(default_factory=list)
    name_sources: List[str] = Field(default_factory=list)
    gender_sources: List[str] = Field(default_factory=list)
    ceo_role_sources: List[str] = Field(default_factory=list)
    appointment_sources: List[str] = Field(default_factory=list)
    f500_sources: List[str] = Field(default_factory=list)
    age_sources: List[str] = Field(default_factory=list)
    birth_sources: List[str] = Field(default_factory=list)
    mba_sources: List[str] = Field(default_factory=list)
    mba_year_sources: List[str] = Field(default_factory=list)
    prior_role_sources: List[str] = Field(default_factory=list)
    experience_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_executive_info() -> str:
    return """
    Extract the identity and supporting details for the executive described in the answer. Return a single JSON object with the following fields:

    Required identity & role fields:
    - name: The executive’s full name stated in the answer.
    - current_company: The name of the company of which she is CEO.
    - role_title: The role title used in the answer (should be CEO).
    - industry_kind: The industry description for the company (e.g., "healthcare", "health insurance").
    - appointment_date: The stated date/month/year of the CEO appointment if present.
    - appointment_year: The stated year of the CEO appointment (just the year).
    
    Age & birth context:
    - age_at_appointment: The age at appointment as stated in the answer, or a claimed "under 45" statement. If absent, return null.
    - birth_date: If the answer provides birth date, extract it; else null.
    - birth_year: If the answer provides birth year, extract it; else null.

    Education:
    - mba_school: The MBA school name (e.g., "Chicago Booth", "Harvard Business School").
    - mba_year: The MBA graduation year (just the year).

    Prior role & company:
    - prior_company: The company where she held a previous C-level or VP-level role before joining her current company.
    - prior_role_title: The title she held there (e.g., "Chief Product Officer", "VP").
    - prior_company_category: A brief category or description of that company (e.g., "healthcare technology", "analytics").

    Experience duration:
    - experience_years_prior: The stated total years of professional experience prior to being appointed CEO, as mentioned in the answer (keep as string; can be an approximate or range like "12" or "10-15").

    Source URLs explicitly present in the answer:
    - general_sources: All general URLs cited that are relevant to the executive.
    - name_sources: URLs specifically supporting the identity/name.
    - gender_sources: URLs supporting the executive being female (e.g., pronouns usage or explicit statements).
    - ceo_role_sources: URLs supporting that she is CEO of the stated company.
    - appointment_sources: URLs supporting the appointment date/year.
    - f500_sources: URLs supporting that the company is a Fortune 500 and in healthcare/health insurance.
    - age_sources: URLs supporting age-at-appointment claim.
    - birth_sources: URLs providing birth date/year info.
    - mba_sources: URLs supporting the MBA school fact.
    - mba_year_sources: URLs supporting the MBA graduation year.
    - prior_role_sources: URLs supporting the prior role and prior company classification (health tech/analytics).
    - experience_sources: URLs supporting total years of experience prior to CEO appointment.

    Rules:
    - Extract ONLY what is explicitly stated in the answer; do not invent information.
    - If any field is missing, set it to null (for strings) or an empty list (for sources).
    - For URLs, extract actual URLs present in the answer (including markdown links). Ignore invalid/malformed URLs.
    - If multiple executives are mentioned, choose the single executive the answer ultimately identifies as meeting the constraints (or the first one if ambiguous).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def combine_sources(*arrays: Optional[List[str]]) -> List[str]:
    """Combine and deduplicate source URL arrays, ignoring empties."""
    seen = set()
    out: List[str] = []
    for arr in arrays:
        if not arr:
            continue
        for url in arr:
            if url and url not in seen:
                seen.add(url)
                out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification sub-tree builders                                              #
# --------------------------------------------------------------------------- #
async def build_identify_executive_branch(
    evaluator: Evaluator,
    parent_node,
    info: ExecutiveInfo,
) -> None:
    """
    Build the 'Identify_Executive' parallel critical branch with all verification leaves.
    """

    identify_node = evaluator.add_parallel(
        id="Identify_Executive",
        desc="Provide an executive identity that satisfies all constraints.",
        parent=parent_node,
        critical=True,
    )

    # 1) Executive_Name existence (custom critical gate)
    name_exists = bool(info.name and info.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="Executive_Name",
        desc="Answer states the executive’s name.",
        parent=identify_node,
        critical=True,
    )

    # 2) Gender_Requirement (female)
    gender_node = evaluator.add_leaf(
        id="Gender_Requirement",
        desc="The executive is female.",
        parent=identify_node,
        critical=True,
    )
    gender_sources = combine_sources(info.gender_sources, info.general_sources, info.name_sources)
    gender_claim = f"The identified executive '{info.name or ''}' is female."
    await evaluator.verify(
        claim=gender_claim,
        node=gender_node,
        sources=gender_sources,
        additional_instruction=(
            "Treat the claim as supported if the cited sources use she/her pronouns or explicitly "
            "state that the individual is female."
        ),
    )

    # 3) Fortune_500_Healthcare_CEO
    f500_node = evaluator.add_leaf(
        id="Fortune_500_Healthcare_CEO",
        desc="The executive is CEO of a Fortune 500 healthcare or health insurance company.",
        parent=identify_node,
        critical=True,
    )
    f500_sources = combine_sources(info.ceo_role_sources, info.f500_sources, info.general_sources)
    f500_claim = (
        f"The identified executive '{info.name or ''}' is CEO of '{info.current_company or ''}', "
        f"which is a Fortune 500 company in healthcare or health insurance."
    )
    await evaluator.verify(
        claim=f500_claim,
        node=f500_node,
        sources=f500_sources,
        additional_instruction=(
            "To pass, the sources must support BOTH that she is CEO of the named company AND that the company "
            "is in the Fortune 500 and belongs to healthcare/health insurance."
        ),
    )

    # 4) Appointment_Timing (2020–2024 inclusive)
    appoint_node = evaluator.add_leaf(
        id="Appointment_Timing",
        desc="The executive was appointed to the CEO position between 2020 and 2024 (inclusive).",
        parent=identify_node,
        critical=True,
    )
    appointment_sources = combine_sources(info.appointment_sources, info.ceo_role_sources, info.general_sources)
    appoint_year_text = info.appointment_year or ""
    appoint_claim = (
        f"The executive was appointed CEO in {appoint_year_text}, which falls between 2020 and 2024 inclusive."
    )
    await evaluator.verify(
        claim=appoint_claim,
        node=appoint_node,
        sources=appointment_sources,
        additional_instruction=(
            "Verify the appointment year from the sources and confirm it is within 2020–2024 inclusive."
        ),
    )

    # 5) Age_at_Appointment (under 45)
    age_node = evaluator.add_leaf(
        id="Age_at_Appointment",
        desc="The executive was under 45 years old at the time of CEO appointment.",
        parent=identify_node,
        critical=True,
    )
    age_sources = combine_sources(info.age_sources, info.birth_sources, info.appointment_sources, info.general_sources)
    age_claim = (
        "At the time she was appointed CEO, she was under 45 years old."
    )
    await evaluator.verify(
        claim=age_claim,
        node=age_node,
        sources=age_sources,
        additional_instruction=(
            "If the sources provide date of birth (or birth year) plus the appointment year, use that to confirm "
            "the age is strictly less than 45 at appointment; age 45 does NOT satisfy 'under 45'."
        ),
    )

    # 6) MBA_Top_Tier_School
    mba_school_node = evaluator.add_leaf(
        id="MBA_Top_Tier_School",
        desc="The executive holds an MBA from a top-tier business school (e.g., Harvard, Stanford, Wharton, Chicago Booth, or an equivalent top-tier school).",
        parent=identify_node,
        critical=True,
    )
    mba_school_sources = combine_sources(info.mba_sources, info.general_sources)
    mba_school_claim = (
        f"She holds an MBA from {info.mba_school or ''}, which is a top-tier business school."
    )
    await evaluator.verify(
        claim=mba_school_claim,
        node=mba_school_node,
        sources=mba_school_sources,
        additional_instruction=(
            "Consider top-tier programs commonly recognized among leading rankings: Harvard Business School, Stanford GSB, "
            "UPenn Wharton, MIT Sloan, Chicago Booth, Northwestern Kellogg, Columbia Business School, UC Berkeley Haas, "
            "Dartmouth Tuck, Yale SOM, among equivalent elite programs."
        ),
    )

    # 7) Prior_HealthTech_Or_Analytics_Leadership
    prior_role_node = evaluator.add_leaf(
        id="Prior_HealthTech_Or_Analytics_Leadership",
        desc="Before joining the current company, the executive held a C-level or VP-level position (or equivalent) at a major healthcare technology or analytics company.",
        parent=identify_node,
        critical=True,
    )
    prior_sources = combine_sources(info.prior_role_sources, info.general_sources)
    prior_role_claim = (
        f"Prior to joining {info.current_company or ''}, she held a {info.prior_role_title or ''} role at {info.prior_company or ''}, "
        f"which is a major healthcare technology or analytics company."
    )
    await evaluator.verify(
        claim=prior_role_claim,
        node=prior_role_node,
        sources=prior_sources,
        additional_instruction=(
            "Major health tech/analytics companies include (examples): Optum/UnitedHealth Group, IQVIA, Change Healthcare, Cerner/Oracle Health, "
            "Epic, Verily, Truveta, etc. The sources should clearly support the prior role and the company's health tech/analytics nature."
        ),
    )

    # 8) Total_Experience_Duration (10–15 years prior to CEO appointment)
    exp_node = evaluator.add_leaf(
        id="Total_Experience_Duration",
        desc="The executive had between 10 and 15 years of total professional experience before being appointed CEO.",
        parent=identify_node,
        critical=True,
    )
    exp_sources = combine_sources(info.experience_sources, info.general_sources)
    exp_claim = (
        "Before being appointed CEO, she had between 10 and 15 years of total professional experience."
    )
    await evaluator.verify(
        claim=exp_claim,
        node=exp_node,
        sources=exp_sources,
        additional_instruction=(
            "Treat the claim as supported if the sources state a duration within 10–15 years inclusive; "
            "allow approximate phrasing (e.g., 'around 12 years')."
        ),
    )


async def build_mba_year_leaf(
    evaluator: Evaluator,
    parent_node,
    info: ExecutiveInfo,
) -> None:
    """
    Build the 'MBA_Graduation_Year' critical leaf node verifying the MBA graduation year.
    """
    mba_year_node = evaluator.add_leaf(
        id="MBA_Graduation_Year",
        desc="State the calendar year the identified executive earned her MBA degree.",
        parent=parent_node,
        critical=True,
    )
    mba_year_sources = combine_sources(info.mba_year_sources, info.mba_sources, info.general_sources)
    mba_year_text = info.mba_year or ""
    mba_year_claim = f"She earned her MBA in {mba_year_text}."
    await evaluator.verify(
        claim=mba_year_claim,
        node=mba_year_node,
        sources=mba_year_sources,
        additional_instruction=(
            "Verify the MBA graduation year. If sources provide month/year, confirm the year."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the young Fortune 500 healthcare female CEO identification task.
    """
    # Initialize evaluator and root (critical sequential as per rubric)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Create top-level critical container node "Complete_Answer"
    complete_node = evaluator.add_sequential(
        id="Complete_Answer",
        desc="Answer identifies the executive who meets all stated constraints and states the year she earned her MBA.",
        parent=root,
        critical=True,
    )

    # Extract structured info
    exec_info = await evaluator.extract(
        prompt=prompt_extract_executive_info(),
        template_class=ExecutiveInfo,
        extraction_name="executive_info",
    )

    # Build Identify_Executive branch (parallel critical)
    await build_identify_executive_branch(evaluator, complete_node, exec_info)

    # Build MBA_Graduation_Year leaf (critical)
    await build_mba_year_leaf(evaluator, complete_node, exec_info)

    # Return evaluation summary
    return evaluator.get_summary()