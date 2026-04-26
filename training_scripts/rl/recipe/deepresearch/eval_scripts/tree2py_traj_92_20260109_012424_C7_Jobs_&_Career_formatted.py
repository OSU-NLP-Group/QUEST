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
TASK_ID = "tech_remote_benefits_2026"
TASK_DESCRIPTION = (
    "Identify a technology company that offers fully remote positions in the United States and provides ALL of the following employee benefits "
    "(as of January 2026):\n"
    "1. 100% remote positions available (no required office attendance)\n"
    "2. Medical/health insurance coverage\n"
    "3. Dental insurance coverage\n"
    "4. Vision insurance coverage\n"
    "5. 401(k) retirement plan with employer matching\n"
    "6. Paid time off (vacation/PTO)\n"
    "7. Paid parental leave\n"
    "8. Professional development budget or learning stipend\n"
    "9. Home office equipment stipend or setup budget\n"
    "10. Flexible working hours\n"
    "11. Company-provided laptop and work equipment\n"
    "12. Performance bonuses, profit sharing, or equity compensation\n"
    "13. Positions available at mid-level or senior level\n"
    "14. Operates in the technology sector\n"
    "15. All benefits publicly documented on the company's careers or benefits pages\n\n"
    "Provide the company name and reference URLs documenting these benefits."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CompanyExtraction(BaseModel):
    company_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_info() -> str:
    return (
        "Extract the following from the answer:\n"
        "1) company_name: The specific company identified in the response.\n"
        "2) reference_urls: A list of all URLs provided in the answer that serve as references. These should include official company pages "
        "(e.g., careers, benefits, remote-work policy, job listings) and any other URLs cited in support of the claims.\n"
        "Rules:\n"
        "- Only include URLs that are explicitly present in the answer text. Do not infer or invent URLs.\n"
        "- Include full URLs. If a URL lacks protocol, prepend http://.\n"
        "- If the company name is not specified, return null.\n"
        "- If no URLs are provided, return an empty list."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def safe_company_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the company"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_company_benefits(
    evaluator: Evaluator,
    root_node,
    extracted: CompanyExtraction,
) -> None:
    """
    Build the verification tree and execute verification for all rubric leaves.
    """
    company = safe_company_name(extracted.company_name)
    sources = extracted.reference_urls

    # Main critical parallel node encapsulating all requirements
    main_node = evaluator.add_parallel(
        id="Technology_Company_Remote_Benefits",
        desc="Identify a technology company with 100% remote (US) roles and a benefits package meeting all specified criteria, "
             "with public official documentation and provided reference URLs (as of January 2026).",
        parent=root_node,
        critical=True,
    )

    # Existence checks (custom nodes) - critical
    evaluator.add_custom_node(
        result=bool(extracted.company_name and extracted.company_name.strip()),
        id="Company_Identified",
        desc="Response identifies a specific company by name.",
        parent=main_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(sources),
        id="Reference_URLs_Provided",
        desc="Response provides reference URL(s) supporting the claims.",
        parent=main_node,
        critical=True,
    )

    # Prepare leaf nodes and batch verification tuples
    claims_and_nodes: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Leaf: US_Remote_Positions
    leaf_us_remote = evaluator.add_leaf(
        id="US_Remote_Positions",
        desc="Company offers 100% remote positions available in the United States with no required office attendance.",
        parent=main_node,
        critical=True,
    )
    claim_us_remote = (
        f"{company} offers fully remote positions in the United States with no required office attendance "
        f"(e.g., remote-first, remote-only, or similar policy explicitly stated)."
    )
    ins_us_remote = (
        "Verify on official pages (careers, remote policy, job listings) that roles are fully remote in the U.S. "
        "Accept explicit statements such as 'remote-first', 'remote-only', '100% remote', no office requirement."
    )
    claims_and_nodes.append((claim_us_remote, sources, leaf_us_remote, ins_us_remote))

    # Leaf: As_Of_Jan_2026 (simple verify against the answer)
    leaf_as_of = evaluator.add_leaf(
        id="As_Of_Jan_2026",
        desc="Response asserts/verifies the listed benefits and remote-work policy are current as of January 2026.",
        parent=main_node,
        critical=True,
    )
    claim_as_of = (
        "The answer explicitly asserts or indicates that the benefits and remote-work policy are current as of January 2026 "
        "(e.g., 'as of January 2026', 'current in Jan 2026', or equivalent phrasing)."
    )
    ins_as_of = (
        "Judge only based on the answer content. Accept reasonable phrasing indicating currency in January 2026."
    )
    claims_and_nodes.append((claim_as_of, None, leaf_as_of, ins_as_of))

    # Leaf: Medical_Insurance
    leaf_med = evaluator.add_leaf(
        id="Medical_Insurance",
        desc="Company provides medical/health insurance coverage to employees.",
        parent=main_node,
        critical=True,
    )
    claim_med = f"{company} provides medical or health insurance coverage to employees."
    ins_med = "Look for 'medical insurance', 'health insurance', 'healthcare coverage' on official benefits pages."
    claims_and_nodes.append((claim_med, sources, leaf_med, ins_med))

    # Leaf: Dental_Coverage
    leaf_dental = evaluator.add_leaf(
        id="Dental_Coverage",
        desc="Company provides dental insurance coverage.",
        parent=main_node,
        critical=True,
    )
    claim_dental = f"{company} provides dental insurance coverage."
    ins_dental = "Look for 'dental insurance' or 'dental plan' on official benefits pages."
    claims_and_nodes.append((claim_dental, sources, leaf_dental, ins_dental))

    # Leaf: Vision_Coverage
    leaf_vision = evaluator.add_leaf(
        id="Vision_Coverage",
        desc="Company provides vision insurance coverage.",
        parent=main_node,
        critical=True,
    )
    claim_vision = f"{company} provides vision insurance coverage."
    ins_vision = "Look for 'vision insurance' or 'vision plan' on official benefits pages."
    claims_and_nodes.append((claim_vision, sources, leaf_vision, ins_vision))

    # Leaf: Retirement_401k
    leaf_401k = evaluator.add_leaf(
        id="Retirement_401k",
        desc="Company offers a 401(k) retirement plan with employer matching contribution.",
        parent=main_node,
        critical=True,
    )
    claim_401k = f"{company} offers a 401(k) plan with employer matching contributions."
    ins_401k = "Look for '401(k)', 'matching', 'employer match' on official benefits pages."
    claims_and_nodes.append((claim_401k, sources, leaf_401k, ins_401k))

    # Leaf: Paid_Time_Off
    leaf_pto = evaluator.add_leaf(
        id="Paid_Time_Off",
        desc="Company provides paid vacation time or PTO to employees.",
        parent=main_node,
        critical=True,
    )
    claim_pto = f"{company} provides paid time off (PTO) or paid vacation."
    ins_pto = "Look for 'PTO', 'paid vacation', 'paid time off' on official benefits pages."
    claims_and_nodes.append((claim_pto, sources, leaf_pto, ins_pto))

    # Leaf: Parental_Leave_Policy
    leaf_parental = evaluator.add_leaf(
        id="Parental_Leave_Policy",
        desc="Company offers paid parental leave.",
        parent=main_node,
        critical=True,
    )
    claim_parental = f"{company} offers paid parental leave."
    ins_parental = "Look for 'paid parental leave', 'maternity leave', 'paternity leave' with pay."
    claims_and_nodes.append((claim_parental, sources, leaf_parental, ins_parental))

    # Leaf: Learning_Development
    leaf_learning = evaluator.add_leaf(
        id="Learning_Development",
        desc="Company provides a professional development budget, learning stipend, or educational benefits.",
        parent=main_node,
        critical=True,
    )
    claim_learning = (
        f"{company} provides a professional development budget, learning stipend, or educational reimbursement/benefits."
    )
    ins_learning = "Look for 'learning stipend', 'education reimbursement', 'professional development budget'."
    claims_and_nodes.append((claim_learning, sources, leaf_learning, ins_learning))

    # Leaf: Home_Office_Support
    leaf_home = evaluator.add_leaf(
        id="Home_Office_Support",
        desc="Company provides a home office equipment stipend or budget for remote work setup.",
        parent=main_node,
        critical=True,
    )
    claim_home = f"{company} provides a home office equipment stipend or setup budget for remote employees."
    ins_home = "Look for 'home office stipend', 'equipment stipend', 'office setup budget', 'work-from-home stipend'."
    claims_and_nodes.append((claim_home, sources, leaf_home, ins_home))

    # Leaf: Flexible_Hours
    leaf_flex = evaluator.add_leaf(
        id="Flexible_Hours",
        desc="Company allows flexible working hours or schedule flexibility.",
        parent=main_node,
        critical=True,
    )
    claim_flex = f"{company} allows flexible working hours or schedule flexibility."
    ins_flex = "Look for 'flexible schedule', 'flexible hours', 'work when you want', core hours, or similar policy."
    claims_and_nodes.append((claim_flex, sources, leaf_flex, ins_flex))

    # Leaf: Equipment_Provided
    leaf_equipment = evaluator.add_leaf(
        id="Equipment_Provided",
        desc="Company provides a laptop and necessary work equipment.",
        parent=main_node,
        critical=True,
    )
    claim_equipment = f"{company} provides a company laptop and necessary work equipment to employees."
    ins_equipment = "Look for 'company-provided laptop', 'work equipment provided', 'hardware provided', 'MacBook', 'PC'."
    claims_and_nodes.append((claim_equipment, sources, leaf_equipment, ins_equipment))

    # Leaf: Performance_Incentives
    leaf_perf = evaluator.add_leaf(
        id="Performance_Incentives",
        desc="Company offers performance bonuses, profit sharing, or equity compensation.",
        parent=main_node,
        critical=True,
    )
    claim_perf = f"{company} offers performance bonuses, profit sharing, or equity (stock options/RSUs) as compensation."
    ins_perf = "Look for 'bonus', 'profit sharing', 'equity', 'stock options', 'RSUs' on official pages."
    claims_and_nodes.append((claim_perf, sources, leaf_perf, ins_perf))

    # Leaf: Experience_Level
    leaf_experience = evaluator.add_leaf(
        id="Experience_Level",
        desc="Company has positions available at mid-level or senior level (not exclusively entry-level).",
        parent=main_node,
        critical=True,
    )
    claim_experience = (
        f"{company} has job openings at mid-level or senior level (e.g., 'Senior', 'Staff', 'Lead', 'Principal'), not exclusively entry-level."
    )
    ins_experience = (
        "Check job listings for titles including 'Senior', 'Sr.', 'Staff', 'Lead', 'Principal', 'Manager'. "
        "Any official job page showing roles above entry-level suffices."
    )
    claims_and_nodes.append((claim_experience, sources, leaf_experience, ins_experience))

    # Leaf: Technology_Sector
    leaf_tech = evaluator.add_leaf(
        id="Technology_Sector",
        desc="Company operates in the technology sector.",
        parent=main_node,
        critical=True,
    )
    claim_tech = f"{company} operates in the technology sector (e.g., software, hardware, SaaS, AI, IT services)."
    ins_tech = "Verify via official pages that products/services are technology-related; accept obvious tech company indicators."
    claims_and_nodes.append((claim_tech, sources, leaf_tech, ins_tech))

    # Leaf: Public_Documentation_Official
    leaf_public = evaluator.add_leaf(
        id="Public_Documentation_Official",
        desc="All listed benefits are publicly documented on the company's official careers or benefits pages.",
        parent=main_node,
        critical=True,
    )
    claim_public = (
        "These reference URLs are official company careers or benefits pages that publicly document employee benefits and/or remote-work policy."
    )
    ins_public = (
        "Confirm the URLs belong to the company's official domain and are careers/benefits/help/policy pages. "
        "Third-party aggregators (e.g., Glassdoor, Indeed) should not count as official documentation."
    )
    claims_and_nodes.append((claim_public, sources, leaf_public, ins_public))

    # Execute batch verification for all leaves
    await evaluator.batch_verify(claims_and_nodes)


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
    Evaluate the agent's answer for the 'tech_remote_benefits_2026' task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract company name and reference URLs from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_company_info(),
        template_class=CompanyExtraction,
        extraction_name="company_and_urls",
    )

    # Build verification tree and run checks
    await verify_company_benefits(evaluator, root, extracted_info)

    # Return summary
    return evaluator.get_summary()