import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tech_rotation_program_requirements_eval"
TASK_DESCRIPTION = (
    "Identify a Fortune 500 or established enterprise technology company that offers a full-time technology "
    "rotational development program meeting all of the following requirements: (1) The program must be a "
    "structured rotational development program with multiple rotations across different teams or business areas, "
    "(2) The program must require a bachelor's degree in a STEM field or related technical discipline, "
    "(3) The program must be designed for recent graduates or early career professionals with 0-3 years of "
    "post-graduation work experience, (4) The company must offer a signing bonus for new program participants, "
    "(5) The company must provide 401k retirement benefits with employer matching, (6) The program must include "
    "relocation assistance for eligible candidates, (7) The company must offer comprehensive health insurance "
    "benefits, (8) The program must include structured professional development, mentorship, or training components, "
    "(9) Candidates must be legally authorized to work in the United States, (10) The company must have technology "
    "operations in multiple US locations or major tech hubs. Provide the company name and the specific program name, "
    "along with supporting evidence for how the program meets each requirement."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    company_name: Optional[str] = None
    program_name: Optional[str] = None

    # General/primary program URLs (job posting, program overview page, careers page specific to this program)
    program_urls: List[str] = Field(default_factory=list)

    # Evidence URLs per requirement (only extract URLs explicitly present in the answer)
    program_type_urls: List[str] = Field(default_factory=list)       # Full-time technology rotational program evidence
    structure_urls: List[str] = Field(default_factory=list)          # Multiple structured rotations evidence
    degree_urls: List[str] = Field(default_factory=list)             # Bachelor's degree in STEM/related technical
    experience_urls: List[str] = Field(default_factory=list)         # Recent grads / 0-3 years experience
    company_type_urls: List[str] = Field(default_factory=list)       # Fortune 500 or established enterprise tech company
    signing_bonus_urls: List[str] = Field(default_factory=list)      # Signing bonus offered
    retirement_urls: List[str] = Field(default_factory=list)         # 401(k) with employer match
    relocation_urls: List[str] = Field(default_factory=list)         # Relocation assistance
    health_urls: List[str] = Field(default_factory=list)             # Comprehensive health insurance benefits
    development_urls: List[str] = Field(default_factory=list)        # Structured PD/mentorship/training
    authorization_urls: List[str] = Field(default_factory=list)      # US work authorization required
    geographic_urls: List[str] = Field(default_factory=list)         # Tech operations in multiple US locations/hubs


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_info() -> str:
    return """
    Extract the key information for the identified company's full-time technology rotational development program.
    You must only extract information explicitly mentioned in the answer.

    Required fields:
    - company_name: The company name clearly stated in the answer.
    - program_name: The specific program name (not just a generic category).
    - program_urls: All URLs in the answer that directly point to the program overview page, a job posting for the program, or a company careers page that is specifically about this program.

    For each requirement below, extract ALL URLs that serve as direct evidence from the answer text. Include duplicates only once. If none are present in the answer, return an empty list for that field:
    - program_type_urls: URLs that show it is a full-time technology rotational development program (not internship, not co-op, not a standard single-role entry-level job).
    - structure_urls: URLs that demonstrate the program includes multiple structured rotations across different teams or business areas.
    - degree_urls: URLs that show the program requires a bachelor's degree in a STEM field or related technical discipline.
    - experience_urls: URLs that show the program is for recent graduates or early career professionals (0–3 years experience).
    - company_type_urls: URLs that show the company is a Fortune 500 or an established enterprise technology company (e.g., Fortune listing, credible 3rd-party profile, official statements).
    - signing_bonus_urls: URLs that indicate new program participants are offered a signing bonus (job posting, compensation page, or official statement).
    - retirement_urls: URLs that show 401(k) retirement benefits with employer matching.
    - relocation_urls: URLs that show relocation assistance is offered for eligible candidates.
    - health_urls: URLs that show comprehensive health insurance benefits are offered.
    - development_urls: URLs that show structured professional development, mentorship, or training components are part of the program.
    - authorization_urls: URLs that show candidates must be legally authorized to work in the United States.
    - geographic_urls: URLs that show the company has technology operations in multiple US locations or major tech hubs.

    Rules:
    - Extract only URLs that appear in the answer (plain or markdown links). Do not invent URLs.
    - If a URL is missing a protocol, prepend "http://".
    - If a requirement has no URL evidence in the answer, return [] for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def flatten_unique(lists: List[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            uu = (u or "").strip()
            if not uu:
                continue
            if uu not in seen:
                seen.add(uu)
                out.append(uu)
    return out


def safe_name(value: Optional[str], fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def build_additional_instruction(base: str, has_sources: bool) -> str:
    if has_sources:
        return base
    # If no URLs are provided by the answer, force the judge to fail due to lack of evidence.
    return (
        base
        + "\nIMPORTANT: The answer did not provide any supporting URLs for this requirement. "
          "According to the evaluation rules, you must judge this claim as not supported (Incorrect) due to lack of evidence."
    )


async def add_and_verify_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    base_instruction: str,
) -> None:
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=True,  # All children under the critical requirements node must be critical
    )
    add_ins = build_additional_instruction(base_instruction, has_sources=len(urls) > 0)
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=urls if len(urls) > 0 else None,
        additional_instruction=add_ins,
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
    Evaluate whether the response identifies a qualifying company & full-time technology rotational development program
    meeting all specified requirements, with supporting evidence.
    """
    # 1) Initialize evaluator
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

    # Add the critical requirements node under root (root itself is non-critical by design)
    req_root = evaluator.add_parallel(
        id="Technology_Rotational_Program_Requirements",
        desc="Evaluate whether the response identifies a qualifying company and specific full-time technology rotational development program and provides supporting evidence that it meets every stated requirement.",
        parent=root,
        critical=True,
    )

    # 2) Extract structured data from the answer
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program_info(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction"
    )

    company = safe_name(extracted.company_name, "the company")
    program = safe_name(extracted.program_name, "the program")

    # 3) Build sources (use requirement-specific URLs, fallback to program_urls if needed)
    def src(*lists: List[str]) -> List[str]:
        return flatten_unique(list(lists) + [extracted.program_urls])

    # 4) Add leaf nodes and verifications according to rubric

    # Identifies_Company_Name (existence check)
    evaluator.add_custom_node(
        result=bool(extracted.company_name and extracted.company_name.strip()),
        id="Identifies_Company_Name",
        desc="Response clearly states the company name.",
        parent=req_root,
        critical=True
    )

    # Identifies_Specific_Program_Name (existence check)
    evaluator.add_custom_node(
        result=bool(extracted.program_name and extracted.program_name.strip()),
        id="Identifies_Specific_Program_Name",
        desc="Response clearly states the specific program name (not just a general careers page or generic program category).",
        parent=req_root,
        critical=True
    )

    # Program_Type_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Program_Type_With_Evidence",
        "Provides evidence that the position is a full-time technology rotational development program (not an internship or standard entry-level role).",
        claim=f"The {company} {program} is a full-time technology rotational development program (not an internship, co-op, or single-role entry-level job).",
        urls=src(extracted.program_type_urls),
        base_instruction=(
            "Verify on the provided page(s) that the role is explicitly full-time and a technology rotational development program. "
            "Reject if it is an internship, co-op, apprenticeship, or a single non-rotational entry-level job."
        ),
    )

    # Program_Structure_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Program_Structure_With_Evidence",
        "Provides evidence that the program includes multiple structured rotations across different teams or business areas.",
        claim=f"The {program} includes multiple structured rotations across different teams or business areas.",
        urls=src(extracted.structure_urls),
        base_instruction=(
            "Look for explicit mention of multiple rotations (e.g., 2-4 rotations) and that they occur across different teams or functions."
        ),
    )

    # Degree_Requirement_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Degree_Requirement_With_Evidence",
        "Provides evidence that the program requires a bachelor's degree in a STEM field or related technical discipline.",
        claim=f"The {program} requires a bachelor's degree in a STEM field or related technical discipline (e.g., Computer Science, Engineering, Data Science).",
        urls=src(extracted.degree_urls),
        base_instruction=(
            "Accept language indicating a bachelor's degree in STEM or related technical disciplines; reject if degree is unspecified, non-technical only, or not required."
        ),
    )

    # Experience_Level_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Experience_Level_With_Evidence",
        "Provides evidence that the program is designed for recent graduates or early career professionals with 0–3 years of post-graduation work experience.",
        claim=f"The {program} targets recent graduates or early-career candidates with approximately 0–3 years of post-graduation work experience.",
        urls=src(extracted.experience_urls),
        base_instruction=(
            "Look for explicit mention of experience bands like 0–3 years, new/early career, or recent graduates within a few years of graduation."
        ),
    )

    # Company_Type_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Company_Type_With_Evidence",
        "Provides evidence that the company is an established enterprise or a Fortune 500 organization.",
        claim=f"The company {company} is a Fortune 500 company or an established enterprise technology company.",
        urls=src(extracted.company_type_urls),
        base_instruction=(
            "Prefer explicit statements or listings (e.g., Fortune 500 list or credible sources). "
            "If multiple sources disagree, rely on the most authoritative reference."
        ),
    )

    # Signing_Bonus_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Signing_Bonus_With_Evidence",
        "Provides evidence that new program participants are offered a signing bonus.",
        claim=f"New participants in the {program} are offered a signing bonus.",
        urls=src(extracted.signing_bonus_urls),
        base_instruction=(
            "Look for explicit mention of a sign-on/signing bonus in the job posting, compensation pages, or official benefits documentation."
        ),
    )

    # Retirement_Benefits_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Retirement_Benefits_With_Evidence",
        "Provides evidence that the company provides 401(k) retirement benefits with employer matching.",
        claim=f"The company {company} provides 401(k) retirement benefits with employer matching.",
        urls=src(extracted.retirement_urls),
        base_instruction=(
            "The evidence must indicate a 401(k) plan with employer match; generic retirement language without match is insufficient."
        ),
    )

    # Relocation_Assistance_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Relocation_Assistance_With_Evidence",
        "Provides evidence that the program includes relocation assistance for eligible candidates.",
        claim=f"The {program} includes relocation assistance for eligible candidates.",
        urls=src(extracted.relocation_urls),
        base_instruction=(
            "Look for 'relocation assistance' or similar benefits; ensure it's applicable to this program or eligible new hires."
        ),
    )

    # Health_Insurance_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Health_Insurance_With_Evidence",
        "Provides evidence that the company offers comprehensive health insurance benefits.",
        claim=f"The company {company} offers comprehensive health insurance benefits (e.g., medical, dental, vision).",
        urls=src(extracted.health_urls),
        base_instruction=(
            "Evidence should cover health benefits beyond minimal coverage; official benefits pages are acceptable."
        ),
    )

    # Professional_Development_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Professional_Development_With_Evidence",
        "Provides evidence that the program includes structured professional development, mentorship, or training components.",
        claim=f"The {program} includes structured professional development, mentorship, or training components.",
        urls=src(extracted.development_urls),
        base_instruction=(
            "Look for explicit program structure elements such as formal training, mentorship, cohort learning, bootcamps, or rotational training plans."
        ),
    )

    # Work_Authorization_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Work_Authorization_With_Evidence",
        "Provides evidence that candidates must be legally authorized to work in the United States.",
        claim=f"Candidates for the {program} must be legally authorized to work in the United States.",
        urls=src(extracted.authorization_urls),
        base_instruction=(
            "Accept statements like 'must be authorized to work in the U.S.' or 'no sponsorship available' when clearly tied to the program."
        ),
    )

    # Geographic_Presence_With_Evidence
    await add_and_verify_leaf(
        evaluator,
        req_root,
        "Geographic_Presence_With_Evidence",
        "Provides evidence that the company has technology operations in multiple U.S. locations or major tech hubs.",
        claim=f"The company {company} has technology operations in multiple U.S. locations or major tech hubs (e.g., Bay Area, Seattle, NYC, Austin, Boston, Atlanta, Chicago, Dallas, etc.).",
        urls=src(extracted.geographic_urls),
        base_instruction=(
            "Look for careers/location pages, job postings across multiple U.S. cities, or official statements indicating tech operations in multiple U.S. hubs."
        ),
    )

    # 5) Return evaluation summary
    return evaluator.get_summary()