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
TASK_ID = "ai_accelerators_investment_criteria_v1"
TASK_DESCRIPTION = """Identify two startup accelerator programs that specialize in artificial intelligence (AI) or machine learning (ML) technologies and meet the following investment criteria:

1. The program must provide at least $150,000 in total funding (cash investment) to each accepted startup
2. The program must take no more than 8% total equity in the participating companies
3. The program must have a fixed duration between 3 and 6 months
4. The program must be based in the United States or explicitly accept applications from US-based startups

For each accelerator program, provide:
- The official program name
- The exact funding amount provided to startups
- The exact equity percentage taken
- The program duration in months
- A brief description of their AI/ML specialization or focus
- A direct link to the official program website or official page documenting these terms

All information must be verifiable through official program documentation.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    name: Optional[str] = None
    funding_amount: Optional[str] = None
    equity_percentage: Optional[str] = None
    duration: Optional[str] = None  # e.g., "3 months", "12 weeks", "3–4 months"
    ai_ml_focus: Optional[str] = None
    official_url: Optional[str] = None
    extra_official_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    From the answer, extract up to all accelerator programs mentioned (later we will only use the first two).
    For each program, extract the following fields exactly as stated in the answer:
    - name: Official program name (string)
    - funding_amount: The exact cash investment amount promised per startup (string as written)
    - equity_percentage: The exact equity percentage taken by the program (string as written)
    - duration: The program duration as written (e.g., "3 months", "12 weeks", "3–4 months") (string)
    - ai_ml_focus: A brief description (1–2 short phrases) of AI/ML specialization or a dedicated AI/ML track (string)
    - official_url: A single direct URL to the official program webpage that documents these terms. Prefer the organization’s own domain (official site), not news or third-party pages.
    - extra_official_urls: Any other official URLs in the answer for this program (array of strings)

    Rules:
    - Only extract information explicitly present in the answer. Do not invent details.
    - For URLs, extract the actual link targets. If missing protocol, prepend http://
    - If a field is not present for a program, set it to null (or [] for arrays).
    - Ensure official_url is the most direct, authoritative page for the program terms when present in the answer.

    Return JSON with:
    {
      "programs": [
        {
          "name": ...,
          "funding_amount": ...,
          "equity_percentage": ...,
          "duration": ...,
          "ai_ml_focus": ...,
          "official_url": ...,
          "extra_official_urls": [...]
        },
        ...
      ]
    }
    """


# --------------------------------------------------------------------------- #
# Helper: build sources                                                       #
# --------------------------------------------------------------------------- #
def build_sources(program: ProgramItem) -> List[str]:
    urls: List[str] = []
    if program.official_url and str(program.official_url).strip():
        urls.append(program.official_url.strip())
    for u in program.extra_official_urls or []:
        if u and str(u).strip():
            urls.append(str(u).strip())
    # Deduplicate while preserving order
    deduped = []
    seen = set()
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification for a single program                                           #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    program_node,
    program: ProgramItem,
    index: int,  # 1-based index for readability in IDs/descriptions
) -> None:
    """
    Build verification subtree for a single program according to the rubric.
    """
    id_prefix = f"p{index}"
    sources = build_sources(program)
    sources_or_none: Any = sources if len(sources) > 0 else None

    # ---------------- Program Identification (critical, parallel) ----------------
    ident_node = evaluator.add_parallel(
        id=f"{id_prefix}_program_identification",
        desc="Verify program identity and core eligibility",
        parent=program_node,
        critical=True
    )

    # Existence of core info (critical gate): require program name and at least one official URL
    has_required = (program.name is not None and str(program.name).strip() != "") and (len(sources) > 0)
    evaluator.add_custom_node(
        result=has_required,
        id=f"{id_prefix}_required_info",
        desc="Program name and an official URL are provided in the answer",
        parent=ident_node,
        critical=True
    )

    # AI/ML specialization (critical leaf)
    ai_ml_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_ai_ml_specialization",
        desc="Program explicitly specializes in or has a dedicated track for AI, ML, or related AI technologies",
        parent=ident_node,
        critical=True
    )
    ai_ml_claim = (
        f"The accelerator program named '{program.name or 'the program'}' explicitly specializes in "
        f"artificial intelligence or machine learning, or provides a dedicated AI/ML track or cohort."
    )
    await evaluator.verify(
        claim=ai_ml_claim,
        node=ai_ml_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Only count evidence from an official program or organization webpage (not news or third-party sites). "
            "Look for explicit statements such as 'AI', 'machine learning', 'ML', 'generative AI', or 'deep learning' "
            "indicating specialization or a dedicated AI/ML track. A general-purpose accelerator with no dedicated "
            "AI/ML focus should be considered NOT specialized."
        )
    )

    # US eligibility (critical leaf)
    us_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_us_eligibility",
        desc="Program is based in the United States or explicitly accepts applications from US-based startups",
        parent=ident_node,
        critical=True
    )
    us_claim = (
        "The accelerator is based in the United States OR explicitly accepts applications from U.S.-based startups."
    )
    await evaluator.verify(
        claim=us_claim,
        node=us_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Rely only on the official program/organization website. Accept if the page states a U.S. location, U.S. "
            "presence, or clearly indicates that U.S.-based startups are eligible to apply (including remote/global "
            "programs that explicitly accept U.S. startups). If geography is unspecified, treat as NOT supported."
        )
    )

    # ---------------- Investment Terms (critical, parallel) ----------------
    invest_node = evaluator.add_parallel(
        id=f"{id_prefix}_investment_terms",
        desc="Verify program investment terms meet all financial and structural requirements",
        parent=program_node,
        critical=True
    )

    # Minimum funding >= $150,000 (critical leaf)
    min_funding_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_minimum_funding",
        desc="Program provides at least $150,000 in total funding to each accepted startup",
        parent=invest_node,
        critical=True
    )
    min_funding_claim = (
        "The accelerator provides at least $150,000 in total cash funding to each accepted startup "
        "(exclude only non-cash perks like credits or discounts)."
    )
    await evaluator.verify(
        claim=min_funding_claim,
        node=min_funding_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Use only the official program/organization website. Treat cash investment, stipends, and standard "
            "instruments like SAFE/convertible notes that deliver cash to the company as cash funding. Do NOT count "
            "non-cash perks (e.g., cloud credits). Do NOT count optional, non-guaranteed prize/follow-on funding."
        )
    )

    # Maximum equity <= 8% (critical leaf)
    max_equity_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_maximum_equity",
        desc="Program takes no more than 8% total equity",
        parent=invest_node,
        critical=True
    )
    max_equity_claim = (
        "The accelerator takes no more than 8% total equity for its standard/core program terms."
    )
    await evaluator.verify(
        claim=max_equity_claim,
        node=max_equity_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Use only the official program/organization website. Consider the equity taken for the core program terms "
            "only. Do not include equity or ownership that may result from optional or separate follow-on investments."
        )
    )

    # Program duration between 3 and 6 months inclusive (critical leaf)
    duration_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_program_duration",
        desc="Program has a fixed duration between 3 and 6 months (inclusive)",
        parent=invest_node,
        critical=True
    )
    duration_claim = (
        "The accelerator program has a fixed duration between 3 and 6 months inclusive."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=sources_or_none,
        additional_instruction=(
            "Use only the official program/organization website. Accept equivalent week ranges (e.g., 12–26 weeks). "
            "If a discrete fixed duration within 3–6 months is clearly stated (including ranges entirely within this "
            "interval), count as supported. If duration is outside this interval or not stated, treat as NOT supported."
        )
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
    Evaluate an answer for the AI/ML accelerator investment criteria task.
    """
    # Initialize evaluator (root is parallel: two programs evaluated independently)
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

    # Extract structured program info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="accelerator_programs_extracted"
    )

    programs: List[ProgramItem] = list(extracted.programs or [])
    # Use only the first two programs; pad with empty entries if fewer are provided
    while len(programs) < 2:
        programs.append(ProgramItem())
    programs = programs[:2]

    # Build subtrees for two programs
    program1_node = evaluator.add_parallel(
        id="accelerator_program_1",
        desc="First qualifying accelerator program",
        parent=root,
        critical=False
    )
    await verify_program(evaluator, program1_node, programs[0], index=1)

    program2_node = evaluator.add_parallel(
        id="accelerator_program_2",
        desc="Second qualifying accelerator program",
        parent=root,
        critical=False
    )
    await verify_program(evaluator, program2_node, programs[1], index=2)

    # Return structured evaluation summary
    return evaluator.get_summary()