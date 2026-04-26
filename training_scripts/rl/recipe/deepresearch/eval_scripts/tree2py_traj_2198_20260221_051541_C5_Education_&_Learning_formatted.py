import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nsf_top20_abet_coop_universities"
TASK_DESCRIPTION = """
Identify 4 public universities in the United States that meet all of the following criteria:

1. Ranked in the top 20 by total research and development (R&D) expenditures for fiscal year 2023, as reported by the National Science Foundation's Higher Education Research and Development (HERD) survey
2. Offer at least one ABET-accredited undergraduate (Bachelor's degree level) engineering program
3. Have an established undergraduate cooperative education (co-op) program available to engineering students, involving alternating periods of full-time work experience and academic study

For each university, provide:
- The full official name of the university
- The university's rank in the NSF HERD survey for FY 2023 total R&D expenditures
- A direct link to the NSF HERD rankings page or university profile showing the FY 2023 ranking
- A direct link to the ABET accredited programs database showing at least one accredited undergraduate engineering program at the university
- A direct link to the university's official webpage describing the cooperative education program for engineering students
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    nsf_rank: Optional[str] = None  # Keep as string for robustness (e.g., "14", "14th", "Top 20")
    nsf_herd_url: Optional[str] = None
    abet_url: Optional[str] = None
    coop_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to 6 universities mentioned in the answer that are claimed to meet all the requirements.
    For each university, extract the following fields exactly as presented in the answer text:

    - university_name: Full official university name
    - nsf_rank: The FY 2023 NSF HERD total R&D expenditures ranking for the university (keep as-is, e.g., "14", "14th", or similar). If not explicitly provided in the answer, set to null.
    - nsf_herd_url: A direct URL to either the NSF HERD rankings page or the NSF profile/page that shows the FY 2023 ranking for that university. If multiple are listed in the answer, choose the most specific one that shows the ranking. If missing, set to null.
    - abet_url: A direct URL to the ABET accredited program(s) database page that shows at least one undergraduate (Baccalaureate) engineering program for this university. If missing, set to null.
    - coop_url: A direct URL to the university’s official webpage describing the cooperative education (co-op) program for engineering students. If missing, set to null.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer. Do not infer or create URLs.
    - Accept plain URLs or markdown links (extract the actual URL).
    - Include the protocol (http:// or https://). If missing, prepend http://.
    - If the answer provides more than 4 universities, still extract them all; the evaluator will use the first 4.
    - If a field is missing for a university, set it to null.
    - Do not include shortened or redirector links if the final destination URL is shown elsewhere in the answer.
    - The ABET URL must point to the ABET database (e.g., abet.org) and show at least one undergraduate engineering (Baccalaureate) program.
    - The co-op URL must be an official university domain page describing the cooperative education program for engineering students.
    """.strip()


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    """
    Build verification subtree and run checks for one university.
    """
    # Create the university parallel node (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"{['First','Second','Third','Fourth','Fifth','Sixth'][idx] if idx < 6 else f'#{idx+1}'} qualifying university with all required attributes",
        parent=parent_node,
        critical=False
    )

    # Prepare convenience variables
    name = uni.university_name or ""
    nsf_rank = uni.nsf_rank or ""
    nsf_url = uni.nsf_herd_url
    abet_url = uni.abet_url
    coop_url = uni.coop_url

    # 1) Reference URLs presence (critical) - existence/format check only
    refs_ok = bool(nsf_url and abet_url and coop_url)
    if refs_ok:
        # Basic URL sanity: should start with http
        refs_ok = all(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))
                      for u in [nsf_url, abet_url, coop_url])

    evaluator.add_custom_node(
        result=refs_ok,
        id=f"u{idx+1}_Reference_URLs",
        desc="Appropriate reference URLs are provided to verify the university's NSF ranking, ABET accreditation status, and co-op program information",
        parent=uni_node,
        critical=True
    )

    # 2) Create four leaf nodes for content verifications (all critical)
    nsf_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_NSF_Top_20_Ranking",
        desc="University is ranked in the top 20 by total R&D expenditures for fiscal year 2023 according to NSF HERD survey",
        parent=uni_node,
        critical=True
    )

    public_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_Public_Institution",
        desc="University is a public institution (state-funded, not private)",
        parent=uni_node,
        critical=True
    )

    abet_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_ABET_Accreditation",
        desc="University offers at least one ABET-accredited undergraduate (Bachelor's degree) engineering program",
        parent=uni_node,
        critical=True
    )

    coop_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_Cooperative_Education_Program",
        desc="University has an established undergraduate cooperative education (co-op) program available to engineering students that involves alternating periods of full-time work and academic study",
        parent=uni_node,
        critical=True
    )

    # 3) Build claims and run verifications
    # NSF Top 20 verification
    rank_fragment = f" Its numerical rank is given as '{nsf_rank}'." if nsf_rank else ""
    nsf_claim = (
        f"On the provided NSF HERD page, the institution '{name}' is listed within the top 20 for total "
        f"research and development (R&D) expenditures for FY 2023.{rank_fragment}"
    )
    nsf_additional = (
        "Verify specifically for FY 2023 and 'total R&D expenditures' (not limited to federal or other subcategories). "
        "Accept either a consolidated ranking table or an NSF institutional profile page if it clearly shows the FY 2023 rank. "
        "If a number is shown, it must be 20 or less to pass."
    )

    # Public institution verification
    public_claim = (
        f"The institution '{name}' is a public (state) university, not a private institution."
    )
    public_additional = (
        "Use any provided pages (NSF link, ABET database page, or official co-op page) to determine whether the institution is public. "
        "Look for phrases like 'public university', 'public research university', or indications of state control. "
        "If the pages do not clearly indicate public status, do not assume; treat as not supported."
    )
    public_sources: List[str] = []
    for possible in [nsf_url, abet_url, coop_url]:
        if isinstance(possible, str) and possible.strip():
            public_sources.append(possible)

    # ABET accreditation verification
    abet_claim = (
        f"The provided ABET database page shows at least one accredited undergraduate (Baccalaureate/Bachelor's) "
        f"engineering program for '{name}'."
    )
    abet_additional = (
        "Check that the page lists an accredited program at the baccalaureate level (e.g., 'Baccalaureate', 'Bachelor of Science'). "
        "Programs may be under ABET's EAC or ETAC; ensure the degree level is undergraduate."
    )

    # Co-op program verification
    coop_claim = (
        f"The provided official university page describes a cooperative education (co-op) program for undergraduate engineering students "
        f"that involves alternating full-time work periods with academic study terms."
    )
    coop_additional = (
        "Look for explicit mentions of 'co-op' or 'cooperative education' for engineering students, and that it uses alternating "
        "work/study terms (e.g., full-time paid work rotations interleaved with academic semesters)."
    )

    # Run verifications in parallel for this university
    claims_and_sources = [
        (nsf_claim, nsf_url, nsf_leaf, nsf_additional),
        (public_claim, public_sources if public_sources else None, public_leaf, public_additional),
        (abet_claim, abet_url, abet_leaf, abet_additional),
        (coop_claim, coop_url, coop_leaf, coop_additional),
    ]
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the task of identifying 4 public universities that satisfy:
    - Top 20 in NSF HERD FY2023 total R&D expenditures
    - At least one ABET-accredited undergraduate engineering program
    - Established engineering co-op program with alternating work/study
    And provide the three reference URLs for verification.
    """
    # Initialize evaluator with a parallel root to allow partial credit across universities
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract the universities data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Keep only the first 4 (pad with empty placeholders if fewer)
    universities: List[UniversityItem] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build and verify each university subtree
    verify_tasks = []
    for i in range(4):
        verify_tasks.append(verify_university(evaluator, evaluator.root, universities[i], i))

    # Run all universities' verifications concurrently
    await asyncio.gather(*verify_tasks)

    # Return evaluation summary
    return evaluator.get_summary()