import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "cs_phd_ai_ml_programs"
TASK_DESCRIPTION = """
Identify three Computer Science PhD programs in the United States that specialize in Artificial Intelligence (AI) or Machine Learning (ML) and meet all specified institutional, program, funding, faculty, and cross-program requirements.
"""


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class ProgramEntry(BaseModel):
    # Core identity fields
    university_name: Optional[str] = None
    program_name: Optional[str] = None
    state: Optional[str] = None
    sector: Optional[str] = None  # "public" or "private" (normalize if necessary)

    # Institutional metrics (values are strings to be permissive; verification relies on sources)
    rd_expenditure: Optional[str] = None  # e.g., "$1.2B" or "1,200,000,000"
    rd_ranking: Optional[str] = None      # e.g., "Top 10", "rank #14"
    ai_center_name: Optional[str] = None

    # Funding/Facts (free-form strings)
    stipend_amount: Optional[str] = None
    ai_faculty_count: Optional[str] = None

    # Evidence URLs
    rd_sources: List[str] = Field(default_factory=list)                # NSF HERD / institutional report for R&D + ranking
    ai_center_urls: List[str] = Field(default_factory=list)            # University AI center page
    nsf_funding_urls: List[str] = Field(default_factory=list)          # NSF awards db or university report
    ranking_urls: List[str] = Field(default_factory=list)              # US News / QS or equivalent
    specialization_urls: List[str] = Field(default_factory=list)       # Program specialization/track/area page
    credit_urls: List[str] = Field(default_factory=list)               # Program handbook/page for credit hours
    exam_urls: List[str] = Field(default_factory=list)                 # Comp/qual/candidacy exam policy
    committee_urls: List[str] = Field(default_factory=list)            # Committee size requirement
    funding_policy_urls: List[str] = Field(default_factory=list)       # Guaranteed funding policy
    stipend_urls: List[str] = Field(default_factory=list)              # Stipend amount evidence
    faculty_roster_urls: List[str] = Field(default_factory=list)       # Faculty roster + research areas
    publication_urls: List[str] = Field(default_factory=list)          # DBLP/GS/profiles with top-tier pubs


class ProgramsExtraction(BaseModel):
    programs: List[ProgramEntry] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_programs() -> str:
    return """
    You must extract up to THREE Computer Science PhD programs in the United States that the answer proposes as satisfying the detailed AI/ML-focused requirements. Extract ONLY what is explicitly provided in the answer. Do not invent URLs or values.

    For each proposed program, extract:
    - university_name: The university name
    - program_name: The specific PhD program name (e.g., "PhD in Computer Science")
    - state: The U.S. state where the university is located (e.g., "California")
    - sector: Whether the university is public or private (use "public" or "private" if explicitly stated; otherwise copy what is written)

    Institutional metrics (values as strings; exact numbers/ranks if stated, else copy the phrasing):
    - rd_expenditure: The reported FY 2024 total R&D expenditure if stated (string; can include symbols or units)
    - rd_ranking: The FY 2024 national ranking for total R&D expenditure if stated (string)
    - ai_center_name: Name of an affiliated AI research center/institute/lab, if named

    Evidence URLs (extract complete URLs exactly as shown; these must appear in the answer):
    - rd_sources: List of URLs that support R&D expenditure >= $1B and top-30 national ranking (prefer NSF HERD or official institutional reports)
    - ai_center_urls: University AI center/institute/lab page URLs
    - nsf_funding_urls: URLs evidencing NSF funding for AI/ML (NSF award database pages or official university reports)
    - ranking_urls: URLs showing the program's top-15 national ranking in CS or AI/ML (U.S. News, QS, or equivalent)
    - specialization_urls: URLs showing the AI/ML specialization/track/concentration in the PhD program
    - credit_urls: URLs showing total PhD credit hours requirement
    - exam_urls: URLs showing comprehensive/qualifying/candidacy exam requirement
    - committee_urls: URLs showing dissertation committee size requirement
    - funding_policy_urls: URLs showing guaranteed funding policy (tuition + stipend) for admitted PhD students
    - stipend_urls: URLs showing the current annual stipend amount
    - faculty_roster_urls: URLs listing faculty and their research areas (to count at least 10 AI/ML faculty)
    - publication_urls: URLs evidencing faculty publications in top-tier venues (NeurIPS, ICML, CVPR, AAAI, etc.)

    Additional facts if explicitly provided:
    - stipend_amount: The annual stipend amount text (e.g., "$34,000")
    - ai_faculty_count: The number of faculty in AI/ML (as written, e.g., "12")

    Rules:
    - Return a JSON with a "programs" array (max length 3). If more than 3 programs are present, include only the first three.
    - If any field is missing, set it to null (for strings) or an empty list (for URL lists).
    - Extract URLs exactly as shown (including protocol). If the answer uses markdown links, extract the underlying URL.
    - Do NOT synthesize or assume any URL or information that is not explicitly provided in the answer.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _has_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    valid = [u for u in urls if isinstance(u, str) and u.strip()]
    return len(valid) > 0


def _normalize_sector(val: Optional[str]) -> Optional[str]:
    if not val:
        return None
    s = val.strip().lower()
    if "public" in s:
        return "public"
    if "private" in s:
        return "private"
    return None


async def _verify_leaf_with_sources(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    additional_instruction: str = "None",
    extra_prereq: Optional[List[Any]] = None,
) -> None:
    """
    Convenience wrapper: add a leaf, fail immediately if no sources, otherwise verify by URLs.
    """
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
        status="initialized",
        score=0.0,
    )

    if not _has_urls(sources):
        # No sources -> immediate fail (binary leaf requirement)
        leaf.status = "failed"
        leaf.score = 0.0
        return

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction,
        extra_prerequisites=extra_prereq,
    )


# -----------------------------------------------------------------------------
# Program verification
# -----------------------------------------------------------------------------
async def verify_program(evaluator: Evaluator, parent, prog: ProgramEntry, idx: int) -> None:
    """
    Build the verification subtree for a single program candidate.
    """
    pfx = f"P{idx+1}"
    # Program-level container (non-critical to allow partial credit between programs)
    program_node = evaluator.add_parallel(
        id=f"Program_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} qualifying PhD program meeting all requirements",
        parent=parent,
        critical=False,
    )

    # --------------------------- Institutional Requirements ---------------------------
    inst_node = evaluator.add_parallel(
        id=f"{pfx}_Institutional_Requirements",
        desc=f"The university hosting Program {idx+1} meets institutional research metrics",
        parent=program_node,
        critical=True,
    )

    # R&D Expenditure + Ranking
    rd_node = evaluator.add_parallel(
        id=f"{pfx}_RD_Expenditure_Verification",
        desc="Verify university's R&D expenditure meets threshold and ranking requirements",
        parent=inst_node,
        critical=True,
    )

    # Reference (official NSF HERD or institutional report) - verify first
    await _verify_leaf_with_sources(
        evaluator,
        parent=rd_node,
        node_id=f"{pfx}_RD_Reference",
        desc="Provide official NSF HERD Survey data or institutional report verifying R&D expenditure and ranking",
        claim="These sources are official NSF HERD Survey pages or official institutional reports that document the university's total R&D expenditures and its FY 2024 national ranking.",
        sources=prog.rd_sources,
        critical=True,
        additional_instruction="Check whether the page is from NSF HERD (official) or an official institutional research report with FY 2024 data.",
    )

    # Expenditure >= $1B
    await _verify_leaf_with_sources(
        evaluator,
        parent=rd_node,
        node_id=f"{pfx}_RD_Expenditure",
        desc="University reported total R&D expenditures of at least $1 billion in FY 2024",
        claim="According to the provided sources, the university's total R&D expenditures in FY 2024 were at least $1 billion.",
        sources=prog.rd_sources,
        critical=True,
        additional_instruction="Look for FY 2024 totals. Accept values equal to or greater than $1,000,000,000; minor rounding/formatting variations are acceptable.",
        extra_prereq=[evaluator.find_node(f"{pfx}_RD_Reference")],
    )

    # Top-30 ranking
    await _verify_leaf_with_sources(
        evaluator,
        parent=rd_node,
        node_id=f"{pfx}_RD_Ranking",
        desc="University ranks within the top 30 nationally for total R&D expenditures in FY 2024",
        claim="According to the provided sources, the university ranks within the top 30 nationally for total R&D expenditures in FY 2024.",
        sources=prog.rd_sources,
        critical=True,
        additional_instruction="Look for national rank or an explicit position (<= 30) for FY 2024.",
        extra_prereq=[evaluator.find_node(f"{pfx}_RD_Reference")],
    )

    # AI Center presence
    ai_center_node = evaluator.add_parallel(
        id=f"{pfx}_AI_Center_Verification",
        desc="Verify university has an affiliated AI research center",
        parent=inst_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=ai_center_node,
        node_id=f"{pfx}_AI_Center_Reference",
        desc="Provide official university webpage for the AI research center",
        claim="This source is an official university webpage for an AI research center, institute, or laboratory affiliated with the university.",
        sources=prog.ai_center_urls,
        critical=True,
        additional_instruction="The page should be hosted on the university's domain and clearly describe an AI/ML research center, institute, or lab.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=ai_center_node,
        node_id=f"{pfx}_AI_Center",
        desc="University has an affiliated AI research center, institute, or laboratory",
        claim="The university has an affiliated AI research center, institute, or laboratory, as evidenced by the provided official webpage(s).",
        sources=prog.ai_center_urls,
        critical=True,
        additional_instruction="The page should indicate an active AI/ML research center/lab connected to the university.",
        extra_prereq=[evaluator.find_node(f"{pfx}_AI_Center_Reference")],
    )

    # NSF funding for AI/ML
    nsf_node = evaluator.add_parallel(
        id=f"{pfx}_NSF_Funding_Verification",
        desc="Verify university received NSF funding for AI or ML research",
        parent=inst_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=nsf_node,
        node_id=f"{pfx}_NSF_Reference",
        desc="Provide evidence of NSF AI/ML research funding (NSF award database or university report)",
        claim="These sources are official NSF award database entries or official university reports documenting NSF funding for AI or ML research.",
        sources=prog.nsf_funding_urls,
        critical=True,
        additional_instruction="Prefer NSF award pages (nsf.gov/awardsearch) or official university research news/reports explicitly naming NSF grants in AI/ML.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=nsf_node,
        node_id=f"{pfx}_NSF_Funding",
        desc="University received NSF funding for AI or ML research",
        claim="The university has received NSF funding for AI or machine learning research.",
        sources=prog.nsf_funding_urls,
        critical=True,
        additional_instruction="The source(s) should clearly tie NSF-funded projects to AI or ML research activities at the university.",
        extra_prereq=[evaluator.find_node(f"{pfx}_NSF_Reference")],
    )

    # --------------------------- Program Requirements ---------------------------
    prog_req_node = evaluator.add_parallel(
        id=f"{pfx}_Program_Requirements",
        desc=f"Program {idx+1} meets academic program quality and structure requirements",
        parent=program_node,
        critical=True,
    )

    # Program ranking top 15
    pr_rank_node = evaluator.add_parallel(
        id=f"{pfx}_Program_Ranking_Verification",
        desc="Verify program ranking meets top 15 requirement",
        parent=prog_req_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=pr_rank_node,
        node_id=f"{pfx}_Ranking_Reference",
        desc="Provide official ranking source (U.S. News, QS, or equivalent) documenting program ranking",
        claim="This source is an official ranking page from U.S. News & World Report, QS, or an equivalent recognized ranking that documents the program's ranking.",
        sources=prog.ranking_urls,
        critical=True,
        additional_instruction="Accept U.S. News graduate CS rankings or QS subject rankings; ensure the ranking is for Computer Science or AI/ML and is credible/official.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=pr_rank_node,
        node_id=f"{pfx}_Program_Ranking",
        desc="PhD program ranked within top 15 nationally for Computer Science or AI/ML",
        claim="According to the provided ranking source(s), the PhD program is ranked within the top 15 in the United States for Computer Science or AI/ML.",
        sources=prog.ranking_urls,
        critical=True,
        additional_instruction="If the source is QS (global), ensure that the claim refers to a top-15 position within the U.S.; otherwise, prefer U.S. News national rankings.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Ranking_Reference")],
    )

    # AI specialization
    ai_spec_node = evaluator.add_parallel(
        id=f"{pfx}_AI_Specialization_Verification",
        desc="Verify program offers AI or ML specialization",
        parent=prog_req_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=ai_spec_node,
        node_id=f"{pfx}_Specialization_Reference",
        desc="Provide official program webpage documenting AI/ML specialization",
        claim="This source is an official program or department webpage documenting an AI or ML specialization/track/concentration within the PhD program.",
        sources=prog.specialization_urls,
        critical=True,
        additional_instruction="The page should explicitly mention an AI or ML specialization, track, or concentration for PhD students.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=ai_spec_node,
        node_id=f"{pfx}_AI_Specialization",
        desc="Program explicitly offers AI or ML specialization, track, or concentration",
        claim="The program explicitly offers a specialization, track, or concentration in Artificial Intelligence or Machine Learning.",
        sources=prog.specialization_urls,
        critical=True,
        additional_instruction="Confirm that AI/ML specialization applies to the PhD degree (not only MS/BS).",
        extra_prereq=[evaluator.find_node(f"{pfx}_Specialization_Reference")],
    )

    # Credit hours
    credit_node = evaluator.add_parallel(
        id=f"{pfx}_Credit_Hours_Verification",
        desc="Verify program credit hour requirements",
        parent=prog_req_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=credit_node,
        node_id=f"{pfx}_Credit_Reference",
        desc="Provide official program webpage or handbook stating credit hour requirements",
        claim="This source is an official program webpage, handbook, or catalog page that states total PhD credit hour requirements.",
        sources=prog.credit_urls,
        critical=True,
        additional_instruction="Ensure it specifies total credit hours for the PhD (including coursework and research).",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=credit_node,
        node_id=f"{pfx}_Credit_Hours",
        desc="Program requires between 48 and 96 total credit hours for PhD completion",
        claim="The program requires between 48 and 96 total credit hours for PhD completion, including coursework and dissertation research.",
        sources=prog.credit_urls,
        critical=True,
        additional_instruction="Accept ranges or totals falling within [48, 96]. If presented as minimums/maximums, ensure the feasible total lies within the range.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Credit_Reference")],
    )

    # Comprehensive exam requirement
    exam_node = evaluator.add_parallel(
        id=f"{pfx}_Comprehensive_Exam_Verification",
        desc="Verify program requires comprehensive examination",
        parent=prog_req_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=exam_node,
        node_id=f"{pfx}_Exam_Reference",
        desc="Provide official program webpage or handbook documenting examination requirement",
        claim="This source is an official program or department page that documents a comprehensive, qualifying, or candidacy examination requirement for the PhD.",
        sources=prog.exam_urls,
        critical=True,
        additional_instruction="Look for language indicating comprehensive/qualifying/candidacy exam as a requirement for advancing in the PhD.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=exam_node,
        node_id=f"{pfx}_Comprehensive_Exam",
        desc="Program requires comprehensive, qualifying, or candidacy examination",
        claim="The PhD program requires students to pass a comprehensive, qualifying, or candidacy examination.",
        sources=prog.exam_urls,
        critical=True,
        additional_instruction="Confirm that it is a required milestone for PhD students.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Exam_Reference")],
    )

    # Committee size requirement
    comm_node = evaluator.add_parallel(
        id=f"{pfx}_Committee_Size_Verification",
        desc="Verify dissertation committee size requirement",
        parent=prog_req_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=comm_node,
        node_id=f"{pfx}_Committee_Reference",
        desc="Provide official program webpage or handbook stating committee size requirement",
        claim="This source is an official program or graduate school policy page that states dissertation committees must have at least a specified minimum number of members.",
        sources=prog.committee_urls,
        critical=True,
        additional_instruction="Prefer official program or graduate school policies specifying committee composition.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=comm_node,
        node_id=f"{pfx}_Committee_Size",
        desc="Dissertation committees must have at least 4 members",
        claim="Doctoral dissertation committees consist of at least 4 members.",
        sources=prog.committee_urls,
        critical=True,
        additional_instruction="Confirm the minimum is 4 or more. If ranges/variations exist, ensure the base minimum meets 4.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Committee_Reference")],
    )

    # --------------------------- Funding Requirements ---------------------------
    fund_node = evaluator.add_parallel(
        id=f"{pfx}_Funding_Requirements",
        desc=f"Program {idx+1} provides adequate financial support",
        parent=program_node,
        critical=True,
    )

    # Guaranteed funding
    guar_node = evaluator.add_parallel(
        id=f"{pfx}_Guaranteed_Funding_Verification",
        desc="Verify program provides guaranteed funding",
        parent=fund_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=guar_node,
        node_id=f"{pfx}_Funding_Reference",
        desc="Provide official program webpage documenting guaranteed funding policy",
        claim="This source is an official program or department webpage documenting guaranteed funding (tuition + stipend) for admitted full-time PhD students.",
        sources=prog.funding_policy_urls,
        critical=True,
        additional_instruction="Look for explicit guarantees (e.g., 5 years of full support) that include both tuition and stipend for full-time PhD students.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=guar_node,
        node_id=f"{pfx}_Guaranteed_Funding",
        desc="Program provides guaranteed funding (tuition + stipend) for admitted PhD students",
        claim="The program provides guaranteed funding packages that cover both tuition and a stipend for admitted full-time PhD students.",
        sources=prog.funding_policy_urls,
        critical=True,
        additional_instruction="If conditions apply (e.g., satisfactory progress), the guarantee still counts as long as it is a standing program policy.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Funding_Reference")],
    )

    # Stipend amount >= $30,000
    stipend_node = evaluator.add_parallel(
        id=f"{pfx}_Stipend_Amount_Verification",
        desc="Verify stipend amount meets minimum requirement",
        parent=fund_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=stipend_node,
        node_id=f"{pfx}_Stipend_Reference",
        desc="Provide official source documenting stipend amount (program webpage, admitted student portal, or publicly available stipend database)",
        claim="This source documents the current annual stipend amount for PhD students in this program.",
        sources=prog.stipend_urls,
        critical=True,
        additional_instruction="Prefer official departmental pages, graduate division pages, or reliable public stipend databases. PDFs/handbooks acceptable.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=stipend_node,
        node_id=f"{pfx}_Stipend_Amount",
        desc="Annual stipend is at least $30,000",
        claim="The annual stipend for PhD students is at least $30,000.",
        sources=prog.stipend_urls,
        critical=True,
        additional_instruction="If monthly or 9/12-month is shown, compute or infer the annualized amount from the page; small rounding differences are acceptable.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Stipend_Reference")],
    )

    # --------------------------- Faculty Requirements ---------------------------
    fac_node = evaluator.add_parallel(
        id=f"{pfx}_Faculty_Requirements",
        desc=f"Program {idx+1} has sufficient high-quality faculty in AI/ML",
        parent=program_node,
        critical=True,
    )

    # Faculty count >= 10
    fac_count_node = evaluator.add_parallel(
        id=f"{pfx}_Faculty_Count_Verification",
        desc="Verify program has sufficient AI/ML faculty",
        parent=fac_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=fac_count_node,
        node_id=f"{pfx}_Faculty_Reference",
        desc="Provide official department webpage listing faculty and their research areas",
        claim="This source lists faculty members and their primary research areas so that AI/ML-focused faculty can be identified and counted.",
        sources=prog.faculty_roster_urls,
        critical=True,
        additional_instruction="Prefer official department faculty listings with research area tags; AI/ML-related areas should be clearly indicated.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=fac_count_node,
        node_id=f"{pfx}_Faculty_Count",
        desc="At least 10 faculty members have primary research areas in AI or ML",
        claim="At least 10 faculty members in the program have primary research areas in Artificial Intelligence or Machine Learning.",
        sources=prog.faculty_roster_urls,
        critical=True,
        additional_instruction="Names and areas do not need to be enumerated in the answer; the page(s) should make counting or verification feasible.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Faculty_Reference")],
    )

    # Publication venues (top-tier)
    pub_node = evaluator.add_parallel(
        id=f"{pfx}_Publication_Venues_Verification",
        desc="Verify faculty publish in top-tier venues",
        parent=fac_node,
        critical=True,
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=pub_node,
        node_id=f"{pfx}_Publication_Reference",
        desc="Provide evidence of recent faculty publications in top-tier venues (faculty profiles, Google Scholar, or DBLP)",
        claim="These sources (e.g., faculty profiles, Google Scholar, DBLP) provide evidence of recent faculty publications.",
        sources=prog.publication_urls,
        critical=True,
        additional_instruction="Look for publications in NeurIPS, ICML, CVPR, AAAI, or equivalent A*/A venues in the last several years.",
    )

    await _verify_leaf_with_sources(
        evaluator,
        parent=pub_node,
        node_id=f"{pfx}_Publication_Venues",
        desc="Faculty actively publish in top-tier conferences (A* or A-ranked: NeurIPS, ICML, CVPR, AAAI, or equivalent)",
        claim="Faculty in the program actively publish in top-tier A*/A-ranked computer science conferences such as NeurIPS, ICML, CVPR, or AAAI (or equivalent).",
        sources=prog.publication_urls,
        critical=True,
        additional_instruction="At least some faculty should have publications in those venues; accept reasonable variants or equivalent top-tier venues.",
        extra_prereq=[evaluator.find_node(f"{pfx}_Publication_Reference")],
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point to evaluate an answer for the AI/ML-focused CS PhD programs task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # Extract structured information for up to 3 programs
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep first 3; pad if needed
    programs: List[ProgramEntry] = list(extracted.programs[:3])
    while len(programs) < 3:
        programs.append(ProgramEntry())

    # Build program subtrees
    for i in range(3):
        await verify_program(evaluator, root, programs[i], i)

    # --------------------------- Cross-Program Requirements ---------------------------
    cross_node = evaluator.add_parallel(
        id="Cross_Program_Requirements",
        desc="Requirements that apply across all three selected programs collectively",
        parent=root,
        critical=True,
    )

    # Gather states and sectors for cross checks
    states = []
    sectors = []
    for p in programs:
        if p.state and p.state.strip():
            states.append(p.state.strip())
        norm_sector = _normalize_sector(p.sector)
        if norm_sector:
            sectors.append(norm_sector)

    # Geographic diversity: at least two distinct states
    evaluator.add_custom_node(
        result=len(set([s.lower() for s in states])) >= 2,
        id="Geographic_Diversity",
        desc="The three programs collectively represent at least 2 different U.S. states",
        parent=cross_node,
        critical=True,
    )

    # At least one public
    evaluator.add_custom_node(
        result=any(s == "public" for s in sectors),
        id="Public_University_Inclusion",
        desc="At least one program is from a public university",
        parent=cross_node,
        critical=True,
    )

    # At least one private
    evaluator.add_custom_node(
        result=any(s == "private" for s in sectors),
        id="Private_University_Inclusion",
        desc="At least one program is from a private university",
        parent=cross_node,
        critical=True,
    )

    # Ground-truth/expectation record (not used for scoring, just for context)
    evaluator.add_ground_truth({
        "expected_num_programs": 3,
        "institutional_requirements": [
            "FY2024 total R&D >= $1B (NSF HERD) and top-30 nationally",
            "Affiliated AI research center/institute/lab (official page)",
            "NSF-funded AI/ML research (NSF awards DB or official report)"
        ],
        "program_requirements": [
            "Top-15 national ranking in CS or AI/ML (US News/ QS acceptable)",
            "Explicit AI/ML specialization for PhD",
            "Total credits between 48 and 96",
            "Comprehensive/qualifying/candidacy exam required",
            "Dissertation committee size >= 4"
        ],
        "funding_requirements": [
            "Guaranteed funding (tuition + stipend) for admitted full-time PhD students",
            "Annual stipend >= $30,000"
        ],
        "faculty_requirements": [
            ">= 10 AI/ML faculty (primary research areas)",
            "Active publications in top-tier A*/A venues (NeurIPS/ICML/CVPR/AAAI or equivalent)"
        ],
        "cross_program_requirements": [
            "At least 2 different U.S. states across the 3 programs",
            "At least one public university and at least one private university"
        ]
    })

    # Return evaluation summary
    return evaluator.get_summary()