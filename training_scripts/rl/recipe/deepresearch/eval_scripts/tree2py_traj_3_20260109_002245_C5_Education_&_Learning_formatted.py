import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "affordable_aacsb_analytics_masters_under_15k"
TASK_DESCRIPTION = """
I am seeking to enroll in an affordable online master's degree program in analytics or business analytics. I need to find four distinct AACSB-accredited online master's programs in Analytics, Business Analytics, or Data Analytics where the total program tuition is under $15,000 and no GRE or GMAT is required for admission.

For each program, please provide:
1. The institution name
2. The specific program name (e.g., "MBA in Data Analytics", "MS in Business Analytics")
3. A direct URL link to the program's official webpage
4. Confirmation that the program is 100% online
5. Confirmation that the institution or business school holds AACSB accreditation
6. The total program tuition cost
7. The total credit hours required for the degree
8. Confirmation that GRE/GMAT is not required (or that a waiver is explicitly available for qualified applicants)
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ProgramItem(BaseModel):
    # Required identity fields
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    official_url: Optional[str] = None

    # Delivery modality (online)
    online_modality_text: Optional[str] = None
    online_info_urls: List[str] = Field(default_factory=list)

    # Accreditation
    aacsb_accreditation_text: Optional[str] = None
    aacsb_urls: List[str] = Field(default_factory=list)

    regional_accreditation_text: Optional[str] = None
    regional_accreditation_urls: List[str] = Field(default_factory=list)

    # Tuition and credits
    total_tuition: Optional[str] = None
    tuition_urls: List[str] = Field(default_factory=list)

    total_credits: Optional[str] = None
    credits_urls: List[str] = Field(default_factory=list)

    # GRE/GMAT
    gre_gmat_policy_text: Optional[str] = None
    gre_gmat_urls: List[str] = Field(default_factory=list)

    # Extra URLs mentioned for this program (if any)
    additional_urls: List[str] = Field(default_factory=list)


class ProgramsExtraction(BaseModel):
    programs: List[ProgramItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
Extract up to 6 distinct online master's programs referenced in the answer that relate to Analytics, Business Analytics, Data Analytics, or Data Science (including MBA with Business Analytics concentration), capturing fields exactly as they appear. Return a JSON object:
{
  "programs": [
    {
      "institution_name": string or null,
      "program_name": string or null,
      "official_url": string or null,

      "online_modality_text": string or null,
      "online_info_urls": [urls...],

      "aacsb_accreditation_text": string or null,
      "aacsb_urls": [urls...],

      "regional_accreditation_text": string or null,
      "regional_accreditation_urls": [urls...],

      "total_tuition": string or null,
      "tuition_urls": [urls...],

      "total_credits": string or null,
      "credits_urls": [urls...],

      "gre_gmat_policy_text": string or null,
      "gre_gmat_urls": [urls...],

      "additional_urls": [urls...]
    },
    ...
  ]
}

Rules:
- Extract only information explicitly present in the answer; do not invent any details.
- Include only valid URLs present in the answer (plain or markdown links). If none are provided for a field's evidence, return an empty array for that field.
- Keep numbers as strings exactly as written (e.g., "$12,500", "30 credits"). Do not normalize.
- If a field is missing, set it to null (or an empty array for URL lists).
- "official_url" must be the most direct official program webpage if provided in the answer; otherwise null.
- For "online_modality_text", prefer phrases like "100% online", "fully online", "no campus visits" if the answer provides them verbatim.
- For accreditation URLs, include AACSB directory links and/or official business school accreditation pages if provided in the answer.
- For GRE/GMAT policy, include any admissions page link(s) that mention test requirements or waivers if present in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_institution_name(name: str) -> str:
    if not name:
        return ""
    # Lowercase, remove punctuation and excess whitespace for distinctness comparison
    s = name.lower()
    s = re.sub(r"[\s\-\.,&'’“”\"/]+", " ", s).strip()
    return s


def gather_sources(primary_url: Optional[str], extra_lists: List[List[str]]) -> List[str]:
    """Collect and deduplicate sources. Always include primary_url if provided."""
    urls: List[str] = []
    if primary_url and isinstance(primary_url, str) and primary_url.strip():
        urls.append(primary_url.strip())
    for lst in extra_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and u.strip():
                urls.append(u.strip())
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def parse_money_value_str_to_number(value: Optional[str]) -> Optional[float]:
    """Best-effort to parse a money string like '$12,345' or 'USD 14999' to a float. Returns None if not parsable."""
    if not value or not isinstance(value, str):
        return None
    s = value
    # Remove currency symbols and words
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        if s == "" or s == "." or s == "-":
            return None
        return float(s)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification logic per program                                              #
# --------------------------------------------------------------------------- #
async def verify_single_program(
    evaluator: Evaluator,
    parent_node,
    program: ProgramItem,
    index_1_based: int,
) -> None:
    """
    Construct verification nodes for a single program as per rubric.
    Parent node is the program container under root (parallel aggregation).
    """

    # 1) Provides the institution name (critical) — check existence only
    node_inst = evaluator.add_custom_node(
        result=(program.institution_name is not None and str(program.institution_name).strip() != ""),
        id=f"program_{index_1_based}_institution_name",
        desc="Provides the institution name",
        parent=parent_node,
        critical=True,
    )

    # 2) Provides the specific program name and it is a master's degree in Analytics/Business Analytics/Data Analytics/Data Science (critical)
    node_prog_name_level = evaluator.add_leaf(
        id=f"program_{index_1_based}_program_name_and_level",
        desc="Provides the specific program name and it is a master's degree in Analytics, Business Analytics, Data Analytics, Data Science, or a clearly equivalent analytics-focused title",
        parent=parent_node,
        critical=True,
    )
    prog_name = program.program_name or ""
    # We'll verify against the official program URL (preferred) or any provided additional URLs
    sources_prog_name = gather_sources(program.official_url, [program.additional_urls])
    claim_prog_level = (
        f"The program is a master's-level degree in Analytics or Business Analytics or Data Analytics or Data Science "
        f"(including equivalent titles such as MS in Data Analytics, MS in Business Analytics, M.S., MBA with a Business Analytics concentration), "
        f"and the program name provided is '{prog_name}'."
    )
    await evaluator.verify(
        claim=claim_prog_level,
        node=node_prog_name_level,
        sources=sources_prog_name,
        additional_instruction=(
            "Accept reasonable variants of master's degree naming (MS, M.S., MSc, MBA with analytics concentration, "
            "MPS in Data Analytics, etc.). The page should clearly indicate graduate-level master's program and a focus "
            "on analytics/data analytics/data science/business analytics."
        ),
    )

    # 3) Provides a direct, accessible URL to the official program webpage (critical) — existence check
    node_official_url = evaluator.add_custom_node(
        result=(program.official_url is not None and program.official_url.strip().lower().startswith(("http://", "https://"))),
        id=f"program_{index_1_based}_official_program_url",
        desc="Provides a direct, accessible URL to the program's official webpage",
        parent=parent_node,
        critical=True,
    )

    # 4) Official info confirms 100% online / fully online (critical) — verify via URLs
    node_online = evaluator.add_leaf(
        id=f"program_{index_1_based}_delivery_mode",
        desc="Official information confirms the program is 100% online / fully online",
        parent=parent_node,
        critical=True,
    )
    sources_online = gather_sources(program.official_url, [program.online_info_urls, program.additional_urls])
    claim_online = (
        "The program is delivered fully online (100% online), meaning students can complete the degree without required campus attendance."
    )
    await evaluator.verify(
        claim=claim_online,
        node=node_online,
        sources=sources_online,
        additional_instruction=(
            "Look for explicit language such as '100% online', 'fully online', 'no campus visits required'. "
            "If the page only says 'online' but indicates required on-campus intensives/residencies, then it's not fully online."
        ),
    )

    # 5) Confirms the institution is regionally accredited (critical) — verify via URLs
    node_regional = evaluator.add_leaf(
        id=f"program_{index_1_based}_regional_accreditation",
        desc="Confirms the institution is regionally accredited",
        parent=parent_node,
        critical=True,
    )
    sources_regional = gather_sources(program.official_url, [program.regional_accreditation_urls, program.additional_urls])
    inst_name_for_claim = program.institution_name or "the institution"
    claim_regional = (
        f"{inst_name_for_claim} is regionally accredited by a recognized U.S. regional accreditor "
        f"(e.g., HLC, MSCHE, SACSCOC, WSCUC, NECHE, NWCCU)."
    )
    await evaluator.verify(
        claim=claim_regional,
        node=node_regional,
        sources=sources_regional,
        additional_instruction=(
            "Accept evidence from the institution's accreditation page or the regional accreditor listing. "
            "If only national or programmatic accreditation is shown without regional accreditation, mark as unsupported."
        ),
    )

    # 6) Confirms AACSB accreditation (critical) — verify via URLs
    node_aacsb = evaluator.add_leaf(
        id=f"program_{index_1_based}_aacsb_accreditation",
        desc="Confirms the institution/business school/program holds AACSB accreditation",
        parent=parent_node,
        critical=True,
    )
    sources_aacsb = gather_sources(program.official_url, [program.aacsb_urls, program.additional_urls])
    claim_aacsb = (
        "The institution's business school (or the relevant program/college) is accredited by AACSB (Association to Advance Collegiate Schools of Business)."
    )
    await evaluator.verify(
        claim=claim_aacsb,
        node=node_aacsb,
        sources=sources_aacsb,
        additional_instruction=(
            "Prefer explicit mentions or listings of AACSB accreditation, e.g., on the school's site or AACSB's official directory. "
            "Mentions of other accreditations (e.g., ACBSP, IACBE) do not satisfy AACSB."
        ),
    )

    # 7) States the total program tuition is a complete-program total and under $15,000 (critical) — verify via URLs
    node_tuition = evaluator.add_leaf(
        id=f"program_{index_1_based}_total_tuition",
        desc="States the total program tuition as a complete-program total (not only per-credit without a total), and the total is under $15,000, verifiable via official sources",
        parent=parent_node,
        critical=True,
    )
    sources_tuition = gather_sources(program.official_url, [program.tuition_urls, program.additional_urls])
    tuition_str = program.total_tuition or ""
    tuition_val = parse_money_value_str_to_number(program.total_tuition)
    # We'll phrase the claim to include both 'complete-program total' and 'under $15,000'
    if tuition_val is not None:
        under_15k_clause = "and this amount is under $15,000 USD"
    else:
        under_15k_clause = "and this amount is under $15,000 USD (based on the total explicitly shown)"
    claim_tuition = (
        f"The source shows a complete-program total tuition for the degree (not just a per-credit rate). "
        f"The total program tuition is '{tuition_str}' {under_15k_clause}."
    )
    await evaluator.verify(
        claim=claim_tuition,
        node=node_tuition,
        sources=sources_tuition,
        additional_instruction=(
            "To pass: the page(s) must show an explicit full-program tuition total. If only per-credit cost is shown without a total, "
            "and the answer's total is not explicitly shown in official sources, mark as unsupported. "
            "If multiple totals exist (e.g., in-state/out-of-state), the stated total must be under $15,000."
        ),
    )

    # 8) States total credit hours required for degree completion (critical) — verify via URLs
    node_credits = evaluator.add_leaf(
        id=f"program_{index_1_based}_credit_hours",
        desc="States the total credit hours required for degree completion",
        parent=parent_node,
        critical=True,
    )
    sources_credits = gather_sources(program.official_url, [program.credits_urls, program.additional_urls])
    credits_str = program.total_credits or ""
    claim_credits = f"The program requires '{credits_str}' total credit hours to complete."
    await evaluator.verify(
        claim=claim_credits,
        node=node_credits,
        sources=sources_credits,
        additional_instruction=(
            "Look for explicit program credit totals in catalog, curriculum, or program overview pages. "
            "Accept small textual variants like '30 credits' vs '30 credit hours'."
        ),
    )

    # 9) Confirms GRE/GMAT not required or an explicit waiver is available (critical) — verify via URLs
    node_tests = evaluator.add_leaf(
        id=f"program_{index_1_based}_gre_gmat_policy",
        desc="Confirms GRE/GMAT is not required for admission or that an explicit waiver is available for qualified applicants",
        parent=parent_node,
        critical=True,
    )
    sources_tests = gather_sources(program.official_url, [program.gre_gmat_urls, program.additional_urls])
    gre_text = program.gre_gmat_policy_text or ""
    claim_tests = (
        "GRE/GMAT is not strictly required for admission to this program OR an explicit waiver policy is available for qualified applicants. "
        f"Policy excerpt (if provided in the answer): '{gre_text}'."
    )
    await evaluator.verify(
        claim=claim_tests,
        node=node_tests,
        sources=sources_tests,
        additional_instruction=(
            "Accept policies that say 'GRE/GMAT not required', 'test optional', or 'waivers available' with clear criteria. "
            "If the page indicates GRE/GMAT is required without waivers, mark as unsupported."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    Evaluate an answer for four distinct AACSB-accredited fully-online master's programs
    in analytics/business analytics/data analytics/data science with total tuition < $15,000
    and no GRE/GMAT requirement (or explicit waiver).
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
        default_model=model,
    )

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=ProgramsExtraction,
        extraction_name="programs_extraction",
    )

    # Keep only the first 4 programs for evaluation; pad with empty if fewer
    programs = list(extracted.programs[:4])
    while len(programs) < 4:
        programs.append(ProgramItem())  # placeholders

    # Build 4 program nodes (parallel) and verify each
    program_nodes = []
    for i in range(4):
        prog_node = evaluator.add_parallel(
            id=f"program_{i+1}",
            desc=f"Program {i+1} (one qualifying program)",
            parent=root,
            critical=False,
        )
        program_nodes.append(prog_node)

    # Verify all programs (can be done sequentially in code; the framework aggregates in parallel)
    for i, program in enumerate(programs, start=1):
        await verify_single_program(evaluator, program_nodes[i - 1], program, i)

    # Distinct institutions check (critical at root)
    # Require 4 non-empty institution names and all distinct (case-insensitive, basic normalization)
    inst_names = [p.institution_name for p in programs]
    non_empty = [n for n in inst_names if isinstance(n, str) and n.strip() != ""]
    normalized = [normalize_institution_name(n) for n in non_empty]
    distinct_ok = (len(non_empty) == 4) and (len(set(normalized)) == 4)

    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_institutions",
        desc="All four programs are from distinct institutions (no duplicate universities across program_1–program_4)",
        parent=root,
        critical=True,
    )

    # Record some custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_program_count": len(extracted.programs),
            "institutions_first_four": inst_names,
        },
        info_type="debug",
        info_name="extraction_debug_info",
    )

    # Return final summary
    return evaluator.get_summary()