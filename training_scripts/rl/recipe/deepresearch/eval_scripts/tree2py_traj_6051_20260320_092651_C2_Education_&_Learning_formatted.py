import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "top_bschool_deans_2026"
TASK_DESCRIPTION = (
    "Identify three current deans of top-ranked business schools (schools appearing in at least one major global or "
    "U.S. MBA ranking such as QS World University Rankings, U.S. News & World Report, Financial Times, Times Higher "
    "Education, or Poets & Quants) who hold doctoral degrees (PhD, Ed.D., or equivalent terminal degree). For each "
    "dean, provide: (1) The name of the business school and university, (2) The dean's full name, (3) Confirmation "
    "that they hold a doctoral degree, specifying the type (PhD, Ed.D., etc.), (4) Their complete educational "
    "background, including all degrees (undergraduate and graduate) with the institutions that granted them, and "
    "(5) A direct link to their official biography page on the business school's website or a credible biographical "
    "source. The deans identified must be current as of March 2026."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class EducationEntry(BaseModel):
    degree: Optional[str] = None            # e.g., PhD, Ed.D., MBA, BA
    field: Optional[str] = None             # e.g., Finance, Education Leadership
    institution: Optional[str] = None       # e.g., University of X
    year: Optional[str] = None              # e.g., 2004 (keep string to be lenient)


class DeanItem(BaseModel):
    business_school: Optional[str] = None
    university: Optional[str] = None
    dean_name: Optional[str] = None
    doctoral_degree_type: Optional[str] = None  # e.g., PhD in Economics; Ed.D. in Higher Ed
    education: List[EducationEntry] = Field(default_factory=list)
    official_bio_url: Optional[str] = None      # Prefer official business school/university faculty page
    ranking_urls: List[str] = Field(default_factory=list)     # Links to QS / USNews / FT / THE / Poets&Quants pages or school pages citing them
    additional_urls: List[str] = Field(default_factory=list)  # Other credible supporting URLs mentioned in the answer


class DeansExtraction(BaseModel):
    deans: List[DeanItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_deans() -> str:
    return """
    From the answer, extract up to three business school deans with the following fields. If the answer lists more than three, take the first three; if fewer, include only those provided.

    For each dean, extract:
    - business_school: The name of the business school (e.g., "Wharton School", "Harvard Business School").
    - university: The university name associated with the business school (e.g., "University of Pennsylvania").
    - dean_name: The dean's full name as stated.
    - doctoral_degree_type: The doctoral degree type and (optionally) field (e.g., "PhD in Economics", "Ed.D. in Higher Education").
      If not specified, set to null.
    - education: List all degrees (undergraduate and graduate) with the granting institutions, and include the field/major and year if available.
      Each item should be an object with fields: degree, field, institution, year (any may be null).
    - official_bio_url: A direct link to the dean’s official biography page on the business school/university website; if multiple are present, choose the most official-looking one.
      If none is present in the answer, set to null.
    - ranking_urls: URLs that directly support that the business school appears in at least one recognized MBA ranking (QS, U.S. News & World Report, Financial Times, Times Higher Education, or Poets & Quants).
      These can be the ranking sites themselves OR the business school/university page explicitly citing those rankings. Include only URLs explicitly given in the answer; do not invent.
    - additional_urls: Any other credible support URLs provided in the answer (e.g., university news releases, press, Poets & Quants profile pages, etc.) not already used as the official_bio_url or ranking_urls.

    IMPORTANT:
    - Only include URLs that are explicitly present in the answer (plain URL or markdown link). Do not infer or fabricate URLs.
    - Keep strings as-is from the answer; do not normalize or expand them.
    - If any field is missing, return null (for strings) or an empty array (for lists).
    - Return a JSON object with a top-level key "deans" which is an array of up to three objects as described.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _format_education_for_claim(entries: List[EducationEntry]) -> str:
    """
    Turn the structured education entries into a concise, readable list for verification claims.
    """
    parts: List[str] = []
    for e in entries:
        deg = (e.degree or "").strip()
        fld = (e.field or "").strip()
        inst = (e.institution or "").strip()
        yr = (e.year or "").strip()
        segs = []
        if deg:
            segs.append(deg)
        if fld:
            segs.append(f"in {fld}")
        if inst:
            segs.append(f"from {inst}")
        if yr:
            segs.append(f"({yr})")
        part = " ".join(segs).strip()
        if not part:
            continue
        parts.append(part)
    return "; ".join(parts) if parts else "No education entries provided"


def _concat_sources(*maybe_lists_or_strs: Any) -> List[str]:
    """
    Concatenate multiple source containers into a flattened list of strings.
    Ignores None; flattens lists; filters empty strings.
    """
    out: List[str] = []
    for obj in maybe_lists_or_strs:
        if obj is None:
            continue
        if isinstance(obj, str):
            if obj.strip():
                out.append(obj.strip())
        elif isinstance(obj, list):
            for x in obj:
                if isinstance(x, str) and x.strip():
                    out.append(x.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_one_dean(
    evaluator: Evaluator,
    parent_node,
    dean: DeanItem,
    index_zero_based: int,
) -> None:
    """
    Build verification sub-tree for a single dean.
    """
    idx = index_zero_based + 1
    dean_node = evaluator.add_parallel(
        id=f"Dean_{idx}",
        desc=f"{['First','Second','Third'][index_zero_based]} business school dean meeting all specified criteria",
        parent=parent_node,
        critical=False,  # Allow partial credit per dean
    )

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(dean.business_school) and bool(dean.university),
        id=f"School_and_University_Name_{idx}",
        desc=f"The name of the business school and university are provided",
        parent=dean_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(dean.dean_name),
        id=f"Dean_Full_Name_{idx}",
        desc=f"The dean's full name is provided",
        parent=dean_node,
        critical=True
    )

    # Helpful existence gate for URL (not in rubric but prevents meaningless URL verifications)
    bio_url_present = evaluator.add_custom_node(
        result=bool(dean.official_bio_url),
        id=f"Bio_URL_Provided_{idx}",
        desc=f"Official biography URL is provided for dean #{idx}",
        parent=dean_node,
        critical=True
    )

    # Official source verification (critical)
    official_source_leaf = evaluator.add_leaf(
        id=f"Official_Source_{idx}",
        desc=(
            "Information is verified through official university website, business school faculty directory, "
            "or credible biographical source with URL provided"
        ),
        parent=dean_node,
        critical=True
    )
    official_claim = (
        f"This page is an official or credible biography/profile page for {dean.dean_name or 'the dean'} "
        f"at {dean.business_school or 'the business school'} ({dean.university or 'the university'})."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_source_leaf,
        sources=dean.official_bio_url,
        additional_instruction=(
            "Accept if the URL is clearly an official business school/university page (often on a .edu or official "
            "school domain) or a widely recognized credible biography page (e.g., Poets & Quants, major reputable "
            "media). The page should present a faculty/dean bio with roles and background."
        ),
    )

    # Top-ranked business school verification (critical)
    top_school_leaf = evaluator.add_leaf(
        id=f"Top_School_Verification_{idx}",
        desc=(
            "The business school appears in at least one recognized global or U.S. MBA ranking (QS, U.S. News, "
            "Financial Times, THE, or Poets & Quants)"
        ),
        parent=dean_node,
        critical=True
    )
    ranking_claim = (
        f"The business school '{dean.business_school or 'the business school'}' appears in at least one "
        "recognized MBA ranking (QS, U.S. News & World Report, Financial Times, Times Higher Education, or Poets & Quants)."
    )
    ranking_sources = _concat_sources(dean.ranking_urls, dean.official_bio_url, dean.additional_urls)
    await evaluator.verify(
        claim=ranking_claim,
        node=top_school_leaf,
        sources=ranking_sources,
        additional_instruction=(
            "Prefer evidence directly from the ranking sites (QS, U.S. News & World Report, Financial Times, "
            "Times Higher Education, Poets & Quants). Alternatively, accept the official business school/university "
            "page if it explicitly cites any of these rankings. If none of the provided URLs substantiate this, mark as not supported."
        ),
    )

    # Current dean status as of March 2026 (critical)
    current_dean_leaf = evaluator.add_leaf(
        id=f"Current_Dean_Status_{idx}",
        desc=f"The identified individual is the current dean of the business school as of March 2026",
        parent=dean_node,
        critical=True
    )
    dean_title_claim = (
        f"As of March 2026, {dean.dean_name or 'the named person'} is the current dean "
        f"of {dean.business_school or 'the business school'} at {dean.university or 'the university'}."
    )
    dean_title_sources = _concat_sources(dean.official_bio_url, dean.additional_urls)
    await evaluator.verify(
        claim=dean_title_claim,
        node=current_dean_leaf,
        sources=dean_title_sources,
        additional_instruction=(
            "On the provided page(s), look for clear indications that the person currently holds the Dean (or Interim/Acting Dean) "
            "role as of 2026 (present-tense language, no replacement announcement). If the page indicates a past dean role or a "
            "successor has been appointed before March 2026, do not accept."
        ),
    )

    # Doctoral degree specified type existence (critical)
    evaluator.add_custom_node(
        result=bool(dean.doctoral_degree_type),
        id=f"Degree_Type_Specified_{idx}",
        desc="The type of doctoral degree is explicitly specified (e.g., PhD, Ed.D.)",
        parent=dean_node,
        critical=True
    )

    # Doctoral degree verification (critical)
    doctoral_leaf = evaluator.add_leaf(
        id=f"Doctoral_Degree_{idx}",
        desc="The dean holds a doctoral degree (PhD, Ed.D., or equivalent terminal degree)",
        parent=dean_node,
        critical=True
    )
    degree_txt = dean.doctoral_degree_type or "a doctoral degree"
    doctoral_claim = (
        f"The page states that {dean.dean_name or 'the dean'} holds {degree_txt}, which is a doctoral degree "
        f"(e.g., PhD/DPhil, Ed.D., DBA, ScD/DSc, DrPH, EngD, or similarly recognized research/academic doctorate)."
    )
    doctoral_sources = _concat_sources(dean.official_bio_url, dean.additional_urls)
    await evaluator.verify(
        claim=doctoral_claim,
        node=doctoral_leaf,
        sources=doctoral_sources,
        additional_instruction=(
            "Accept only if the page explicitly mentions a doctoral degree (e.g., PhD/DPhil, Ed.D., DBA, ScD/DSc, DrPH, EngD, "
            "DMan/DMgt). Do NOT accept professional-only degrees such as JD, MD, DO, DDS/DMD as satisfying this requirement "
            "unless the page also lists a separate doctoral research/academic degree."
        ),
    )

    # Complete educational background verification (critical)
    edu_leaf = evaluator.add_leaf(
        id=f"Educational_Background_{idx}",
        desc="Complete educational background is provided including all degrees (undergraduate and graduate) with granting institutions",
        parent=dean_node,
        critical=True
    )
    edu_str = _format_education_for_claim(dean.education)
    edu_claim = (
        f"The official/credible biography page(s) list the following education for {dean.dean_name or 'the dean'}: {edu_str}. "
        "The provided list in the answer aligns with and fully covers all degrees shown on the page(s) (order/formatting differences are acceptable)."
    )
    edu_sources = _concat_sources(dean.official_bio_url, dean.additional_urls)
    await evaluator.verify(
        claim=edu_claim,
        node=edu_leaf,
        sources=edu_sources,
        additional_instruction=(
            "Verify that each degree (with institution) mentioned in the claim appears on the page(s). "
            "If the page shows additional degrees not included in the claim, treat the claim as not fully supported. "
            "Minor formatting/ordering differences are acceptable; focus on coverage and correctness."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the 'top_bschool_deans_2026' task and return the verification summary.
    """
    # Initialize evaluator (root as parallel to allow partial credit across deans)
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

    # Extract structured deans info
    extracted: DeansExtraction = await evaluator.extract(
        prompt=prompt_extract_deans(),
        template_class=DeansExtraction,
        extraction_name="deans_extraction",
    )

    # Add a top-level task node (non-critical to avoid critical-child constraint and allow partial credit)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify three current deans of top-ranked business schools who hold doctoral degrees, and provide their complete educational backgrounds with official source verification",
        parent=root,
        critical=False
    )

    # Normalize to exactly 3 slots
    deans = list(extracted.deans or [])
    while len(deans) < 3:
        deans.append(DeanItem())  # pad with empty items
    if len(deans) > 3:
        deans = deans[:3]

    evaluator.add_custom_info(
        info={"extracted_deans_count": len(extracted.deans) if extracted.deans else 0},
        info_type="extraction_stats",
        info_name="extraction_stats"
    )

    # Verify each dean subtree
    for i in range(3):
        await verify_one_dean(evaluator, task_node, deans[i], i)

    # Final summary
    return evaluator.get_summary()