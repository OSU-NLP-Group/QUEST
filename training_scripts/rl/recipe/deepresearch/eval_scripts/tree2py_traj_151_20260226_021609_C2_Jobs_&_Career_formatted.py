import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "mi_elem_teacher_positions"
TASK_DESCRIPTION = (
    "You are seeking elementary teaching positions in Michigan. Find TWO currently open elementary teaching positions "
    "in two different Michigan school districts. For each position, provide: (1) the job title and school district name, "
    "(2) the direct URL to apply for that specific position, and (3) one specific employee benefit or support program "
    "mentioned in the job posting or on the district's employment/benefits website."
)


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class PositionItem(BaseModel):
    job_title: Optional[str] = None
    district_name: Optional[str] = None
    posting_url: Optional[str] = None
    apply_url: Optional[str] = None
    benefit: Optional[str] = None
    benefit_source_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_positions() -> str:
    return """
    Extract up to five elementary teaching positions (as presented in the answer). For each position, extract the following fields exactly as provided in the answer:
    - job_title: The job title of the position (e.g., "Elementary Teacher", "3rd Grade Teacher").
    - district_name: The Michigan school district name (e.g., "Ann Arbor Public Schools"). If only a school name is given, extract the district/school system name as written in the answer.
    - posting_url: The direct URL to the job posting page (if provided).
    - apply_url: The direct URL to apply for that specific position (if provided; can be an applicant tracking system link).
    - benefit: One specific employee benefit or support program mentioned (e.g., "health insurance", "tuition reimbursement", "new teacher mentoring").
    - benefit_source_url: A URL where that benefit is mentioned (either the job posting page or a district employment/benefits page), if provided.
    
    Rules:
    - Extract only what is explicitly present in the answer text; do not invent missing information.
    - If a field is missing for a position, set it to null.
    - Return a JSON object with an array field `positions`, where each element has the fields listed above.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _build_sources_for_position(pos: PositionItem) -> List[str]:
    urls = []
    for u in [pos.apply_url, pos.posting_url, pos.benefit_source_url]:
        if _nonempty(u):
            urls.append(u.strip())  # type: ignore
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# -----------------------------------------------------------------------------
# Verification logic for a single position
# -----------------------------------------------------------------------------
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    idx: int,
) -> None:
    """
    Build verification subtree for one position:
    - Existence checks (custom critical gates)
    - Role & District verification (leaf, critical)
    - Application URL verification (leaf, critical)
    - Employee Benefit verification (leaf, critical)
    """
    position_human_idx = idx + 1

    # Parent node for this position (non-critical so partial credit across positions is allowed)
    pos_node = evaluator.add_parallel(
        id=f"position_{idx+1}",
        desc=("First elementary teaching position with complete required information"
              if idx == 0 else
              "Second elementary teaching position in a different Michigan district with complete required information"),
        parent=parent_node,
        critical=False
    )

    # Existence gates (custom critical nodes) for clarity and auto-skip behavior
    role_dist_exists = evaluator.add_custom_node(
        result=_nonempty(pos.job_title) and _nonempty(pos.district_name),
        id=f"position_{idx+1}_role_district_provided",
        desc=f"Position #{position_human_idx}: job title and district name are provided",
        parent=pos_node,
        critical=True
    )

    app_url_exists = evaluator.add_custom_node(
        result=_nonempty(pos.apply_url) or _nonempty(pos.posting_url),
        id=f"position_{idx+1}_application_url_provided",
        desc=f"Position #{position_human_idx}: direct apply or job posting URL is provided",
        parent=pos_node,
        critical=True
    )

    benefit_exists = evaluator.add_custom_node(
        result=_nonempty(pos.benefit) and (_nonempty(pos.benefit_source_url) or _nonempty(pos.posting_url)),
        id=f"position_{idx+1}_benefit_provided",
        desc=f"Position #{position_human_idx}: at least one benefit with a source URL is provided",
        parent=pos_node,
        critical=True
    )

    # Prepare sources for verification
    sources = _build_sources_for_position(pos)

    # 1) Role and District verification
    role_dist_leaf = evaluator.add_leaf(
        id=f"Position_{position_human_idx}_Role_and_District",
        desc=("Position is for elementary teaching (grades K-6) in a Michigan school district, "
              "with job title and district name provided")
             if idx == 0 else
             ("Position is for elementary teaching (grades K-6) in a different Michigan school district than Position 1, "
              "with job title and district name provided"),
        parent=pos_node,
        critical=True,
    )
    job_title = pos.job_title or ""
    district_name = pos.district_name or ""
    claim_role_dist = (
        f"The webpage shows a currently open elementary teaching position (K–6 range acceptable, including 'Elementary', "
        f"'K-5', 'K-6', 'K-8', or grade-specific labels like '3rd Grade') in Michigan for the district named '{district_name}', "
        f"and the job title on the page matches or is equivalent to '{job_title}'."
    )
    await evaluator.verify(
        claim=claim_role_dist,
        node=role_dist_leaf,
        sources=sources,
        additional_instruction=(
            "Accept reasonable variants of 'elementary' (e.g., 'K-5', 'K-6', 'K-8', 'Primary', or grade labels such as '2nd Grade'). "
            "Confirm the posting is for a Michigan district (e.g., MI address, Michigan mention, or district known to be in MI). "
            "Also allow small differences in job title phrasing if they are equivalent (e.g., 'Elementary Classroom Teacher'). "
            "Treat the job as 'currently open' if the page shows an active posting or application instructions (e.g., Apply button)."
        ),
    )

    # 2) Application URL verification
    app_leaf = evaluator.add_leaf(
        id=f"Position_{position_human_idx}_Application_URL",
        desc="Direct URL to apply for the specific position or to access the job posting is provided",
        parent=pos_node,
        critical=True,
    )
    claim_app = (
        f"This URL is either a direct apply link or the job posting page for the specific position titled '{job_title}' "
        f"in the '{district_name}' school district."
    )
    # Prefer using both apply_url and posting_url if available
    app_sources = []
    if _nonempty(pos.apply_url):
        app_sources.append(pos.apply_url.strip())  # type: ignore
    if _nonempty(pos.posting_url):
        app_sources.append(pos.posting_url.strip())  # type: ignore

    await evaluator.verify(
        claim=claim_app,
        node=app_leaf,
        sources=app_sources if app_sources else None,
        additional_instruction=(
            "Consider typical applicant tracking systems (e.g., Frontline/AppliTrack/NEOGOV) pages with an Apply/Sign In button "
            "as valid apply links. The page should clearly correspond to this position (matching title or unique posting ID). "
            "If it's the posting page, it must provide a way to apply or clear application instructions."
        ),
    )

    # 3) Employee benefit verification
    benefit_leaf = evaluator.add_leaf(
        id=f"Position_{position_human_idx}_Employee_Benefit",
        desc=("At least one specific employee benefit or support program is identified from the job posting or district "
              "employment/benefits website"),
        parent=pos_node,
        critical=True,
    )
    benefit_txt = pos.benefit or ""
    # Verify via benefit_source_url first; fall back to posting_url if needed
    benefit_sources = []
    if _nonempty(pos.benefit_source_url):
        benefit_sources.append(pos.benefit_source_url.strip())  # type: ignore
    if _nonempty(pos.posting_url):
        benefit_sources.append(pos.posting_url.strip())  # type: ignore

    claim_benefit = (
        f"The page explicitly mentions the employee benefit or support program '{benefit_txt}' for employees of "
        f"the {district_name} school district or for this position."
    )
    await evaluator.verify(
        claim=claim_benefit,
        node=benefit_leaf,
        sources=benefit_sources if benefit_sources else (sources if sources else None),
        additional_instruction=(
            "Accept common benefit phrasing variants (e.g., 'medical/health insurance', 'dental/vision', 'retirement', "
            "'professional development', 'tuition reimbursement', 'mentoring/induction for new teachers'). "
            "The benefit must be specifically stated on the cited page."
        ),
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
    Entry point used by the evaluation framework.
    """
    evaluator = Evaluator()

    # IMPORTANT: We set the root as non-critical PARALLEL to allow partial credit across the two positions.
    # The provided JSON marks the root as critical, but that would require all children to be critical (by framework constraint)
    # and would also disallow partial credit. Adjusted here for robust evaluation while preserving leaf criticality.
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

    # Extract structured positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Keep first 2 positions; pad if fewer
    positions: List[PositionItem] = list(extracted.positions[:2])
    while len(positions) < 2:
        positions.append(PositionItem())

    # Verify Position 1
    await verify_position(evaluator, root, positions[0], idx=0)

    # Verify Position 2
    await verify_position(evaluator, root, positions[1], idx=1)

    # Additional check: Position 2 must be in a different district than Position 1
    # Make it a critical check under Position 2's node (so Position 2 fails if same district).
    pos2_parent = evaluator.find_node("position_2")
    if pos2_parent is None:
        # Fallback: if ID got deduplicated, find by desc
        for node_id in evaluator.get_all_node_ids():
            node = evaluator.find_node(node_id)
            if node and node.desc.startswith("Second elementary teaching position"):
                pos2_parent = node
                break

    d1 = (positions[0].district_name or "").strip()
    d2 = (positions[1].district_name or "").strip()

    # We use a custom node (binary) for the "different districts" logical check.
    evaluator.add_custom_node(
        result=bool(d1) and bool(d2) and (d1.lower() != d2.lower()),
        id="Position_2_Different_District",
        desc=f"Position 2 district '{d2 or 'N/A'}' is different from Position 1 district '{d1 or 'N/A'}'",
        parent=pos2_parent or root,
        critical=True,
    )

    # Record some helpful custom info
    evaluator.add_custom_info(
        info={
            "positions_extracted_count": len(extracted.positions),
            "position_1_sources": _build_sources_for_position(positions[0]),
            "position_2_sources": _build_sources_for_position(positions[1]),
        },
        info_type="debug",
        info_name="extraction_and_sources_overview",
    )

    return evaluator.get_summary()