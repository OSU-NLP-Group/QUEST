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
TASK_ID = "ohio_ce_professions"
TASK_DESCRIPTION = (
    "For career planning purposes, identify 4 different licensed professions in Ohio that require continuing "
    "education for license renewal. For each profession, provide: (1) the profession name, (2) the total continuing "
    "education hours required per renewal period, (3) the specific ethics or professional conduct hours required (if any), "
    "and (4) the renewal period in years. Each profession must be distinct and selected from regulated professions such as "
    "healthcare, legal, accounting, engineering, teaching, or social work. Provide official source URLs for each profession's requirements."
)

ALLOWED_FIELDS = {
    "healthcare",
    "legal",
    "accounting",
    "engineering",
    "teaching",
    "social work"
}


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ProfessionItem(BaseModel):
    profession_name: Optional[str] = None
    category: Optional[str] = None  # Expected to be one of ALLOWED_FIELDS
    ce_hours_total: Optional[str] = None  # Keep as string; can be "30", "24 per biennium", etc.
    ethics_hours: Optional[str] = None  # String, e.g., "3", "N/A", "None", "Not required"; null if not stated
    renewal_period_years: Optional[str] = None  # String like "2", "3", "biennial (2)"
    source_urls: List[str] = Field(default_factory=list)


class ProfessionsExtraction(BaseModel):
    professions: List[ProfessionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_professions() -> str:
    return """
    Extract all professions mentioned in the answer that correspond to Ohio state-licensed professions requiring continuing education (CE) for license renewal.
    For each profession, extract the following fields:
    - profession_name: The explicit profession name (e.g., "Registered Nurse", "Attorney", "CPA", "Professional Engineer", "Teacher", "Social Worker").
    - category: Map the profession into ONE of the allowed categories exactly as one of:
        ["healthcare", "legal", "accounting", "engineering", "teaching", "social work"].
      If you are unsure or the profession does not fit any of these, return null.
    - ce_hours_total: The total CE hours required per renewal period, exactly as stated in the answer (can be numeric or textual, e.g., "30", "24 per biennium").
    - ethics_hours: The specific ethics or professional conduct hours required (if any), exactly as stated in the answer.
      If the answer explicitly states that ethics/professional conduct is "none", "not required", or "N/A", set this field to one of those strings.
      If the answer does not mention ethics or professional conduct at all, return null.
    - renewal_period_years: The renewal period length in years, preferably as a number string (e.g., "2", "3").
      If the answer uses terms like "biennial" or "annual", convert to a number string when possible (e.g., "biennial" -> "2", "annual" -> "1"), else include the textual value.
    - source_urls: A list of URLs provided in the answer that support the profession’s CE requirements. Extract only valid URLs that are explicitly present. Prefer official Ohio regulator/board URLs if provided.

    Return a JSON object with:
    {
      "professions": [
        { profession fields ... },
        ...
      ]
    }

    IMPORTANT:
    - Do NOT invent any values. If any field is not present in the answer, return null for that field.
    - Extract all professions mentioned; do not filter. The evaluator will select the first 4 later.
    - For source_urls, extract only URLs explicitly given in the answer text (plain or markdown).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch.isspace())


def ethics_is_na(value: Optional[str]) -> bool:
    if not value:
        return False
    v = value.strip().lower()
    return v in {"n/a", "na", "none", "not applicable", "not required", "no ethics requirement"}


def sources_provided(item: ProfessionItem) -> bool:
    return bool(item.source_urls and len(item.source_urls) > 0)


def category_is_allowed(category: Optional[str]) -> bool:
    if not category:
        return False
    return category.strip().lower() in ALLOWED_FIELDS


# --------------------------------------------------------------------------- #
# Verification logic per profession                                           #
# --------------------------------------------------------------------------- #
async def verify_profession(
    evaluator: Evaluator,
    parent_node,
    item: ProfessionItem,
    idx: int,
) -> None:
    """
    Build and verify nodes for a single profession.
    """
    pnum = idx + 1
    pnode = evaluator.add_parallel(
        id=f"profession_{pnum}",
        desc=f"Profession {pnum} (one Ohio-licensed profession requiring CE)",
        parent=parent_node,
        critical=False  # Allow partial credit across professions
    )

    # Existence: profession name provided
    name_exists_node = evaluator.add_custom_node(
        result=bool(item.profession_name and item.profession_name.strip()),
        id=f"p{pnum}_profession_name",
        desc="Profession name is provided",
        parent=pnode,
        critical=True
    )

    # Existence: at least one source URL provided (useful prerequisite for URL-based verifications)
    sources_exist_node = evaluator.add_custom_node(
        result=sources_provided(item),
        id=f"p{pnum}_sources_provided",
        desc="At least one source URL is provided for this profession",
        parent=pnode,
        critical=True
    )

    # Regulated field category membership
    category_node = evaluator.add_custom_node(
        result=category_is_allowed(item.category),
        id=f"p{pnum}_regulated_field_category",
        desc="Profession falls within the regulated fields listed in the prompt (e.g., healthcare, legal, accounting, engineering, teaching, or social work)",
        parent=pnode,
        critical=True
    )

    # Ohio licensed and CE required (verify via sources)
    ohio_ce_node = evaluator.add_leaf(
        id=f"p{pnum}_ohio_licensed_and_ce_required",
        desc="Profession is state-licensed in Ohio and requires continuing education for license renewal",
        parent=pnode,
        critical=True
    )
    claim_ohio_ce = (
        f"The profession '{item.profession_name or ''}' is licensed by the State of Ohio and requires continuing "
        f"education for license renewal."
    )
    await evaluator.verify(
        claim=claim_ohio_ce,
        node=ohio_ce_node,
        sources=item.source_urls,
        additional_instruction=(
            "Check the provided URL(s) to confirm that this profession is a state-licensed profession in Ohio and that "
            "continuing education is required for license renewal. Treat official Ohio boards/regulators (e.g., "
            "ohio.gov domains, Ohio licensing boards) as authoritative. If the pages are clearly unrelated or do not "
            "mention CE requirements, mark as not supported."
        ),
        extra_prerequisites=[sources_exist_node]
    )

    # Total CE hours (verify via sources)
    ce_hours_node = evaluator.add_leaf(
        id=f"p{pnum}_total_ce_hours",
        desc="Total continuing education hours required per renewal period are stated",
        parent=pnode,
        critical=True
    )
    claim_ce_hours = (
        f"The total continuing education hours required per renewal period for '{item.profession_name or ''}' in Ohio "
        f"is '{item.ce_hours_total or ''}'."
    )
    await evaluator.verify(
        claim=claim_ce_hours,
        node=ce_hours_node,
        sources=item.source_urls,
        additional_instruction=(
            "Verify the total CE hours per renewal period as stated. Allow reasonable textual variants like 'biennium' "
            "or 'per 2-year period.' The value should match or be equivalent to what the official/regulatory page states."
        ),
        extra_prerequisites=[sources_exist_node, name_exists_node]
    )

    # Ethics/professional conduct hours (verify via sources; allow 'N/A'/'None'/not required)
    ethics_node = evaluator.add_leaf(
        id=f"p{pnum}_ethics_hours",
        desc="Ethics/professional conduct hours requirement is stated, or explicitly noted as not applicable",
        parent=pnode,
        critical=True
    )
    if ethics_is_na(item.ethics_hours):
        claim_ethics = (
            f"For '{item.profession_name or ''}' in Ohio, there is no specific ethics or professional conduct hours "
            f"requirement for renewal."
        )
    else:
        claim_ethics = (
            f"For '{item.profession_name or ''}' in Ohio, the ethics/professional conduct hours requirement is "
            f"'{item.ethics_hours or ''}'."
        )
    await evaluator.verify(
        claim=claim_ethics,
        node=ethics_node,
        sources=item.source_urls,
        additional_instruction=(
            "Confirm whether a specific ethics/professional conduct hours requirement is present. If the answer indicates "
            "none/N/A/not required, verify that the source supports that there is no specific ethics hours requirement. "
            "If a value is provided, verify that the source shows the same value or an equivalent requirement."
        ),
        extra_prerequisites=[sources_exist_node, name_exists_node]
    )

    # Renewal period in years (verify via sources)
    renewal_node = evaluator.add_leaf(
        id=f"p{pnum}_renewal_period_years",
        desc="Renewal period length is stated in years",
        parent=pnode,
        critical=True
    )
    claim_renewal = (
        f"The renewal period length for '{item.profession_name or ''}' in Ohio is '{item.renewal_period_years or ''}' years."
    )
    await evaluator.verify(
        claim=claim_renewal,
        node=renewal_node,
        sources=item.source_urls,
        additional_instruction=(
            "Verify the renewal period length in years. If the source uses terms like 'biennial' or 'annual', allow equivalence "
            "(biennial=2 years, annual=1 year). The claim should be consistent with the official/regulatory page."
        ),
        extra_prerequisites=[sources_exist_node, name_exists_node]
    )

    # Official sources present and authoritative (verify via URLs)
    official_sources_node = evaluator.add_leaf(
        id=f"p{pnum}_official_sources",
        desc="At least one official Ohio licensing-board/regulator (or similarly authoritative) source URL is provided supporting the stated requirements",
        parent=pnode,
        critical=True
    )
    claim_official = (
        f"The provided source URL(s) for '{item.profession_name or ''}' are official Ohio licensing/regulator pages or "
        f"similarly authoritative sources that explicitly state the CE requirements."
    )
    await evaluator.verify(
        claim=claim_official,
        node=official_sources_node,
        sources=item.source_urls,
        additional_instruction=(
            "Judge whether the URLs appear to be official or authoritative (e.g., ohio.gov domains, Ohio licensing board "
            "pages, Supreme Court of Ohio for attorneys, Accountancy Board of Ohio for CPAs, Ohio Engineering & Surveying "
            "Board for engineers, Ohio Department of Education & Workforce for teachers, State Medical Board of Ohio for healthcare, "
            "Ohio Counselor, Social Worker & Marriage and Family Therapist Board, etc.). The page(s) should clearly state the CE requirements."
        ),
        extra_prerequisites=[sources_exist_node, name_exists_node]
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
    Evaluate an answer for the Ohio CE professions task.
    """
    # Initialize evaluator (root node non-critical to avoid the strict child-critical constraint)
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

    # Extract professions
    extracted: ProfessionsExtraction = await evaluator.extract(
        prompt=prompt_extract_professions(),
        template_class=ProfessionsExtraction,
        extraction_name="professions_extraction"
    )

    # Record allowed categories in summary for transparency
    evaluator.add_custom_info(
        info={"allowed_categories": sorted(list(ALLOWED_FIELDS))},
        info_type="config",
        info_name="allowed_categories"
    )

    # Set-level requirements node (critical)
    set_node = evaluator.add_parallel(
        id="set_level_requirements",
        desc="Set-level requirements for the full list of professions",
        parent=root,
        critical=True
    )

    # Count professions mentioned in the answer (with names)
    num_named = sum(1 for p in extracted.professions if p.profession_name and p.profession_name.strip())

    provides_four_node = evaluator.add_custom_node(
        result=(num_named == 4),
        id="provides_four_professions",
        desc="Response provides exactly 4 professions",
        parent=set_node,
        critical=True
    )

    # Select the first 4 professions (for downstream checks), padding if fewer
    selected: List[ProfessionItem] = []
    for p in extracted.professions[:4]:
        selected.append(p)
    while len(selected) < 4:
        selected.append(ProfessionItem())  # placeholder

    # Distinctness of the four selected professions
    selected_names = [normalize_name(p.profession_name) for p in selected]
    all_distinct = (
        len(selected) == 4 and
        all(n for n in selected_names) and
        len(set(selected_names)) == 4
    )
    distinct_node = evaluator.add_custom_node(
        result=all_distinct,
        id="all_professions_distinct",
        desc="All 4 professions are distinct from one another",
        parent=set_node,
        critical=True
    )

    # Verify each of the four professions
    for idx, item in enumerate(selected):
        await verify_profession(evaluator, root, item, idx)

    # Return evaluation summary
    return evaluator.get_summary()