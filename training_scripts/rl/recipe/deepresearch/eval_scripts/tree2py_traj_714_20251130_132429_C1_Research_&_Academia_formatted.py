import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "uneswa_scholar_profile"
TASK_DESCRIPTION = (
    "Identify one current faculty member at the University of Eswatini who has a publicly accessible Google Scholar profile. "
    "Provide the following information: (1) The direct URL to their Google Scholar profile, (2) The current h-index displayed on their profile, "
    "and (3) The current i10-index displayed on their profile."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacultyProfileExtraction(BaseModel):
    faculty_name: Optional[str] = None
    scholar_url: Optional[str] = None
    affiliation_text: Optional[str] = None
    affiliation_sources: List[str] = Field(default_factory=list)
    h_index: Optional[str] = None
    i10_index: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_faculty_profile() -> str:
    return """
    Extract exactly the information the answer provides about a single University of Eswatini (UNESWA) faculty member with a Google Scholar profile.

    Return the following fields:
    - faculty_name: The researcher's full name, exactly as written in the answer text.
    - scholar_url: The direct Google Scholar profile URL explicitly listed in the answer. It must be a profile link of the form https://scholar.google.com/citations?user=... (or similar scholar.google.* domain with '/citations' and a 'user=' parameter). If the answer does not include such a URL explicitly, return null.
    - affiliation_text: Any explicit textual claim in the answer that indicates they are a current faculty member at the University of Eswatini (UNESWA). If not stated, return null.
    - affiliation_sources: All URLs explicitly included in the answer that support their University of Eswatini affiliation (e.g., department page, staff directory, personal page at UNESWA). If none are provided, return an empty list.
    - h_index: The current h-index value stated in the answer text (prefer the 'All' column value if the answer distinguishes). If the answer does not provide a number, return null.
    - i10_index: The current i10-index value stated in the answer text (prefer the 'All' column value if the answer distinguishes). If the answer does not provide a number, return null.

    IMPORTANT:
    - Only extract URLs and values explicitly present in the answer text.
    - Do not invent URLs or numbers.
    - If the answer lists multiple candidates, extract the first one only.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_valid_scholar_profile_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    return ("scholar.google." in u) and ("/citations" in u) and ("user=" in u)


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if not u2:
            continue
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_identify_phase(
    evaluator: Evaluator,
    parent_node,
    extracted: FacultyProfileExtraction,
):
    """
    Build and verify the 'identify_eligible_faculty_and_profile' group:
    - current faculty affiliation at University of Eswatini
    - public Google Scholar profile exists (accessible)
    - valid Scholar profile URL format
    """
    identify_node = evaluator.add_parallel(
        id="identify_eligible_faculty_and_profile",
        desc="Identify an eligible researcher and their Google Scholar profile",
        parent=parent_node,
        critical=True,
    )

    # 1) Valid Scholar profile URL (format check only; direct custom result)
    valid_url = is_valid_scholar_profile_url(extracted.scholar_url)
    valid_url_node = evaluator.add_custom_node(
        result=valid_url,
        id="valid_google_scholar_profile_url",
        desc="Provide the direct Google Scholar profile URL in the format https://scholar.google.com/citations?user=...",
        parent=identify_node,
        critical=True,
    )

    # 2) Public, accessible Google Scholar profile page (content/evidence check)
    public_profile_node = evaluator.add_leaf(
        id="public_google_scholar_profile_exists",
        desc="Researcher has a publicly accessible Google Scholar profile",
        parent=identify_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This URL is a publicly accessible Google Scholar profile page for an individual researcher (not a search page or error page), and it displays the Citations metrics table (including h-index and i10-index).",
        node=public_profile_node,
        sources=extracted.scholar_url,
        additional_instruction=(
            "Confirm that the page is a Google Scholar profile page (with a profile name and the metrics table). "
            "It must be accessible without login, i.e., a normal public profile."
        ),
        extra_prerequisites=[valid_url_node],
    )

    # 3) Current faculty affiliation check (University of Eswatini)
    current_aff_node = evaluator.add_leaf(
        id="current_faculty_affiliation",
        desc="Researcher is a current faculty member at the University of Eswatini",
        parent=identify_node,
        critical=True,
    )
    name_part = f"{extracted.faculty_name} " if extracted.faculty_name else ""
    claim_aff = f"{name_part}is a current faculty member at the University of Eswatini."
    sources = dedup_urls(([extracted.scholar_url] if extracted.scholar_url else []) + (extracted.affiliation_sources or []))
    await evaluator.verify(
        claim=claim_aff,
        node=current_aff_node,
        sources=sources,
        additional_instruction=(
            "Accept 'University of Eswatini' synonyms or legacy names: 'UNESWA', and the former name 'University of Swaziland' (UNISWA). "
            "Faculty includes roles such as Lecturer, Senior Lecturer, Professor, etc. "
            "Evidence may come from the Scholar profile affiliation line and/or an official department/staff directory page. "
            "If sources clearly indicate current employment at University of Eswatini, mark as supported."
        ),
        extra_prerequisites=[valid_url_node],
    )

    return {
        "identify_node": identify_node,
        "valid_url_node": valid_url_node,
        "public_profile_node": public_profile_node,
        "current_aff_node": current_aff_node,
    }


async def verify_metrics_phase(
    evaluator: Evaluator,
    parent_node,
    extracted: FacultyProfileExtraction,
    deps: List,
):
    """
    Build and verify the 'extract_required_metrics' group:
    - h-index (All) equals the value provided in the answer
    - i10-index (All) equals the value provided in the answer
    """
    metrics_node = evaluator.add_parallel(
        id="extract_required_metrics",
        desc="Extract the required citation metrics from the provided Google Scholar profile",
        parent=parent_node,
        critical=True,
    )

    # h-index metric
    if extracted.h_index and str(extracted.h_index).strip():
        h_node = evaluator.add_leaf(
            id="h_index_metric",
            desc="Provide the current h-index value displayed on the Google Scholar profile",
            parent=metrics_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"On the Google Scholar profile page, the h-index (All) displayed equals {extracted.h_index}.",
            node=h_node,
            sources=extracted.scholar_url,
            additional_instruction=(
                "Check the metrics table on the profile page. Use the 'All' column for h-index. "
                "Allow minor formatting differences (e.g., spacing). "
                "Do not use the 'Since' column; match the 'All' value."
            ),
            extra_prerequisites=deps,
        )
    else:
        # Missing value in the answer -> fail this critical leaf
        evaluator.add_leaf(
            id="h_index_metric",
            desc="Provide the current h-index value displayed on the Google Scholar profile",
            parent=metrics_node,
            critical=True,
            score=0.0,
            status="failed",
        )

    # i10-index metric
    if extracted.i10_index and str(extracted.i10_index).strip():
        i10_node = evaluator.add_leaf(
            id="i10_index_metric",
            desc="Provide the current i10-index value displayed on the Google Scholar profile",
            parent=metrics_node,
            critical=True,
        )
        await evaluator.verify(
            claim=f"On the Google Scholar profile page, the i10-index (All) displayed equals {extracted.i10_index}.",
            node=i10_node,
            sources=extracted.scholar_url,
            additional_instruction=(
                "Check the metrics table on the profile page. Use the 'All' column for i10-index. "
                "Allow minor formatting differences (e.g., spacing). "
                "Do not use the 'Since' column; match the 'All' value."
            ),
            extra_prerequisites=deps,
        )
    else:
        # Missing value in the answer -> fail this critical leaf
        evaluator.add_leaf(
            id="i10_index_metric",
            desc="Provide the current i10-index value displayed on the Google Scholar profile",
            parent=metrics_node,
            critical=True,
            score=0.0,
            status="failed",
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
    Evaluate an answer for the UNESWA Google Scholar profile task.
    """
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

    # Extract structured data from the agent's answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_faculty_profile(),
        template_class=FacultyProfileExtraction,
        extraction_name="faculty_profile_extraction",
    )

    # Phase 1: Identify eligible faculty and profile
    identify_results = await verify_identify_phase(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted,
    )

    # Phase 2: Extract and verify required metrics (depends on profile validity/access)
    deps = [identify_results["valid_url_node"], identify_results["public_profile_node"]]
    await verify_metrics_phase(
        evaluator=evaluator,
        parent_node=root,
        extracted=extracted,
        deps=deps,
    )

    return evaluator.get_summary()