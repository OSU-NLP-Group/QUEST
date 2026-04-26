import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "r1_2025_universities"
TASK_DESCRIPTION = """
Identify four universities that were newly designated as R1 (Research 1: Very High Spending and Doctorate Production) institutions in the 2025 Carnegie Classification update, which was released in February 2025. For each of the four universities, provide the following information:

1. Official University Name: The complete, official name of the university
2. State Location: The state where the university's main campus is located
3. R1 Designation Announcement: A direct link to an official university news article, press release, or statement that confirms the university received the R1 designation in the 2025 Carnegie Classification update
4. Announcement Date: The date when the R1 designation announcement or news release was published
5. Quote from Announcement: A direct quote from the official announcement that explicitly mentions the R1 designation, Carnegie Classification, or the university's achievement of Research 1 status
6. Research Webpage: A link to the university's official research office, graduate school, or research programs webpage

Ensure that all four universities you identify are distinct institutions that genuinely received R1 designation for the first time in the 2025 Carnegie Classification update (not universities that previously held R1 status).
""".strip()


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    official_name: Optional[str] = None
    state: Optional[str] = None
    r1_announcement_url: Optional[str] = None
    announcement_date: Optional[str] = None
    designation_quote: Optional[str] = None
    research_webpage_url: Optional[str] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) universities from the answer that are claimed to have been newly designated as R1 in the 2025 Carnegie Classification update (released in February 2025).
    For each university, extract the following fields exactly as presented in the answer:
    - official_name: The complete, official university name.
    - state: The U.S. state for the university’s main campus.
    - r1_announcement_url: A direct URL to an official university-domain announcement/news/press page confirming the R1 designation in the 2025 update.
    - announcement_date: The publication date of the announcement/news page (string as shown in the answer; do not normalize).
    - designation_quote: A direct quote from the announcement that explicitly mentions “R1”, “Carnegie Classification”, “Research 1”, or equivalent. Keep punctuation and quotes as provided in the answer.
    - research_webpage_url: A direct URL to an official research office, graduate school, or research programs page on the university’s official domain.

    Rules:
    - Collect the first four universities if more than four are present.
    - If any field is missing for a university, set that field to null.
    - For URLs, extract the actual full URLs shown in the answer (including http/https). If a URL is missing the protocol, prepend http://.
    - Do not invent or infer data not present in the answer.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def ordinal_word(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def has_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def normalize_university_name(name: Optional[str]) -> str:
    if not name:
        return ""
    n = name.strip().lower()
    if n.startswith("the "):
        n = n[4:]
    # collapse multiple spaces
    n = " ".join(n.split())
    return n


# --------------------------------------------------------------------------- #
# Verification per university                                                 #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    """
    Build and evaluate verification leaves for a single university.
    """
    # Create a parallel node for this university (non-critical to allow partial credit per university)
    uni_node = evaluator.add_parallel(
        id=f"university_{index+1}",
        desc=f"{ordinal_word(index+1)} newly designated R1 university with all required information",
        parent=parent_node,
        critical=False
    )

    # Prepare fields
    name = (uni.official_name or "").strip()
    state = (uni.state or "").strip()
    ann_url = (uni.r1_announcement_url or "").strip() or None
    ann_date = (uni.announcement_date or "").strip()
    quote = (uni.designation_quote or "").strip()
    research_url = (uni.research_webpage_url or "").strip() or None

    # Create leaf nodes according to rubric (all critical under this university)
    # 1) Newly designated R1 in 2025 update (first time)
    node_new_r1 = evaluator.add_leaf(
        id=f"university_{index+1}_newly_designated_r1",
        desc="Verify the university received R1 designation for the first time in the 2025 Carnegie Classification update (not previously R1)",
        parent=uni_node,
        critical=True
    )

    # 2) Official university name provided and correct per official pages
    node_name = evaluator.add_leaf(
        id=f"university_{index+1}_university_name",
        desc="Provide the official university name",
        parent=uni_node,
        critical=True
    )

    # 3) State location provided and correct
    node_state = evaluator.add_leaf(
        id=f"university_{index+1}_state_location",
        desc="Provide the state where the main campus is located",
        parent=uni_node,
        critical=True
    )

    # 4) R1 announcement link is an official university page confirming the designation
    node_announce_ref = evaluator.add_leaf(
        id=f"university_{index+1}_r1_announcement_reference",
        desc="Provide a direct link to an official university news article, press release, or statement from an official university domain confirming the R1 designation",
        parent=uni_node,
        critical=True
    )

    # 5) Announcement date correct
    node_announce_date = evaluator.add_leaf(
        id=f"university_{index+1}_announcement_date",
        desc="Provide the date of the R1 designation announcement or news release",
        parent=uni_node,
        critical=True
    )

    # 6) Quote present and mentions R1/Carnegie
    node_quote = evaluator.add_leaf(
        id=f"university_{index+1}_designation_quote",
        desc="Provide a direct quote from the announcement that explicitly mentions the R1 designation or Carnegie Classification",
        parent=uni_node,
        critical=True
    )

    # 7) Research webpage is official research/graduate/programs page
    node_research = evaluator.add_leaf(
        id=f"university_{index+1}_research_webpage",
        desc="Provide a link from an official university domain to the university's official research office, graduate school, or research programs webpage",
        parent=uni_node,
        critical=True
    )

    # Collect claims for batch verification
    claims: List[tuple[str, Optional[str] | List[str], Any, Optional[str]]] = []

    # newly_designated_r1: requires announcement URL evidence
    if has_url(ann_url):
        claim_new = (
            f"This official announcement page explicitly states that "
            f"{name if name else 'the university'} received R1 (Research 1: Very High Spending and Doctorate Production "
            f"or equivalent 'R1: Very high research activity') designation in the 2025 Carnegie Classification update, "
            f"and that this is the first time the university achieved R1 status."
        )
        add_ins_new = (
            "Accept equivalent phrasing such as 'elevated to R1', 'newly classified as R1', 'achieved R1 status', "
            "'joined the R1 ranks', etc. The page should reference the 2025 Carnegie Classification update "
            "(often published in February 2025). If the page clearly indicates the institution moved into R1 in 2025, "
            "treat that as 'first time' unless the page explicitly says it previously held R1 status."
        )
        claims.append((claim_new, ann_url, node_new_r1, add_ins_new))
    else:
        node_new_r1.score = 0.0
        node_new_r1.status = "failed"

    # university_name: verify with announcement and/or research page
    name_sources: List[str] = []
    if has_url(ann_url):
        name_sources.append(ann_url)  # type: ignore[arg-type]
    if has_url(research_url):
        name_sources.append(research_url)  # type: ignore[arg-type]
    if name and name_sources:
        claim_name = f"The official university name is '{name}'."
        add_ins_name = (
            "Confirm the official name as shown on the university's own pages. "
            "Allow minor variations such as the optional 'The' prefix, but the core official name should match."
        )
        claims.append((claim_name, name_sources, node_name, add_ins_name))
    else:
        node_name.score = 0.0
        node_name.status = "failed"

    # state_location: verify with announcement and/or research page
    if state and name_sources:
        claim_state = (
            f"The university's main campus is located in the U.S. state of {state}."
        )
        add_ins_state = (
            "Use address/location details on the official pages to confirm the state. "
            "If only the city is shown, use common knowledge to determine the state only if the page makes it clear "
            "(e.g., 'Tucson, AZ' implies Arizona). Minor formatting variants or abbreviations are acceptable."
        )
        claims.append((claim_state, name_sources, node_state, add_ins_state))
    else:
        node_state.score = 0.0
        node_state.status = "failed"

    # r1_announcement_reference: requires announcement URL
    if has_url(ann_url):
        claim_ref = (
            "This URL belongs to an official university domain (e.g., *.edu or official subdomain) and is a news/press/"
            "announcement page that confirms the university received the R1 designation in the 2025 Carnegie Classification update."
        )
        add_ins_ref = (
            "Check that the domain is clearly owned by the university (e.g., *.edu or official university subdomain). "
            "The page content should explicitly mention the R1 designation or the 2025 Carnegie Classification update."
        )
        claims.append((claim_ref, ann_url, node_announce_ref, add_ins_ref))
    else:
        node_announce_ref.score = 0.0
        node_announce_ref.status = "failed"

    # announcement_date: requires both date and announcement URL
    if ann_date and has_url(ann_url):
        claim_date = (
            f"The announcement page was published on {ann_date}."
        )
        add_ins_date = (
            "Use the posted/published date displayed on the page. If both 'posted' and 'updated' dates appear, "
            "use the posted/published date. Accept reasonable date formatting differences."
        )
        claims.append((claim_date, ann_url, node_announce_date, add_ins_date))
    else:
        node_announce_date.score = 0.0
        node_announce_date.status = "failed"

    # designation_quote: requires quote and announcement URL
    if quote and has_url(ann_url):
        claim_quote = (
            f"The announcement page contains the following quotation (allowing minor punctuation/whitespace differences): \"{quote}\" "
            f"and the quoted text explicitly references either 'R1', 'Research 1', 'Carnegie Classification', or 'Very high research activity'."
        )
        add_ins_quote = (
            "Verify that the page includes this sentence or a very close variant. Minor punctuation or whitespace differences are acceptable. "
            "The quotation must unambiguously reference the R1 designation or the Carnegie Classification."
        )
        claims.append((claim_quote, ann_url, node_quote, add_ins_quote))
    else:
        node_quote.score = 0.0
        node_quote.status = "failed"

    # research_webpage: requires research URL
    if has_url(research_url):
        claim_research = (
            "This URL is on an official university-owned domain and is a page for the university's research office, "
            "graduate school, or research programs."
        )
        add_ins_research = (
            "Confirm that the page clearly identifies itself as the research office, graduate school, or official research programs page. "
            "The domain should be an official university domain (e.g., *.edu or recognized official subdomain)."
        )
        claims.append((claim_research, research_url, node_research, add_ins_research))
    else:
        node_research.score = 0.0
        node_research.status = "failed"

    # Run verifications in parallel for those with sources prepared
    if claims:
        await evaluator.batch_verify(claims)


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
    Evaluate an answer for the 2025 Carnegie R1 newly designated universities task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    # Note: The provided JSON marks the root as critical with non-critical children, which violates the
    # framework constraint that all children of a critical node must also be critical.
    # Here we initialize the root as NON-CRITICAL (default) to allow partial credit across universities.
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Parallel across universities
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

    # Extract up to 4 universities
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Select first four; if fewer, pad with empty entries
    selected: List[UniversityItem] = list(extracted.universities[:4])
    while len(selected) < 4:
        selected.append(UniversityItem())

    # Add a distinctness check at root (critical) to enforce "four distinct universities"
    normalized_names = [normalize_university_name(u.official_name) for u in selected]
    unique_nonempty = set([n for n in normalized_names if n])
    distinct_ok = len(unique_nonempty) == 4
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinct_universities",
        desc="Four universities are distinct (no duplicates among the four).",
        parent=root,
        critical=True
    )

    # Build per-university verification nodes
    for i in range(4):
        await verify_university(evaluator, root, selected[i], i)

    # Return structured result
    return evaluator.get_summary()