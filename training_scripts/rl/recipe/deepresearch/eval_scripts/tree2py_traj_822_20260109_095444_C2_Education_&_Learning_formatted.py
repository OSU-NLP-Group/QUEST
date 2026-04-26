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
TASK_ID = "anki_step1_study"
TASK_DESCRIPTION = """
A medical school's curriculum committee is evaluating whether to recommend Anki spaced repetition software to students preparing for USMLE Step 1. To support evidence-based decision-making, find a peer-reviewed research study published in 2023 or later that specifically examined Anki's effectiveness for medical students preparing for this exam. From the study, provide: (1) the comparative performance metrics between students who used Anki and those who did not (such as exam failure rates, pass rates, or scores), (2) the sample sizes for both groups, and (3) the source URL where the study can be accessed.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StudyExtraction(BaseModel):
    """Structured extraction for a single qualified study, as presented in the agent's answer."""
    study_title: Optional[str] = None
    journal_or_venue: Optional[str] = None
    publication_year: Optional[str] = None

    primary_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)

    # Content details required
    intervention_desc: Optional[str] = None            # e.g., "Anki spaced repetition"
    population_desc: Optional[str] = None              # e.g., "medical students preparing for USMLE Step 1"
    metrics_text: Optional[str] = None                 # e.g., "Failure rate: 6% (Anki) vs 14% (non-Anki)"
    sample_size_anki: Optional[str] = None             # e.g., "n=120" or "120"
    sample_size_nonanki: Optional[str] = None          # e.g., "n=98" or "98"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_study_info() -> str:
    return """
    Your task is to extract details of a single peer‑reviewed research study described in the answer that examines Anki spaced repetition for medical students preparing for USMLE Step 1 (published in 2023 or later). If multiple studies are mentioned, select the first one that best matches the criteria. If no such study is present, return nulls.

    Extract the following fields exactly as stated in the answer:
    1. study_title: The full title of the study/article.
    2. journal_or_venue: The journal or academic venue name (e.g., "Academic Medicine", "BMC Medical Education").
    3. publication_year: The publication year (numbers only if possible, otherwise the exact text).

    4. primary_url: The main URL where the study can be accessed (publisher page, journal page, or repository link). If multiple URLs are given, choose the most direct/official one as primary.
    5. additional_urls: A list of any other URLs mentioned that also point to the study (e.g., PubMed, DOI page, institutional repository).

    6. intervention_desc: The exact description of the intervention related to Anki (e.g., "Anki spaced repetition", "use of Anki flashcards").
    7. population_desc: A brief phrase describing the study population related to USMLE Step 1 (e.g., "medical students preparing for USMLE Step 1").
    8. metrics_text: The exact comparative performance metrics between Anki users and non-users as stated in the answer. Include units (%, points) and both groups (e.g., "Pass rate: 93% (Anki) vs 85% (non-Anki)"; "Mean Step 1 score: 232 vs 225"). If not stated, return null.
    9. sample_size_anki: The sample size for the Anki user group (prefer the number only; if not possible, include text like "n=120"). If missing, return null.
    10. sample_size_nonanki: The sample size for the non‑Anki group (same format as above). If missing, return null.

    RULES:
    - Extract only what is explicitly present in the answer.
    - URLs must be explicitly present in the answer (plain, markdown, DOI, PubMed, etc.). Do not invent URLs.
    - If a field is not provided, return null (or empty list for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def collect_sources(study: StudyExtraction) -> List[str]:
    """Collect all candidate source URLs from extraction, deduped and non-empty."""
    urls: List[str] = []
    if study.primary_url and study.primary_url.strip():
        urls.append(study.primary_url.strip())
    for u in study.additional_urls:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    uniq_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq_urls.append(u)
    return uniq_urls


def normalize_year(year_text: Optional[str]) -> Optional[int]:
    """Try to normalize a year string to an integer."""
    if not year_text:
        return None
    try:
        # Extract first 4-digit year if present
        import re
        m = re.search(r"\b(19|20)\d{2}\b", year_text)
        if m:
            return int(m.group(0))
        return int(year_text.strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, study: StudyExtraction) -> None:
    """
    Build and execute the verification steps according to the rubric tree.
    """
    # Top-level critical sequential node (the main analysis task)
    top_node = evaluator.add_sequential(
        id="Research_Study_Analysis",
        desc="Find and analyze a peer-reviewed research study examining Anki's effectiveness for medical students preparing for USMLE Step 1, published in 2023 or later, and extract required comparative outcomes, sample sizes, and an access URL.",
        parent=evaluator.root,
        critical=True
    )

    # Child 1: Study Qualification (parallel, all critical checks)
    qual_node = evaluator.add_parallel(
        id="Study_Qualification",
        desc="The identified study meets all eligibility constraints from the prompt.",
        parent=top_node,
        critical=True
    )

    sources = collect_sources(study)
    pub_year_int = normalize_year(study.publication_year)

    # 1.a Intervention_Is_Anki
    leaf_intervention = evaluator.add_leaf(
        id="Intervention_Is_Anki",
        desc="The study specifically examines Anki spaced repetition software as the educational intervention.",
        parent=qual_node,
        critical=True
    )
    claim_intervention = "The study specifically examines Anki spaced repetition software as the educational intervention for learners."
    await evaluator.verify(
        claim=claim_intervention,
        node=leaf_intervention,
        sources=sources,
        additional_instruction="Check the study page to confirm that Anki (the software/platform) is explicitly studied as the intervention (e.g., use of Anki flashcards/spaced repetition). Consider synonymous phrasing such as 'Anki'/'spaced repetition using Anki'."
    )

    # 1.b Population_Is_Step1_Medical_Students
    leaf_population = evaluator.add_leaf(
        id="Population_Is_Step1_Medical_Students",
        desc="The study focuses on medical students preparing for the USMLE Step 1 examination.",
        parent=qual_node,
        critical=True
    )
    claim_population = "The study focuses on medical students preparing for the USMLE Step 1 exam."
    await evaluator.verify(
        claim=claim_population,
        node=leaf_population,
        sources=sources,
        additional_instruction="Verify the study population is medical students specifically preparing for USMLE Step 1 (allow phrasing like 'preclinical medical students', 'students studying for USMLE Step 1')."
    )

    # 1.c Peer_Reviewed_Source
    leaf_peer = evaluator.add_leaf(
        id="Peer_Reviewed_Source",
        desc="The study is published in a peer-reviewed academic source.",
        parent=qual_node,
        critical=True
    )
    claim_peer = "This study is published in a peer-reviewed academic journal or peer-reviewed proceedings."
    await evaluator.verify(
        claim=claim_peer,
        node=leaf_peer,
        sources=sources,
        additional_instruction="Confirm evidence of peer-review: presence of a journal name (with volume/issue), publisher's academic journal page, indexing like PubMed with 'Journal Article', or other explicit indicators of peer review. Preprints not peer-reviewed should NOT count."
    )

    # 1.d Recency_2023_or_Later
    leaf_recency = evaluator.add_leaf(
        id="Recency_2023_or_Later",
        desc="The study is published in 2023 or later.",
        parent=qual_node,
        critical=True
    )
    if pub_year_int is not None:
        claim_recency = f"The study was published in {pub_year_int}, which is 2023 or later."
    else:
        claim_recency = "The study was published in 2023 or later."
    await evaluator.verify(
        claim=claim_recency,
        node=leaf_recency,
        sources=sources,
        additional_instruction="Use the article's publication year/date on the page (online first/epub ahead of print acceptable) to confirm the year is 2023 or later."
    )

    # Child 2: Data Extraction (parallel, all critical checks)
    data_node = evaluator.add_parallel(
        id="Data_Extraction",
        desc="The answer extracts all required information from the qualified study.",
        parent=top_node,
        critical=True
    )

    # 2.a Quantitative_Comparative_Performance_Metrics
    leaf_metrics = evaluator.add_leaf(
        id="Quantitative_Comparative_Performance_Metrics",
        desc="The answer provides quantitative comparative performance metrics between Anki users and non-users (e.g., failure rates, pass rates, or exam scores).",
        parent=data_node,
        critical=True
    )
    if study.metrics_text and study.metrics_text.strip():
        claim_metrics = f"The study reports the following quantitative comparative metrics between Anki users and non-users: {study.metrics_text.strip()}."
    else:
        claim_metrics = "The study reports quantitative comparative performance metrics between Anki users and non-users (e.g., failure rates, pass rates, or exam scores)."
    await evaluator.verify(
        claim=claim_metrics,
        node=leaf_metrics,
        sources=sources,
        additional_instruction="Verify that the page explicitly reports comparative outcomes between Anki users and non-users (such as pass/fail rates or mean scores). Reasonable rounding/format variants are acceptable."
    )

    # 2.b Group_Sample_Sizes
    leaf_samples = evaluator.add_leaf(
        id="Group_Sample_Sizes",
        desc="The answer reports sample sizes for both the Anki-using group and the non-Anki group.",
        parent=data_node,
        critical=True
    )
    if (study.sample_size_anki and study.sample_size_anki.strip()) and (study.sample_size_nonanki and study.sample_size_nonanki.strip()):
        claim_samples = f"The study reports sample sizes for both groups: Anki users = {study.sample_size_anki.strip()}, non-Anki group = {study.sample_size_nonanki.strip()}."
    else:
        claim_samples = "The study reports sample sizes for both the Anki-using group and the non‑Anki group."
    await evaluator.verify(
        claim=claim_samples,
        node=leaf_samples,
        sources=sources,
        additional_instruction="Verify that the page provides sample sizes (n) for both groups (Anki users vs non‑Anki). Minor formatting (e.g., 'n=120') should be accepted."
    )

    # 2.c Verifiable_Source_URL
    leaf_url = evaluator.add_leaf(
        id="Verifiable_Source_URL",
        desc="A verifiable source URL is provided where the study can be accessed.",
        parent=data_node,
        critical=True
    )
    if sources:
        title_for_claim = study.study_title or "the study"
        claim_url = f"At least one of the provided URLs directly hosts or provides access to the peer‑reviewed study titled '{title_for_claim}'."
        await evaluator.verify(
            claim=claim_url,
            node=leaf_url,
            sources=sources,
            additional_instruction="Confirm the URL opens a page that corresponds to the study (publisher page, journal landing page, full text or abstract). The page should display the study title and bibliographic information."
        )
    else:
        # Fall back to simple verification of URL presence in the answer text
        claim_url = "The answer includes a valid access URL where the study can be accessed."
        await evaluator.verify(
            claim=claim_url,
            node=leaf_url,
            sources=None,
            additional_instruction="Check the answer text to confirm that at least one valid study access URL is present."
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
    Evaluate an agent's answer for the Anki/USMLE Step 1 study task using the Mind2Web2 framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root node strategy (non-critical); main logic lives under a critical child node
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

    # Extract structured study info from the answer
    study_info = await evaluator.extract(
        prompt=prompt_extract_study_info(),
        template_class=StudyExtraction,
        extraction_name="study_extraction",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, study_info)

    # Return standard summary
    return evaluator.get_summary()