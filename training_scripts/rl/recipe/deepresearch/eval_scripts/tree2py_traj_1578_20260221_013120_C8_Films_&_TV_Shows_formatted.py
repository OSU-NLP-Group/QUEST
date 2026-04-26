import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "music_docs_2025"
TASK_DESCRIPTION = (
    "Identify three music documentaries released in 2025 that meet ALL of the following criteria:\n\n"
    "1. The director has won either an Academy Award for Best Documentary Feature for a music-related documentary OR a Critics Choice Documentary Award for Best Music Documentary\n"
    "2. The documentary is available for streaming on a major platform (Amazon Prime Video, Hulu, Disney+, or Netflix) with a streaming release date between January 2025 and February 2026\n"
    "3. The documentary focuses on a music artist or band whose career peak or formation occurred before the year 2000\n"
    "4. The documentary premiered at a major film festival (such as Sundance, Telluride, or Venice) OR had a theatrical release in 2025\n"
    "5. The documentary has a runtime of at least 100 minutes\n\n"
    "For each documentary, provide the title, director's name, the subject of the documentary, the streaming platform and release date, and the film festival or theatrical release information."
)

ALLOWED_STREAMING_PLATFORMS = [
    "Amazon Prime Video", "Prime Video", "Amazon Prime",
    "Hulu",
    "Disney+", "Disney Plus",
    "Netflix"
]
STREAMING_RANGE_START = "2025-01-01"
STREAMING_RANGE_END = "2026-02-28"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DocumentaryItem(BaseModel):
    title: Optional[str] = None
    director: Optional[str] = None
    subject: Optional[str] = None
    release_year: Optional[str] = None

    streaming_platform: Optional[str] = None
    streaming_release_date: Optional[str] = None
    streaming_url: Optional[str] = None
    streaming_sources: List[str] = Field(default_factory=list)

    festival_or_theatrical_info: Optional[str] = None
    festival_or_theatrical_url: Optional[str] = None
    release_sources: List[str] = Field(default_factory=list)

    runtime: Optional[str] = None
    runtime_sources: List[str] = Field(default_factory=list)

    subject_sources: List[str] = Field(default_factory=list)
    director_award_sources: List[str] = Field(default_factory=list)

    sources: List[str] = Field(default_factory=list)


class DocumentariesExtraction(BaseModel):
    documentaries: List[DocumentaryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_documentaries() -> str:
    return (
        "Extract up to three music documentaries mentioned in the answer that the agent claims meet the specified 2025 criteria.\n"
        "For each documentary, extract the following fields exactly as presented in the answer:\n"
        "- title: The documentary title\n"
        "- director: Director's full name\n"
        "- subject: The music artist or band the documentary focuses on\n"
        "- release_year: The year of the film's release (festival premiere or theatrical release year as given)\n"
        "- streaming_platform: The named streaming platform (e.g., Amazon Prime Video, Hulu, Disney+, Netflix)\n"
        "- streaming_release_date: The stated streaming release date\n"
        "- streaming_url: The URL to the streaming platform page, if provided\n"
        "- streaming_sources: All URLs that support the streaming platform availability or streaming date for this film\n"
        "- festival_or_theatrical_info: A brief description of the premiere festival or theatrical release info\n"
        "- festival_or_theatrical_url: A URL that supports the premiere or theatrical release info, if provided\n"
        "- release_sources: All URLs that support the release year information for this film\n"
        "- runtime: The film runtime as stated (keep the exact format, e.g., '102 minutes' or '1h 42m')\n"
        "- runtime_sources: All URLs that support the runtime information for this film\n"
        "- subject_sources: All URLs that support the subject's career timeline (formation year or career peak) for pre-2000 verification\n"
        "- director_award_sources: All URLs that support the director's award credentials (Academy Award for Best Documentary Feature for a music-related film OR Critics Choice Best Music Documentary)\n"
        "- sources: Any additional URLs cited about this film (e.g., official pages, festival schedule, reviews, Wikipedia)\n\n"
        "Return the data in a JSON object with a 'documentaries' array. If the answer lists more than three, include only the first three. If a field is missing, set it to null (for single values) or an empty array (for list of URLs). Extract only URLs explicitly present in the answer; do not invent or infer URLs."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def compose_sources(*parts: Any) -> List[str]:
    """
    Compose a unique list of URLs from mixed inputs (strings, lists, None),
    preserving order and filtering out falsy values.
    """
    seen = set()
    result: List[str] = []
    for p in parts:
        if not p:
            continue
        if isinstance(p, str):
            val = p.strip()
            if val and val not in seen:
                seen.add(val)
                result.append(val)
        elif isinstance(p, list):
            for s in p:
                if not s:
                    continue
                val = s.strip()
                if val and val not in seen:
                    seen.add(val)
                    result.append(val)
    return result


def ordinal_label(index: int) -> str:
    return ["First", "Second", "Third"][index] if 0 <= index < 3 else f"Item {index + 1}"


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_documentary(
    evaluator: Evaluator,
    parent_node,
    doc: DocumentaryItem,
    idx: int,
) -> None:
    """
    Build verification nodes and run checks for one documentary.
    """
    label = ordinal_label(idx)
    doc_node = evaluator.add_parallel(
        id=f"Documentary_{idx + 1}",
        desc=f"{label} music documentary meeting all criteria",
        parent=parent_node,
        critical=False
    )

    # Information completeness (critical)
    completeness_ok = all([
        bool(doc.title and doc.title.strip()),
        bool(doc.director and doc.director.strip()),
        bool(doc.subject and doc.subject.strip()),
        bool(doc.streaming_platform and doc.streaming_platform.strip()),
        bool(doc.streaming_release_date and doc.streaming_release_date.strip()),
        bool(doc.festival_or_theatrical_info and doc.festival_or_theatrical_info.strip()),
    ])
    evaluator.add_custom_node(
        result=completeness_ok,
        id=f"doc_{idx}_Information_Completeness",
        desc="The answer provides all required information: title, director's name, subject of the documentary, streaming platform and release date, and film festival or theatrical release information",
        parent=doc_node,
        critical=True
    )

    # Director Award Credentials (critical)
    director_award_node = evaluator.add_leaf(
        id=f"doc_{idx}_Director_Award_Credentials",
        desc="The documentary's director has won an Academy Award for Best Documentary Feature for a music-related documentary OR a Critics Choice Documentary Award for Best Music Documentary",
        parent=doc_node,
        critical=True
    )
    sources_award = compose_sources(doc.director_award_sources, doc.sources)
    # Source prerequisite: require URLs for award verification
    prereq_award = evaluator.add_custom_node(
        result=len(sources_award) > 0,
        id=f"doc_{idx}_Director_Award_Sources_Provided",
        desc="Award credential sources are provided",
        parent=evaluator.root,  # Attach to root to avoid gating siblings
        critical=True
    )
    award_claim = (
        f"Director {doc.director or ''} has previously won either: "
        f"(a) the Academy Award for Best Documentary Feature for a music-related documentary, "
        f"or (b) the Critics Choice Documentary Award for Best Music Documentary."
    )
    await evaluator.verify(
        claim=award_claim,
        node=director_award_node,
        sources=sources_award,
        additional_instruction=(
            "Use award pages or credible sources to confirm the director's win. "
            "For the Academy Award condition, confirm that the winning documentary is about music (e.g., an artist/band or music subject). "
            "For Critics Choice, confirm the category 'Best Music Documentary'. "
            "If neither condition can be supported, mark as not supported."
        ),
        extra_prerequisites=[prereq_award]
    )

    # Release Year 2025 (critical)
    release_2025_node = evaluator.add_leaf(
        id=f"doc_{idx}_Release_Year_2025",
        desc="The documentary was released in 2025",
        parent=doc_node,
        critical=True
    )
    sources_release = compose_sources(doc.release_sources, doc.festival_or_theatrical_url, doc.sources)
    prereq_release = evaluator.add_custom_node(
        result=len(sources_release) > 0,
        id=f"doc_{idx}_Release_Year_Sources_Provided",
        desc="Release year sources are provided",
        parent=evaluator.root,
        critical=True
    )
    release_claim = (
        f"The documentary '{doc.title or ''}' had its release in 2025. "
        f"This can be evidenced by a 2025 festival premiere or a 2025 theatrical release."
    )
    await evaluator.verify(
        claim=release_claim,
        node=release_2025_node,
        sources=sources_release,
        additional_instruction="Confirm that the film's initial public release occurred in 2025 via festival premiere or theatrical release.",
        extra_prerequisites=[prereq_release]
    )

    # Streaming Availability 2025–2026 on major platform (critical)
    streaming_node = evaluator.add_leaf(
        id=f"doc_{idx}_Streaming_Availability_2025_2026",
        desc="The documentary is available for streaming on a major platform (Amazon Prime Video, Hulu, Disney+, or Netflix) with a streaming release date between January 2025 and February 2026",
        parent=doc_node,
        critical=True
    )
    sources_streaming = compose_sources(doc.streaming_url, doc.streaming_sources, doc.sources)
    prereq_streaming = evaluator.add_custom_node(
        result=len(sources_streaming) > 0,
        id=f"doc_{idx}_Streaming_Sources_Provided",
        desc="Streaming availability sources are provided",
        parent=evaluator.root,
        critical=True
    )
    streaming_claim = (
        f"The documentary '{doc.title or ''}' is available for streaming on {doc.streaming_platform or ''} "
        f"with a streaming release date of {doc.streaming_release_date or ''}, "
        f"and that date falls between {STREAMING_RANGE_START} and {STREAMING_RANGE_END}. "
        f"Also, the platform must be one of: Amazon Prime Video, Hulu, Disney+, or Netflix."
    )
    await evaluator.verify(
        claim=streaming_claim,
        node=streaming_node,
        sources=sources_streaming,
        additional_instruction=(
            "Verify availability and the specific streaming release date using the provided streaming page or credible sources. "
            "You are additionally asked to check (using your general knowledge) whether the named platform is in the allowed set "
            f"{ALLOWED_STREAMING_PLATFORMS}. If it is not, mark the claim as not supported even if availability/date are correct."
        ),
        extra_prerequisites=[prereq_streaming]
    )

    # Subject Artist/Band pre-2000 (critical)
    subject_pre2000_node = evaluator.add_leaf(
        id=f"doc_{idx}_Subject_Artist_Pre_2000",
        desc="The documentary focuses on a music artist or band whose career peak or formation occurred before the year 2000",
        parent=doc_node,
        critical=True
    )
    sources_subject = compose_sources(doc.subject_sources, doc.sources)
    prereq_subject = evaluator.add_custom_node(
        result=len(sources_subject) > 0,
        id=f"doc_{idx}_Subject_Sources_Provided",
        desc="Subject career timeline sources are provided",
        parent=evaluator.root,
        critical=True
    )
    subject_claim = (
        f"The documentary '{doc.title or ''}' focuses on {doc.subject or ''}, "
        "and this subject either formed before the year 2000 (for a band) or had their career peak before 2000 (for an individual artist)."
    )
    await evaluator.verify(
        claim=subject_claim,
        node=subject_pre2000_node,
        sources=sources_subject,
        additional_instruction=(
            "Use credible sources (e.g., Wikipedia, AllMusic, official pages) to confirm formation year (for bands) or timeframe of peak success (for artists). "
            "If you cannot confirm pre-2000 formation/peak, mark as not supported."
        ),
        extra_prerequisites=[prereq_subject]
    )

    # Festival or Theatrical in 2025 (critical)
    festival_or_theatrical_node = evaluator.add_leaf(
        id=f"doc_{idx}_Festival_or_Theatrical_2025",
        desc="The documentary premiered at a major film festival (Sundance, Telluride, Venice, or similar) OR had a theatrical release in 2025",
        parent=doc_node,
        critical=True
    )
    sources_festival = compose_sources(doc.festival_or_theatrical_url, doc.release_sources, doc.sources)
    prereq_festival = evaluator.add_custom_node(
        result=len(sources_festival) > 0,
        id=f"doc_{idx}_Festival_Theatrical_Sources_Provided",
        desc="Festival/theatrical sources are provided",
        parent=evaluator.root,
        critical=True
    )
    festival_claim = (
        f"The documentary '{doc.title or ''}' either premiered at a major film festival "
        "(Sundance, Telluride, Venice, or similar) or had a theatrical release in 2025."
    )
    await evaluator.verify(
        claim=festival_claim,
        node=festival_or_theatrical_node,
        sources=sources_festival,
        additional_instruction="Confirm at least one of: major festival premiere or theatrical release in 2025 using credible sources.",
        extra_prerequisites=[prereq_festival]
    )

    # Runtime minimum 100 minutes (critical)
    runtime_node = evaluator.add_leaf(
        id=f"doc_{idx}_Runtime_Minimum_100_Minutes",
        desc="The documentary has a runtime of at least 100 minutes",
        parent=doc_node,
        critical=True
    )
    sources_runtime = compose_sources(doc.runtime_sources, doc.streaming_sources, doc.sources)
    prereq_runtime = evaluator.add_custom_node(
        result=len(sources_runtime) > 0,
        id=f"doc_{idx}_Runtime_Sources_Provided",
        desc="Runtime sources are provided",
        parent=evaluator.root,
        critical=True
    )
    runtime_claim = f"The documentary '{doc.title or ''}' has a runtime of at least 100 minutes."
    await evaluator.verify(
        claim=runtime_claim,
        node=runtime_node,
        sources=sources_runtime,
        additional_instruction="Confirm the runtime from credible sources (platform page, official page, databases). 100 minutes exactly qualifies.",
        extra_prerequisites=[prereq_runtime]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2025 music documentaries criteria task.
    """
    # Initialize evaluator with parallel root (non-critical)
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

    # Add context info for criteria (optional)
    evaluator.add_custom_info(
        info={
            "allowed_streaming_platforms": ALLOWED_STREAMING_PLATFORMS,
            "streaming_date_range": {"start": STREAMING_RANGE_START, "end": STREAMING_RANGE_END},
            "required_fields": [
                "title", "director", "subject",
                "streaming_platform", "streaming_release_date",
                "festival_or_theatrical_info"
            ]
        },
        info_type="criteria",
        info_name="evaluation_criteria"
    )

    # Extract structured documentaries info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_documentaries(),
        template_class=DocumentariesExtraction,
        extraction_name="documentaries_extraction"
    )

    # Build top-level node that mirrors rubric root (optional under the evaluator's root)
    rubric_root = evaluator.add_parallel(
        id="Music_Documentaries_2025",
        desc="Evaluate whether the provided music documentaries from 2025 meet all specified criteria",
        parent=root,
        critical=False
    )

    # Ensure exactly 3 documentaries by padding or truncating
    docs: List[DocumentaryItem] = list(extracted.documentaries[:3])
    while len(docs) < 3:
        docs.append(DocumentaryItem())

    # Verify each documentary (parallel children)
    for i, doc in enumerate(docs):
        await verify_documentary(evaluator, rubric_root, doc, i)

    # Return summary
    return evaluator.get_summary()