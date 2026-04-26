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
TASK_ID = "mit_photonic_processor_2024"
TASK_DESCRIPTION = """
What are the key technical details and performance specifications of the photonic processor developed by MIT researchers and published in Nature Photonics in December 2024 that enables ultrafast AI computations?
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PublicationInfo(BaseModel):
    venue: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    senior_author: Optional[str] = None
    lead_author: Optional[str] = None
    publication_sources: List[str] = Field(default_factory=list)


class TechnicalInnovation(BaseModel):
    nofu_definition: Optional[str] = None
    nofu_mechanism: Optional[str] = None
    nofu_sources: List[str] = Field(default_factory=list)


class PerformanceSpecs(BaseModel):
    computation_time: Optional[str] = None
    training_accuracy: Optional[str] = None
    inference_accuracy: Optional[str] = None
    performance_sources: List[str] = Field(default_factory=list)


class FabricationInfo(BaseModel):
    fabrication_process: Optional[str] = None
    fabrication_sources: List[str] = Field(default_factory=list)


class PhotonicProcessorExtraction(BaseModel):
    publication: PublicationInfo = Field(default_factory=PublicationInfo)
    innovation: TechnicalInnovation = Field(default_factory=TechnicalInnovation)
    performance: PerformanceSpecs = Field(default_factory=PerformanceSpecs)
    fabrication: FabricationInfo = Field(default_factory=FabricationInfo)
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_photonic_processor_details() -> str:
    return """
    Extract the structured information about the MIT photonic processor described in the answer. We need four groups of details and any URLs cited as sources. Follow these instructions precisely.

    1) Publication Information:
       - venue: The publication venue where the work was published (e.g., Nature Photonics).
       - date: The publication date string as provided (e.g., "December 2, 2024").
       - title: The paper title (e.g., "Single-chip photonic deep neural network with forward-only training").
       - senior_author: The senior author's name (e.g., "Dirk Englund").
       - lead_author: The lead author's name (e.g., "Saumil Bandyopadhyay").
       - publication_sources: An array of URLs the answer cites for publication details (Nature page, MIT news, DOI, etc.).

    2) Core Technical Innovation:
       - nofu_definition: A short sentence describing the core innovation (NOFUs) combining electronics and optics to perform nonlinear operations on-chip.
       - nofu_mechanism: A short sentence describing how NOFUs siphon a small amount of light to photodiodes converting optical signals to electric current, eliminating the need for external amplifiers.
       - nofu_sources: An array of URLs the answer cites that support the innovation and mechanism.

    3) Performance Specifications:
       - computation_time: The stated computation time from the answer (e.g., "less than 0.5 nanoseconds").
       - training_accuracy: The stated training accuracy (e.g., ">96%").
       - inference_accuracy: The stated inference accuracy (e.g., ">92%").
       - performance_sources: An array of URLs the answer cites that support these metrics.

    4) Fabrication Process:
       - fabrication_process: A short phrase/sentence summarizing fabrication approach (e.g., "fabricated using commercial foundry processes and CMOS-compatible infrastructure").
       - fabrication_sources: An array of URLs the answer cites that support fabrication claims.

    5) all_sources:
       - all_sources: Extract ALL URLs anywhere in the answer (including markdown links). Include Nature pages, DOI pages, MIT News, arXiv, press releases, or any other links.

    Rules for URL extraction:
    - Extract only URLs explicitly present in the answer (plain or markdown link target).
    - Include full URLs with protocol; if missing, prepend "http://".
    - Ignore malformed URLs.
    - If no URLs appear for a specific section, return an empty array for that section.
    - If no URLs appear in the overall answer, return an empty array for 'all_sources'.

    If any field is missing from the answer, set it to null (or empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(primary: List[str], fallback_all: List[str]) -> List[str]:
    """Merge specific sources with all_sources, preserving order and uniqueness."""
    seen = set()
    merged: List[str] = []
    for url in (primary or []):
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    for url in (fallback_all or []):
        if url and url not in seen:
            merged.append(url)
            seen.add(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_publication(
    evaluator: Evaluator,
    parent_node,
    data: PhotonicProcessorExtraction
) -> None:
    pub_node = evaluator.add_parallel(
        id="Publication_Information",
        desc="Verify publication venue/date, paper title, and authorship.",
        parent=parent_node,
        critical=True
    )
    pub_sources = _merge_sources(data.publication.publication_sources, data.all_sources)

    # Publication venue and date
    venue_date_leaf = evaluator.add_leaf(
        id="Publication_Venue_and_Date",
        desc="Published in Nature Photonics on December 2, 2024.",
        parent=pub_node,
        critical=True
    )
    claim_venue_date = "This work was published in Nature Photonics on December 2, 2024."
    await evaluator.verify(
        claim=claim_venue_date,
        node=venue_date_leaf,
        sources=pub_sources,
        additional_instruction="Allow date formats like '2 December 2024' or 'Dec 2, 2024'. Confirm the venue is Nature Photonics."
    )

    # Paper title
    title_leaf = evaluator.add_leaf(
        id="Paper_Title",
        desc="Paper title is 'Single-chip photonic deep neural network with forward-only training'.",
        parent=pub_node,
        critical=True
    )
    claim_title = "The paper title is 'Single-chip photonic deep neural network with forward-only training'."
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=pub_sources,
        additional_instruction="Check the exact or near-exact title string on the cited Nature/DOI page. Minor punctuation or hyphenation differences are acceptable."
    )

    # Senior author
    senior_leaf = evaluator.add_leaf(
        id="Senior_Author",
        desc="Senior author is Dirk Englund.",
        parent=pub_node,
        critical=True
    )
    claim_senior = "The senior author is Dirk Englund."
    await evaluator.verify(
        claim=claim_senior,
        node=senior_leaf,
        sources=pub_sources,
        additional_instruction="Confirm the role as senior/PI if explicitly indicated; otherwise accept as corresponding or lead investigator when clearly implied. Allow minor name variants."
    )

    # Lead author
    lead_leaf = evaluator.add_leaf(
        id="Lead_Author",
        desc="Lead author is Saumil Bandyopadhyay.",
        parent=pub_node,
        critical=True
    )
    claim_lead = "The lead author is Saumil Bandyopadhyay."
    await evaluator.verify(
        claim=claim_lead,
        node=lead_leaf,
        sources=pub_sources,
        additional_instruction="Confirm that Saumil Bandyopadhyay is listed as lead/first author. Allow middle initials or minor spelling variants."
    )


async def build_and_verify_technical_innovation(
    evaluator: Evaluator,
    parent_node,
    data: PhotonicProcessorExtraction
) -> None:
    tech_node = evaluator.add_parallel(
        id="Core_Technical_Innovation",
        desc="Verify the core innovation (NOFUs) and how they work.",
        parent=parent_node,
        critical=True
    )
    tech_sources = _merge_sources(data.innovation.nofu_sources, data.all_sources)

    # NOFU definition
    nofu_def_leaf = evaluator.add_leaf(
        id="NOFU_Definition",
        desc="Core technical innovation involves nonlinear optical function units (NOFUs) combining electronics and optics to perform nonlinear operations on-chip.",
        parent=tech_node,
        critical=True
    )
    claim_nofu_def = (
        "The core innovation uses nonlinear optical function units (NOFUs) that combine electronics and optics to perform nonlinear operations on-chip."
    )
    await evaluator.verify(
        claim=claim_nofu_def,
        node=nofu_def_leaf,
        sources=tech_sources,
        additional_instruction="Confirm that the processor's nonlinearities are implemented via NOFUs integrating electronics and photonics on the same chip."
    )

    # NOFU mechanism
    nofu_mech_leaf = evaluator.add_leaf(
        id="NOFU_Mechanism",
        desc="NOFUs siphon off a small amount of light to photodiodes that convert optical signals to electric current, eliminating the need for external amplifiers.",
        parent=tech_node,
        critical=True
    )
    claim_nofu_mech = (
        "The NOFUs siphon a small portion of light to photodiodes that convert optical signals into electric current, eliminating the need for external amplifiers."
    )
    await evaluator.verify(
        claim=claim_nofu_mech,
        node=nofu_mech_leaf,
        sources=tech_sources,
        additional_instruction="Confirm that the NOFU design uses on-chip photodiodes tapping optical power to generate electrical signals, avoiding off-chip amplification."
    )


async def build_and_verify_performance(
    evaluator: Evaluator,
    parent_node,
    data: PhotonicProcessorExtraction
) -> None:
    perf_node = evaluator.add_parallel(
        id="Performance_Specifications",
        desc="Verify computation time and accuracy metrics.",
        parent=parent_node,
        critical=True
    )
    perf_sources = _merge_sources(data.performance.performance_sources, data.all_sources)

    # Computation time
    comp_time_leaf = evaluator.add_leaf(
        id="Computation_Time",
        desc="Processor completes key computations in less than 0.5 nanoseconds (less than half a nanosecond).",
        parent=perf_node,
        critical=True
    )
    claim_comp_time = "The processor completes its key computations in less than 0.5 nanoseconds."
    await evaluator.verify(
        claim=claim_comp_time,
        node=comp_time_leaf,
        sources=perf_sources,
        additional_instruction="Accept phrasing such as 'sub-nanosecond', 'under half a nanosecond', or '< 0.5 ns'. Confirm the timing refers to the processor's core compute operation."
    )

    # Training accuracy
    train_acc_leaf = evaluator.add_leaf(
        id="Training_Accuracy",
        desc="Processor achieves training accuracy greater than 96 percent.",
        parent=perf_node,
        critical=True
    )
    claim_train_acc = "The processor achieves training accuracy of at least 96 percent."
    await evaluator.verify(
        claim=claim_train_acc,
        node=train_acc_leaf,
        sources=perf_sources,
        additional_instruction="Confirm training accuracy reported is ≥ 96%. Allow minor rounding differences."
    )

    # Inference accuracy
    infer_acc_leaf = evaluator.add_leaf(
        id="Inference_Accuracy",
        desc="Processor achieves inference accuracy greater than 92 percent.",
        parent=perf_node,
        critical=True
    )
    claim_infer_acc = "The processor achieves inference accuracy of at least 92 percent."
    await evaluator.verify(
        claim=claim_infer_acc,
        node=infer_acc_leaf,
        sources=perf_sources,
        additional_instruction="Confirm inference accuracy reported is ≥ 92%. Allow minor rounding differences."
    )


async def build_and_verify_fabrication(
    evaluator: Evaluator,
    parent_node,
    data: PhotonicProcessorExtraction
) -> None:
    fab_node = evaluator.add_parallel(
        id="Fabrication_Process",
        desc="Verify fabrication/manufacturing approach.",
        parent=parent_node,
        critical=True
    )
    fab_sources = _merge_sources(data.fabrication.fabrication_sources, data.all_sources)

    # Commercial foundry + CMOS-compatible
    fab_leaf = evaluator.add_leaf(
        id="Commercial_Foundry_CMOS",
        desc="Circuit fabricated using commercial foundry processes and CMOS-compatible infrastructure.",
        parent=fab_node,
        critical=True
    )
    claim_fab = "The circuit was fabricated using commercial foundry processes and CMOS-compatible infrastructure."
    await evaluator.verify(
        claim=claim_fab,
        node=fab_leaf,
        sources=fab_sources,
        additional_instruction="Confirm that the device was built in a commercial foundry and is CMOS-compatible (process and infrastructure)."
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
    Evaluate the agent's answer about the MIT photonic processor (Nature Photonics, Dec 2024).
    Builds a critical parallel verification tree covering publication details, core innovation, performance specs, and fabrication approach.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The main rubric node is parallel
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_photonic_processor_details(),
        template_class=PhotonicProcessorExtraction,
        extraction_name="photonic_processor_extraction",
    )

    # Add ground truth expectations for transparency
    evaluator.add_ground_truth({
        "publication_expected": {
            "venue": "Nature Photonics",
            "date": "December 2, 2024",
            "title": "Single-chip photonic deep neural network with forward-only training",
            "senior_author": "Dirk Englund",
            "lead_author": "Saumil Bandyopadhyay"
        },
        "innovation_expected": {
            "nofu_definition": "NOFUs combine electronics and optics to perform nonlinear operations on-chip",
            "nofu_mechanism": "NOFUs siphon light to photodiodes converting optical signals to electric current, eliminating external amplifiers"
        },
        "performance_expected": {
            "computation_time": "< 0.5 ns",
            "training_accuracy": "≥ 96%",
            "inference_accuracy": "≥ 92%"
        },
        "fabrication_expected": {
            "process": "commercial foundry; CMOS-compatible infrastructure"
        }
    })

    # Build main critical node under root to satisfy critical-child constraint
    main_node = evaluator.add_parallel(
        id="MIT_Photonic_Processor_Details",
        desc="Verify publication information, core technical innovation, performance specifications, and fabrication approach for the MIT photonic processor described in the constraints.",
        parent=root,
        critical=True
    )

    # Build and verify each critical group
    await build_and_verify_publication(evaluator, main_node, extracted)
    await build_and_verify_technical_innovation(evaluator, main_node, extracted)
    await build_and_verify_performance(evaluator, main_node, extracted)
    await build_and_verify_fabrication(evaluator, main_node, extracted)

    # Return structured summary
    return evaluator.get_summary()