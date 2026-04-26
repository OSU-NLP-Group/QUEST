import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "willow_d7_lifetime_config"
TASK_DESCRIPTION = (
    "What is the logical qubit lifetime achieved by Google's Willow quantum processor using a distance-7 surface code "
    "implementation, and what is the complete qubit configuration (data qubits, measure qubits, and leakage removal "
    "qubits) used in this distance-7 implementation? Provide your answer with citations to the primary research publication."
)

# Expected values (used to guide verification claims and instructions)
EXPECTED = {
    "processor_name": "Willow",
    "total_qubits": "105",
    "qubit_technology": "superconducting transmon qubits",
    "manufacturing_location": "Santa Barbara, California",
    "code_distance": "distance-7 surface code",
    "data_qubits": "49",
    "measure_qubits": "48",
    "leakage_qubits": "4",
    "lifetime_text": "291 ± 6 μs",
    "improvement_text": "2.4 ± 0.3",
    "journal": "Nature",
    "pub_date": "December 9, 2024",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProcessorInfo(BaseModel):
    name: Optional[str] = None
    total_qubits: Optional[str] = None
    qubit_technology: Optional[str] = None
    manufacturing_location: Optional[str] = None


class ImplementationInfo(BaseModel):
    code_distance: Optional[str] = None  # e.g., "distance-7 surface code", "d=7"
    data_qubits: Optional[str] = None
    measure_qubits: Optional[str] = None
    leakage_removal_qubits: Optional[str] = None


class LogicalLifetimeInfo(BaseModel):
    lifetime_text: Optional[str] = None  # e.g., "291 ± 6 μs"
    improvement_text: Optional[str] = None  # e.g., "2.4 ± 0.3"


class PublicationInfo(BaseModel):
    journal: Optional[str] = None
    publication_date: Optional[str] = None  # Accepts formats like "December 9, 2024" or "2024-12-09"
    primary_publication_urls: List[str] = Field(default_factory=list)  # Explicit URLs cited in the answer
    primary_doi: Optional[str] = None  # DOI string if provided (e.g., "10.1038/s41586-024-XXXXX")
    formal_reference_text: Optional[str] = None  # Any formal/bibliographic reference text if provided


class WillowD7Extraction(BaseModel):
    processor: Optional[ProcessorInfo] = None
    implementation: Optional[ImplementationInfo] = None
    logical_lifetime: Optional[LogicalLifetimeInfo] = None
    publication: Optional[PublicationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_willow_d7() -> str:
    return """
Extract the following structured information from the answer text. Do not infer or invent any missing info—return null for any field not explicitly present in the answer.

1) processor:
   - name: The processor name (e.g., "Willow"); do not include extra descriptors.
   - total_qubits: The total number of qubits reported for Willow; return as a string (e.g., "105").
   - qubit_technology: The qubit type/technology (e.g., "superconducting transmon qubits" or "transmon").
   - manufacturing_location: The reported manufacturing location (e.g., "Santa Barbara, California" or "Santa Barbara, CA").

2) implementation:
   - code_distance: The code type and distance string for the implementation; capture phrasing if present (e.g., "distance-7 surface code", "d=7").
   - data_qubits: The number of data qubits in the distance-7 configuration (as string).
   - measure_qubits: The number of measure/stabilizer qubits in the distance-7 configuration (as string).
   - leakage_removal_qubits: The number of leakage removal qubits (as string), if provided.

3) logical_lifetime:
   - lifetime_text: The reported logical qubit lifetime including uncertainty and units if provided (e.g., "291 ± 6 μs"). Keep the original notation (±, units).
   - improvement_text: The comparison factor between logical lifetime and best physical qubit including uncertainty if provided (e.g., "2.4 ± 0.3").

4) publication:
   - journal: The journal name if given (e.g., "Nature").
   - publication_date: The publication date string if present (e.g., "December 9, 2024" or "2024-12-09").
   - primary_publication_urls: All explicit URLs cited for the primary research publication (e.g., Nature or DOI-resolver links). Only extract actual URLs present in the answer.
   - primary_doi: The DOI string if present (e.g., "10.1038/s41586-024-XXXXX"); do not return as URL; just the DOI identifier.
   - formal_reference_text: Any formal bibliographic reference text if present.

Notes:
- Return numbers as strings. Accept common synonyms (e.g., "measure qubits" vs "stabilizer qubits").
- For URLs, only include explicit URLs that appear in the answer (including markdown links).
- If a field is not present in the answer, set it to null (or empty array for URLs).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_doi_to_url(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    d = doi.strip()
    if d.lower().startswith("doi:"):
        d = d.split(":", 1)[1].strip()
    # Remove leading URL if given, keep only identifier for uniformity
    if d.lower().startswith("https://doi.org/") or d.lower().startswith("http://doi.org/"):
        return d
    if d:
        return f"https://doi.org/{d}"
    return None


def build_primary_sources(pub: Optional[PublicationInfo]) -> List[str]:
    urls: List[str] = []
    if not pub:
        return urls
    if pub.primary_publication_urls:
        urls.extend([u for u in pub.primary_publication_urls if isinstance(u, str) and u.strip() != ""])
    doi_url = normalize_doi_to_url(pub.primary_doi)
    if doi_url:
        urls.append(doi_url)
    # Deduplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_processor_constraints(
    evaluator: Evaluator,
    parent_node,
    primary_sources: List[str],
) -> None:
    proc_node = evaluator.add_parallel(
        id="Processor_Constraints",
        desc="Verify the processor identity and required processor attributes.",
        parent=parent_node,
        critical=True,
    )

    # Processor_Is_Willow
    leaf = evaluator.add_leaf(
        id="Processor_Is_Willow",
        desc="Answer identifies the quantum processor as Google's Willow chip.",
        parent=proc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly identifies the quantum processor as Google's 'Willow' chip.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Focus on whether the answer text clearly names the processor as 'Willow'. "
            "If sources are provided, also ensure at least one source refers to the processor as 'Willow'. "
            "Allow phrasing variants (e.g., 'Google's Willow processor', 'Willow quantum processor')."
        ),
    )

    # Processor_Total_Qubits_105
    leaf = evaluator.add_leaf(
        id="Processor_Total_Qubits_105",
        desc="Answer states Willow has 105 total qubits.",
        parent=proc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly states that the Willow processor has 105 total qubits (total count = 105).",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Confirm the answer text reports the total qubit count as 105. "
            "If sources are provided, also check that at least one source clearly indicates a total of 105 qubits. "
            "Treat '105 qubits', 'total of 105 qubits', or equivalent wording as a match."
        ),
    )

    # Processor_Qubit_Technology_Transmon
    leaf = evaluator.add_leaf(
        id="Processor_Qubit_Technology_Transmon",
        desc="Answer states Willow uses superconducting transmon qubits.",
        parent=proc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that Willow uses superconducting transmon qubits (i.e., transmons).",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Accept reasonable variants such as 'transmon qubits', 'superconducting transmons', or equivalent. "
            "If sources are available, also verify the sources describe Willow as using transmon-based superconducting qubits."
        ),
    )

    # Processor_Manufactured_Santa_Barbara_CA
    leaf = evaluator.add_leaf(
        id="Processor_Manufactured_Santa_Barbara_CA",
        desc="Answer states Willow was manufactured in Santa Barbara, California.",
        parent=proc_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Willow processor was manufactured in Santa Barbara, California.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Accept 'Santa Barbara, CA' as equivalent to 'Santa Barbara, California'. "
            "If sources are provided, also ensure at least one source associates Willow with being manufactured in Santa Barbara."
        ),
    )


async def verify_implementation_constraints(
    evaluator: Evaluator,
    parent_node,
    primary_sources: List[str],
) -> None:
    impl_node = evaluator.add_parallel(
        id="Implementation_Constraints",
        desc="Verify the implementation is the required distance-7 surface code and configuration.",
        parent=parent_node,
        critical=True,
    )

    # Uses_Distance_7_Surface_Code
    leaf = evaluator.add_leaf(
        id="Uses_Distance_7_Surface_Code",
        desc="Answer specifies the implementation is a distance-7 surface code.",
        parent=impl_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer specifies that the implementation uses a distance-7 surface code (a.k.a. d=7).",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Allow 'distance-7', 'd=7', or equivalent phrasing. "
            "If sources are provided, also ensure that the source explicitly indicates a distance-7 surface code."
        ),
    )

    # Distance_7_Data_Qubits_49
    leaf = evaluator.add_leaf(
        id="Distance_7_Data_Qubits_49",
        desc="Answer specifies the distance-7 configuration uses exactly 49 data qubits.",
        parent=impl_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the distance-7 configuration uses exactly 49 data qubits.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "If sources are provided, also verify that at least one source clearly shows 49 data qubits for the distance-7 layout."
        ),
    )

    # Distance_7_Measure_Qubits_48
    leaf = evaluator.add_leaf(
        id="Distance_7_Measure_Qubits_48",
        desc="Answer specifies the distance-7 configuration uses exactly 48 measure qubits.",
        parent=impl_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the distance-7 configuration uses exactly 48 measure (stabilizer) qubits.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Accept 'measurement qubits', 'stabilizer qubits', 'check qubits' as synonyms for measure qubits. "
            "If sources are provided, also verify a count of 48 measure/stabilizer qubits is stated."
        ),
    )

    # Distance_7_Leakage_Removal_Qubits_4
    leaf = evaluator.add_leaf(
        id="Distance_7_Leakage_Removal_Qubits_4",
        desc="Answer specifies the distance-7 configuration uses exactly 4 leakage removal qubits.",
        parent=impl_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the distance-7 configuration uses exactly 4 leakage removal qubits (LRUs).",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Allow 'leakage removal units/qubits', 'LRUs', or equivalent phrasing. "
            "If sources are provided, also verify that at least one source clearly indicates 4 leakage removal qubits."
        ),
    )


async def verify_logical_lifetime_constraints(
    evaluator: Evaluator,
    parent_node,
    primary_sources: List[str],
) -> None:
    life_node = evaluator.add_parallel(
        id="Logical_Lifetime_Constraints",
        desc="Verify the required logical lifetime result and comparison factor.",
        parent=parent_node,
        critical=True,
    )

    # Logical_Qubit_Lifetime_291_plusminus_6_us
    leaf = evaluator.add_leaf(
        id="Logical_Qubit_Lifetime_291_plusminus_6_us",
        desc="Answer reports the distance-7 logical qubit lifetime as 291 ± 6 μs.",
        parent=life_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer reports the distance-7 logical qubit lifetime as 291 ± 6 microseconds (μs).",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Treat 'μs', 'us', and 'microseconds' as equivalent. "
            "Allow formatting variants like '291±6 μs'. "
            "If sources are provided, also ensure that at least one source explicitly reports '291 ± 6 μs' as the logical lifetime."
        ),
    )

    # Lifetime_Improvement_Factor_2_4_plusminus_0_3
    leaf = evaluator.add_leaf(
        id="Lifetime_Improvement_Factor_2_4_plusminus_0_3",
        desc="Answer states the logical qubit lifetime exceeds the best constituent physical qubit by a factor of 2.4 ± 0.3.",
        parent=life_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the logical qubit lifetime exceeds the best constituent physical qubit by a factor of 2.4 ± 0.3.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Allow phrasings such as '2.4× (±0.3)' or '2.4 ± 0.3 times'. "
            "If sources are provided, also verify the same improvement factor is reported."
        ),
    )


async def verify_publication_constraints(
    evaluator: Evaluator,
    parent_node,
    extracted_pub: Optional[PublicationInfo],
    primary_sources: List[str],
) -> None:
    pub_node = evaluator.add_parallel(
        id="Publication_Constraints_and_Citation",
        desc="Verify the publication constraints and that the primary research publication is cited.",
        parent=parent_node,
        critical=True,
    )

    # Published_In_Nature
    leaf = evaluator.add_leaf(
        id="Published_In_Nature",
        desc="Answer indicates the results are published in Nature journal.",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly indicates that the results are published in the journal Nature.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "If sources are provided, also verify the cited primary publication is indeed from Nature (nature.com or clearly labeled Nature)."
        ),
    )

    # Nature_Publication_Date_Dec_9_2024
    leaf = evaluator.add_leaf(
        id="Nature_Publication_Date_Dec_9_2024",
        desc="Answer specifies the Nature article publication date is December 9, 2024.",
        parent=pub_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer specifies that the Nature article's publication date is December 9, 2024.",
        node=leaf,
        sources=primary_sources,
        additional_instruction=(
            "Accept date formatting variants like '9 December 2024' or '2024-12-09' as equivalent to 'December 9, 2024'. "
            "If sources are provided, also verify that at least one source shows the publication date as 2024-12-09 or its equivalent."
        ),
    )

    # Cites_Primary_Research_Publication (existence of citation in the answer)
    has_any_citation = False
    if extracted_pub:
        urls_present = bool(extracted_pub.primary_publication_urls)
        doi_present = bool(extracted_pub.primary_doi and extracted_pub.primary_doi.strip())
        ref_present = bool(extracted_pub.formal_reference_text and extracted_pub.formal_reference_text.strip())
        has_any_citation = urls_present or doi_present or ref_present

    evaluator.add_custom_node(
        result=has_any_citation,
        id="Cites_Primary_Research_Publication",
        desc="Answer includes citations to the primary research publication supporting the lifetime and configuration claims (e.g., DOI, URL, or formal reference).",
        parent=pub_node,
        critical=True,
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
    # Initialize evaluator with a parallel root (default)
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

    # Extraction
    extracted: WillowD7Extraction = await evaluator.extract(
        prompt=prompt_extract_willow_d7(),
        template_class=WillowD7Extraction,
        extraction_name="willow_d7_extraction",
    )

    # Build primary publication sources (URLs)
    primary_sources = build_primary_sources(extracted.publication if extracted else None)

    # Record helpful info in summary
    evaluator.add_custom_info(
        info={"primary_sources": primary_sources},
        info_type="sources",
        info_name="extracted_primary_sources"
    )

    # Add ground truth info for transparency (not used directly in verification)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED,
            "note": "These are the expected target values used for verification guidance."
        },
        gt_type="ground_truth"
    )

    # Build main critical node as per rubric
    main_node = evaluator.add_parallel(
        id="Distance_7_Logical_Qubit_Achievement",
        desc="Verify the answer about Google Willow's distance-7 logical qubit lifetime and qubit configuration, with proper citation to the primary research publication.",
        parent=root,
        critical=True,
    )

    # Sub verifications
    await verify_processor_constraints(evaluator, main_node, primary_sources)
    await verify_implementation_constraints(evaluator, main_node, primary_sources)
    await verify_logical_lifetime_constraints(evaluator, main_node, primary_sources)
    await verify_publication_constraints(evaluator, main_node, extracted.publication if extracted else None, primary_sources)

    # Return structured summary
    return evaluator.get_summary()