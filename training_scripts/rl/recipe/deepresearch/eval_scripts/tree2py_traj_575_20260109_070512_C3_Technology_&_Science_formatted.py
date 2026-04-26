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
TASK_ID = "google_qai_chip_dec2024"
TASK_DESCRIPTION = (
    "In December 2024, Google Quantum AI announced a breakthrough quantum chip that demonstrated quantum error "
    "correction below the surface code threshold. Identify this chip and provide the following sequential information: "
    "(1) What is the name of this quantum chip? (2) How many qubits does it contain in total? (3) What is the highest "
    "surface code distance that the chip successfully demonstrated in its error correction experiments? "
    "(4) What is the mean T1 coherence time (in microseconds) reported for the Quantum Error Correction configuration "
    "(Chip 1)? (5) How many error correction cycles per second does the QEC configuration achieve? "
    "(6) Provide the URL to the official specification sheet PDF document published by Google Quantum AI for this chip."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldEvidence(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ChipInfoExtraction(BaseModel):
    chip_name: FieldEvidence = Field(default_factory=FieldEvidence)
    total_qubit_count: FieldEvidence = Field(default_factory=FieldEvidence)
    highest_surface_code_distance: FieldEvidence = Field(default_factory=FieldEvidence)
    mean_t1_chip1_qec_us: FieldEvidence = Field(default_factory=FieldEvidence)
    qec_cycles_per_second: FieldEvidence = Field(default_factory=FieldEvidence)
    spec_sheet_pdf_url: Optional[str] = None
    additional_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_info() -> str:
    return """
    Extract the requested information about the Google Quantum AI quantum chip referenced in the answer. Return a JSON object matching the following schema:

    {
      "chip_name": { "value": str|null, "sources": [url, ...] },
      "total_qubit_count": { "value": str|null, "sources": [url, ...] },
      "highest_surface_code_distance": { "value": str|null, "sources": [url, ...] },
      "mean_t1_chip1_qec_us": { "value": str|null, "sources": [url, ...] },
      "qec_cycles_per_second": { "value": str|null, "sources": [url, ...] },
      "spec_sheet_pdf_url": str|null,
      "additional_urls": [url, ...]
    }

    Field requirements:
    1) chip_name.value: The stated name of the quantum chip.
       chip_name.sources: All URLs cited in the answer that directly support the chip name (e.g., official spec sheet PDF, Google Quantum AI posts, press releases).
    2) total_qubit_count.value: The total number of qubits on the chip as stated in the answer (keep formatting as in the answer, e.g., "1,024" or "1024").
       total_qubit_count.sources: URLs used to support this count.
    3) highest_surface_code_distance.value: The highest surface code distance demonstrated (e.g., "d=25" or "25").
       highest_surface_code_distance.sources: URLs used to support this detail.
    4) mean_t1_chip1_qec_us.value: The mean T1 coherence time for the Quantum Error Correction configuration (Chip 1), in microseconds, as stated in the answer (e.g., "170 μs" or "170").
       mean_t1_chip1_qec_us.sources: URLs used to support this.
    5) qec_cycles_per_second.value: The number of error-correction cycles per second for the QEC configuration (Chip 1), as stated in the answer (e.g., "2500", "2.5 kHz", or "2,500 cps").
       qec_cycles_per_second.sources: URLs used to support this.
    6) spec_sheet_pdf_url: The single URL to the official specification sheet PDF published by Google Quantum AI for this chip (must be a direct link to a PDF). If multiple URLs are given in the answer, select the most official one (prefer quantumai.google or a Google-hosted PDF).
    7) additional_urls: Any other URLs cited in the answer relevant to this chip or details (exclude duplicates already placed in the field-level sources).

    Rules:
    - Extract only what is explicitly present in the answer. Do not invent values or URLs.
    - For each 'sources' array, include only URLs explicitly present in the answer. If none, return an empty list.
    - For spec_sheet_pdf_url, include only a single, direct PDF URL if present; otherwise return null.
    - Normalize obvious malformed URLs and ensure they have a protocol (http:// or https://).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(value: Optional[str]) -> str:
    return value.strip() if isinstance(value, str) else ""


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if not s:
            continue
        # Keep only http/https URLs
        if not (s.startswith("http://") or s.startswith("https://")):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _gather_sources(field: FieldEvidence, extracted: ChipInfoExtraction) -> List[str]:
    urls: List[str] = []
    # Prefer the official spec sheet if present
    if extracted.spec_sheet_pdf_url:
        urls.append(extracted.spec_sheet_pdf_url)
    # Include field-specific sources
    if field and field.sources:
        urls.extend(field.sources)
    # Include any additional URLs mentioned
    if extracted.additional_urls:
        urls.extend(extracted.additional_urls)
    return _dedupe_urls(urls)


async def _add_and_verify_value_leaf(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    value: Optional[str],
    urls: List[str],
    claim_text: str,
    additional_instruction: str,
) -> bool:
    """
    Add a critical leaf node and verify the claim against the provided URLs.
    If value or URLs are missing, the leaf fails early without calling the verifier.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=True,
    )

    value_clean = _safe_str(value)
    if not value_clean:
        node.score = 0.0
        node.status = "failed"
        return False

    urls_clean = _dedupe_urls(urls)
    if not urls_clean:
        node.score = 0.0
        node.status = "failed"
        return False

    # Use the provided claim text directly (already formatted with the value)
    return await evaluator.verify(
        claim=claim_text,
        node=node,
        sources=urls_clean,
        additional_instruction=additional_instruction,
    )


async def _add_and_verify_spec_pdf_leaf(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    spec_pdf_url: Optional[str],
    chip_name_value: Optional[str],
) -> bool:
    """
    Verify that the provided URL is the official Google Quantum AI spec sheet PDF for the chip.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=True,
    )

    url_clean = _safe_str(spec_pdf_url)
    if not url_clean or not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        node.score = 0.0
        node.status = "failed"
        return False

    chip_name_text = _safe_str(chip_name_value)
    if chip_name_text:
        claim = (
            f"This URL is the official Google Quantum AI PDF specification sheet for the chip named '{chip_name_text}'."
        )
    else:
        claim = (
            "This URL is the official Google Quantum AI PDF specification sheet for the chip announced in December 2024 "
            "that demonstrated quantum error correction below the surface code threshold."
        )

    add_ins = (
        "Confirm this is a PDF specification or 'spec sheet' document published by Google/Google Quantum AI. "
        "Treat domains like quantumai.google, google.com, or Google-hosted storage as official. "
        "The PDF content should explicitly reference Google Quantum AI and the chip name (if provided) "
        "and should look like a technical specification sheet. If the URL is not a PDF, or not official, mark as not supported."
    )

    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=url_clean,
        additional_instruction=add_ins,
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate the answer for the Google Quantum AI December 2024 chip task.
    """

    # Initialize evaluator (root is non-critical by design; we'll add a critical sequential node beneath)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root container
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

    # Add the critical sequential node representing the rubric root
    seq_root = evaluator.add_sequential(
        id="google_quantum_chip_dec_2024_response",
        desc="Provide all requested information about the Google Quantum AI quantum chip announced in December 2024 that demonstrated quantum error correction below the surface code threshold.",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted: ChipInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_chip_info(),
        template_class=ChipInfoExtraction,
        extraction_name="chip_info_extraction",
    )

    # 1) Chip name
    chip_name_value = _safe_str(extracted.chip_name.value)
    chip_name_sources = _gather_sources(extracted.chip_name, extracted)
    chip_name_claim = (
        f"The Google Quantum AI chip announced in December 2024 that demonstrated quantum error correction below the "
        f"surface code threshold is named '{chip_name_value}'."
    )
    chip_name_add_ins = (
        "Use the provided sources, prioritizing the official specification sheet and Google Quantum AI publications. "
        "Confirm the official chip name exactly (allowing minor punctuation, capitalization, or hyphenation variants). "
        "Do not rely on the answer text itself as evidence."
    )
    await _add_and_verify_value_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="chip_name",
        desc="State the name of the quantum chip announced in December 2024 by Google Quantum AI.",
        value=chip_name_value,
        urls=chip_name_sources,
        claim_text=chip_name_claim,
        additional_instruction=chip_name_add_ins,
    )

    # 2) Total qubit count
    tq_value = _safe_str(extracted.total_qubit_count.value)
    tq_sources = _gather_sources(extracted.total_qubit_count, extracted)
    tq_claim = f"The total number of qubits on the chip is {tq_value}."
    tq_add_ins = (
        "Verify the total or overall physical qubit count on the chip from the sources (prefer the official spec sheet). "
        "Accept formatting differences (e.g., '1,024' vs '1024'). If the source uses grouped numbers or qualifiers, "
        "ensure it clearly states the total qubit count."
    )
    await _add_and_verify_value_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="total_qubit_count",
        desc="State the total number of qubits on the identified chip.",
        value=tq_value,
        urls=tq_sources,
        claim_text=tq_claim,
        additional_instruction=tq_add_ins,
    )

    # 3) Highest surface code distance
    scd_value = _safe_str(extracted.highest_surface_code_distance.value)
    scd_sources = _gather_sources(extracted.highest_surface_code_distance, extracted)
    scd_claim = (
        f"The highest surface code distance successfully demonstrated in the chip’s error correction experiments is {scd_value}."
    )
    scd_add_ins = (
        "Look for phrasing like 'surface code distance', 'code distance', or 'd=...' that indicates the highest achieved distance. "
        "Ensure the claim corresponds to the peak distance demonstrated, not a planned or theoretical target."
    )
    await _add_and_verify_value_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="highest_surface_code_distance",
        desc="Report the highest surface code distance demonstrated in the chip’s error correction experiments.",
        value=scd_value,
        urls=scd_sources,
        claim_text=scd_claim,
        additional_instruction=scd_add_ins,
    )

    # 4) Mean T1 coherence time (QEC Chip 1)
    t1_value = _safe_str(extracted.mean_t1_chip1_qec_us.value)
    t1_sources = _gather_sources(extracted.mean_t1_chip1_qec_us, extracted)
    t1_claim = (
        f"The mean T1 coherence time for the Quantum Error Correction configuration (Chip 1) is {t1_value} microseconds."
    )
    t1_add_ins = (
        "Confirm the 'mean T1' (average T1) for the QEC configuration (Chip 1) from the specification or supporting sources. "
        "Allow unit variants such as 'μs' and equivalence if the document provides clearly convertible units. "
        "If only median is provided and mean is not, do not accept unless the document explicitly equates them."
    )
    await _add_and_verify_value_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="mean_t1_chip1_qec",
        desc="Report the mean T1 coherence time (in microseconds) for the Quantum Error Correction configuration (Chip 1).",
        value=t1_value,
        urls=t1_sources,
        claim_text=t1_claim,
        additional_instruction=t1_add_ins,
    )

    # 5) QEC cycles per second
    cps_value = _safe_str(extracted.qec_cycles_per_second.value)
    cps_sources = _gather_sources(extracted.qec_cycles_per_second, extracted)
    cps_claim = (
        f"The Quantum Error Correction configuration (Chip 1) achieves {cps_value} error correction cycles per second."
    )
    cps_add_ins = (
        "Verify the QEC cycle rate (cycles per second). Sources may express this as 'kHz' (e.g., 2.5 kHz = 2500 cycles/s). "
        "Accept equivalent expressions if clearly the same rate. Ensure the value corresponds specifically to the QEC configuration (Chip 1)."
    )
    await _add_and_verify_value_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="qec_cycles_per_second",
        desc="State how many error correction cycles per second the QEC configuration (Chip 1) achieves.",
        value=cps_value,
        urls=cps_sources,
        claim_text=cps_claim,
        additional_instruction=cps_add_ins,
    )

    # 6) Spec sheet PDF URL
    await _add_and_verify_spec_pdf_leaf(
        evaluator=evaluator,
        parent=seq_root,
        leaf_id="spec_sheet_pdf_url",
        desc="Provide the URL to the official Google Quantum AI specification sheet PDF for this chip.",
        spec_pdf_url=extracted.spec_sheet_pdf_url,
        chip_name_value=chip_name_value,
    )

    # Return evaluation summary
    return evaluator.get_summary()