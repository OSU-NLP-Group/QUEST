import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "online_mba_aacsb_under_28k"
TASK_DESCRIPTION = (
    "I am exploring affordable online MBA options and want to compare programs from AACSB-accredited institutions. "
    "Identify five distinct AACSB-accredited online MBA programs where the total program tuition cost is under $28,000. "
    "The programs must be fully online with no required campus visits. For each program, provide: "
    "(1) the university name, (2) a direct link to the official program webpage, and (3) the exact total tuition cost for completing the entire MBA program."
)

TUITION_THRESHOLD = 28000.0


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class MBAProgram(BaseModel):
    """One program entry extracted from the answer."""
    university_name: Optional[str] = None
    program_url: Optional[str] = None

    # Evidence URLs explicitly mentioned in the answer
    aacsb_sources: List[str] = Field(default_factory=list)              # AACSB listing and/or official page stating AACSB
    accreditation_sources: List[str] = Field(default_factory=list)      # Regional accreditation evidence (HLC, SACSCOC, etc.)
    online_novisit_sources: List[str] = Field(default_factory=list)     # Evidence of fully online, no campus visits
    tuition_sources: List[str] = Field(default_factory=list)            # Official page that states the exact total tuition

    # Stated total program tuition (string, as shown in the answer)
    total_tuition: Optional[str] = None


class MBAProgramsExtraction(BaseModel):
    """Top-level extraction: list of programs."""
    programs: List[MBAProgram] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract each AACSB-accredited online MBA program mentioned in the answer. For every program, return the following fields:
    - university_name: The university name exactly as stated in the answer.
    - program_url: A direct link to the official MBA program webpage, if present in the answer. If multiple URLs are provided, choose the URL most clearly corresponding to the official MBA program page (not a search results page or aggregator).
    - aacsb_sources: An array of URLs cited in the answer that explicitly verify AACSB accreditation (e.g., AACSB Accredited Schools directory page or the school's official page stating AACSB). Extract only actual URLs that appear in the answer.
    - accreditation_sources: An array of URLs cited in the answer that verify U.S. regional accreditation for the university (e.g., HLC, SACSCOC, MSCHE, WSCUC, NECHE, NWCCU). Extract only actual URLs that appear in the answer.
    - online_novisit_sources: An array of URLs cited in the answer that confirm the MBA program is fully online with no required campus visits/residency. Extract only actual URLs that appear in the answer. If the answer states this but does not provide a URL, return an empty array.
    - tuition_sources: An array of URLs cited in the answer that provide the exact total program tuition for completing the entire MBA. Extract only actual URLs that appear in the answer.
    - total_tuition: The exact total tuition amount for the entire MBA program as stated in the answer (string format, e.g., "$27,500", "USD 26,000", "approximately $25k", "24,900").

    Rules:
    - Do not invent information. Only extract content explicitly present in the answer.
    - Extract only valid URLs that appear in the answer (plain URLs or markdown links). If the answer mentions a source without a URL, do not include it.
    - If the answer lists more than five programs, still extract them all; the evaluator will select the first five later.
    - If any field is missing for a program, return null (for strings) or an empty array (for sources).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_university_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Simple normalization: lowercase, strip spaces, collapse inner whitespace
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def parse_amount_to_float(text: Optional[str]) -> Optional[float]:
    """
    Parse a textual tuition amount into a float in USD if possible.
    Handles formats like:
    - "$27,500"
    - "USD 26,000"
    - "27k"
    - "25,000 - 27,000"
    - "approx. $24,900"
    Returns the upper bound if a range is detected.
    """
    if not text:
        return None

    s = text.strip().lower()

    # Replace common currency markers
    s = s.replace("usd", "").replace("us$", "").replace("$", "").replace("dollars", "")
    s = s.replace(",", "").replace("approximately", "").replace("approx.", "").replace("approx", "")

    # Handle ranges "a - b" or "a to b"
    range_match = re.findall(r"(\d+(?:\.\d+)?\s*(?:k)?)\s*(?:-|to|–|—)\s*(\d+(?:\.\d+)?\s*(?:k)?)", s)
    if range_match:
        a_str, b_str = range_match[0]
        def to_num(val: str) -> Optional[float]:
            val = val.strip()
            if val.endswith("k"):
                try:
                    return float(val[:-1]) * 1000.0
                except:
                    return None
            try:
                return float(val)
            except:
                return None
        a = to_num(a_str)
        b = to_num(b_str)
        if a is not None and b is not None:
            return max(a, b)

    # Handle single "Nk" format (e.g., 27k)
    k_match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", s)
    if k_match:
        try:
            return float(k_match.group(1)) * 1000.0
        except:
            pass

    # Handle plain number
    num_match = re.search(r"(\d+(?:\.\d+)?)", s)
    if num_match:
        try:
            return float(num_match.group(1))
        except:
            return None

    return None


def combine_sources(*args: List[str | None]) -> List[str]:
    """Combine multiple iterables of URLs into a unique list, dropping None/empty."""
    seen = set()
    out: List[str] = []
    for lst in args:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            u = url.strip()
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_program(
    evaluator: Evaluator,
    parent_node,
    program: MBAProgram,
    index: int,
) -> None:
    """
    Build verification sub-tree and run checks for a single program.
    All children under this program node are critical checks as specified by the rubric.
    """
    prog_id = f"program_{index+1}"
    prog_node = evaluator.add_parallel(
        id=prog_id,
        desc=f"Program {index+1} satisfies all stated constraints and includes all required fields",
        parent=parent_node,
        critical=True  # Root is critical; its children must also be critical per framework rules
    )

    # 1) University name provided (existence check)
    evaluator.add_custom_node(
        result=bool(program.university_name and program.university_name.strip()),
        id=f"p{index+1}_university_name_provided",
        desc="Provides the university name",
        parent=prog_node,
        critical=True
    )

    # 2) Direct official program URL (verify page is official program page, not a search or aggregator)
    url_node = evaluator.add_leaf(
        id=f"p{index+1}_direct_official_program_url",
        desc="Provides a direct link to the official MBA program webpage (not a search result or non-program page)",
        parent=prog_node,
        critical=True
    )
    url_claim = (
        f"This webpage is the official university page for the online MBA program, not a search results page, news article, or third-party aggregator."
    )
    # If URL missing, pass None to sources; verification will likely fail; that's acceptable for this critical check.
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=program.program_url if program.program_url else None,
        additional_instruction=(
            "Check that the page content clearly represents the university's official Online MBA program page "
            "(e.g., on *.edu domains or official school subdomains). It should present program details (curriculum, "
            "admissions, tuition, etc.). If the URL is missing or the page is a search engine results page, marketing aggregator, "
            "or non-program page, mark as not supported."
        ),
    )

    # 3) AACSB accredited with verification
    aacsb_node = evaluator.add_leaf(
        id=f"p{index+1}_aacsb_accredited_with_verification",
        desc="States the program/school is AACSB-accredited and provides a verifiable citation (AACSB listing and/or official university accreditation page)",
        parent=prog_node,
        critical=True
    )
    aacsb_claim = (
        "The business school or institution offering this online MBA is AACSB-accredited."
    )
    aacsb_urls = program.aacsb_sources
    if aacsb_urls and len(aacsb_urls) > 0:
        await evaluator.verify(
            claim=aacsb_claim,
            node=aacsb_node,
            sources=aacsb_urls,
            additional_instruction=(
                "Prefer the AACSB Accredited Schools directory or official university pages explicitly stating AACSB accreditation. "
                "If none of the URLs substantiate AACSB accreditation, mark as not supported."
            )
        )
    else:
        # No citation provided -> fail this critical check
        evaluator.add_custom_node(
            result=False,
            id=f"p{index+1}_aacsb_accredited_with_verification_no_sources",
            desc="No AACSB verification source provided in the answer",
            parent=prog_node,
            critical=True
        )

    # 4) Regionally accredited in the United States (verification)
    regional_node = evaluator.add_leaf(
        id=f"p{index+1}_regionally_accredited_us",
        desc="States the university is regionally accredited in the United States with a verifiable citation",
        parent=prog_node,
        critical=True
    )
    regional_claim = (
        "The university is regionally accredited in the United States by a recognized accreditor (e.g., HLC, SACSCOC, MSCHE, WSCUC, NECHE, NWCCU)."
    )
    regional_urls = program.accreditation_sources
    if regional_urls and len(regional_urls) > 0:
        await evaluator.verify(
            claim=regional_claim,
            node=regional_node,
            sources=regional_urls,
            additional_instruction=(
                "Look for clear statements from recognized U.S. regional accreditors or the university's official accreditation page. "
                "If no valid citation is provided, mark as not supported."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"p{index+1}_regionally_accredited_us_no_sources",
            desc="No regional accreditation verification source provided in the answer",
            parent=prog_node,
            critical=True
        )

    # 5) Fully online with no required campus visits/residency (verification)
    online_node = evaluator.add_leaf(
        id=f"p{index+1}_fully_online_no_visits",
        desc="Confirms the MBA is 100% online with no required campus visits/residency, with a verifiable citation",
        parent=prog_node,
        critical=True
    )
    online_claim = "This MBA program is fully online and requires no campus visits or residencies."
    online_urls = combine_sources([program.program_url] if program.program_url else [], program.online_novisit_sources)
    if online_urls and len(online_urls) > 0:
        await evaluator.verify(
            claim=online_claim,
            node=online_node,
            sources=online_urls,
            additional_instruction=(
                "The evidence must explicitly indicate the program is fully online and that there are no required campus visits/residencies. "
                "If the official program page implies required on-campus components, mark as not supported."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"p{index+1}_fully_online_no_visits_no_sources",
            desc="No verification source provided for fully online with no campus visits",
            parent=prog_node,
            critical=True
        )

    # 6) Exact total tuition cost cited from an official source (verification)
    tuition_exact_node = evaluator.add_leaf(
        id=f"p{index+1}_total_tuition_exact_official_source",
        desc="Provides the exact total tuition cost to complete the entire MBA for out-of-state students and cites an official university source for that total",
        parent=prog_node,
        critical=True
    )
    tuition_claim = (
        f"The total tuition for the entire online MBA program is '{program.total_tuition}'."
        if program.total_tuition else
        "The exact total tuition for the entire online MBA program is stated on the provided official source."
    )
    tuition_urls = combine_sources([program.program_url] if program.program_url else [], program.tuition_sources)
    if tuition_urls and len(tuition_urls) > 0:
        await evaluator.verify(
            claim=tuition_claim,
            node=tuition_exact_node,
            sources=tuition_urls,
            additional_instruction=(
                "Verify that the page explicitly lists the exact total program tuition for completing the entire MBA (not just per-credit or per-semester figures). "
                "Favor out-of-state amounts if different residency rates exist. If only per-credit rates are shown and no total is provided, mark as not supported."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"p{index+1}_total_tuition_exact_official_source_no_sources",
            desc="No official source URL provided for exact total tuition",
            parent=prog_node,
            critical=True
        )

    # 7) Total tuition under $28,000 (constraint check on the stated total)
    under_node = evaluator.add_custom_node(
        result=(lambda v: (v is not None and v < TUITION_THRESHOLD))(parse_amount_to_float(program.total_tuition)),
        id=f"p{index+1}_total_tuition_under_28000",
        desc="The stated total out-of-state tuition cost is under $28,000",
        parent=prog_node,
        critical=True
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
    model: str = "o4-mini"
) -> Dict:
    """
    Entry point for evaluating an answer against the AACSB online MBA under $28,000 rubric.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,   # Root strategy per rubric
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
    # Set root as critical to reflect rubric; do this before adding children
    root.critical = True

    # Extract programs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=MBAProgramsExtraction,
        extraction_name="extracted_programs"
    )

    raw_count = len(extracted.programs)

    # Global critical checks at root
    evaluator.add_custom_node(
        result=(raw_count == 5),
        id="global_item_count",
        desc="Response includes exactly five MBA programs (no more, no fewer)",
        parent=root,
        critical=True
    )

    # Work with first five programs for detailed verification
    programs = extracted.programs[:5]
    # Pad with empties if fewer than 5 so tree still builds deterministically
    while len(programs) < 5:
        programs.append(MBAProgram())

    # Distinct university names among the five
    normalized_names = [normalize_university_name(p.university_name) for p in programs]
    # Ensure all five are non-null and unique
    distinct_check = (
        all(n is not None and n.strip() != "" for n in normalized_names) and
        len(set(normalized_names)) == 5
    )
    evaluator.add_custom_node(
        result=distinct_check,
        id="global_distinct_universities",
        desc="All five programs are from distinct universities (no duplicates)",
        parent=root,
        critical=True
    )

    # Add a custom info block for debugging uniqueness
    evaluator.add_custom_info(
        info={
            "raw_program_count_in_answer": raw_count,
            "first_five_universities": [p.university_name for p in programs]
        },
        info_type="debug",
        info_name="extraction_summary"
    )

    # Verify each program
    for idx, prog in enumerate(programs):
        await verify_program(evaluator, root, prog, idx)

    # Return summary
    return evaluator.get_summary()