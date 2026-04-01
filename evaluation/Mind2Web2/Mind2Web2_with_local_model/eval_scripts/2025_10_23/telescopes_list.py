import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from datetime import datetime
from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy
from mind2web2 import LLMClient

TASK_ID = "telescopes_list"
TASK_DESCRIPTION = """
Identify 5 astronomy instruments that became operational between 2005 and the present, and that have contributed to at least three scientific results since becoming operational. Instruments may include space telescopes, ground-based observatories, or instrument networks (e.g., interferometer arrays).

For each instrument, provide the following information:
1. Instrument Name
2. Instrument Type (e.g., space telescope, radio array, gravitational wave detector)
3. Year Became Operational
4. Operating Agency or Organization (e.g., NASA, ESA, ALMA Collaboration)
5. Three results detected or observed using the instrument
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class InstrumentNamesList(BaseModel):
    """List of instrument names extracted from the answer"""
    instrument_names: List[str] = Field(default_factory=list, description="List of astronomy instrument names")


class InstrumentBasicInfo(BaseModel):
    """Basic information about a single astronomy instrument"""
    name: Optional[str] = Field(default=None, description="Name of the astronomy instrument")
    type: Optional[str] = Field(default=None, description="Type of instrument")
    operational_year: Optional[str] = Field(default=None, description="Year became operational")
    agency: Optional[str] = Field(default=None, description="Operating agency (first one if multiple)")


class InstrumentLinks(BaseModel):
    """URLs related to an instrument"""
    urls: List[str] = Field(default_factory=list, description="All URLs related to this instrument")


class InstrumentResults(BaseModel):
    """Scientific results from an instrument"""
    results: List[str] = Field(default_factory=list, description="List of scientific results")


def prompt_extract_instrument_names() -> str:
    """Extract just the names of instruments"""
    return """
    Extract ONLY the names of astronomy instruments mentioned in the answer.

    List each instrument name exactly as it appears in the text.
    Include all instruments mentioned, in the order they appear.
    """


def prompt_extract_basic_info(instrument_name: str) -> str:
    """Extract basic information for a specific instrument"""
    return f"""
    For the astronomy instrument "{instrument_name}", extract the following information:

    - name: The instrument name (should be "{instrument_name}")
    - type: The type of instrument (e.g., space telescope, radio array)
    - operational_year: The year it became operational (as string)
    - agency: The operating agency or organization (if multiple are listed, extract ONLY the first one)

    Extract exactly as written in the text.
    """


def prompt_extract_urls(instrument_name: str) -> str:
    """Extract all URLs related to a specific instrument"""
    return f"""
    Extract ALL URLs that are associated with the astronomy instrument "{instrument_name}".

    Include any URL that appears near or in relation to this instrument's information.
    Extract complete URLs including the protocol (http:// or https://).
    """


def prompt_extract_results(instrument_name: str) -> str:
    """Extract scientific results for a specific instrument"""
    return f"""
    Extract the scientific results or discoveries made using the astronomy instrument "{instrument_name}".

    List each result as a separate item.
    Include all results mentioned for this instrument.
    Extract the results exactly as described in the text.
    """


async def verify_single_instrument(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        instrument_name: str,
        instrument_index: int,
) -> None:
    """Verify a single astronomy instrument with step-by-step extraction"""

    # Create node for this instrument
    instrument_node = evaluator.add_parallel(
        id=f"instrument_{instrument_index}",
        desc=f"Instrument {instrument_index}: {instrument_name}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # Step 1: Extract basic information
    basic_info = await evaluator.extract(
        prompt=prompt_extract_basic_info(instrument_name),
        template_class=InstrumentBasicInfo,
        extraction_name=f"instrument_{instrument_index}_basic_info",
        source=None,
    )

    # Step 2: Extract URLs
    urls_info = await evaluator.extract(
        prompt=prompt_extract_urls(instrument_name),
        template_class=InstrumentLinks,
        extraction_name=f"instrument_{instrument_index}_urls",
        source=None,
    )

    # Step 3: Extract scientific results
    results_info = await evaluator.extract(
        prompt=prompt_extract_results(instrument_name),
        template_class=InstrumentResults,
        extraction_name=f"instrument_{instrument_index}_results",
        source=None,
    )

    # Combined existence check for all required fields
    all_fields_exist = evaluator.add_custom_node(
        result=bool(
            basic_info.name and basic_info.name.strip() and
            basic_info.type and basic_info.type.strip() and
            basic_info.operational_year and basic_info.operational_year.strip() and
            basic_info.agency and basic_info.agency.strip() and
            results_info.results and len(results_info.results) >= 3 and
            urls_info.urls and len(urls_info.urls) > 0
        ),
        id=f"instrument_{instrument_index}_all_fields_exist",
        desc=f"Instrument {instrument_index} has all required fields (name, type, year, agency, 3+ results, URLs)",
        parent=instrument_node,
        critical=True,
    )

    # Verify operational year is valid (2005 or later)
    year_valid_node = evaluator.add_leaf(
        id=f"instrument_{instrument_index}_operational_year_valid",
        desc=f"Instrument {instrument_index} operational year is between 2005 and present",
        parent=instrument_node,
        critical=True,
    )

    # if basic_info.operational_year:

    current_year = datetime.now().year
    claim = f"The year {basic_info.operational_year} is between 2005 and {current_year}"
    await evaluator.verify(
        claim=claim,
        node=year_valid_node,
        additional_instruction=f"Check if the year is 2005 or later and not in the future (after {current_year}). For year ranges, check the starting year."
    )
    # Verify operational year with URLs
    year_verified_node = evaluator.add_leaf(
        id=f"instrument_{instrument_index}_operational_year_verified",
        desc=f"Instrument {instrument_index} operational year verified by sources",
        parent=instrument_node,
        critical=True,
    )

    # if basic_info.operational_year and urls_info.urls:
    claim = f"The astronomy instrument '{basic_info.name}' became operational in {basic_info.operational_year}"
    await evaluator.verify(
        claim=claim,
        node=year_verified_node,
        sources=urls_info.urls,
        additional_instruction="Verify that the instrument became operational in the stated year according to the source."
    )

    # Verify agency with URLs
    agency_verified_node = evaluator.add_leaf(
        id=f"instrument_{instrument_index}_agency_verified",
        desc=f"Instrument {instrument_index} operating agency verified by sources",
        parent=instrument_node,
        critical=True,
    )

    # if basic_info.agency and urls_info.urls:
    claim = f"The astronomy instrument '{basic_info.name}' is operated by {basic_info.agency}"
    await evaluator.verify(
        claim=claim,
        node=agency_verified_node,
        sources=urls_info.urls,
        additional_instruction="Verify that the stated agency or organization operates this instrument. Allow reasonable variations in the names of instrucments or agencies. For example, 'NASA' and 'National Aeronautics and Space Administration' should be considered equivalent. If the instrument is operated by a collaboration, verify that the given agency extracted from the answer is in the list."
    )

    # Verify scientific results (first 3 only)
    results_container = evaluator.add_parallel(
        id=f"instrument_{instrument_index}_results_verification",
        desc=f"Instrument {instrument_index} scientific results verification",
        parent=instrument_node,
        critical=False,  # Non-critical to allow partial credit for results
    )

    # Verify first 3 results
    num_results_to_verify = min(3, len(results_info.results) if results_info.results else 0)

    for i in range(3):  # Always create 3 nodes
        if i < num_results_to_verify and urls_info.urls:
            result_node = evaluator.add_leaf(
                id=f"instrument_{instrument_index}_result_{i + 1}_verified",
                desc=f"Instrument {instrument_index} result {i + 1} verified",
                parent=results_container,
                critical=True,
            )

            result_text = results_info.results[i] if results_info and results_info.results else "Not available. Give it a False"
            claim = f"The astronomy instrument '{basic_info.name}' contributed to or enabled the following scientific result: {result_text}"

            await evaluator.verify(
                claim=claim,
                node=result_node,
                sources=urls_info.urls,
                additional_instruction="Verify that this instrument was involved in or enabled this scientific discovery/result."
            )
        else:
            # Create failed node for missing result
            evaluator.add_leaf(
                id=f"instrument_{instrument_index}_result_{i + 1}_verified",
                desc=f"Instrument {instrument_index} result {i + 1} not provided",
                parent=results_container,
                critical=True,
                score=0.0,
                status="failed"
            )


async def create_placeholder_instrument(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        instrument_index: int,
) -> None:
    """Create placeholder nodes for a missing instrument"""

    instrument_node = evaluator.add_parallel(
        id=f"instrument_{instrument_index}",
        desc=f"Instrument {instrument_index}: Not provided",
        parent=parent_node,
        critical=False,
    )

    # All fields missing
    evaluator.add_leaf(
        id=f"instrument_{instrument_index}_all_fields_exist",
        desc=f"Instrument {instrument_index} has all required fields",
        parent=instrument_node,
        critical=True,
        score=0.0,
        status="failed"
    )

    # Other checks also failed
    for check in ["operational_year_valid", "operational_year_verified", "agency_verified"]:
        evaluator.add_leaf(
            id=f"instrument_{instrument_index}_{check}",
            desc=f"Instrument {instrument_index} {check.replace('_', ' ')}",
            parent=instrument_node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Results verification container with failed results
    results_container = evaluator.add_parallel(
        id=f"instrument_{instrument_index}_results_verification",
        desc=f"Instrument {instrument_index} scientific results verification",
        parent=instrument_node,
        critical=False,
    )

    for i in range(3):
        evaluator.add_leaf(
            id=f"instrument_{instrument_index}_result_{i + 1}_verified",
            desc=f"Instrument {instrument_index} result {i + 1} not provided",
            parent=results_container,
            critical=True,
            score=0.0,
            status="failed"
        )


async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for astronomy instruments task
    """

    # 1. Initialize evaluator
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

    # 2. Extract instrument names first
    names_list = await evaluator.extract(
        prompt=prompt_extract_instrument_names(),
        template_class=InstrumentNamesList,
        extraction_name="instrument_names_extraction",
        source=None,
    )

    # 3. Check if any instruments were extracted
    instruments_exist = evaluator.add_custom_node(
        result=bool(names_list.instrument_names and len(names_list.instrument_names) > 0),
        id="instruments_list_exists",
        desc="At least one instrument name was identified",
        parent=root,
        critical=True,
    )

    # 4. Create instruments verification container
    instruments_container = evaluator.add_parallel(
        id="instruments_verification",
        desc="Verification of all astronomy instruments",
        parent=root,
        critical=False,  # Non-critical to allow partial credit
    )

    # 5. Process first 5 instruments
    num_to_verify = 5
    for i in range(num_to_verify):
        if i < len(names_list.instrument_names):
            # Verify existing instrument
            await verify_single_instrument(
                evaluator,
                instruments_container,
                names_list.instrument_names[i],
                i + 1
            )
        else:
            # Create placeholder for missing instrument
            await create_placeholder_instrument(
                evaluator,
                instruments_container,
                i + 1
            )

    # 6. Return evaluation results
    return evaluator.get_summary()