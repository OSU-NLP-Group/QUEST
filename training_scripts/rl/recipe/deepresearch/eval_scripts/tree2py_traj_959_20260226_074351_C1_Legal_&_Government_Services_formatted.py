import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_venezuela_envoy_2026"
TASK_DESCRIPTION = (
    "Identify the US Chargé d'Affaires who arrived in Caracas in February 2026 to reopen the United States diplomatic mission in Venezuela after seven years of severed ties. "
    "Provide the representative's full name and the specific date of arrival."
)

TARGET_MONTH = 2
TARGET_YEAR = 2026


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class EnvoyExtraction(BaseModel):
    full_name: Optional[str] = None
    arrival_date: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_envoy_info() -> str:
    return """
    From the answer, extract the following for the event described as the US Chargé d'Affaires arriving in Caracas in February 2026 to reopen the US diplomatic mission in Venezuela:
    - full_name: The full name of the US Chargé d'Affaires (e.g., given name and family name). If missing, return null.
    - arrival_date: The specific arrival date provided in the answer (e.g., "February 14, 2026", "Feb 14, 2026", or ISO-like "2026-02-14"). If the answer only mentions a month without a specific day, return null.
    - source_urls: All URLs explicitly included in the answer that are cited to support this identification and date (e.g., links to credible news sites or official US government pages). Only include actual URLs mentioned in the answer text. If none are provided, return an empty list.

    Notes:
    - If multiple names or dates are present, choose the one that corresponds to the arrival in Caracas in February 2026 for reopening the US mission.
    - Do not invent URLs; strictly extract those present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions: date parsing and checks                                   #
# --------------------------------------------------------------------------- #
def _normalize_date_text(s: str) -> str:
    """Normalize common variations to help parsing."""
    s = s.strip()
    # Remove ordinal suffixes: 1st -> 1, 2nd -> 2, 3rd -> 3, 4th -> 4
    s = re.sub(r"(\b\d{1,2})(st|nd|rd|th)\b", r"\1", s, flags=re.IGNORECASE)
    # Normalize abbreviated month with dot: "Feb." -> "Feb"
    s = re.sub(r"\b(Feb)\.\b", r"\1", s, flags=re.IGNORECASE)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s


def _try_parse_date(date_text: str) -> Optional[datetime]:
    """Attempt to parse a date string using a set of common formats."""
    if not date_text:
        return None
    s = _normalize_date_text(date_text)

    patterns = [
        "%B %d, %Y",   # February 14, 2026
        "%b %d, %Y",   # Feb 14, 2026
        "%Y-%m-%d",    # 2026-02-14
        "%d %B %Y",    # 14 February 2026
        "%d %b %Y",    # 14 Feb 2026
        "%B %d %Y",    # February 14 2026
        "%b %d %Y",    # Feb 14 2026
        "%m/%d/%Y",    # 02/14/2026
        "%d/%m/%Y",    # 14/02/2026 (unlikely in US-centric answers, but acceptable)
    ]

    for fmt in patterns:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue

    return None


def is_specific_date_in_feb_2026(date_text: Optional[str]) -> bool:
    """Return True if the provided date_text parses to a specific date in February 2026."""
    if not date_text or not isinstance(date_text, str) or not date_text.strip():
        return False
    dt = _try_parse_date(date_text)
    if not dt:
        return False
    return dt.year == TARGET_YEAR and dt.month == TARGET_MONTH


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_name_verification(
    evaluator: Evaluator,
    parent_node,
    envoy: EnvoyExtraction
) -> None:
    """
    Build the 'Representative_Name_Provided' sequential branch:
      1) Existence of name in the answer (critical)
      2) Presence of at least one source URL (critical)
      3) Verification via cited sources that this person is indeed the US Chargé d'Affaires who arrived in Caracas in Feb 2026 to reopen the mission (critical)
    """
    name_node = evaluator.add_sequential(
        id="Representative_Name_Provided",
        desc="A full name is provided for the US Chargé d'Affaires who arrived in Caracas in February 2026 to reopen the diplomatic mission, and this name can be verified through reliable news sources",
        parent=parent_node,
        critical=True
    )

    # 1) Name existence
    name_exists = bool(envoy.full_name and envoy.full_name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="name_present",
        desc="The answer provides the envoy's full name",
        parent=name_node,
        critical=True
    )

    # 2) Sources provided (at least one URL)
    has_sources = bool(envoy.source_urls and any(u.strip() for u in envoy.source_urls))
    evaluator.add_custom_node(
        result=has_sources,
        id="name_sources_provided",
        desc="At least one source URL is provided to support the envoy's identification",
        parent=name_node,
        critical=True
    )

    # 3) Verify name against sources
    name_verify_leaf = evaluator.add_leaf(
        id="name_supported_by_sources",
        desc="The identified envoy's name is supported by the cited sources",
        parent=name_node,
        critical=True
    )

    full_name = envoy.full_name or ""
    claim = (
        f"The US Chargé d'Affaires who arrived in Caracas in February {TARGET_YEAR} to reopen the U.S. diplomatic mission in Venezuela is {full_name}."
    )
    await evaluator.verify(
        claim=claim,
        node=name_verify_leaf,
        sources=envoy.source_urls,
        additional_instruction=(
            "Rely strictly on the provided webpages. Confirm that they explicitly identify the person as the U.S. Chargé d’Affaires (allow diacritics variations such as 'chargé d’affaires' vs 'charge d'affaires') "
            f"and that the arrival occurred in Caracas in February {TARGET_YEAR} in the context of reopening or reestablishing the U.S. diplomatic mission. "
            "Minor phrasing differences are acceptable."
        ),
    )


async def build_date_verification(
    evaluator: Evaluator,
    parent_node,
    envoy: EnvoyExtraction
) -> None:
    """
    Build the 'Arrival_Date_Provided' sequential branch:
      1) Date string exists in the answer (critical)
      2) The date is a specific day in February 2026 (critical)
      3) Sources provided (critical)
      4) Verify the specific arrival date via cited sources (critical)
    """
    date_node = evaluator.add_sequential(
        id="Arrival_Date_Provided",
        desc="A specific arrival date in February 2026 is provided, and this date can be verified through reliable news sources",
        parent=parent_node,
        critical=True
    )

    # 1) Date existence
    date_exists = bool(envoy.arrival_date and envoy.arrival_date.strip())
    evaluator.add_custom_node(
        result=date_exists,
        id="date_present",
        desc="The answer provides a specific arrival date",
        parent=date_node,
        critical=True
    )

    # 2) Date is in February 2026 (and is specific day)
    in_feb_2026 = is_specific_date_in_feb_2026(envoy.arrival_date)
    evaluator.add_custom_node(
        result=in_feb_2026,
        id="date_in_feb_2026",
        desc=f"The provided date is a specific day in February {TARGET_YEAR}",
        parent=date_node,
        critical=True
    )

    # 3) Sources provided (at least one URL)
    has_sources = bool(envoy.source_urls and any(u.strip() for u in envoy.source_urls))
    evaluator.add_custom_node(
        result=has_sources,
        id="date_sources_provided",
        desc="At least one source URL is provided to support the arrival date",
        parent=date_node,
        critical=True
    )

    # 4) Verify date against sources
    date_verify_leaf = evaluator.add_leaf(
        id="date_supported_by_sources",
        desc="The provided arrival date is supported by the cited sources",
        parent=date_node,
        critical=True
    )

    date_str = envoy.arrival_date or ""
    claim = (
        f"The arrival date for the U.S. Chargé d'Affaires in Caracas to reopen the U.S. diplomatic mission is {date_str}, and this took place in February {TARGET_YEAR}."
    )
    await evaluator.verify(
        claim=claim,
        node=date_verify_leaf,
        sources=envoy.source_urls,
        additional_instruction=(
            f"Verify that the webpages explicitly support the stated arrival date ({date_str}) for the U.S. Chargé d'Affaires arriving in Caracas in February {TARGET_YEAR}. "
            "Allow minor phrasing variations (e.g., 'arrived on', 'arrival on'). "
            "If a source only mentions a relative day (e.g., 'Monday') without a date and cannot be unambiguously tied to a specific calendar date in February 2026, consider it insufficient."
        ),
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
    Evaluate an answer for the US Chargé d'Affaires in Caracas (February 2026) identification task.
    """
    # Initialize evaluator (root is non-critical by design)
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

    # Extract structured information from the answer
    envoy_info = await evaluator.extract(
        prompt=prompt_extract_envoy_info(),
        template_class=EnvoyExtraction,
        extraction_name="envoy_extraction",
    )

    # Record custom info (parsed date attempt for debugging/traceability)
    parsed_dt = _try_parse_date(envoy_info.arrival_date) if envoy_info and envoy_info.arrival_date else None
    evaluator.add_custom_info(
        info={
            "extracted_full_name": envoy_info.full_name,
            "extracted_arrival_date": envoy_info.arrival_date,
            "extracted_source_urls": envoy_info.source_urls,
            "parsed_arrival_date_iso": parsed_dt.strftime("%Y-%m-%d") if parsed_dt else None,
        },
        info_type="extraction_debug",
    )

    # Build top-level critical node representing the rubric root (since the framework's root is non-critical)
    top_node = evaluator.add_parallel(
        id="US_Venezuela_Envoy_Identification",
        desc="Correctly identify the US Chargé d'Affaires who arrived in Caracas in February 2026 to reopen the diplomatic mission",
        parent=root,
        critical=True,
    )

    # Build and run verification branches
    await build_name_verification(evaluator, top_node, envoy_info)
    await build_date_verification(evaluator, top_node, envoy_info)

    # Return evaluation summary
    return evaluator.get_summary()