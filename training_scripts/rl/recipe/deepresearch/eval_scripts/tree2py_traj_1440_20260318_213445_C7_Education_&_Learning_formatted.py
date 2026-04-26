import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wcpss_board_dec2025"
TASK_DESCRIPTION = """
As of December 2025, provide comprehensive information about the Wake County Public School System (WCPSS) Board of Education. Your response must include: 
(1) The total number of members serving on the board, 
(2) The number of districts these members represent, 
(3) The term length (in years) for board members, 
(4) The full name of the current Board Chair, 
(5) The district number represented by the Board Chair, 
(6) The year the current Board Chair first assumed office as a board member (not as Chair), 
(7) The full name of the current Board Vice Chair, 
(8) The district number represented by the Board Vice Chair, 
(9) The year the current Board Vice Chair first assumed office as a board member (not as Vice Chair), 
(10) The full name of the newest board member (the member most recently sworn in), 
(11) The district number represented by this newest member, 
(12) The exact date (month, day, and year) when this newest member was sworn in, 
(13) The full name of the first Asian-American individual to serve on the WCPSS board, and 
(14) The district number represented by this first Asian-American member. 
All information must be accurate as of December 2025 and must be supported by verifiable sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SourcedValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BoardOverview(BaseModel):
    total_members: Optional[SourcedValue] = None
    number_districts: Optional[SourcedValue] = None
    term_length_years: Optional[SourcedValue] = None


class OfficerInfo(BaseModel):
    full_name: Optional[SourcedValue] = None
    district_number: Optional[SourcedValue] = None
    first_assumed_office_year: Optional[SourcedValue] = None


class NewestMemberInfo(BaseModel):
    full_name: Optional[SourcedValue] = None
    district_number: Optional[SourcedValue] = None
    sworn_in_date: Optional[SourcedValue] = None


class FirstAsianAmericanInfo(BaseModel):
    full_name: Optional[SourcedValue] = None
    district_number: Optional[SourcedValue] = None


class WCPSSBoardExtraction(BaseModel):
    board_overview: Optional[BoardOverview] = None
    board_chair: Optional[OfficerInfo] = None
    board_vice_chair: Optional[OfficerInfo] = None
    newest_member: Optional[NewestMemberInfo] = None
    first_asian_american_member: Optional[FirstAsianAmericanInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_wcpss_board_info() -> str:
    return """
    Extract the requested WCPSS Board of Education information AS PRESENTED IN THE ANSWER. 
    For each required field, return an object with:
      - value: the exact value stated in the answer (string; do not normalize; if a number appears, keep it as a string)
      - sources: a list of URL(s) explicitly cited in the answer that support this specific field. 
                 If a single citation supports multiple fields, duplicate that URL in each field's 'sources'.
                 Only include URLs that appear in the answer (plain or markdown links). Do not invent URLs.

    Required structure:
    {
      "board_overview": {
        "total_members": { "value": string or null, "sources": [urls...] },
        "number_districts": { "value": string or null, "sources": [urls...] },
        "term_length_years": { "value": string or null, "sources": [urls...] }
      },
      "board_chair": {
        "full_name": { "value": string or null, "sources": [urls...] },
        "district_number": { "value": string or null, "sources": [urls...] },
        "first_assumed_office_year": { "value": string or null, "sources": [urls...] }
      },
      "board_vice_chair": {
        "full_name": { "value": string or null, "sources": [urls...] },
        "district_number": { "value": string or null, "sources": [urls...] },
        "first_assumed_office_year": { "value": string or null, "sources": [urls...] }
      },
      "newest_member": {
        "full_name": { "value": string or null, "sources": [urls...] },
        "district_number": { "value": string or null, "sources": [urls...] },
        "sworn_in_date": { "value": string or null, "sources": [urls...] }
      },
      "first_asian_american_member": {
        "full_name": { "value": string or null, "sources": [urls...] },
        "district_number": { "value": string or null, "sources": [urls...] }
      }
    }

    Notes:
    - If any field is missing in the answer, set its value to null and sources to [].
    - For dates, keep the exact format as presented (e.g., "December 3, 2025").
    - For district numbers, keep the exact string (e.g., "District 6" or "6").
    - For term length, keep the exact phrasing (e.g., "four years", "4", or "staggered four-year terms").
    - Only extract URLs explicitly present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sv_value(sv: Optional[SourcedValue]) -> str:
    return (sv.value or "").strip() if sv else ""


def _sv_sources(sv: Optional[SourcedValue]) -> List[str]:
    if not sv or not sv.sources:
        return []
    return [u for u in sv.sources if isinstance(u, str) and u.strip()]


def _all_required_sourced_values(ex: WCPSSBoardExtraction) -> List[Tuple[str, Optional[SourcedValue]]]:
    items: List[Tuple[str, Optional[SourcedValue]]] = []
    if ex.board_overview:
        items.append(("total_members", ex.board_overview.total_members))
        items.append(("number_districts", ex.board_overview.number_districts))
        items.append(("term_length_years", ex.board_overview.term_length_years))
    else:
        items.extend([
            ("total_members", None),
            ("number_districts", None),
            ("term_length_years", None),
        ])

    if ex.board_chair:
        items.append(("chair_full_name", ex.board_chair.full_name))
        items.append(("chair_district_number", ex.board_chair.district_number))
        items.append(("chair_first_assumed_year", ex.board_chair.first_assumed_office_year))
    else:
        items.extend([
            ("chair_full_name", None),
            ("chair_district_number", None),
            ("chair_first_assumed_year", None),
        ])

    if ex.board_vice_chair:
        items.append(("vice_full_name", ex.board_vice_chair.full_name))
        items.append(("vice_district_number", ex.board_vice_chair.district_number))
        items.append(("vice_first_assumed_year", ex.board_vice_chair.first_assumed_office_year))
    else:
        items.extend([
            ("vice_full_name", None),
            ("vice_district_number", None),
            ("vice_first_assumed_year", None),
        ])

    if ex.newest_member:
        items.append(("newest_full_name", ex.newest_member.full_name))
        items.append(("newest_district_number", ex.newest_member.district_number))
        items.append(("newest_sworn_in_date", ex.newest_member.sworn_in_date))
    else:
        items.extend([
            ("newest_full_name", None),
            ("newest_district_number", None),
            ("newest_sworn_in_date", None),
        ])

    if ex.first_asian_american_member:
        items.append(("first_asian_american_full_name", ex.first_asian_american_member.full_name))
        items.append(("first_asian_american_district_number", ex.first_asian_american_member.district_number))
    else:
        items.extend([
            ("first_asian_american_full_name", None),
            ("first_asian_american_district_number", None),
        ])
    return items


def _every_required_field_has_source(ex: WCPSSBoardExtraction) -> bool:
    required = _all_required_sourced_values(ex)
    # Each required fact must have a non-empty value and at least one source URL.
    for field_name, sv in required:
        if not sv or not (sv.value and str(sv.value).strip()):
            return False
        urls = _sv_sources(sv)
        if len(urls) == 0:
            return False
    return True


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_timeframe(evaluator: Evaluator, parent_node) -> None:
    # Leaf: Timeframe stated
    timeframe_leaf = evaluator.add_leaf(
        id="Timeframe_AsOf_December_2025_Stated",
        desc="Response explicitly indicates the information is accurate/current as of December 2025.",
        parent=parent_node,
        critical=True
    )
    timeframe_claim = (
        "The answer explicitly states that the information is accurate/current as of December 2025 "
        "(allow acceptable equivalents like 'as of Dec. 2025', 'current through December 2025', or "
        "'updated December 2025')."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        additional_instruction="Judge based only on the provided answer text. Accept reasonable phrasing variations."
    )


async def build_and_verify_reliable_sourcing(evaluator: Evaluator, parent_node, extracted: WCPSSBoardExtraction) -> None:
    # Leaf (custom): All required facts have at least one source URL
    all_sourced = _every_required_field_has_source(extracted)
    evaluator.add_custom_node(
        result=all_sourced,
        id="Reliable_Verifiable_Sourcing_For_All_Required_Facts",
        desc="Each required fact/field is supported by at least one verifiable citation (e.g., URL to an authoritative/reputable source).",
        parent=parent_node,
        critical=True
    )

    # Also record a small breakdown for debugging
    required = _all_required_sourced_values(extracted)
    details = {
        "total_required_fields": len(required),
        "fields_with_value_and_source": sum(
            1 for _, sv in required if sv and (sv.value and str(sv.value).strip()) and len(_sv_sources(sv)) > 0
        ),
        "fields_missing_or_no_source": [
            name for name, sv in required if (not sv) or (not (sv.value and str(sv.value).strip())) or len(_sv_sources(sv)) == 0
        ],
    }
    evaluator.add_custom_info(details, "sourcing_completeness", "sourcing_overview")


async def build_and_verify_board_overview(evaluator: Evaluator, parent_node, ex: WCPSSBoardExtraction) -> None:
    node = evaluator.add_parallel(
        id="Board_Overview",
        desc="Provides the required board-wide numeric/governance facts.",
        parent=parent_node,
        critical=True
    )
    bo = ex.board_overview or BoardOverview()

    # Total number of members
    leaf_total = evaluator.add_leaf(
        id="Total_Number_Of_Board_Members",
        desc="States the total number of members serving on the board.",
        parent=node,
        critical=True
    )
    total_val = _sv_value(bo.total_members)
    await evaluator.verify(
        claim=f"As of December 2025, the total number of members serving on the Wake County Public School System (WCPSS) Board of Education is {total_val}.",
        node=leaf_total,
        sources=_sv_sources(bo.total_members),
        additional_instruction="Verify on official WCPSS Board pages or other reputable sources cited in the answer."
    )

    # Number of districts represented
    leaf_districts = evaluator.add_leaf(
        id="Number_Of_Districts_Represented",
        desc="States the number of districts these members represent.",
        parent=node,
        critical=True
    )
    num_districts_val = _sv_value(bo.number_districts)
    await evaluator.verify(
        claim=f"As of December 2025, WCPSS Board members represent {num_districts_val} districts.",
        node=leaf_districts,
        sources=_sv_sources(bo.number_districts),
        additional_instruction="The claimed number of districts should be explicitly supported by the cited pages."
    )

    # Term length
    leaf_term = evaluator.add_leaf(
        id="Board_Member_Term_Length_Years",
        desc="States the term length (in years) for board members.",
        parent=node,
        critical=True
    )
    term_val = _sv_value(bo.term_length_years)
    # Phrase claim to tolerate variants like "four-year terms" vs "4 years"
    await evaluator.verify(
        claim=f"As of December 2025, the term length for WCPSS Board members is {term_val} years (or an equivalent phrasing, e.g., 'four-year terms').",
        node=leaf_term,
        sources=_sv_sources(bo.term_length_years),
        additional_instruction="Allow minor phrasing variants such as 'four-year terms' vs '4 years'."
    )


async def build_and_verify_officer(
    evaluator: Evaluator,
    parent_node,
    officer_info: Optional[OfficerInfo],
    group_id: str,
    group_desc: str,
    full_name_leaf_id: str,
    district_leaf_id: str,
    first_assumed_year_leaf_id: str
) -> None:
    node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=True
    )
    oi = officer_info or OfficerInfo()

    # Full name
    leaf_name = evaluator.add_leaf(
        id=full_name_leaf_id,
        desc=f"Gives the full name of the current {group_desc.split(' ')[-2]} {group_desc.split(' ')[-1]}.".strip(),  # Keep rubric intention
        parent=node,
        critical=True
    )
    name_val = _sv_value(oi.full_name)
    await evaluator.verify(
        claim=f"As of December 2025, the current WCPSS Board of Education {group_desc.split(' ')[-1]} is {name_val}.",
        node=leaf_name,
        sources=_sv_sources(oi.full_name),
        additional_instruction="The cited source should clearly indicate the officer role (Chair or Vice Chair) with the person's name."
    )

    # District number
    leaf_district = evaluator.add_leaf(
        id=district_leaf_id,
        desc=f"States the district number represented by the {group_desc.split(' ')[-1]}.",
        parent=node,
        critical=True
    )
    district_val = _sv_value(oi.district_number)
    await evaluator.verify(
        claim=f"{name_val} represents District {district_val} on the WCPSS Board of Education.",
        node=leaf_district,
        sources=_sv_sources(oi.district_number),
        additional_instruction="Accept minor formatting variants like 'District 6' vs '6'. The citation must support the district."
    )

    # First assumed office year as member
    leaf_year = evaluator.add_leaf(
        id=first_assumed_year_leaf_id,
        desc=f"States the year the current {group_desc.split(' ')[-1]} first assumed office as a board member (not as {group_desc.split(' ')[-1]}).",
        parent=node,
        critical=True
    )
    year_val = _sv_value(oi.first_assumed_office_year)
    await evaluator.verify(
        claim=f"{name_val} first assumed office as a WCPSS board member in {year_val}.",
        node=leaf_year,
        sources=_sv_sources(oi.first_assumed_office_year),
        additional_instruction="The cited page should clearly indicate the year first elected/assumed office as a board member."
    )


async def build_and_verify_newest_member(evaluator: Evaluator, parent_node, nm_info: Optional[NewestMemberInfo]) -> None:
    node = evaluator.add_parallel(
        id="Newest_Board_Member",
        desc="Provides the required information about the newest board member (most recently sworn in) as of December 2025.",
        parent=parent_node,
        critical=True
    )
    nm = nm_info or NewestMemberInfo()

    # Full name
    leaf_name = evaluator.add_leaf(
        id="Newest_Member_Full_Name",
        desc="Gives the full name of the newest board member (most recently sworn in).",
        parent=node,
        critical=True
    )
    name_val = _sv_value(nm.full_name)
    await evaluator.verify(
        claim=f"As of December 2025, the newest (most recently sworn-in) WCPSS board member is {name_val}.",
        node=leaf_name,
        sources=_sv_sources(nm.full_name),
        additional_instruction="The cited source should clearly indicate that this person was the most recently sworn-in member."
    )

    # District number
    leaf_district = evaluator.add_leaf(
        id="Newest_Member_District_Number",
        desc="States the district number represented by this newest member.",
        parent=node,
        critical=True
    )
    district_val = _sv_value(nm.district_number)
    await evaluator.verify(
        claim=f"{name_val} represents District {district_val} on the WCPSS Board of Education.",
        node=leaf_district,
        sources=_sv_sources(nm.district_number),
        additional_instruction="Accept minor variants like 'District 3' vs '3'. The citation must support the district."
    )

    # Sworn-in date (exact)
    leaf_date = evaluator.add_leaf(
        id="Newest_Member_Sworn_In_Date_Exact",
        desc="States the exact sworn-in date (month, day, year) for the newest member.",
        parent=node,
        critical=True
    )
    sworn_date_val = _sv_value(nm.sworn_in_date)
    await evaluator.verify(
        claim=f"{name_val} was sworn in as a WCPSS board member on {sworn_date_val}.",
        node=leaf_date,
        sources=_sv_sources(nm.sworn_in_date),
        additional_instruction="The cited source should explicitly provide the sworn-in date with month, day, and year."
    )


async def build_and_verify_first_asian_american(
    evaluator: Evaluator, parent_node, faa_info: Optional[FirstAsianAmericanInfo]
) -> None:
    node = evaluator.add_parallel(
        id="First_Asian_American_Board_Member",
        desc="Provides the required information about the first Asian-American individual to serve on the WCPSS board.",
        parent=parent_node,
        critical=True
    )
    faa = faa_info or FirstAsianAmericanInfo()

    # Full name
    leaf_name = evaluator.add_leaf(
        id="First_Asian_American_Member_Full_Name",
        desc="Gives the full name of the first Asian-American individual to serve on the WCPSS board.",
        parent=node,
        critical=True
    )
    name_val = _sv_value(faa.full_name)
    await evaluator.verify(
        claim=f"The first Asian-American individual to serve on the WCPSS Board of Education is {name_val}.",
        node=leaf_name,
        sources=_sv_sources(faa.full_name),
        additional_instruction="The citation should explicitly support 'first Asian-American' or a clearly equivalent statement."
    )

    # District number
    leaf_district = evaluator.add_leaf(
        id="First_Asian_American_Member_District_Number",
        desc="States the district number represented by this first Asian-American board member.",
        parent=node,
        critical=True
    )
    district_val = _sv_value(faa.district_number)
    await evaluator.verify(
        claim=f"{name_val} represented District {district_val} on the WCPSS Board of Education.",
        node=leaf_district,
        sources=_sv_sources(faa.district_number),
        additional_instruction="The cited source should support the district represented by this person."
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
    Evaluate an answer for the WCPSS Board of Education (as of Dec 2025) task and return a structured result.
    """
    # 1) Initialize evaluator (root is non-critical; we create a critical top-level node below)
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

    # 2) Extract structured data from the answer
    extracted: WCPSSBoardExtraction = await evaluator.extract(
        prompt=prompt_extract_wcpss_board_info(),
        template_class=WCPSSBoardExtraction,
        extraction_name="wcpss_board_extraction"
    )

    # 3) Build critical top-level node reflecting rubric
    top = evaluator.add_parallel(
        id="WCPSS_Board_Information_AsOf_December_2025",
        desc="Provides the required WCPSS Board of Education information as of December 2025, with verifiable sourcing.",
        parent=root,
        critical=True
    )

    # 4) Build and verify leaves/groups according to rubric

    # 4.1 Timeframe leaf
    await build_and_verify_timeframe(evaluator, top)

    # 4.2 Reliable sourcing leaf (custom check that each required field has ≥1 source URL)
    await build_and_verify_reliable_sourcing(evaluator, top, extracted)

    # 4.3 Board overview (three leaves)
    await build_and_verify_board_overview(evaluator, top, extracted)

    # 4.4 Board Chair (three leaves)
    await build_and_verify_officer(
        evaluator=evaluator,
        parent_node=top,
        officer_info=extracted.board_chair,
        group_id="Board_Chair",
        group_desc="Provides the required information about the current Board Chair.",
        full_name_leaf_id="Board_Chair_Full_Name",
        district_leaf_id="Board_Chair_District_Number",
        first_assumed_year_leaf_id="Board_Chair_First_Assumed_Office_Year_As_Member"
    )

    # 4.5 Board Vice Chair (three leaves)
    await build_and_verify_officer(
        evaluator=evaluator,
        parent_node=top,
        officer_info=extracted.board_vice_chair,
        group_id="Board_Vice_Chair",
        group_desc="Provides the required information about the current Board Vice Chair.",
        full_name_leaf_id="Board_Vice_Chair_Full_Name",
        district_leaf_id="Board_Vice_Chair_District_Number",
        first_assumed_year_leaf_id="Board_Vice_Chair_First_Assumed_Office_Year_As_Member"
    )

    # 4.6 Newest Board Member (three leaves)
    await build_and_verify_newest_member(evaluator, top, extracted.newest_member)

    # 4.7 First Asian-American Board Member (two leaves)
    await build_and_verify_first_asian_american(evaluator, top, extracted.first_asian_american_member)

    # 5) Return evaluation summary
    return evaluator.get_summary()