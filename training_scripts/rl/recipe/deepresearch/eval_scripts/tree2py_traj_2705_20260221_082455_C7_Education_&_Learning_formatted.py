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
TASK_ID = "texas_school_districts_identification"
TASK_DESCRIPTION = """
Identify three specific Texas public school districts that meet the following criteria, and for each district provide the requested information as of the 2024-2025 school year:

District A: A Texas school district that is ranked 7th largest in the state by enrollment. Provide: (1) the district name, (2) the current student enrollment number, (3) the total number of campuses, and (4) the county where the district is primarily located.

District B: A Texas school district with an enrollment between 45,000 and 50,000 students. Provide: (1) the district name, (2) the exact enrollment number as of the 2023-2024 school year, (3) the total number of schools/campuses, and (4) the primary county or counties served.

District C: A Texas school district that serves the city of Pearland and has an enrollment between 20,000 and 25,000 students. Provide: (1) the district name, (2) the current enrollment number, (3) the total number of campuses, and (4) the main city or cities served by the district.

For each district, all information must be supported by official sources or reliable news reports.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictAExtraction(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    # Ranking support
    rank_7th_sources: List[str] = Field(default_factory=list)

    # Enrollment current (2024–2025)
    enrollment_2024_2025: Optional[str] = None
    enrollment_2024_2025_sources: List[str] = Field(default_factory=list)

    # Campus count current (2024–2025)
    campus_count_2024_2025: Optional[str] = None
    campus_count_2024_2025_sources: List[str] = Field(default_factory=list)

    # Primary county (where district is primarily located)
    primary_county: Optional[str] = None
    primary_county_sources: List[str] = Field(default_factory=list)


class DistrictBExtraction(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    # Enrollment 2023–2024 exact
    enrollment_2023_2024: Optional[str] = None
    enrollment_2023_2024_sources: List[str] = Field(default_factory=list)

    # Campus count (total schools/campuses)
    campus_count: Optional[str] = None
    campus_count_sources: List[str] = Field(default_factory=list)

    # Primary counties served
    primary_counties_served: List[str] = Field(default_factory=list)
    primary_counties_sources: List[str] = Field(default_factory=list)


class DistrictCExtraction(BaseModel):
    name: Optional[str] = None
    name_sources: List[str] = Field(default_factory=list)

    # Serves the city of Pearland
    serves_pearland_sources: List[str] = Field(default_factory=list)

    # Enrollment current (2024–2025)
    enrollment_2024_2025: Optional[str] = None
    enrollment_2024_2025_sources: List[str] = Field(default_factory=list)

    # Campus count current (2024–2025)
    campus_count_2024_2025: Optional[str] = None
    campus_count_2024_2025_sources: List[str] = Field(default_factory=list)

    # Main city or cities served
    main_cities_served: List[str] = Field(default_factory=list)
    main_cities_sources: List[str] = Field(default_factory=list)


class TexasDistrictsExtraction(BaseModel):
    district_a: Optional[DistrictAExtraction] = None
    district_b: Optional[DistrictBExtraction] = None
    district_c: Optional[DistrictCExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract structured information for three Texas public school districts labeled (or logically identifiable as) District A, District B, and District C from the provided answer. Return a JSON object with fields `district_a`, `district_b`, and `district_c`. For each district, extract the following:

    General source rules:
    - For EVERY fact extracted (name, ranking, enrollment, campus count, county/cities served), also extract the citation URLs explicitly provided in the answer that support that fact.
    - Extract only valid, explicit URLs present in the answer (including markdown links). Do not invent URLs.
    - If a fact is mentioned but no URL is provided for it in the answer, leave the corresponding `*_sources` field as an empty array.
    - If a fact itself is not provided in the answer, set that fact field to null (and the sources list to an empty array).

    District A (7th largest in Texas by enrollment):
    - name (string) and name_sources (array of URLs)
    - rank_7th_sources (array of URLs that explicitly support “ranked 7th largest in Texas by enrollment”)
    - enrollment_2024_2025 (string as written in the answer, e.g., "73,200") and enrollment_2024_2025_sources (array of URLs)
    - campus_count_2024_2025 (string, e.g., "71") and campus_count_2024_2025_sources (array of URLs)
    - primary_county (string, e.g., "Montgomery County") and primary_county_sources (array of URLs)

    District B (enrollment between 45,000 and 50,000 as of 2023–2024):
    - name (string) and name_sources (array of URLs)
    - enrollment_2023_2024 (string as written in the answer) and enrollment_2023_2024_sources (array of URLs)
    - campus_count (string) and campus_count_sources (array of URLs)
    - primary_counties_served (array of strings) and primary_counties_sources (array of URLs)

    District C (serves the city of Pearland; enrollment between 20,000 and 25,000 as of 2024–2025):
    - name (string) and name_sources (array of URLs)
    - serves_pearland_sources (array of URLs that show the district serves the city of Pearland)
    - enrollment_2024_2025 (string as written in the answer) and enrollment_2024_2025_sources (array of URLs)
    - campus_count_2024_2025 (string) and campus_count_2024_2025_sources (array of URLs)
    - main_cities_served (array of strings) and main_cities_sources (array of URLs)

    Notes:
    - Keep numbers as strings exactly as presented in the answer (e.g., "73,000+" or "approx. 48,500"). Do not convert to numeric in the extraction.
    - If the answer provides multiple URLs for a fact, include all of them in the corresponding sources array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty_str(s: Optional[str]) -> str:
    return s.strip() if (s is not None) else ""

def _has_sources(urls: List[str]) -> bool:
    return bool(urls) and len(urls) > 0

def _join_list(items: List[str]) -> str:
    return ", ".join([i.strip() for i in items if i and i.strip()])

def _all_required_sources_present(extracted: TexasDistrictsExtraction) -> bool:
    """
    Global source presence check:
    Return True only if every required fact for A/B/C has at least one citation URL present in the answer.
    This is a strict presence check; detailed correspondence is separately verified per leaf with URLs.
    """
    # District A requirements
    ok_a = True
    if extracted.district_a is None:
        ok_a = False
    else:
        a = extracted.district_a
        ok_a = (
            _nonempty_str(a.name) != "" and _has_sources(a.name_sources) and
            _has_sources(a.rank_7th_sources) and
            _nonempty_str(a.enrollment_2024_2025) != "" and _has_sources(a.enrollment_2024_2025_sources) and
            _nonempty_str(a.campus_count_2024_2025) != "" and _has_sources(a.campus_count_2024_2025_sources) and
            _nonempty_str(a.primary_county) != "" and _has_sources(a.primary_county_sources)
        )

    # District B requirements
    ok_b = True
    if extracted.district_b is None:
        ok_b = False
    else:
        b = extracted.district_b
        ok_b = (
            _nonempty_str(b.name) != "" and _has_sources(b.name_sources) and
            _nonempty_str(b.enrollment_2023_2024) != "" and _has_sources(b.enrollment_2023_2024_sources) and
            _nonempty_str(b.campus_count) != "" and _has_sources(b.campus_count_sources) and
            len(b.primary_counties_served) > 0 and _has_sources(b.primary_counties_sources)
        )

    # District C requirements
    ok_c = True
    if extracted.district_c is None:
        ok_c = False
    else:
        c = extracted.district_c
        ok_c = (
            _nonempty_str(c.name) != "" and _has_sources(c.name_sources) and
            _has_sources(c.serves_pearland_sources) and
            _nonempty_str(c.enrollment_2024_2025) != "" and _has_sources(c.enrollment_2024_2025_sources) and
            _nonempty_str(c.campus_count_2024_2025) != "" and _has_sources(c.campus_count_2024_2025_sources) and
            len(c.main_cities_served) > 0 and _has_sources(c.main_cities_sources)
        )

    return ok_a and ok_b and ok_c


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_global_source_requirement(evaluator: Evaluator, parent_node, extracted: TexasDistrictsExtraction) -> None:
    """
    Global source presence requirement - critical.
    Ensures every required fact across A/B/C has at least one citation URL. This is a strict presence gate.
    """
    result = _all_required_sources_present(extracted)
    evaluator.add_custom_node(
        result=result,
        id="Global_Source_Requirement",
        desc="All reported district facts are supported by official sources or reliable news reports (citations provided and correspond to the stated facts).",
        parent=parent_node,
        critical=True
    )


async def verify_district_a(evaluator: Evaluator, parent_node, a: Optional[DistrictAExtraction]) -> None:
    node = evaluator.add_parallel(
        id="District_A",
        desc="District A requirements and requested attributes.",
        parent=parent_node,
        critical=False
    )

    # Name provided gate
    name_provided = evaluator.add_custom_node(
        result=(a is not None and _nonempty_str(a.name) != "" and _has_sources(a.name_sources)),
        id="A_Name_Provided",
        desc="District A name is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Name verify
    name_leaf = evaluator.add_leaf(
        id="A_District_Name",
        desc="Provide District A district name.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district's official name is '{_nonempty_str(a.name)}'.",
        node=name_leaf,
        sources=a.name_sources if a else [],
        additional_instruction="Verify that the cited official page or reliable report clearly shows the district's official name."
    )

    # Ranked 7th gate
    rank_gate = evaluator.add_custom_node(
        result=(a is not None and _has_sources(a.rank_7th_sources)),
        id="A_Rank_Sources_Provided",
        desc="District A 'ranked 7th largest' claim has citation(s).",
        parent=node,
        critical=True
    )
    # Ranked 7th verify
    rank_leaf = evaluator.add_leaf(
        id="A_Ranked_7th_Largest",
        desc="District A is ranked 7th largest in Texas by enrollment, supported by a citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_nonempty_str(a.name)}' is ranked 7th largest Texas school district by enrollment.",
        node=rank_leaf,
        sources=a.rank_7th_sources if a else [],
        additional_instruction="The source should explicitly state '7th largest' statewide by enrollment or equivalent wording for the specified timeframe."
    )

    # Enrollment current gate
    enroll_gate = evaluator.add_custom_node(
        result=(a is not None and _nonempty_str(a.enrollment_2024_2025) != "" and _has_sources(a.enrollment_2024_2025_sources)),
        id="A_Enrollment_Current_Provided",
        desc="District A current (2024–2025) enrollment is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Enrollment current supported
    enroll_supported_leaf = evaluator.add_leaf(
        id="A_Enrollment_Current_Supported",
        desc="District A current (2024–2025) enrollment number is supported by cited source(s).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current (2024–2025) enrollment of '{_nonempty_str(a.name)}' is {_nonempty_str(a.enrollment_2024_2025)} students.",
        node=enroll_supported_leaf,
        sources=a.enrollment_2024_2025_sources if a else [],
        additional_instruction="Allow reasonable rounding or formatting (commas, plus signs). Confirm the figure corresponds to 2024–2025."
    )
    # Enrollment > 73,000 logical check
    enroll_range_leaf = evaluator.add_leaf(
        id="A_Enrollment_Current_And_Over_73000",
        desc="Provide District A current (2024–2025) enrollment number with citation, and the value is > 73,000.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The enrollment figure '{_nonempty_str(a.enrollment_2024_2025)}' is greater than 73,000.",
        node=enroll_range_leaf,
        additional_instruction="Extract the numeric value from the string and judge whether it is strictly greater than 73,000. Consider reasonable rounding."
    )

    # Campus count current gate
    campus_gate = evaluator.add_custom_node(
        result=(a is not None and _nonempty_str(a.campus_count_2024_2025) != "" and _has_sources(a.campus_count_2024_2025_sources)),
        id="A_Campus_Count_Current_Provided",
        desc="District A current (2024–2025) campus count is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Campus count supported
    campus_supported_leaf = evaluator.add_leaf(
        id="A_Campus_Count_Current_Supported",
        desc="District A current (2024–2025) total number of campuses is supported by cited source(s).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_nonempty_str(a.name)}' has {_nonempty_str(a.campus_count_2024_2025)} total campuses.",
        node=campus_supported_leaf,
        sources=a.campus_count_2024_2025_sources if a else [],
        additional_instruction="Confirm the number refers to total schools/campuses and corresponds to the 2024–2025 timeframe."
    )
    # Campus count equals 71 logical check
    campus_equals_leaf = evaluator.add_leaf(
        id="A_Campus_Count_Current_And_Equals_71",
        desc="Provide District A current (2024–2025) total number of campuses with citation, and the value is 71.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The campus count '{_nonempty_str(a.campus_count_2024_2025)}' equals 71.",
        node=campus_equals_leaf,
        additional_instruction="Extract the numeric value and judge equality to 71. Be tolerant to minor textual formatting."
    )

    # Primary county gate
    county_gate = evaluator.add_custom_node(
        result=(a is not None and _nonempty_str(a.primary_county) != "" and _has_sources(a.primary_county_sources)),
        id="A_Primary_County_Provided",
        desc="District A primary county is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Primary county equals Montgomery verify
    county_leaf = evaluator.add_leaf(
        id="A_Primary_County_Equals_Montgomery",
        desc="Provide the county where District A is primarily located with citation, and it is Montgomery County, Texas.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The district is primarily located in Montgomery County, Texas.",
        node=county_leaf,
        sources=a.primary_county_sources if a else [],
        additional_instruction="Confirm the primary location is Montgomery County, Texas on the cited pages (official or reliable sources)."
    )


async def verify_district_b(evaluator: Evaluator, parent_node, b: Optional[DistrictBExtraction]) -> None:
    node = evaluator.add_parallel(
        id="District_B",
        desc="District B requirements and requested attributes.",
        parent=parent_node,
        critical=False
    )

    # Name provided gate
    name_provided = evaluator.add_custom_node(
        result=(b is not None and _nonempty_str(b.name) != "" and _has_sources(b.name_sources)),
        id="B_Name_Provided",
        desc="District B name is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Name verify
    name_leaf = evaluator.add_leaf(
        id="B_District_Name",
        desc="Provide District B district name.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district's official name is '{_nonempty_str(b.name)}'.",
        node=name_leaf,
        sources=b.name_sources if b else [],
        additional_instruction="Verify the official district name on the cited official page or reliable report."
    )

    # Enrollment 2023–2024 gate
    enroll_gate = evaluator.add_custom_node(
        result=(b is not None and _nonempty_str(b.enrollment_2023_2024) != "" and _has_sources(b.enrollment_2023_2024_sources)),
        id="B_Enrollment_2023_2024_Provided",
        desc="District B enrollment (2023–2024) is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Enrollment supported
    enroll_supported_leaf = evaluator.add_leaf(
        id="B_Enrollment_2023_2024_Supported",
        desc="District B enrollment (2023–2024) is supported by cited source(s).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The 2023–2024 enrollment of '{_nonempty_str(b.name)}' is {_nonempty_str(b.enrollment_2023_2024)} students.",
        node=enroll_supported_leaf,
        sources=b.enrollment_2023_2024_sources if b else [],
        additional_instruction="Confirm the figure corresponds to the 2023–2024 school year and allow reasonable rounding or formatting."
    )
    # Enrollment range check
    enroll_range_leaf = evaluator.add_leaf(
        id="B_Enrollment_2023_2024_In_Range",
        desc="Provide District B exact enrollment as of the 2023–2024 school year with citation, and the value is between 45,000 and 50,000 (inclusive).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The enrollment figure '{_nonempty_str(b.enrollment_2023_2024)}' is between 45,000 and 50,000 inclusive.",
        node=enroll_range_leaf,
        additional_instruction="Extract the numeric value and judge whether it lies in [45,000, 50,000]. Consider rounding and thousand separators."
    )

    # Campus count gate
    campus_gate = evaluator.add_custom_node(
        result=(b is not None and _nonempty_str(b.campus_count) != "" and _has_sources(b.campus_count_sources)),
        id="B_Campus_Count_Provided",
        desc="District B campus count is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Campus count verify
    campus_leaf = evaluator.add_leaf(
        id="B_Campus_Count",
        desc="Provide District B total number of schools/campuses with citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_nonempty_str(b.name)}' has {_nonempty_str(b.campus_count)} total schools/campuses.",
        node=campus_leaf,
        sources=b.campus_count_sources if b else [],
        additional_instruction="Confirm total number of schools/campuses (not programs), using official or reliable sources."
    )

    # Primary counties gate
    counties_gate = evaluator.add_custom_node(
        result=(b is not None and len(b.primary_counties_served) > 0 and _has_sources(b.primary_counties_sources)),
        id="B_Primary_Counties_Provided",
        desc="District B primary county/counties served are provided with citation(s).",
        parent=node,
        critical=True
    )
    # Primary counties verify
    counties_leaf = evaluator.add_leaf(
        id="B_Primary_Counties_Served",
        desc="Provide District B primary county or counties served with citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district primarily serves the following counties: {_join_list(b.primary_counties_served) if b else ''}.",
        node=counties_leaf,
        sources=b.primary_counties_sources if b else [],
        additional_instruction="Accept reasonable county descriptions; verify that the cited sources indicate the district's primary service area includes these counties."
    )


async def verify_district_c(evaluator: Evaluator, parent_node, c: Optional[DistrictCExtraction]) -> None:
    node = evaluator.add_parallel(
        id="District_C",
        desc="District C requirements and requested attributes.",
        parent=parent_node,
        critical=False
    )

    # Name provided gate
    name_provided = evaluator.add_custom_node(
        result=(c is not None and _nonempty_str(c.name) != "" and _has_sources(c.name_sources)),
        id="C_Name_Provided",
        desc="District C name is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Name verify
    name_leaf = evaluator.add_leaf(
        id="C_District_Name",
        desc="Provide District C district name.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district's official name is '{_nonempty_str(c.name)}'.",
        node=name_leaf,
        sources=c.name_sources if c else [],
        additional_instruction="Verify the official district name on the cited official page or reliable report."
    )

    # Serves Pearland gate
    pearland_gate = evaluator.add_custom_node(
        result=(c is not None and _has_sources(c.serves_pearland_sources)),
        id="C_Serves_Pearland_Sources_Provided",
        desc="District C 'serves the city of Pearland' claim has citation(s).",
        parent=node,
        critical=True
    )
    # Serves Pearland verify
    pearland_leaf = evaluator.add_leaf(
        id="C_Serves_Pearland",
        desc="District C serves the city of Pearland, supported by a citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_nonempty_str(c.name)}' serves the city of Pearland.",
        node=pearland_leaf,
        sources=c.serves_pearland_sources if c else [],
        additional_instruction="Confirm the district boundary/service area includes the city of Pearland."
    )

    # Enrollment current gate
    enroll_gate = evaluator.add_custom_node(
        result=(c is not None and _nonempty_str(c.enrollment_2024_2025) != "" and _has_sources(c.enrollment_2024_2025_sources)),
        id="C_Enrollment_Current_Provided",
        desc="District C current (2024–2025) enrollment is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Enrollment supported
    enroll_supported_leaf = evaluator.add_leaf(
        id="C_Enrollment_Current_Supported",
        desc="District C current (2024–2025) enrollment number is supported by cited source(s).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current (2024–2025) enrollment of '{_nonempty_str(c.name)}' is {_nonempty_str(c.enrollment_2024_2025)} students.",
        node=enroll_supported_leaf,
        sources=c.enrollment_2024_2025_sources if c else [],
        additional_instruction="Allow reasonable rounding or formatting (commas, plus signs). Confirm the figure corresponds to 2024–2025."
    )
    # Enrollment range check
    enroll_range_leaf = evaluator.add_leaf(
        id="C_Enrollment_Current_In_Range",
        desc="Provide District C current (2024–2025) enrollment number with citation, and the value is between 20,000 and 25,000 (inclusive).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The enrollment figure '{_nonempty_str(c.enrollment_2024_2025)}' is between 20,000 and 25,000 inclusive.",
        node=enroll_range_leaf,
        additional_instruction="Extract the numeric value and judge whether it lies in [20,000, 25,000]. Consider rounding and thousand separators."
    )

    # Campus count current gate
    campus_gate = evaluator.add_custom_node(
        result=(c is not None and _nonempty_str(c.campus_count_2024_2025) != "" and _has_sources(c.campus_count_2024_2025_sources)),
        id="C_Campus_Count_Current_Provided",
        desc="District C current (2024–2025) campus count is provided with citation(s).",
        parent=node,
        critical=True
    )
    # Campus count supported
    campus_leaf = evaluator.add_leaf(
        id="C_Campus_Count_Current",
        desc="Provide District C current (2024–2025) total number of campuses with citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"'{_nonempty_str(c.name)}' has {_nonempty_str(c.campus_count_2024_2025)} total campuses.",
        node=campus_leaf,
        sources=c.campus_count_2024_2025_sources if c else [],
        additional_instruction="Confirm total number of schools/campuses for 2024–2025 using official or reliable sources."
    )

    # Main cities served gate
    cities_gate = evaluator.add_custom_node(
        result=(c is not None and len(c.main_cities_served) > 0 and _has_sources(c.main_cities_sources)),
        id="C_Main_Cities_Served_Provided",
        desc="District C main city/cities served are provided with citation(s).",
        parent=node,
        critical=True
    )
    # Main cities served verify
    cities_leaf = evaluator.add_leaf(
        id="C_Main_Cities_Served",
        desc="Identify the main city or cities served by District C with citation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The district's main city/cities served include: {_join_list(c.main_cities_served) if c else ''}.",
        node=cities_leaf,
        sources=c.main_cities_sources if c else [],
        additional_instruction="Verify that the cited sources explicitly mention these city/cities as being served by the district."
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
    Evaluate an answer for the Texas school districts identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: A, B, C independent; global source requirement applies
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
    # Important note: The framework requires that critical parents have all critical children.
    # Thus we set root as non-critical (default) to allow a mix of critical and non-critical child nodes.

    # Extract districts info from answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=TexasDistrictsExtraction,
        extraction_name="texas_districts_extraction"
    )

    # Add a custom info entry to record the numeric thresholds used in evaluation
    evaluator.add_custom_info(
        {
            "A_enrollment_threshold_gt": 73000,
            "B_enrollment_range_inclusive": [45000, 50000],
            "C_enrollment_range_inclusive": [20000, 25000],
            "school_years": {
                "A_enrollment": "2024–2025",
                "B_enrollment": "2023–2024",
                "C_enrollment": "2024–2025"
            }
        },
        info_type="thresholds",
        info_name="evaluation_thresholds"
    )

    # Build global source requirement (critical)
    await build_global_source_requirement(evaluator, root, extracted)

    # Build district subtrees
    await verify_district_a(evaluator, root, extracted.district_a)
    await verify_district_b(evaluator, root, extracted.district_b)
    await verify_district_c(evaluator, root, extracted.district_c)

    # Return summary
    return evaluator.get_summary()