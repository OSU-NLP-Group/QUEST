import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_isd_multi_constraint"
TASK_DESCRIPTION = """
Identify a Texas Independent School District (ISD) that meets all of the following criteria:

1. Has student enrollment between 40,000 and 80,000 students
2. Serves 10 or more separate municipalities or cities
3. Operates 60 or more school campuses total
4. Covers a geographic area of at least 100 square miles
5. Has a total property tax rate for the 2024-2025 school year below $1.00 per $100 valuation
6. Serves students from Pre-Kindergarten through Grade 12
7. Ranks among the top 25 largest school districts in Texas by enrollment
8. Serves communities within or adjacent to one of Texas's major metropolitan areas (Dallas-Fort Worth, Houston, Austin, or San Antonio)
9. Operates at least 10 high school campuses
10. Maintains an active official website with publicly accessible enrollment and demographic information
11. Is governed by an elected Board of Trustees
12. Has school year calendars for 2024-2026 that have been officially approved

Provide the full name of the district and reference URLs supporting each criterion.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DistrictExtraction(BaseModel):
    # Basic identification
    name: Optional[str] = None
    state: Optional[str] = None
    classification: Optional[str] = None

    # Free-text numeric or descriptive fields as captured from the answer (strings preferred for robustness)
    enrollment_text: Optional[str] = None
    municipalities_count_text: Optional[str] = None
    municipalities_list: List[str] = Field(default_factory=list)
    campus_count_text: Optional[str] = None
    area_sq_miles_text: Optional[str] = None
    tax_rate_2024_2025_text: Optional[str] = None
    grade_span_text: Optional[str] = None
    top25_text: Optional[str] = None
    metro_area_text: Optional[str] = None
    high_school_count_text: Optional[str] = None
    official_site_url: Optional[str] = None
    elected_board_text: Optional[str] = None
    calendars_text: Optional[str] = None

    # Per-criterion URLs as explicitly provided in the answer
    urls_texas_location: List[str] = Field(default_factory=list)
    urls_isd_classification: List[str] = Field(default_factory=list)
    urls_enrollment: List[str] = Field(default_factory=list)
    urls_municipalities: List[str] = Field(default_factory=list)
    urls_campuses: List[str] = Field(default_factory=list)
    urls_area: List[str] = Field(default_factory=list)
    urls_tax_rate: List[str] = Field(default_factory=list)
    urls_grade_span: List[str] = Field(default_factory=list)
    urls_top25: List[str] = Field(default_factory=list)
    urls_metro: List[str] = Field(default_factory=list)
    urls_highschools: List[str] = Field(default_factory=list)
    urls_enrollment_demographics: List[str] = Field(default_factory=list)
    urls_elected_board: List[str] = Field(default_factory=list)
    urls_calendars: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_district() -> str:
    return """
    You must extract structured information about exactly one Texas Independent School District (ISD) mentioned in the answer. 
    If multiple districts are mentioned, extract ONLY the first one presented as the identified solution.

    Extract the following fields (strings preferred; keep numbers as they appear, e.g., "about 75,000", "61+", etc.):
    - name: Full official name of the district (e.g., "Katy Independent School District").
    - state: State explicitly stated or implied (e.g., "Texas" or "TX"); return null if not present.
    - classification: District type as stated (e.g., "Independent School District" or "ISD"); return null if not present.
    - enrollment_text: The enrollment figure/range as written in the answer text.
    - municipalities_count_text: The number of municipalities/cities served as text.
    - municipalities_list: A list of municipality/city names if listed; otherwise return [].
    - campus_count_text: Total number of campuses as text.
    - area_sq_miles_text: Geographic area in square miles as text.
    - tax_rate_2024_2025_text: Property tax rate for the 2024–2025 school year as text, ideally including total or M&O + I&S sum.
    - grade_span_text: Grade span as written (e.g., "PK–12", "Pre-K through 12").
    - top25_text: Any statement about top 25 enrollment ranking in Texas.
    - metro_area_text: Any reference to Dallas–Fort Worth, Houston, Austin, or San Antonio metro adjacency/coverage.
    - high_school_count_text: Number of high school campuses as text.
    - official_site_url: The official district website URL if present (e.g., https://www.exampleisd.org/).
    - elected_board_text: Any mention that the Board of Trustees is elected.
    - calendars_text: Any mention that 2024–2026 school year calendars are approved/published.

    Also extract explicit supporting URLs the answer provided for each criterion. 
    IMPORTANT: Only include URLs explicitly present in the answer text (plain link or markdown link). 
    If a field is missing, return an empty list for that URL array.

    - urls_texas_location: URLs supporting that the district is in Texas.
    - urls_isd_classification: URLs supporting that it is an ISD.
    - urls_enrollment: URLs supporting the enrollment figure/range.
    - urls_municipalities: URLs supporting number/list of municipalities served.
    - urls_campuses: URLs supporting total campus count.
    - urls_area: URLs supporting geographic area (sq miles).
    - urls_tax_rate: URLs supporting the 2024–2025 total property tax rate (< $1.00 per $100 valuation).
    - urls_grade_span: URLs supporting the PK–12 grade span.
    - urls_top25: URLs supporting top-25-by-enrollment status in Texas.
    - urls_metro: URLs supporting the district’s being within/adjacent to DFW, Houston, Austin, or San Antonio metro.
    - urls_highschools: URLs supporting ≥10 high school campuses.
    - urls_enrollment_demographics: URLs on the district's official website showing enrollment/demographics are publicly accessible.
    - urls_elected_board: URLs supporting governance by an elected Board of Trustees.
    - urls_calendars: URLs supporting that 2024–2026 school-year calendars are officially approved/published.

    Do not fabricate or infer any URL or value. If something is not explicitly presented in the answer, return null (for a single string) or an empty list (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*args: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in args:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str):
                norm = u.strip()
                if norm and norm not in merged:
                    merged.append(norm)
    return merged


def _name_or_placeholder(name: Optional[str]) -> str:
    return name.strip() if name else "the identified district"


# --------------------------------------------------------------------------- #
# Build tree segments                                                         #
# --------------------------------------------------------------------------- #
async def build_meets_all_criteria(
    evaluator: Evaluator,
    parent_node,
    ext: DistrictExtraction,
) -> None:
    district_name = _name_or_placeholder(ext.name)

    meets_node = evaluator.add_parallel(
        id="Meets_All_Criteria",
        desc="The identified district satisfies every required constraint.",
        parent=parent_node,
        critical=True
    )

    # Define leaves
    leaf_specs = [
        (
            "Located_In_Texas",
            "District is located in Texas.",
            f"The district {district_name} is located in the U.S. state of Texas.",
            ext.urls_texas_location,
            "Confirm the district is a Texas public school district (e.g., TEA listing, official district pages, or credible sources)."
        ),
        (
            "Classified_As_ISD",
            "District is classified as an Independent School District (ISD).",
            f"The district {district_name} is an Independent School District (ISD).",
            ext.urls_isd_classification,
            "The source should explicitly indicate 'Independent School District' or 'ISD' as the district type; do not rely solely on the name."
        ),
        (
            "Enrollment_Between_40000_80000",
            "Student enrollment is between 40,000 and 80,000 students.",
            f"The district {district_name} has student enrollment between 40,000 and 80,000 (allow approximate statements).",
            ext.urls_enrollment,
            "Use the most recent figure presented; accept approximate language like 'about' or ranges as long as they fall within [40,000, 80,000]."
        ),
        (
            "Serves_At_Least_10_Municipalities",
            "District serves 10 or more separate municipalities/cities.",
            f"The district {district_name} serves at least 10 separate municipalities or cities.",
            ext.urls_municipalities,
            "Statements like 'serves X communities/cities' or a listed set of cities count; ensure the count is >= 10."
        ),
        (
            "Operates_At_Least_60_Campuses",
            "District operates 60 or more total school campuses.",
            f"The district {district_name} operates at least 60 total campuses.",
            ext.urls_campuses,
            "Campuses can include elementary, intermediate, junior high/middle, high schools, and specialty campuses; count total >= 60."
        ),
        (
            "Covers_At_Least_100_Sq_Miles",
            "District covers a geographic area of at least 100 square miles.",
            f"The district {district_name} covers at least 100 square miles.",
            ext.urls_area,
            "Accept 'covers over/approximately X square miles' when X >= 100."
        ),
        (
            "Tax_Rate_Below_1_Dollar_2024_2025",
            "Total property tax rate for the 2024–2025 school year is below $1.00 per $100 valuation.",
            f"For the 2024–2025 school year, the total property tax rate of {district_name} is below $1.00 per $100 valuation.",
            ext.urls_tax_rate,
            "Confirm the school year is 2024–2025 and that the total rate (M&O + I&S, or explicitly 'Total') is < 1.00. Accept official notices, financial pages, or board documents."
        ),
        (
            "Serves_PK_Through_12",
            "District serves students from Pre-Kindergarten through Grade 12.",
            f"The district {district_name} serves students from pre-kindergarten (PK) through grade 12.",
            ext.urls_grade_span,
            "Accept equivalent expressions like 'PK–12', 'Pre-K through 12th grade', or 'early childhood through high school'."
        ),
        (
            "Top_25_By_Enrollment_In_Texas",
            "District ranks among the top 25 largest school districts in Texas by enrollment.",
            f"The district {district_name} is among the 25 largest school districts in Texas by enrollment.",
            ext.urls_top25,
            "Accept credible statewide rankings or TEA datasets; ranking may be for a recent year (e.g., 2023–2025)."
        ),
        (
            "Within_Or_Adjacent_To_Major_Metro",
            "District serves communities within or adjacent to one of: Dallas–Fort Worth, Houston, Austin, or San Antonio metro area.",
            f"The district {district_name} serves communities within or adjacent to one of Texas's major metros: Dallas–Fort Worth, Houston, Austin, or San Antonio.",
            ext.urls_metro,
            "Evidence can include district boundary/city locations within the relevant MSA counties or explicit statements on official/credible pages."
        ),
        (
            "At_Least_10_High_Schools",
            "District operates at least 10 high school campuses.",
            f"The district {district_name} operates at least 10 high school campuses.",
            ext.urls_highschools,
            "Count traditional high schools plus early college, magnet, or specialized high schools if the district classifies them as high schools."
        ),
        (
            "Has_Active_Official_Website_With_Enrollment_And_Demographics",
            "District maintains an active official website with publicly accessible enrollment and demographic information.",
            f"The district {district_name} maintains an active official website that publishes enrollment and demographic information publicly.",
            _merge_sources(ext.urls_enrollment_demographics, [ext.official_site_url] if ext.official_site_url else []),
            "Accept district 'About/Statistics/Enrollment/Demographics' pages or PDF profiles hosted on the official domain; the website must be active and accessible."
        ),
        (
            "Governed_By_Elected_Board_Of_Trustees",
            "District is governed by an elected Board of Trustees.",
            f"The district {district_name} is governed by an elected Board of Trustees.",
            ext.urls_elected_board,
            "Look for 'elected', 'trustees', 'board elections' or similar on official or authoritative pages."
        ),
        (
            "Approved_Calendars_2024_2026",
            "District has school year calendars for 2024–2026 that have been officially approved.",
            f"The district {district_name} has officially approved school year calendars covering years 2024 through 2026 (e.g., 2024–2025 and 2025–2026).",
            ext.urls_calendars,
            "Approval can be shown via board-approved calendar documents/pages. Accept separate pages for each year if they explicitly indicate approval."
        ),
    ]

    # Create leaf nodes and run verifications (batch for efficiency)
    claims_and_sources = []
    for node_id, node_desc, claim, sources, add_ins in leaf_specs:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=node_desc,
            parent=meets_node,
            critical=True
        )
        claims_and_sources.append((claim, sources, leaf, add_ins))

    # Batch verify all constraints under this parallel node
    await evaluator.batch_verify(claims_and_sources)


def build_supporting_urls_section(
    evaluator: Evaluator,
    parent_node,
    ext: DistrictExtraction,
) -> None:
    urls_node = evaluator.add_parallel(
        id="Provides_Supporting_URLs_For_Each_Criterion",
        desc="Response provides reference URLs that support each required criterion.",
        parent=parent_node,
        critical=True
    )

    # Helper to add a URL-existence check node
    def add_url_presence_node(node_id: str, desc: str, url_lists: List[List[str]]) -> None:
        merged = _merge_sources(*url_lists)
        evaluator.add_custom_node(
            result=(len(merged) > 0),
            id=node_id,
            desc=desc,
            parent=urls_node,
            critical=True
        )

    # Add per-criterion URL presence checks
    add_url_presence_node(
        "URL_For_Texas_Location",
        "Provides at least one reference URL supporting that the district is located in Texas.",
        [ext.urls_texas_location]
    )
    add_url_presence_node(
        "URL_For_ISD_Classification",
        "Provides at least one reference URL supporting that the district is an ISD.",
        [ext.urls_isd_classification]
    )
    add_url_presence_node(
        "URL_For_Enrollment",
        "Provides at least one reference URL supporting the enrollment figure/range.",
        [ext.urls_enrollment]
    )
    add_url_presence_node(
        "URL_For_Municipalities_Served",
        "Provides at least one reference URL supporting the number/list of municipalities/cities served.",
        [ext.urls_municipalities]
    )
    add_url_presence_node(
        "URL_For_Campus_Count",
        "Provides at least one reference URL supporting total campus count.",
        [ext.urls_campuses]
    )
    add_url_presence_node(
        "URL_For_Geographic_Area",
        "Provides at least one reference URL supporting geographic area (square miles).",
        [ext.urls_area]
    )
    add_url_presence_node(
        "URL_For_Tax_Rate_2024_2025",
        "Provides at least one reference URL supporting the 2024–2025 total property tax rate and that it is below $1.00 per $100 valuation.",
        [ext.urls_tax_rate]
    )
    add_url_presence_node(
        "URL_For_Grade_Span",
        "Provides at least one reference URL supporting PK–12 grade span.",
        [ext.urls_grade_span]
    )
    add_url_presence_node(
        "URL_For_Top_25_Ranking",
        "Provides at least one reference URL supporting top-25-by-enrollment status in Texas.",
        [ext.urls_top25]
    )
    add_url_presence_node(
        "URL_For_Metro_Area_Adjacency",
        "Provides at least one reference URL supporting the district’s relationship to one of the specified major metro areas.",
        [ext.urls_metro]
    )
    add_url_presence_node(
        "URL_For_High_School_Count",
        "Provides at least one reference URL supporting that the district operates at least 10 high school campuses.",
        [ext.urls_highschools]
    )
    add_url_presence_node(
        "URL_For_Official_Website_Enrollment_Demographics",
        "Provides at least one reference URL (may be the official site) supporting that enrollment and demographic information are publicly accessible.",
        [ext.urls_enrollment_demographics, [ext.official_site_url] if ext.official_site_url else []]
    )
    add_url_presence_node(
        "URL_For_Elected_Board",
        "Provides at least one reference URL supporting governance by an elected Board of Trustees.",
        [ext.urls_elected_board]
    )
    add_url_presence_node(
        "URL_For_Approved_Calendars_2024_2026",
        "Provides at least one reference URL supporting that the 2024–2026 calendars are officially approved.",
        [ext.urls_calendars]
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
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    ext: DistrictExtraction = await evaluator.extract(
        prompt=prompt_extract_district(),
        template_class=DistrictExtraction,
        extraction_name="district_info"
    )

    # Top-level sequential critical node matching rubric root
    complete_node = evaluator.add_sequential(
        id="Complete_Identification",
        desc="Response identifies one Texas ISD that satisfies all stated criteria and provides supporting reference URLs for each criterion.",
        parent=root,
        critical=True
    )

    # 1) Full district name provided (existence check)
    evaluator.add_custom_node(
        result=bool(ext.name and ext.name.strip()),
        id="Full_District_Name_Provided",
        desc="Response provides the full official name of the identified school district.",
        parent=complete_node,
        critical=True
    )

    # 2) Meets all criteria (parallel critical): evidence-grounded verifications
    await build_meets_all_criteria(evaluator, complete_node, ext)

    # 3) Provides supporting URLs for each criterion (parallel critical): URL existence checks
    build_supporting_urls_section(evaluator, complete_node, ext)

    # Optional: record constraints as "ground truth-style" context for the run
    evaluator.add_ground_truth({
        "constraints": [
            "Enrollment between 40,000 and 80,000",
            "Serves >= 10 municipalities",
            "Operates >= 60 campuses",
            "Area >= 100 square miles",
            "Total property tax rate 2024–2025 < $1.00 per $100 valuation",
            "PK–12 grade span",
            "Top 25 by enrollment in Texas",
            "Within or adjacent to DFW/Houston/Austin/San Antonio metro",
            "Operates >= 10 high schools",
            "Active official website with enrollment & demographics",
            "Governed by an elected Board of Trustees",
            "Approved calendars for 2024–2026"
        ]
    }, gt_type="rubric_constraints")

    return evaluator.get_summary()