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
TASK_ID = "lunar_eclipse_universities_2026"
TASK_DESCRIPTION = """
Identify four universities that meet ALL of the following criteria for participating in lunar eclipse observation research related to the March 3, 2026 total lunar eclipse:

1. Geographic Location: The university must be located in a region where the March 3, 2026 total lunar eclipse will be visible. According to NASA and astronomical sources, visibility includes western North America, the Pacific region, eastern Asia, and Australia.

2. Faculty Expertise: The university must have at least one faculty member in their physics, astronomy, or astrophysics department with documented research expertise relevant to lunar eclipses, observational astronomy, or celestial phenomena observation.

3. Conference Participation History: The university must have demonstrated active participation (through paper submissions, presentations, or author affiliations) in either the IEEE Aerospace Conference OR the ACM CHI Conference within the past three years (2023-2025). This participation should involve faculty or graduate students from departments related to space science, astronomy, computer science, or human-computer interaction.

4. Technology Integration: The university must have documented programs, courses, or research initiatives that integrate virtual reality (VR) technology or advanced visualization systems into their astronomy education, space science programs, OR human-computer interaction research.

For each identified university, provide:
- University name and location (city, state/province, country)
- Evidence of eclipse visibility from that location
- Name and title of at least one relevant faculty member with their research focus
- Evidence of conference participation (specific paper title, year, and conference name)
- Description of VR/visualization program or initiative with supporting documentation

All information must be supported by verifiable URLs from official university websites, conference proceedings, faculty profile pages, or reputable astronomical sources.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacultyInfo(BaseModel):
    name: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    research_focus: Optional[str] = None
    faculty_urls: List[str] = Field(default_factory=list)
    research_urls: List[str] = Field(default_factory=list)


class ConferenceInfo(BaseModel):
    conference_name: Optional[str] = None  # e.g., "IEEE Aerospace Conference", "ACM CHI Conference"
    year: Optional[str] = None             # Keep as string for robustness (e.g., "2024")
    paper_title: Optional[str] = None
    conference_urls: List[str] = Field(default_factory=list)


class TechnologyInfo(BaseModel):
    program_description: Optional[str] = None
    program_urls: List[str] = Field(default_factory=list)
    application_description: Optional[str] = None
    application_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None  # state/province
    country: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    visibility_urls: List[str] = Field(default_factory=list)
    faculty: Optional[FacultyInfo] = None
    conference: Optional[ConferenceInfo] = None
    technology: Optional[TechnologyInfo] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities from the answer that are claimed to meet the specified criteria for the March 3, 2026 total lunar eclipse research participation.

    For each university, extract the following fields exactly using the JSON structure below:

    universities: [
      {
        "name": string or null,
        "city": string or null,
        "region": string or null,       // state or province; can be null
        "country": string or null,

        "location_urls": [urls...],     // URLs that confirm the university's location (prefer official university pages)
        "visibility_urls": [urls...],   // URLs (e.g., NASA, timeanddate, reputable astronomy sources) confirming eclipse visibility from the location

        "faculty": {
          "name": string or null,       // faculty member name
          "title": string or null,      // e.g., Professor, Associate Professor
          "department": string or null, // physics, astronomy, astrophysics or closely related
          "research_focus": string or null, // brief phrase as given in the answer
          "faculty_urls": [urls...],    // faculty profile page(s) or official departmental listings
          "research_urls": [urls...]    // publication or research activity pages supporting relevance
        },

        "conference": {
          "conference_name": string or null, // "IEEE Aerospace Conference" or "ACM CHI Conference"
          "year": string or null,            // 2023, 2024, or 2025 (as string)
          "paper_title": string or null,     // the exact or close title mentioned in the answer
          "conference_urls": [urls...]       // links to official proceedings pages (IEEE Xplore, ACM DL), conference program pages, or university news verifying the participation
        },

        "technology": {
          "program_description": string or null,     // what VR/advanced visualization program/course/lab/initiative exists
          "program_urls": [urls...],                 // URLs documenting the VR/visualization program (official pages)
          "application_description": string or null, // how the VR/visualization is applied to astronomy/space science or HCI research/education
          "application_urls": [urls...]              // URLs documenting the application; if not distinct, can reuse the program URL(s)
        }
      }
    ]

    Rules:
    - Extract only what is explicitly present in the answer.
    - For any missing field, return null or an empty list as appropriate.
    - All URL fields must be full URLs and must come from the answer text (do not invent).
    - Preserve the order the answer used; if more than four are present, include only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"#{n}")


def _format_location(u: UniversityItem) -> str:
    parts = [p for p in [u.city, u.region, u.country] if p and p.strip()]
    return ", ".join(parts) if parts else "the stated location"


def _nonempty_urls(*url_lists: List[str]) -> List[str]:
    for lst in url_lists:
        if lst and len([u for u in lst if isinstance(u, str) and u.strip() != ""]) > 0:
            return [u for u in lst if isinstance(u, str) and u.strip() != ""]
    return []


# --------------------------------------------------------------------------- #
# Verification for one university                                             #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    uni_idx = idx + 1
    uni_title = f"{_ordinal(uni_idx)} identified university meeting all specified criteria"

    # University node (non-critical to allow partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_idx}",
        desc=uni_title,
        parent=parent_node,
        critical=False
    )

    # 1) Geographic_Visibility (critical)
    geo_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Geographic_Visibility",
        desc="University is located in a region where the March 3, 2026 total lunar eclipse will be visible (western North America, Pacific region, eastern Asia, or Australia)",
        parent=uni_node,
        critical=True
    )

    # 1.1) Location_Verification (critical)
    loc_ver_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Location_Verification",
        desc="University's geographic location is verifiable and falls within the eclipse visibility zone",
        parent=geo_node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Location_URL",
        desc="URL evidence confirming university location",
        parent=loc_ver_node,
        critical=True
    )
    uni_name = uni.name or "the university"
    location_str = _format_location(uni)
    loc_claim = f"The official page or authoritative source confirms that {uni_name} is located in {location_str}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=_nonempty_urls(uni.location_urls),
        additional_instruction="Focus on whether the page confirms the university's location (city/state/country). Accept reasonable variations (e.g., postal address, campus address, or standard abbreviations)."
    )

    # 1.2) Visibility_Confirmation (critical)
    vis_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Visibility_Confirmation",
        desc="Eclipse visibility from the university's location is confirmed through astronomical sources",
        parent=geo_node,
        critical=True
    )
    vis_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Visibility_URL",
        desc="URL evidence confirming eclipse visibility from location",
        parent=vis_node,
        critical=True
    )
    vis_claim = f"According to the cited source, the March 3, 2026 total lunar eclipse will be visible from {location_str} (i.e., the location falls within the visibility region)."
    await evaluator.verify(
        claim=vis_claim,
        node=vis_leaf,
        sources=_nonempty_urls(uni.visibility_urls),
        additional_instruction="Check the map/table/text for the March 3, 2026 total lunar eclipse. Accept visibility if the location lies within a region listed as visible (e.g., western North America, the Pacific, eastern Asia, or Australia). Minor city/region name variants are acceptable."
    )

    # 2) Faculty_Expertise (sequential, critical)
    fac_seq = evaluator.add_sequential(
        id=f"U{uni_idx}_Faculty_Expertise",
        desc="University has faculty members with relevant research expertise in astronomy, astrophysics, or related lunar observation fields",
        parent=uni_node,
        critical=True
    )

    # 2.1) Faculty_Identification (critical)
    fac_id_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Faculty_Identification",
        desc="Specific faculty member(s) identified with appropriate research focus",
        parent=fac_seq,
        critical=True
    )
    fac_url_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Faculty_URL",
        desc="URL evidence of faculty member and their research expertise",
        parent=fac_id_node,
        critical=True
    )
    f = uni.faculty or FacultyInfo()
    fac_title = f.title or "faculty member"
    fac_name = f.name or "the faculty member"
    # Claim: affiliation in relevant department (physics/astronomy/astrophysics or closely related)
    fac_affil_claim = (
        f"{fac_name} is a {fac_title} at {uni_name}, affiliated with a department relevant to physics, astronomy, or astrophysics (or a closely related program)."
    )
    await evaluator.verify(
        claim=fac_affil_claim,
        node=fac_url_leaf,
        sources=_nonempty_urls(f.faculty_urls, f.research_urls),
        additional_instruction="Confirm that the person is a current faculty member at the university and that their affiliation (department/unit) is in or closely related to physics, astronomy, or astrophysics. Accept synonymous unit names (e.g., School of Physics & Astronomy, Department of Astrophysical Sciences)."
    )

    # 2.2) Research_Relevance (critical)
    fac_rel_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Research_Relevance",
        desc="Faculty research is relevant to lunar eclipses, observational astronomy, or celestial phenomena",
        parent=fac_seq,
        critical=True
    )
    res_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Research_URL",
        desc="URL evidence of faculty research activities or publications",
        parent=fac_rel_node,
        critical=True
    )
    research_focus = (f.research_focus or "").strip()
    rel_claim = (
        f"The research interests or publications of {fac_name} include topics relevant to lunar eclipses, observational astronomy, or celestial phenomena observation."
    )
    await evaluator.verify(
        claim=rel_claim,
        node=res_leaf,
        sources=_nonempty_urls(f.research_urls, f.faculty_urls),
        additional_instruction="Look for keywords such as 'observational astronomy', 'eclipse', 'lunar eclipse', 'astronomy instrumentation', 'astrophysics', or clear equivalents on the page(s). It is acceptable if the page lists closely related observational/astronomical topics even if 'lunar eclipse' is not explicitly named."
    )

    # 3) Conference_Participation (critical)
    conf_parent = evaluator.add_parallel(
        id=f"U{uni_idx}_Conference_Participation",
        desc="University has demonstrated participation in either IEEE Aerospace Conference or ACM CHI Conference within the past 3 years (2023-2025)",
        parent=uni_node,
        critical=True
    )
    conf_aff_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Conference_Affiliation",
        desc="Evidence of university affiliation with papers or presentations at target conferences",
        parent=conf_parent,
        critical=True
    )
    conf_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Conference_URL",
        desc="URL evidence of conference participation",
        parent=conf_aff_node,
        critical=True
    )
    c = uni.conference or ConferenceInfo()
    conf_name = (c.conference_name or "the target conference").strip()
    conf_year = (c.year or "one of 2023-2025").strip()
    paper_title = (c.paper_title or "the cited paper or presentation").strip()
    conf_claim = (
        f"In {conf_year}, at the {conf_name}, there is a paper/presentation titled '{paper_title}' "
        f"with at least one author affiliated with {uni_name} (or the paper otherwise clearly indicates {uni_name}'s participation)."
    )
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=_nonempty_urls(c.conference_urls),
        additional_instruction="Verify both the conference (IEEE Aerospace Conference or ACM CHI Conference) and the year (2023, 2024, or 2025). Accept reasonable title variations (e.g., punctuation/casing). The page should clearly show the conference venue and the author affiliation or institutional association with the university."
    )

    # 4) Technology_Integration (critical)
    tech_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Technology_Integration",
        desc="University has programs or initiatives integrating VR technology or advanced visualization in astronomy education or HCI research",
        parent=uni_node,
        critical=True
    )

    # 4.1) Program_Existence (critical)
    prog_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Program_Existence",
        desc="Evidence of VR/visualization programs in relevant departments",
        parent=tech_node,
        critical=True
    )
    prog_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Program_URL",
        desc="URL evidence of VR or visualization programs",
        parent=prog_node,
        critical=True
    )
    t = uni.technology or TechnologyInfo()
    prog_claim = (
        f"{uni_name} has a program, course, lab, facility, or research initiative that integrates virtual reality (VR) or advanced visualization technology."
    )
    await evaluator.verify(
        claim=prog_claim,
        node=prog_leaf,
        sources=_nonempty_urls(t.program_urls),
        additional_instruction="Accept terms like VR, AR, XR, mixed reality, immersive visualization, fulldome/planetarium visualization, CAVE, or similar. The page should belong to the university or an officially affiliated unit."
    )

    # 4.2) Technology_Application (critical)
    app_node = evaluator.add_parallel(
        id=f"U{uni_idx}_Technology_Application",
        desc="VR/visualization technology is applied to astronomy, space science, or HCI education/research",
        parent=tech_node,
        critical=True
    )
    app_leaf = evaluator.add_leaf(
        id=f"U{uni_idx}_Application_URL",
        desc="URL evidence of technology application in relevant fields",
        parent=app_node,
        critical=True
    )
    app_claim = (
        f"The cited program or initiative at {uni_name} applies VR/advanced visualization to astronomy or space-science education/research, "
        f"OR is used in human-computer interaction (HCI) research/education."
    )
    await evaluator.verify(
        claim=app_claim,
        node=app_leaf,
        sources=_nonempty_urls(t.application_urls, t.program_urls),
        additional_instruction="Look for explicit mentions of astronomy/astrophysics/space-science teaching or research using VR/visualization, or HCI research/education that uses VR/XR/visualization. Accept closely related terms and synonyms."
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
    Evaluate an answer for the 2026 lunar eclipse university identification task.
    """
    evaluator = Evaluator()
    # IMPORTANT: Set root as non-critical to allow non-critical children (framework constraint).
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

    # Create an explicit top-level node mirroring the rubric root (non-critical to avoid child constraint)
    task_root = evaluator.add_parallel(
        id="Research_Task_Completion",
        desc="Complete identification and verification of universities meeting all specified research and conference participation criteria related to the March 3, 2026 lunar eclipse observation opportunities",
        parent=root,
        critical=False
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Keep only the first 4; pad if fewer
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build per-university verification trees
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, task_root, uni, idx)

    # Return structured evaluation summary
    return evaluator.get_summary()