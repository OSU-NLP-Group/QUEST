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
TASK_ID = "adult_swim_s3_director_university"
TASK_DESCRIPTION = """
An Adult Swim animated series premiered on October 5, 2025. Season 3 of this series was unique because it was the only season where all episodes were directed by a single director, unlike previous seasons which had multiple directors. This director had previously worked on the series as a storyboard artist before becoming the sole episode director for Season 3. What university did this director attend?
"""

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SeriesInfo(BaseModel):
    series_name: Optional[str] = None
    sources_premiere: List[str] = Field(default_factory=list)
    sources_creators: List[str] = Field(default_factory=list)
    sources_final_season: List[str] = Field(default_factory=list)
    sources_annecy_2025: List[str] = Field(default_factory=list)


class DirectorInfo(BaseModel):
    director_name: Optional[str] = None
    sources_s3_sole_director: List[str] = Field(default_factory=list)
    sources_storyboarder: List[str] = Field(default_factory=list)
    # Optional textual start date as claimed in the answer (not directly verified numerically)
    storyboard_start_text: Optional[str] = None


class EducationInfo(BaseModel):
    university_name: Optional[str] = None
    sources_education_profiles: List[str] = Field(default_factory=list)
    sources_university_info: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    series: Optional[SeriesInfo] = None
    director: Optional[DirectorInfo] = None
    education: Optional[EducationInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_task_info() -> str:
    return """
    Extract structured information needed to verify the task. Only extract details explicitly present in the answer text and the URLs explicitly cited in the answer.

    Return a JSON object with the following nested structure:

    {
      "series": {
        "series_name": string or null,
        "sources_premiere": [list of URLs supporting the series premiere date/time on Adult Swim],
        "sources_creators": [list of URLs supporting that the series was created by Zach Hadel and Michael Cusack],
        "sources_final_season": [list of URLs supporting that Season 3 is designated as the final season],
        "sources_annecy_2025": [list of URLs supporting that a work-in-progress version of Season 3 was screened at the 2025 Annecy International Animation Film Festival with the presence of the creators]
      },
      "director": {
        "director_name": string or null,
        "sources_s3_sole_director": [list of URLs supporting that Season 3 had a single director who directed ALL episodes, and that prior seasons had multiple directors],
        "sources_storyboarder": [list of URLs supporting that the director previously worked on the series as a storyboard artist, ideally including timing information],
        "storyboard_start_text": string or null  // Any claimed start date/phrase for storyboard work (e.g., "May 2021")
      },
      "education": {
        "university_name": string or null,
        "sources_education_profiles": [list of URLs to public professional networking profiles (e.g., LinkedIn) or similar where the director's education is listed],
        "sources_university_info": [list of URLs supporting the university being a real, accredited US institution (e.g., official university site, Wikipedia, accreditation listings)]
      }
    }

    Rules:
    - Extract only URLs explicitly present in the answer (plain URLs or those embedded in markdown).
    - Do not fabricate any data. If something is not provided, return null or an empty list accordingly.
    - Prefer LinkedIn URLs for education profile sources when available.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_series_name(series: Optional[SeriesInfo]) -> str:
    return (series.series_name if series and series.series_name else "the series")


def safe_director_name(director: Optional[DirectorInfo]) -> str:
    return (director.director_name if director and director.director_name else "the director")


def combine_sources(*args: List[str]) -> List[str]:
    combined = []
    for lst in args:
        if lst:
            combined.extend([u for u in lst if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in combined:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_series_constraints(
    evaluator: Evaluator,
    parent_node,
    series: Optional[SeriesInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Verify_Series_And_Season_Constraints",
        desc="Verify the series/season constraints for the referenced Adult Swim animated series.",
        parent=parent_node,
        critical=True
    )

    s_name = safe_series_name(series)

    # Series_Premiere_Matches
    leaf_premiere = evaluator.add_leaf(
        id="Series_Premiere_Matches",
        desc="Series premiered on Adult Swim on October 5, 2025 at 11:30 PM ET/PT.",
        parent=node,
        critical=True
    )
    claim_premiere = f"The series {s_name} premiered on Adult Swim on October 5, 2025 at 11:30 PM ET/PT."
    await evaluator.verify(
        claim=claim_premiere,
        node=leaf_premiere,
        sources=(series.sources_premiere if series else None),
        additional_instruction="Only mark as supported if at least one provided URL explicitly confirms the Adult Swim premiere date of October 5, 2025 at 11:30 PM ET/PT. If no valid source URLs are provided, judge as not supported."
    )

    # Series_Creators_Match
    leaf_creators = evaluator.add_leaf(
        id="Series_Creators_Match",
        desc="Series is created by Zach Hadel and Michael Cusack.",
        parent=node,
        critical=True
    )
    claim_creators = f"The series {s_name} was created by Zach Hadel and Michael Cusack."
    await evaluator.verify(
        claim=claim_creators,
        node=leaf_creators,
        sources=(series.sources_creators if series else None),
        additional_instruction="Only pass if the provided URL(s) explicitly list both Zach Hadel and Michael Cusack as creators. If no valid sources are given, fail."
    )

    # Season3_Is_Final_Season
    leaf_final = evaluator.add_leaf(
        id="Season3_Is_Final_Season",
        desc="Series has a third season that is designated as the final season of the show.",
        parent=node,
        critical=True
    )
    claim_final = f"The series {s_name} has a Season 3 that is designated as the final season."
    await evaluator.verify(
        claim=claim_final,
        node=leaf_final,
        sources=(series.sources_final_season if series else None),
        additional_instruction="Only pass if the provided URL(s) explicitly indicate that Season 3 is the final season. If sources are missing or do not clearly say 'final season', fail."
    )

    # Annecy_2025_WIP_Screening
    leaf_annecy = evaluator.add_leaf(
        id="Annecy_2025_WIP_Screening",
        desc="A work-in-progress version of Season 3 was screened at the 2025 Annecy International Animation Film Festival with the presence of the series creators.",
        parent=node,
        critical=True
    )
    claim_annecy = f"A work-in-progress version of Season 3 of {s_name} was screened at the 2025 Annecy International Animation Film Festival with the presence of the series creators."
    await evaluator.verify(
        claim=claim_annecy,
        node=leaf_annecy,
        sources=(series.sources_annecy_2025 if series else None),
        additional_instruction="Only pass if a provided URL explicitly mentions a Season 3 WIP screening at Annecy 2025 and that the series creators were present. If sources are missing or don't confirm both aspects, fail."
    )


async def build_director_constraints(
    evaluator: Evaluator,
    parent_node,
    series: Optional[SeriesInfo],
    director: Optional[DirectorInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Verify_Director_Constraints",
        desc="Verify all director-related constraints for the Season 3 sole-episode director.",
        parent=parent_node,
        critical=True
    )

    s_name = safe_series_name(series)
    d_name = safe_director_name(director)

    # Season3_Sole_Director_All_Episodes
    leaf_s3_sole = evaluator.add_leaf(
        id="Season3_Sole_Director_All_Episodes",
        desc="Season 3 has a single director who directed ALL episodes in that season, unlike previous seasons which had multiple directors.",
        parent=node,
        critical=True
    )
    claim_s3_sole = f"In Season 3 of {s_name}, {d_name} is the sole director for all episodes; prior seasons had multiple directors."
    await evaluator.verify(
        claim=claim_s3_sole,
        node=leaf_s3_sole,
        sources=(director.sources_s3_sole_director if director else None),
        additional_instruction="Only pass if URLs explicitly confirm both parts: (1) this person directed every S3 episode, and (2) earlier seasons had multiple directors. If no valid sources, fail."
    )

    # Previously_Storyboard_Artist_On_Series
    leaf_storyboard_prev = evaluator.add_leaf(
        id="Previously_Storyboard_Artist_On_Series",
        desc="Director previously worked on the same series as a storyboard artist before becoming the sole director for Season 3.",
        parent=node,
        critical=True
    )
    claim_prev_sb = f"Before directing Season 3 of {s_name}, {d_name} worked on the series as a storyboard artist."
    await evaluator.verify(
        claim=claim_prev_sb,
        node=leaf_storyboard_prev,
        sources=(director.sources_storyboarder if director else None),
        additional_instruction="Confirm that this person served as a storyboard artist on the same series prior to S3 directing. If sources don't clearly state storyboarder role, fail."
    )

    # Storyboard_Start_Date_May2021_Or_Earlier
    leaf_storyboard_date = evaluator.add_leaf(
        id="Storyboard_Start_Date_May2021_Or_Earlier",
        desc="Director started working as a storyboarder on the series in May 2021 or earlier.",
        parent=node,
        critical=True
    )
    claim_sb_date = f"{d_name} started working as a storyboard artist on {s_name} in May 2021 or earlier."
    await evaluator.verify(
        claim=claim_sb_date,
        node=leaf_storyboard_date,
        sources=(director.sources_storyboarder if director else None),
        additional_instruction="Pass only if a provided source explicitly supports that the storyboard work began in May 2021 or earlier (e.g., credits, dated posts, portfolio entries). If ambiguous or later than May 2021, fail."
    )


async def build_university_verification(
    evaluator: Evaluator,
    parent_node,
    director: Optional[DirectorInfo],
    education: Optional[EducationInfo]
) -> None:
    node = evaluator.add_parallel(
        id="Identify_And_Verify_University",
        desc="Provide and verify the university attended by the director per the education constraints.",
        parent=parent_node,
        critical=True
    )

    d_name = safe_director_name(director)
    university = education.university_name if education and education.university_name else None

    # University_Name_Provided (existence check)
    evaluator.add_custom_node(
        result=bool(university and university.strip()),
        id="University_Name_Provided",
        desc="Provide the name of the university attended by the director.",
        parent=node,
        critical=True
    )

    # Education_Verified_From_Public_Professional_Profile
    leaf_profile_verify = evaluator.add_leaf(
        id="Education_Verified_From_Public_Professional_Profile",
        desc="Director's university attendance is identifiable and verifiable from a public professional networking profile (e.g., LinkedIn).",
        parent=node,
        critical=True
    )
    uni_text = university if university else "the university"
    claim_profile = f"According to a public professional networking profile, {d_name} attended {uni_text}."
    await evaluator.verify(
        claim=claim_profile,
        node=leaf_profile_verify,
        sources=(education.sources_education_profiles if education else None),
        additional_instruction=(
            "Only pass if at least one provided URL is a public professional networking profile (ideally LinkedIn) "
            "that explicitly lists the director's education at the named university. If no such profile is provided "
            "or the page does not list the university, fail."
        )
    )

    # University_Is_Accredited_US_Institution
    leaf_accredited = evaluator.add_leaf(
        id="University_Is_Accredited_US_Institution",
        desc="The university is a real, accredited educational institution in the United States.",
        parent=node,
        critical=True
    )
    all_uni_sources = combine_sources(
        education.sources_education_profiles if education else [],
        education.sources_university_info if education else []
    )
    claim_accredited = f"{uni_text} is a real, accredited higher education institution in the United States."
    await evaluator.verify(
        claim=claim_accredited,
        node=leaf_accredited,
        sources=all_uni_sources if all_uni_sources else None,
        additional_instruction=(
            "Verify the institution is real and US-based, with recognized accreditation. Accept clear evidence from "
            "the official university website, Wikipedia (commonly acceptable for basic facts), or recognized "
            "accreditation references. If no sources provided or accreditation/US status is unclear, fail."
        )
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; actual task node below is sequential and critical
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

    # Extract structured information from the answer
    extracted: TaskExtraction = await evaluator.extract(
        prompt=prompt_extract_task_info(),
        template_class=TaskExtraction,
        extraction_name="task_extraction"
    )

    # Build the main "Complete_Task" node (sequential, critical)
    complete_task_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify the university attended by the Season 3 sole-episode director of the specified Adult Swim animated series, verifying all provided constraints.",
        parent=root,
        critical=True
    )

    # Subtree 1: Series and season constraints
    await build_series_constraints(evaluator, complete_task_node, extracted.series)

    # Subtree 2: Director constraints
    await build_director_constraints(evaluator, complete_task_node, extracted.series, extracted.director)

    # Subtree 3: Identify and verify university
    await build_university_verification(evaluator, complete_task_node, extracted.director, extracted.education)

    # Return evaluation summary
    return evaluator.get_summary()