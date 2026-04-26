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
TASK_ID = "chi2022_record_athlete_journey"
TASK_DESCRIPTION = """
Identify the female American marathon runner who set the American women's marathon record at the 2022 Bank of America Chicago Marathon. For this athlete, provide the following information in sequential order:

1. The athlete's full name
2. Her exact finishing time that set the American record at the 2022 Chicago Marathon, and confirmation that this race was certified by USATF, World Athletics, or AIMS
3. Her finishing position and exact time at the 2024 U.S. Olympic Team Trials Marathon held in Orlando, Florida on February 3, 2024
4. Confirmation that she qualified for and was selected to the U.S. Olympic marathon team for the Paris 2024 Olympics
5. Her finishing position and exact time at the women's marathon race during the Paris 2024 Olympic Games
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AthleteExtraction(BaseModel):
    # Athlete identification
    name: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)

    # American record details
    record_time: Optional[str] = None  # Exact finishing time that set the American record
    record_race_name: Optional[str] = None  # Expected: "2022 Bank of America Chicago Marathon" or similar
    record_urls: List[str] = Field(default_factory=list)  # URLs supporting record details and certification

    # Olympic Trials (2024) performance
    trials_position: Optional[str] = None  # e.g., "1st", "second", "2nd place"
    trials_time: Optional[str] = None  # exact time string, e.g., "2:22:10"
    trials_urls: List[str] = Field(default_factory=list)  # URLs supporting Trials results

    # Olympic team selection (Paris 2024)
    selection_urls: List[str] = Field(default_factory=list)  # URLs confirming official team selection

    # Olympic Games (Paris 2024) performance
    olympics_position: Optional[str] = None  # finishing position string
    olympics_time: Optional[str] = None  # exact time string
    olympics_urls: List[str] = Field(default_factory=list)  # URLs supporting Olympic results


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_athlete_info() -> str:
    return """
Extract the requested information exactly as stated in the answer. Return JSON with the following fields:

1) name: The athlete's full name (string).
2) identity_urls: Array of URL(s) that confirm the athlete's identity and that she set the American women's marathon record at Chicago 2022. If none provided, return [].

3) record_time: The exact finishing time that set the American record at the 2022 Bank of America Chicago Marathon (string, e.g., "2:18:29"). If not present, return null.
4) record_race_name: The race name as the answer states it (string; e.g., "2022 Bank of America Chicago Marathon"). If not present, return null.
5) record_urls: Array of URL(s) the answer cites for the record details and/or course certification (USATF, World Athletics, or AIMS). If none provided, return [].

6) trials_position: The finishing position at the 2024 U.S. Olympic Team Trials Marathon (Orlando, Florida; February 3, 2024) as a short string (e.g., "1st", "2nd", "third"). If not present, return null.
7) trials_time: The exact finishing time at that Trials (string, e.g., "2:22:10"). If not present, return null.
8) trials_urls: Array of URL(s) the answer cites for the Trials results. If none provided, return [].

9) selection_urls: Array of URL(s) confirming official selection to represent Team USA in the women's marathon at Paris 2024 (e.g., USATF, Team USA/USOPC, World Athletics or official announcements). If none provided, return [].

10) olympics_position: The athlete's finishing position at the women's marathon at the Paris 2024 Olympics (string). If not present, return null.
11) olympics_time: The athlete's exact finishing time at that Olympic marathon (string). If not present, return null.
12) olympics_urls: Array of URL(s) the answer cites for Olympic marathon results. If none provided, return [].

Rules:
- Extract only what appears in the answer. Do not invent.
- Keep times as plain strings exactly as written (e.g., "2:18:29", not numbers).
- Always return arrays for URL fields. If the answer lists more than one URL for a given section, include them all.
- Accept URLs in any reasonable format (plain links or markdown). Return the actual URL strings.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_sources(*lists: List[str]) -> List[str]:
    """Merge multiple URL lists, remove duplicates, preserve order, strip whitespace."""
    seen = set()
    result: List[str] = []
    for urls in lists:
        if not urls:
            continue
        for u in urls:
            if not isinstance(u, str):
                continue
            uu = u.strip()
            if uu and uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_athlete_identification(evaluator: Evaluator, parent, info: AthleteExtraction):
    """
    Athlete_Identification (parallel)
      - Athlete_Name (critical): existence
      - Identity_Reference (critical): existence of at least one URL
    """
    node = evaluator.add_parallel(
        id="Athlete_Identification",
        desc="Correct identification of the female American marathon runner who set the American record at 2022 Chicago Marathon",
        parent=parent,
        critical=False
    )

    # Athlete_Name: existence check
    evaluator.add_custom_node(
        result=bool(info.name and info.name.strip()),
        id="Athlete_Name",
        desc="The athlete's full name is provided",
        parent=node,
        critical=True
    )

    # Identity_Reference: existence of at least one identity URL
    identity_ref_node = evaluator.add_custom_node(
        result=len(info.identity_urls) > 0,
        id="Identity_Reference",
        desc="Reference URL confirming the athlete's identity and record achievement is provided",
        parent=node,
        critical=True
    )

    return {
        "node": node,
        "identity_ref_node": identity_ref_node,
    }


async def build_american_record_details(evaluator: Evaluator, parent, info: AthleteExtraction):
    """
    American_Record_Details (parallel)
      - Record_Time (critical): verify exact time with sources
      - Race_Information (parallel, critical)
          - Race_Identification (critical): verify race identified as given in the answer
          - Course_Certification (critical): verify certification by USATF/World Athletics/AIMS
          - Race_Reference (critical): existence of at least one URL
    """
    details_node = evaluator.add_parallel(
        id="American_Record_Details",
        desc="Verification of the American record achievement and race certification",
        parent=parent,
        critical=False
    )

    # Race_Information subtree (critical)
    race_info_node = evaluator.add_parallel(
        id="Race_Information",
        desc="Information about the 2022 Chicago Marathon race and certification",
        parent=details_node,
        critical=True
    )

    # Race_Reference: existence
    race_ref_node = evaluator.add_custom_node(
        result=len(info.record_urls) > 0,
        id="Race_Reference",
        desc="Reference URL for the race details and certification is provided",
        parent=race_info_node,
        critical=True
    )

    # Race_Identification: verify that the race (as stated in the answer) is supported by sources
    race_ident_leaf = evaluator.add_leaf(
        id="Race_Identification",
        desc="The race is correctly identified as the 2022 Bank of America Chicago Marathon",
        parent=race_info_node,
        critical=True
    )
    race_ident_claim = (
        f"The American women's marathon record by {info.name or 'the athlete'} was set at the race named "
        f"'{info.record_race_name}'."
    )
    await evaluator.verify(
        claim=race_ident_claim,
        node=race_ident_leaf,
        sources=info.record_urls,
        additional_instruction=(
            "Confirm that the cited pages explicitly identify the record-setting race with the same name (allow minor "
            "formatting variations) as provided in the claim. Focus on whether the record was set at the 2022 Bank of "
            "America Chicago Marathon. Treat variations like 'Chicago Marathon 2022' vs '2022 Bank of America Chicago Marathon' "
            "as equivalent if they clearly refer to the same event."
        )
    )

    # Course_Certification: verify with sources
    course_cert_leaf = evaluator.add_leaf(
        id="Course_Certification",
        desc="Confirmation that the course was certified by USATF, World Athletics, or AIMS is provided",
        parent=race_info_node,
        critical=True
    )
    course_cert_claim = (
        "The course for the 2022 Bank of America Chicago Marathon was certified by at least one of: "
        "USATF, World Athletics, or AIMS (e.g., USATF-certified course, World Athletics Label road race, "
        "AIMS/World Athletics measured/certified)."
    )
    await evaluator.verify(
        claim=course_cert_claim,
        node=course_cert_leaf,
        sources=info.record_urls,
        additional_instruction=(
            "Pass only if the page(s) explicitly indicate course certification/measurement or race sanctioning by "
            "USATF, World Athletics, or AIMS. Accept synonyms/equivalents such as 'World Athletics Gold/Platinum Label', "
            "'AIMS-certified measurement', 'USATF-certified course' or clear official certification language."
        )
    )

    # Record_Time: verify exact record time, using record URLs (and allow fallback to identity URLs if present)
    record_time_leaf = evaluator.add_leaf(
        id="Record_Time",
        desc="The exact finishing time that set the American record is provided",
        parent=details_node,
        critical=True
    )
    record_sources = _merge_sources(info.record_urls, info.identity_urls)
    record_time_claim = (
        f"The exact finishing time that set the American women's marathon record for {info.name or 'the athlete'} at the "
        f"{info.record_race_name or '2022 Bank of America Chicago Marathon'} was {info.record_time}."
    )
    await evaluator.verify(
        claim=record_time_claim,
        node=record_time_leaf,
        sources=record_sources,
        additional_instruction=(
            "Confirm that the exact time string (allowing only trivial formatting differences like presence/absence of "
            "seconds decimals) matches the record-setting result. Do not accept approximate times or ranges; it must be "
            "the official result time associated with the record at the 2022 Chicago Marathon."
        )
    )

    return {
        "node": details_node,
        "race_ref_node": race_ref_node,
    }


async def build_trials_performance(evaluator: Evaluator, parent, info: AthleteExtraction):
    """
    Olympic_Trials_Performance (parallel)
      - Trials_Placement (critical): verify finishing position
      - Trials_Time (critical): verify finishing time
      - Trials_Reference (critical): existence of at least one URL
    """
    trials_node = evaluator.add_parallel(
        id="Olympic_Trials_Performance",
        desc="Verification of performance at the 2024 U.S. Olympic Team Trials Marathon",
        parent=parent,
        critical=False
    )

    # Trials_Reference: existence
    trials_ref_node = evaluator.add_custom_node(
        result=len(info.trials_urls) > 0,
        id="Trials_Reference",
        desc="Reference URL for the 2024 Olympic Trials results is provided",
        parent=trials_node,
        critical=True
    )

    # Trials_Placement
    trials_place_leaf = evaluator.add_leaf(
        id="Trials_Placement",
        desc="The athlete's finishing position at the 2024 Olympic Trials is provided",
        parent=trials_node,
        critical=True
    )
    trials_place_claim = (
        f"At the 2024 U.S. Olympic Team Trials Marathon held in Orlando, Florida on February 3, 2024, "
        f"{info.name or 'the athlete'} finished {info.trials_position} in the women's marathon."
    )
    await evaluator.verify(
        claim=trials_place_claim,
        node=trials_place_leaf,
        sources=info.trials_urls,
        additional_instruction=(
            "Confirm the athlete's finishing placement exactly or with clear equivalence (e.g., '1st' ~ 'first'). "
            "Use the official results page(s) or credible reports."
        )
    )

    # Trials_Time
    trials_time_leaf = evaluator.add_leaf(
        id="Trials_Time",
        desc="The athlete's exact finishing time at the 2024 Olympic Trials is provided",
        parent=trials_node,
        critical=True
    )
    trials_time_claim = (
        f"At the same 2024 U.S. Olympic Team Trials Marathon in Orlando on February 3, 2024, "
        f"{info.name or 'the athlete'} finished with a time of {info.trials_time}."
    )
    await evaluator.verify(
        claim=trials_time_claim,
        node=trials_time_leaf,
        sources=info.trials_urls,
        additional_instruction=(
            "Verify the exact finishing time string; allow minor formatting differences but the numeric time must match. "
            "Prefer official/credible results sources."
        )
    )

    return {
        "node": trials_node,
        "trials_ref_node": trials_ref_node,
    }


async def build_team_selection(evaluator: Evaluator, parent, info: AthleteExtraction, trials_ref_node):
    """
    Olympic_Team_Selection (parallel)
      - Team_Qualification (critical): confirm top-3 at Trials -> qualified
      - Official_Selection (critical): confirm officially selected for Paris 2024
      - Selection_Reference (critical): existence of at least one URL
    """
    sel_node = evaluator.add_parallel(
        id="Olympic_Team_Selection",
        desc="Confirmation of selection to the U.S. Olympic team for Paris 2024",
        parent=parent,
        critical=False
    )

    selection_ref_node = evaluator.add_custom_node(
        result=len(info.selection_urls) > 0,
        id="Selection_Reference",
        desc="Reference URL confirming Olympic team selection is provided",
        parent=sel_node,
        critical=True
    )

    # Team_Qualification: confirm top-3 at Trials implies qualification
    team_qual_leaf = evaluator.add_leaf(
        id="Team_Qualification",
        desc="Confirmation that the athlete finished in the top three at the Olympic Trials and qualified for the Olympic team",
        parent=sel_node,
        critical=True
    )
    team_qual_claim = (
        f"By finishing in the top three in the women's race at the 2024 U.S. Olympic Team Trials Marathon in Orlando on "
        f"February 3, 2024, {info.name or 'the athlete'} qualified for the U.S. Olympic women's marathon team."
    )
    team_qual_sources = _merge_sources(info.trials_urls, info.selection_urls)
    await evaluator.verify(
        claim=team_qual_claim,
        node=team_qual_leaf,
        sources=team_qual_sources,
        additional_instruction=(
            "Confirm both: (a) she finished in the top three at the 2024 Trials (women's marathon), and (b) the top three "
            "finishers qualify for the Olympic team (either explicitly stated or clearly implied on official pages). "
            "If only placement is shown without qualification context, do not pass."
        )
    )

    # Official_Selection: confirm she was officially selected to represent USA at Paris 2024
    official_sel_leaf = evaluator.add_leaf(
        id="Official_Selection",
        desc="Confirmation that the athlete was officially selected to represent the United States in the women's marathon at Paris 2024",
        parent=sel_node,
        critical=True
    )
    official_sel_claim = (
        f"{info.name or 'the athlete'} was officially selected to represent the United States in the women's marathon "
        f"at the Paris 2024 Olympic Games."
    )
    await evaluator.verify(
        claim=official_sel_claim,
        node=official_sel_leaf,
        sources=info.selection_urls,
        additional_instruction=(
            "Look for official selection/roster announcements (e.g., USATF, Team USA/USOPC, or similarly authoritative sources). "
            "The page should explicitly state selection to the Paris 2024 Olympic women's marathon team."
        )
    )

    return {
        "node": sel_node,
        "selection_ref_node": selection_ref_node,
    }


async def build_olympic_performance(evaluator: Evaluator, parent, info: AthleteExtraction):
    """
    Olympic_Performance (parallel)
      - Olympic_Placement (critical): verify Olympic marathon finishing position
      - Olympic_Time (critical): verify Olympic marathon finishing time
      - Olympic_Reference (critical): existence of at least one URL
    """
    oly_node = evaluator.add_parallel(
        id="Olympic_Performance",
        desc="Verification of performance at the Paris 2024 Olympic Games",
        parent=parent,
        critical=False
    )

    olympic_ref_node = evaluator.add_custom_node(
        result=len(info.olympics_urls) > 0,
        id="Olympic_Reference",
        desc="Reference URL for the Paris 2024 Olympic marathon results is provided",
        parent=oly_node,
        critical=True
    )

    # Olympic_Placement
    oly_place_leaf = evaluator.add_leaf(
        id="Olympic_Placement",
        desc="The athlete's finishing position at the Paris 2024 Olympic marathon is provided",
        parent=oly_node,
        critical=True
    )
    oly_place_claim = (
        f"At the women's marathon of the Paris 2024 Olympic Games, {info.name or 'the athlete'} finished "
        f"{info.olympics_position}."
    )
    await evaluator.verify(
        claim=oly_place_claim,
        node=oly_place_leaf,
        sources=info.olympics_urls,
        additional_instruction=(
            "Confirm the placement as stated (allowing typical textual variants, e.g., '1st'/'first'). Use official "
            "Olympic/World Athletics results or equivalently authoritative sources."
        )
    )

    # Olympic_Time
    oly_time_leaf = evaluator.add_leaf(
        id="Olympic_Time",
        desc="The athlete's exact finishing time at the Paris 2024 Olympic marathon is provided",
        parent=oly_node,
        critical=True
    )
    oly_time_claim = (
        f"In that Olympic women's marathon, {info.name or 'the athlete'} recorded an official finishing time of "
        f"{info.olympics_time}."
    )
    await evaluator.verify(
        claim=oly_time_claim,
        node=oly_time_leaf,
        sources=info.olympics_urls,
        additional_instruction=(
            "Verify the exact time string (allow minimal formatting differences). If the athlete did not finish (DNF), "
            "then the claim should not pass."
        )
    )

    return {
        "node": oly_node,
        "olympic_ref_node": olympic_ref_node,
    }


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
    Evaluate an answer for the athlete journey (Chicago 2022 record -> Trials -> Selection -> Olympics) task.
    """
    # Initialize evaluator - root is sequential to reflect the required order of the journey
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

    # Extract athlete information from the answer
    info: AthleteExtraction = await evaluator.extract(
        prompt=prompt_extract_athlete_info(),
        template_class=AthleteExtraction,
        extraction_name="athlete_info"
    )

    # Build verification tree in the specified sequential order
    # 1) Athlete Identification
    identification_ctx = await build_athlete_identification(evaluator, root, info)

    # 2) American Record Details (ensure race reference leaf is created first within this subtree)
    record_ctx = await build_american_record_details(evaluator, root, info)

    # 3) Olympic Trials Performance
    trials_ctx = await build_trials_performance(evaluator, root, info)

    # 4) Olympic Team Selection (may depend on Trials reference implicitly; we already created Trials_Reference)
    selection_ctx = await build_team_selection(evaluator, root, info, trials_ctx["trials_ref_node"])

    # 5) Olympic Performance
    olympic_ctx = await build_olympic_performance(evaluator, root, info)

    # Return final structured evaluation summary
    return evaluator.get_summary()