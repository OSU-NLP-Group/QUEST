import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "horizon_player_2005_2015"
TASK_DESCRIPTION = """
Identify a NCAA Division I men's basketball player who meets all of the following criteria:

1. The player must have attended a university that is currently a member of the Horizon League conference (as of March 2026).
2. The player must have played all four years of their college career (freshman through senior season) at the same university.
3. During their senior season (final year), the player must have averaged at least 20.0 points per game.
4. The player's college basketball career must have concluded between 2005 and 2015 (inclusive).
5. The university where the player competed must currently offer at least one undergraduate degree program in business or a business-related field.

For your answer, provide:
- The player's full name
- The university where they played
- Their senior season scoring average (points per game) with a reference URL
- Confirmation that they played all four years at the same university with a reference URL
- The year their college career concluded
- The name of the conference (Horizon League) with a reference URL confirming the university's current membership
- At least one business or business-related undergraduate program offered by the university with a reference URL
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlayerSelection(BaseModel):
    player_name: Optional[str] = None
    university: Optional[str] = None

    senior_ppg: Optional[str] = None
    senior_season_label: Optional[str] = None  # e.g., "2010-11" or "Senior season"
    senior_ppg_urls: List[str] = Field(default_factory=list)

    four_years_same_univ_statement: Optional[str] = None
    four_years_urls: List[str] = Field(default_factory=list)

    career_conclusion_year: Optional[str] = None

    conference_name: Optional[str] = None
    conference_membership_urls: List[str] = Field(default_factory=list)

    business_program_name: Optional[str] = None
    business_program_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_player_selection() -> str:
    return """
Extract the following structured information exactly as presented in the answer. Do not invent or infer anything not explicitly stated. If a field is not present, return null (or an empty list for URL arrays).

Required fields:
- player_name: Full name of the player selected.
- university: The university where the player competed in college.
- senior_ppg: The numeric points-per-game value for the player's senior (final) college season, as stated in the answer (e.g., "20.3"). Do not include units; only the numeric text if possible.
- senior_season_label: Any season label cited for the senior season PPG (e.g., "2010-11", "Senior year"); if missing, return null.
- senior_ppg_urls: All URLs cited that support the senior season PPG value (array). Include only valid URLs explicitly present in the answer text.
- four_years_same_univ_statement: The sentence or phrase in the answer that claims the player played all four years at the same university; if absent, return null.
- four_years_urls: All URLs cited to support the four-years-at-same-university claim (array).
- career_conclusion_year: The year the player's college career concluded (e.g., "2011" or "2012"). If a range like "2011-12" is given, return the final calendar year if explicitly stated; otherwise, return the free-form string.
- conference_name: The named conference for the school; should be "Horizon League" if provided.
- conference_membership_urls: All URLs cited that confirm the university is a current member of the Horizon League as of March 2026 (array).
- business_program_name: The name of at least one undergraduate degree program in business or a business-related field that the university currently offers (e.g., "BBA in Accounting", "BS in Finance").
- business_program_urls: All URLs cited that support that the university currently offers the named undergraduate business/business-related program(s) (array).

URL extraction rules:
- Extract only URLs explicitly present in the answer.
- If a URL lacks protocol, prepend http://
- Accept both plain URLs and markdown links; extract the actual URL target.
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def parse_numeric_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Find the first plausible float in the string
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def parse_conclusion_year(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    # Prefer a 4-digit year; if multiple, take the last (often the final calendar year in a range)
    years = re.findall(r"\b(19|20)\d{2}\b", text)
    if years:
        try:
            return int(years[-1])
        except Exception:
            pass
    # As a fallback, try direct int
    try:
        return int(text)
    except Exception:
        return None


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any(isinstance(u, str) and u.strip() for u in urls or [])


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_player_and_university_identification(
    evaluator: Evaluator,
    parent_node,
    data: PlayerSelection,
) -> None:
    node = evaluator.add_parallel(
        id="Player_And_University_Identification",
        desc="Provide the player's identity and school.",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.player_name and data.player_name.strip()),
        id="Player_Full_Name_Provided",
        desc="The answer states the player's full name.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.university and data.university.strip()),
        id="University_Name_Provided",
        desc="The answer states the university where the player competed.",
        parent=node,
        critical=True,
    )


async def build_player_eligibility_requirements(
    evaluator: Evaluator,
    parent_node,
    data: PlayerSelection,
) -> None:
    node = evaluator.add_parallel(
        id="Player_Eligibility_Requirements",
        desc="Verify the identified player meets all player-specific constraints and required player outputs are present.",
        parent=parent_node,
        critical=True,
    )

    # Combine any player-related sources we have (senior PPG URLs and four-years URLs)
    combined_player_sources: List[str] = []
    if has_any_url(data.senior_ppg_urls):
        combined_player_sources.extend(data.senior_ppg_urls)
    if has_any_url(data.four_years_urls):
        combined_player_sources.extend(data.four_years_urls)

    # 1) NCAA Division I men's basketball player
    leaf_ncaad1 = evaluator.add_leaf(
        id="NCAA_Division_I_Mens_Basketball_Player",
        desc="The identified person is an NCAA Division I men's basketball player.",
        parent=node,
        critical=True,
    )
    safe_player = data.player_name or "the player"
    safe_univ = data.university or "the university"
    await evaluator.verify(
        claim=f"{safe_player} played NCAA Division I men's basketball for {safe_univ}.",
        node=leaf_ncaad1,
        sources=combined_player_sources if combined_player_sources else None,
        additional_instruction="Use the provided player-related sources (stats/bio pages) to confirm NCAA Division I men's basketball participation for the named university. Do not rely on general knowledge.",
    )

    # 2) Four years at same university WITH URL
    four_years_node = evaluator.add_parallel(
        id="Four_Years_Same_University_With_URL",
        desc="Verify the player played all four years (freshman through senior) at the same university, and provide a reference URL supporting this.",
        parent=node,
        critical=True,
    )

    # URL provided (existence)
    evaluator.add_custom_node(
        result=has_any_url(data.four_years_urls),
        id="Four_Years_Same_University_URL_Provided",
        desc="A reference URL is provided that supports the four-years-at-same-university claim.",
        parent=four_years_node,
        critical=True,
    )

    # Factual verification
    leaf_four_years = evaluator.add_leaf(
        id="Four_Years_Same_University",
        desc="The player played all four college seasons at the same university.",
        parent=four_years_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{safe_player} played all four college seasons (freshman through senior) at {safe_univ}, and did not transfer.",
        node=leaf_four_years,
        sources=data.four_years_urls if has_any_url(data.four_years_urls) else None,
        additional_instruction="Look for season-by-season roster participation or a career summary indicating all four seasons were at the same university with no transfer.",
    )

    # 3) Senior season PPG WITH URL and threshold
    senior_ppg_node = evaluator.add_parallel(
        id="Senior_Season_Scoring_Average_With_URL",
        desc="Provide the senior-season scoring average, ensure it meets the threshold, and provide a reference URL supporting it.",
        parent=node,
        critical=True,
    )

    # PPG provided (existence)
    evaluator.add_custom_node(
        result=bool(data.senior_ppg and data.senior_ppg.strip()),
        id="Senior_Season_PPG_Value_Provided",
        desc="The answer states the player's senior-season points-per-game value.",
        parent=senior_ppg_node,
        critical=True,
    )

    # URL provided (existence)
    evaluator.add_custom_node(
        result=has_any_url(data.senior_ppg_urls),
        id="Senior_Season_PPG_URL_Provided",
        desc="A reference URL is provided that supports the senior-season PPG value.",
        parent=senior_ppg_node,
        critical=True,
    )

    # PPG >= 20.0 (threshold check)
    ppg_val = parse_numeric_float(data.senior_ppg)
    evaluator.add_custom_node(
        result=(ppg_val is not None and ppg_val >= 20.0),
        id="Senior_Season_PPG_GTE_20",
        desc="The stated senior-season points-per-game value is at least 20.0 PPG.",
        parent=senior_ppg_node,
        critical=True,
    )

    # Senior-season (final year) verification using provided URLs
    leaf_senior_is_final = evaluator.add_leaf(
        id="Senior_Season_Is_Final_Year",
        desc="Provided sources support that the season used for the PPG value is the player's senior (final) college season.",
        parent=senior_ppg_node,
        critical=True,
    )
    season_lbl = f" ({data.senior_season_label})" if data.senior_season_label else ""
    ppg_str = data.senior_ppg or "the stated"
    await evaluator.verify(
        claim=f"In the senior (final) college season{season_lbl}, {safe_player} averaged {ppg_str} points per game.",
        node=leaf_senior_is_final,
        sources=data.senior_ppg_urls if has_any_url(data.senior_ppg_urls) else None,
        additional_instruction="Confirm that the PPG value corresponds specifically to the player's senior/final college season, not a different year.",
    )

    # 4) Career conclusion year provided and in range [2005, 2015]
    career_node = evaluator.add_parallel(
        id="Career_Conclusion_Year_Provided_And_In_Range",
        desc="Provide the year the player's college career concluded and verify it is between 2005 and 2015 inclusive.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.career_conclusion_year and data.career_conclusion_year.strip()),
        id="Career_Conclusion_Year_Provided",
        desc="The answer states the year the player's college career concluded.",
        parent=career_node,
        critical=True,
    )

    yr_val = parse_conclusion_year(data.career_conclusion_year)
    evaluator.add_custom_node(
        result=(yr_val is not None and 2005 <= yr_val <= 2015),
        id="Career_Conclusion_Year_Within_2005_2015",
        desc="The stated career conclusion year is between 2005 and 2015 inclusive.",
        parent=career_node,
        critical=True,
    )


async def build_university_requirements(
    evaluator: Evaluator,
    parent_node,
    data: PlayerSelection,
) -> None:
    node = evaluator.add_parallel(
        id="University_Requirements",
        desc="Verify the university meets Horizon League membership and business-program requirements and required outputs are present.",
        parent=parent_node,
        critical=True,
    )

    # Horizon League conference info with URL
    conf_node = evaluator.add_parallel(
        id="Horizon_League_Conference_Info_With_URL",
        desc="Provide the conference name (Horizon League) and a URL confirming the university's current membership as of March 2026.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.conference_name and ("horizon league" in data.conference_name.lower())),
        id="Conference_Name_Horizon_League_Provided",
        desc="The answer explicitly names the conference as the Horizon League.",
        parent=conf_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=has_any_url(data.conference_membership_urls),
        id="Horizon_League_Membership_URL_Provided",
        desc="A reference URL is provided confirming the university's current Horizon League membership.",
        parent=conf_node,
        critical=True,
    )

    leaf_member_now = evaluator.add_leaf(
        id="University_Current_Horizon_League_Member_As_Of_Mar_2026",
        desc="The university is a current Horizon League member as of March 2026.",
        parent=conf_node,
        critical=True,
    )
    safe_univ = data.university or "the university"
    await evaluator.verify(
        claim=f"As of March 2026, {safe_univ} is a current member of the Horizon League.",
        node=leaf_member_now,
        sources=data.conference_membership_urls if has_any_url(data.conference_membership_urls) else None,
        additional_instruction="Confirm that the membership page or official listings show the university as a current member (not historical) of the Horizon League.",
    )

    # Business undergraduate program with URL
    biz_node = evaluator.add_parallel(
        id="Business_Undergraduate_Program_With_URL",
        desc="Provide at least one business/business-related undergraduate program currently offered by the university and a URL supporting it.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(data.business_program_name and data.business_program_name.strip()),
        id="Business_Program_Name_Provided",
        desc="The answer names at least one undergraduate degree program in business or a business-related field offered by the university.",
        parent=biz_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=has_any_url(data.business_program_urls),
        id="Business_Program_URL_Provided",
        desc="A reference URL is provided supporting that the university offers the named business/business-related undergraduate program(s).",
        parent=biz_node,
        critical=True,
    )

    leaf_biz_undergrad = evaluator.add_leaf(
        id="Business_Program_Is_Undergraduate_And_Business_Related",
        desc="Provided sources support that the named program is an undergraduate program and is business or business-related.",
        parent=biz_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The program '{data.business_program_name or 'the named program'}' at {safe_univ} is an undergraduate (bachelor-level) program in business or a closely related field.",
        node=leaf_biz_undergrad,
        sources=data.business_program_urls if has_any_url(data.business_program_urls) else None,
        additional_instruction="Accept business majors such as Accounting, Finance, Management, Marketing, Supply Chain, Business Analytics, or similar business-related bachelor's programs (BBA/BS/BA). The page should clearly indicate undergraduate level.",
    )

    leaf_biz_current = evaluator.add_leaf(
        id="Business_Program_Currently_Offered",
        desc="Provided sources support that the named program is currently offered.",
        parent=biz_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{safe_univ} currently offers the undergraduate program '{data.business_program_name or 'the named program'}'.",
        node=leaf_biz_current,
        sources=data.business_program_urls if has_any_url(data.business_program_urls) else None,
        additional_instruction="The program page or official catalog should indicate it is an active program currently offered (not archived or discontinued).",
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
    model: str = "o4-mini",
) -> Dict:
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; we'll attach a critical sequential node under it
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

    # Extract structured info from the answer
    extracted: PlayerSelection = await evaluator.extract(
        prompt=prompt_extract_player_selection(),
        template_class=PlayerSelection,
        extraction_name="player_selection",
    )

    # Build the top-level critical sequential node as per rubric
    complete_node = evaluator.add_sequential(
        id="Complete_Task",
        desc="Identify a qualifying NCAA Division I men's basketball player from a current Horizon League school and provide all required fields with required references.",
        parent=root,
        critical=True,
    )

    # 1) Player and University identification
    await build_player_and_university_identification(evaluator, complete_node, extracted)

    # 2) Player eligibility requirements
    await build_player_eligibility_requirements(evaluator, complete_node, extracted)

    # 3) University requirements
    await build_university_requirements(evaluator, complete_node, extracted)

    # Return standard summary
    return evaluator.get_summary()