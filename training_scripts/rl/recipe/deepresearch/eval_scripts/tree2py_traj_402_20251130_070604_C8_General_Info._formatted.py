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
TASK_ID = "DWTS_Season_34_Finale_Response"
TASK_DESCRIPTION = (
    "Provide comprehensive information about all five finalist couples from the Dancing with the Stars Season 34 finale "
    "that aired on November 25, 2025. For each of the five finalist couples, include: (1) the celebrity contestant's age, "
    "(2) their professional dance partner's name and age (where age information is available), (3) the total score they "
    "received across all three finale dance performances, and (4) their final placement ranking. Additionally, provide the "
    "name and complete address of the venue where the finale was held."
)

SEASON_NAME = "Dancing with the Stars Season 34"
FINALE_DATE_STR = "November 25, 2025"
AIRTIME_ET_STR = "8:00–11:00 PM ET"
VENUE_NAME_OFFICIAL = "CBS Television City Studios"
VENUE_NAME_ALIAS = "Television City Studios"
VENUE_FULL_ADDRESS = "7800 Beverly Boulevard, Los Angeles, CA 90036"

# Expected couples configuration for reference and descriptions
COUPLES_CONFIG = [
    {
        "id": "couple_1",
        "label": "Robert Irwin & Witney Carson",
        "celebrity": "Robert Irwin",
        "pro": "Witney Carson",
        "expected_total": "89",
        "expected_placement": "1st",
        "expected_celebrity_age": "21",
        "expected_pro_age": "32",
        "pro_age_optional": False,
        "include_nationality": True,
        "expected_nationality": "Australian",
        "ex_field": "robert_witney",
    },
    {
        "id": "couple_2",
        "label": "Alix Earle & Val Chmerkovskiy",
        "celebrity": "Alix Earle",
        "pro": "Val Chmerkovskiy",
        "expected_total": "90",
        "expected_placement": "2nd",
        "expected_celebrity_age": "24",
        "expected_pro_age": "39",
        "pro_age_optional": False,
        "include_nationality": False,
        "ex_field": "alix_val",
    },
    {
        "id": "couple_3",
        "label": "Jordan Chiles & Ezra Sosa",
        "celebrity": "Jordan Chiles",
        "pro": "Ezra Sosa",
        "expected_total": "89",
        "expected_placement": "3rd",
        "expected_celebrity_age": None,  # Must be a correct numeric age as of 11/25/2025
        "expected_pro_age": None,        # Provided if available; otherwise explicitly marked not available
        "pro_age_optional": True,
        "include_nationality": False,
        "ex_field": "jordan_ezra",
    },
    {
        "id": "couple_4",
        "label": "Dylan Efron & Daniella Karagach",
        "celebrity": "Dylan Efron",
        "pro": "Daniella Karagach",
        "expected_total": "88",
        "expected_placement": "4th",
        "expected_celebrity_age": "33",
        "expected_pro_age": "32",
        "pro_age_optional": False,
        "include_nationality": False,
        "ex_field": "dylan_daniella",
    },
    {
        "id": "couple_5",
        "label": "Elaine Hendrix & Alan Bersten",
        "celebrity": "Elaine Hendrix",
        "pro": "Alan Bersten",
        "expected_total": "87",
        "expected_placement": "5th",
        "expected_celebrity_age": "54",
        "expected_pro_age": None,        # Provided if available; otherwise explicitly marked not available
        "pro_age_optional": True,
        "include_nationality": False,
        "ex_field": "elaine_alan",
    },
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoupleInfo(BaseModel):
    celebrity_name: Optional[str] = None
    celebrity_age: Optional[str] = None  # Keep as string for flexibility
    celebrity_nationality: Optional[str] = None
    pro_name: Optional[str] = None
    pro_age: Optional[str] = None
    pro_age_not_available: Optional[bool] = None  # True iff the answer explicitly indicates pro age info is not available
    total_finale_score: Optional[str] = None  # e.g., "89"
    final_placement: Optional[str] = None     # e.g., "1st"
    sources: List[str] = Field(default_factory=list)


class DWTSFinaleExtraction(BaseModel):
    # Finale context and venue
    season_finale_air_date: Optional[str] = None  # e.g., "November 25, 2025"
    airtime_et: Optional[str] = None             # e.g., "8:00–11:00 PM ET"
    venue_name: Optional[str] = None             # e.g., "CBS Television City Studios"
    venue_address: Optional[str] = None          # e.g., full address
    context_sources: List[str] = Field(default_factory=list)

    # Finale format constraints
    finalists_count_statement: Optional[str] = None           # text mentioning exactly five finalists
    dance_rounds_list: List[str] = Field(default_factory=list)  # list of round names stated
    freestyle_perfect_scores_statement: Optional[str] = None  # text indicating all five got 30/30 in freestyle
    format_sources: List[str] = Field(default_factory=list)

    # Couples - fixed five entries
    robert_witney: CoupleInfo = CoupleInfo()
    alix_val: CoupleInfo = CoupleInfo()
    jordan_ezra: CoupleInfo = CoupleInfo()
    dylan_daniella: CoupleInfo = CoupleInfo()
    elaine_alan: CoupleInfo = CoupleInfo()

    # Any additional finale URLs cited in the answer
    finale_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_dwts_finale() -> str:
    return (
        "Extract structured information about the Dancing with the Stars Season 34 finale as presented in the answer.\n"
        "Return a single JSON object matching the DWTSFinaleExtraction schema.\n"
        "- Finale context:\n"
        "  • season_finale_air_date: the stated air date of the referenced finale (e.g., 'November 25, 2025').\n"
        "  • airtime_et: the stated airtime in Eastern Time (e.g., '8:00–11:00 PM ET').\n"
        "  • venue_name: the stated venue name (e.g., 'CBS Television City Studios' or 'Television City Studios').\n"
        "  • venue_address: the stated complete venue address (e.g., '7800 Beverly Boulevard, Los Angeles, CA 90036').\n"
        "  • context_sources: all URLs cited in the answer that support the finale context/venue info.\n"
        "- Finale format constraints:\n"
        "  • finalists_count_statement: any explicit statement that there were exactly five finalists.\n"
        "  • dance_rounds_list: list the named rounds the answer states happened; include entries like 'Judges' Choice', "
        "'Instant Dance Challenge', and 'Freestyle' if present.\n"
        "  • freestyle_perfect_scores_statement: any explicit statement that all five couples received perfect 30/30 for freestyle.\n"
        "  • format_sources: all URLs cited in the answer that support format constraints.\n"
        "- Couples (fixed five): For each, extract the following fields and URLs tied specifically to that couple:\n"
        "  • celebrity_name, celebrity_age (numeric string if given), celebrity_nationality (if stated).\n"
        "  • pro_name, pro_age (numeric string if given). If the answer explicitly indicates pro age info is NOT available, "
        "set pro_age_not_available to true; otherwise false or null.\n"
        "  • total_finale_score: the total score across all three finale dances.\n"
        "  • final_placement: the final ranking placement (e.g., '1st', '2nd', etc.).\n"
        "  • sources: all URLs cited in the answer that support this couple's data.\n"
        "Map the couples to the following fixed fields:\n"
        "  • robert_witney -> Robert Irwin & Witney Carson\n"
        "  • alix_val -> Alix Earle & Val Chmerkovskiy\n"
        "  • jordan_ezra -> Jordan Chiles & Ezra Sosa\n"
        "  • dylan_daniella -> Dylan Efron & Daniella Karagach\n"
        "  • elaine_alan -> Elaine Hendrix & Alan Bersten\n"
        "Rules:\n"
        "  1) Extract only what is explicitly stated in the answer; do not invent values.\n"
        "  2) For URLs, include only valid URLs that appear in the answer. If no URL is provided for a field, return an empty list.\n"
        "  3) Keep ages as strings. If age info is stated as unavailable, set pro_age_not_available true.\n"
        "  4) If a required field is missing for a couple, set it to null. Always include all five couples in the JSON."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def union_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url or not isinstance(url, str):
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_context_and_venue(evaluator: Evaluator, parent_node, ex: DWTSFinaleExtraction) -> None:
    node = evaluator.add_parallel(
        id="Finale_Context_and_Venue",
        desc="Correctly identifies the referenced finale and provides venue name and complete address.",
        parent=parent_node,
        critical=True
    )

    # Existence of supporting sources
    context_sources_present = len(union_sources(ex.context_sources, ex.finale_sources)) > 0
    evaluator.add_custom_node(
        result=context_sources_present,
        id="Context_Sources_Provided",
        desc="Context/venue supporting sources are provided in the answer.",
        parent=node,
        critical=True
    )

    # Season and Finale Date
    leaf_season_date = evaluator.add_leaf(
        id="Season_and_Finale_Date",
        desc=f"States this is the {SEASON_NAME} finale that aired on {FINALE_DATE_STR}.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The referenced finale is {SEASON_NAME} and it aired on {FINALE_DATE_STR}.",
        node=leaf_season_date,
        sources=union_sources(ex.context_sources, ex.finale_sources),
        additional_instruction="Allow reasonable wording variants for the air date. Verify the season and the finale date explicitly on the cited webpages."
    )

    # Finale airtime ET
    leaf_airtime = evaluator.add_leaf(
        id="Finale_Airtime_ET",
        desc=f"States the finale aired from {AIRTIME_ET_STR}.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The finale aired from {AIRTIME_ET_STR}.",
        node=leaf_airtime,
        sources=union_sources(ex.context_sources, ex.finale_sources),
        additional_instruction="Accept formatting variants like '8–11 PM ET' or '8:00 PM to 11:00 PM ET'. Confirm explicitly from the sources."
    )

    # Venue Name
    leaf_venue_name = evaluator.add_leaf(
        id="Venue_Name",
        desc=f"Venue name is identified as {VENUE_NAME_OFFICIAL} (or {VENUE_NAME_ALIAS}).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The finale was held at {VENUE_NAME_OFFICIAL}, also known as {VENUE_NAME_ALIAS}.",
        node=leaf_venue_name,
        sources=union_sources(ex.context_sources, ex.finale_sources),
        additional_instruction="Treat 'Television City Studios' and 'CBS Television City Studios' as equivalent naming variants for the same venue."
    )

    # Venue Complete Address
    leaf_venue_addr = evaluator.add_leaf(
        id="Venue_Complete_Address",
        desc=f"Venue address is stated as {VENUE_FULL_ADDRESS}.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The venue's complete address is {VENUE_FULL_ADDRESS}.",
        node=leaf_venue_addr,
        sources=union_sources(ex.context_sources, ex.finale_sources),
        additional_instruction="Verify that the address listed in sources matches exactly, allowing minor formatting variations."
    )


async def verify_format_constraints(evaluator: Evaluator, parent_node, ex: DWTSFinaleExtraction) -> None:
    node = evaluator.add_parallel(
        id="Finale_Format_Constraints",
        desc="Captures required constraints about finalist count and the finale’s dance format/scoring.",
        parent=parent_node,
        critical=True
    )

    # Existence of format supporting sources
    format_sources_present = len(union_sources(ex.format_sources, ex.context_sources, ex.finale_sources)) > 0
    evaluator.add_custom_node(
        result=format_sources_present,
        id="Format_Sources_Provided",
        desc="Format/scoring supporting sources are provided in the answer.",
        parent=node,
        critical=True
    )

    # Exactly five finalists
    leaf_five_finalists = evaluator.add_leaf(
        id="Exactly_Five_Finalists",
        desc="States (or otherwise makes clear) there were exactly five finalists in the finale.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="There were exactly five finalists in the Season 34 finale.",
        node=leaf_five_finalists,
        sources=union_sources(ex.format_sources, ex.context_sources, ex.finale_sources),
        additional_instruction="Confirm that the finale field consisted of exactly five finalist couples; do not count semifinalists or eliminated contestants."
    )

    # Three dance rounds specified
    leaf_three_rounds = evaluator.add_leaf(
        id="Three_Dance_Rounds_Specified",
        desc="States each finalist performed three dances: Judges' Choice, Instant Dance Challenge, and Freestyle.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Each finalist performed three dances in the finale: Judges' Choice, Instant Dance Challenge, and Freestyle.",
        node=leaf_three_rounds,
        sources=union_sources(ex.format_sources, ex.context_sources, ex.finale_sources),
        additional_instruction="Verify the names of the rounds explicitly as stated on the sources. Allow minor capitalization/formatting differences."
    )

    # Freestyle perfect scores for all five
    leaf_freestyle_30 = evaluator.add_leaf(
        id="Freestyle_Perfect_Scores_All_Five",
        desc="States that all five couples received perfect 30/30 for their freestyle performances.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="All five finalist couples received perfect 30/30 scores for their freestyle performances.",
        node=leaf_freestyle_30,
        sources=union_sources(ex.format_sources, ex.context_sources, ex.finale_sources),
        additional_instruction="Confirm explicitly that each of the five couples received 30/30 for the freestyle round."
    )


async def verify_single_couple(
    evaluator: Evaluator,
    parent_node,
    cfg: Dict[str, Any],
    couple: CoupleInfo
) -> None:
    couple_node = evaluator.add_parallel(
        id=cfg["id"],
        desc=f"Requested attributes for {cfg['label']}.",
        parent=parent_node,
        critical=False  # allow partial credit across couples
    )

    # Gate: required info and sources present
    has_min_info = (
        (couple.celebrity_name is not None and couple.celebrity_name.strip() != "") and
        (couple.pro_name is not None and couple.pro_name.strip() != "") and
        len(couple.sources) > 0
    )
    evaluator.add_custom_node(
        result=has_min_info,
        id=f"{cfg['id']}_required_info_and_sources",
        desc=f"{cfg['label']} has required names and at least one supporting source URL.",
        parent=couple_node,
        critical=True
    )

    # Celebrity age
    leaf_celeb_age = evaluator.add_leaf(
        id=f"{cfg['id']}_Celebrity_Age",
        desc=f"Celebrity ({cfg['celebrity']}) age is correctly stated as of the finale date.",
        parent=couple_node,
        critical=True
    )
    celeb_age_str = couple.celebrity_age or ""
    await evaluator.verify(
        claim=f"{cfg['celebrity']}'s age is {celeb_age_str} as of {FINALE_DATE_STR}.",
        node=leaf_celeb_age,
        sources=couple.sources,
        additional_instruction="Use the source DOB to calculate age as of the finale date if age is not explicitly stated. Allow minor rounding if DOB implies same age."
    )

    # Optional nationality for Robert Irwin
    if cfg.get("include_nationality", False):
        leaf_celeb_nat = evaluator.add_leaf(
            id=f"{cfg['id']}_Celebrity_Nationality",
            desc=f"States {cfg['celebrity']}'s nationality as {cfg['expected_nationality']}.",
            parent=couple_node,
            critical=True
        )
        await evaluator.verify(
            claim=f"{cfg['celebrity']} is {cfg['expected_nationality']}.",
            node=leaf_celeb_nat,
            sources=couple.sources,
            additional_instruction="Confirm nationality from reliable sources; allow synonyms like 'Australian TV personality' to count as Australian."
        )

    # Pro partner name (verify via sources)
    leaf_pro_name = evaluator.add_leaf(
        id=f"{cfg['id']}_Pro_Partner_Name",
        desc=f"Professional partner is correctly named as {cfg['pro']}.",
        parent=couple_node,
        critical=True
    )
    pro_name_str = couple.pro_name or ""
    await evaluator.verify(
        claim=f"The professional partner for {cfg['celebrity']} in Season 34 was {pro_name_str}.",
        node=leaf_pro_name,
        sources=couple.sources,
        additional_instruction=f"Confirm the pairing for Season 34. The expected professional partner is {cfg['pro']}. Treat minor name variants or nicknames as equivalent."
    )

    # Pro partner age (provided OR explicitly marked not available)
    leaf_pro_age = evaluator.add_leaf(
        id=f"{cfg['id']}_Pro_Partner_Age",
        desc="Professional partner age is verified (or answer explicitly indicates age information is not available when permitted).",
        parent=couple_node,
        critical=True
    )
    # Decide verification path:
    if couple.pro_age and couple.pro_age.strip():
        # Verify numeric age via sources
        pro_age_str = couple.pro_age.strip()
        await evaluator.verify(
            claim=f"{cfg['pro']}'s age is {pro_age_str} as of {FINALE_DATE_STR}.",
            node=leaf_pro_age,
            sources=couple.sources,
            additional_instruction="Use DOB to compute age as of the finale date when needed. If the source gives age explicitly, verify it matches exactly."
        )
    else:
        # Either optional or not: if optional allowed, verify the answer stated unavailability; otherwise still try to verify age (will likely fail)
        if cfg.get("pro_age_optional", False):
            await evaluator.verify(
                claim=f"The answer explicitly indicates that {cfg['pro']}'s age information is not available.",
                node=leaf_pro_age,
                sources=None,
                additional_instruction="Check the answer text. Accept phrases like 'age not available', 'not publicly available', 'unknown', or 'not disclosed'."
            )
        else:
            # If not optional, we still attempt to verify empty age against sources (expected to fail).
            await evaluator.verify(
                claim=f"{cfg['pro']}'s age is provided and correct as of {FINALE_DATE_STR}.",
                node=leaf_pro_age,
                sources=couple.sources,
                additional_instruction="Verify that the age is explicitly provided and correct. If no age is present, this should fail."
            )

    # Total finale score
    leaf_total_score = evaluator.add_leaf(
        id=f"{cfg['id']}_Total_Finale_Score",
        desc="Total score across all three finale dances is correctly stated.",
        parent=couple_node,
        critical=True
    )
    total_score_str = couple.total_finale_score or ""
    await evaluator.verify(
        claim=f"The total score across all three finale dances for {cfg['celebrity']} & {cfg['pro']} is {total_score_str}.",
        node=leaf_total_score,
        sources=couple.sources,
        additional_instruction="Confirm the total from sources. If only per-dance scores are given, sum them to validate the total."
    )

    # Final placement
    leaf_final_place = evaluator.add_leaf(
        id=f"{cfg['id']}_Final_Placement",
        desc="Final placement is correctly stated.",
        parent=couple_node,
        critical=True
    )
    final_place_str = couple.final_placement or ""
    await evaluator.verify(
        claim=f"The final placement for {cfg['celebrity']} & {cfg['pro']} was {final_place_str}.",
        node=leaf_final_place,
        sources=couple.sources,
        additional_instruction="Verify the final ranking from reliable sources. Accept ordinal formatting variants like '1st', 'first', etc."
    )


async def verify_all_couples(evaluator: Evaluator, parent_node, ex: DWTSFinaleExtraction) -> None:
    couples_parent = evaluator.add_parallel(
        id="Five_Finalist_Couples",
        desc="Provides the requested attributes for each of the five finalist couples.",
        parent=parent_node,
        critical=False  # allow partial credit across couples
    )

    # Map extraction fields to configs
    field_map = {
        "robert_witney": ex.robert_witney,
        "alix_val": ex.alix_val,
        "jordan_ezra": ex.jordan_ezra,
        "dylan_daniella": ex.dylan_daniella,
        "elaine_alan": ex.elaine_alan,
    }

    for cfg in COUPLES_CONFIG:
        couple_info = field_map.get(cfg["ex_field"], CoupleInfo())
        await verify_single_couple(evaluator, couples_parent, cfg, couple_info)


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
    Build the verification tree, extract structured data from the answer, and run checks.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root aggregates independent major sections
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

    # IMPORTANT: Set root as non-critical to allow a mixture of critical and non-critical children safely
    # Note: The framework enforces that critical parents cannot have non-critical children.
    root.critical = False

    # Extract structured info from the answer
    ex: DWTSFinaleExtraction = await evaluator.extract(
        prompt=prompt_extract_dwts_finale(),
        template_class=DWTSFinaleExtraction,
        extraction_name="dwts_finale_extraction"
    )

    # Optional: Add ground truth info for reference (no scoring impact directly)
    evaluator.add_ground_truth({
        "season": SEASON_NAME,
        "finale_date": FINALE_DATE_STR,
        "airtime_et": AIRTIME_ET_STR,
        "venue_name_official": VENUE_NAME_OFFICIAL,
        "venue_name_alias": VENUE_NAME_ALIAS,
        "venue_full_address": VENUE_FULL_ADDRESS,
        "expected_couples_order": [cfg["label"] for cfg in COUPLES_CONFIG]
    }, gt_type="dwts_ground_truth_reference")

    # Build tree and verify
    await verify_context_and_venue(evaluator, root, ex)
    await verify_format_constraints(evaluator, root, ex)
    await verify_all_couples(evaluator, root, ex)

    # Return summary
    return evaluator.get_summary()