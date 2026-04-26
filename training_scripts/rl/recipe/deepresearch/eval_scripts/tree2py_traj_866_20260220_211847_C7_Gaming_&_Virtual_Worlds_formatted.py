import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wwe2k26_comprehensive_specs"
TASK_DESCRIPTION = (
    "Provide comprehensive specifications for WWE 2K26 by answering the following 14 questions: "
    "1. What is the release date for the Standard Edition? "
    "2. What is the early access date for premium editions (King of Kings Edition, Attitude Era Edition, and Monday Night War Edition)? "
    "3. What are the four platforms the game is available on? "
    "4. What is the price of the Standard Edition in USD? "
    "5. How many playable characters are in the roster (provide the stated range)? "
    "6. How many new match types are introduced in WWE 2K26? "
    "7. How many Create-A-Superstar (CAW) slots are in the Creation Suite? "
    "8. How many Create-An-Image slots are in the Creation Suite? "
    "9. What is the default season length in MyGM mode (in weeks)? "
    "10. How many new GMs are added to MyGM mode? "
    "11. What exclusive Creation Suite feature does the Nintendo Switch 2 version have? "
    "12. Which wrestler's career is the Showcase mode dedicated to? "
    "13. What is the title of the MyRISE mode story? "
    "14. How many Ringside Pass Seasons are planned throughout the year? "
    "All answers must be supported by official sources or announcements from WWE 2K26."
)

# Expected values (treated as ground truth targets)
EXPECTED_SPECS = {
    "Standard_Edition_Release_Date": "March 13, 2026",
    "Premium_Early_Access_Date": "March 6, 2026",
    "Available_Platforms": ["PlayStation 5", "Xbox Series X|S", "Nintendo Switch 2", "PC via Steam"],
    "Standard_Edition_Price": "$69.99 USD",
    "Roster_Size": "400+ Superstars and Legends",
    "New_Match_Types_Count": "4",
    "CAW_Slots_Count": "200",
    "Image_Slots_Count": "2,000",
    "MyGM_Default_Season_Length": "50 weeks",
    "MyGM_New_GMs_Count": "3",
    "Switch2_Creation_Suite_Feature": "exclusive mouse support for Creation Suite face and body painting",
    "Showcase_Mode_Subject": "CM Punk",
    "MyRISE_Story_Title": "The Comeback",
    "Ringside_Pass_Seasons_Count": "6",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WWE2K26Specs(BaseModel):
    # Dates
    standard_edition_release_date: Optional[str] = None
    standard_edition_release_date_sources: List[str] = Field(default_factory=list)

    premium_early_access_date: Optional[str] = None
    premium_early_access_date_sources: List[str] = Field(default_factory=list)

    # Platforms
    available_platforms: List[str] = Field(default_factory=list)
    available_platforms_sources: List[str] = Field(default_factory=list)

    # Price
    standard_edition_price_usd: Optional[str] = None
    standard_edition_price_usd_sources: List[str] = Field(default_factory=list)

    # Roster size
    roster_size_range: Optional[str] = None
    roster_size_range_sources: List[str] = Field(default_factory=list)

    # New match types
    new_match_types_count: Optional[str] = None
    new_match_types_count_sources: List[str] = Field(default_factory=list)

    # Creation Suite slots
    caw_slots_count: Optional[str] = None
    caw_slots_count_sources: List[str] = Field(default_factory=list)

    image_slots_count: Optional[str] = None
    image_slots_count_sources: List[str] = Field(default_factory=list)

    # MyGM details
    mygm_default_season_length_weeks: Optional[str] = None
    mygm_default_season_length_weeks_sources: List[str] = Field(default_factory=list)

    mygm_new_gms_count: Optional[str] = None
    mygm_new_gms_count_sources: List[str] = Field(default_factory=list)

    # Switch 2 feature
    switch2_creation_suite_feature: Optional[str] = None
    switch2_creation_suite_feature_sources: List[str] = Field(default_factory=list)

    # Showcase subject
    showcase_mode_subject: Optional[str] = None
    showcase_mode_subject_sources: List[str] = Field(default_factory=list)

    # MyRISE title
    myrise_story_title: Optional[str] = None
    myrise_story_title_sources: List[str] = Field(default_factory=list)

    # Ringside Pass seasons
    ringside_pass_seasons_count: Optional[str] = None
    ringside_pass_seasons_count_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_specs() -> str:
    return """
    Extract the comprehensive specifications for WWE 2K26 from the answer. For each item below, extract the value exactly as stated in the answer (do not normalize the format), and also extract all cited URLs that specifically support that item (official announcements or pages are preferred). If an item is missing, set the value to null and return an empty array for sources.

    1) standard_edition_release_date (string)
       standard_edition_release_date_sources (array of URLs)

    2) premium_early_access_date (string)
       premium_early_access_date_sources (array of URLs)

    3) available_platforms (array of strings; list each platform verbatim as stated, e.g., "PlayStation 5", "Xbox Series X|S", "Nintendo Switch 2", "PC via Steam")
       available_platforms_sources (array of URLs)

    4) standard_edition_price_usd (string; keep currency formatting exactly as shown, e.g., "$69.99 USD")
       standard_edition_price_usd_sources (array of URLs)

    5) roster_size_range (string; e.g., "400+ Superstars and Legends", "over 400 playable characters")
       roster_size_range_sources (array of URLs)

    6) new_match_types_count (string; allow numeric or spelled-out form, e.g., "4" or "four")
       new_match_types_count_sources (array of URLs)

    7) caw_slots_count (string; e.g., "200")
       caw_slots_count_sources (array of URLs)

    8) image_slots_count (string; e.g., "2,000" or "2000")
       image_slots_count_sources (array of URLs)

    9) mygm_default_season_length_weeks (string; e.g., "50 weeks")
       mygm_default_season_length_weeks_sources (array of URLs)

    10) mygm_new_gms_count (string; e.g., "3" or "three")
        mygm_new_gms_count_sources (array of URLs)

    11) switch2_creation_suite_feature (string; e.g., "exclusive mouse support for Creation Suite face and body painting")
        switch2_creation_suite_feature_sources (array of URLs)

    12) showcase_mode_subject (string; e.g., "CM Punk")
        showcase_mode_subject_sources (array of URLs)

    13) myrise_story_title (string; e.g., "The Comeback")
        myrise_story_title_sources (array of URLs)

    14) ringside_pass_seasons_count (string; e.g., "6" or "six")
        ringside_pass_seasons_count_sources (array of URLs)

    SPECIAL RULES:
    - Sources must be explicitly present as URLs in the answer (plain URLs or markdown links). If no URL is provided for an item, return an empty array for that item's sources.
    - Do not infer or create any values or URLs that are not present in the answer.
    - Preserve original formatting (e.g., "$69.99 USD", "2,000", "March 13, 2026") exactly as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _readable_name(spec_id: str) -> str:
    return spec_id.replace("_", " ").strip()


def _non_empty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_sources(urls: Optional[List[str]]) -> bool:
    return isinstance(urls, list) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_scalar_spec(
    evaluator: Evaluator,
    parent_node,
    spec_id: str,
    spec_claim_text: str,
    answer_value: Optional[str],
    sources: Optional[List[str]],
    expected_value: str,
    match_additional: Optional[str] = None,
    support_additional: Optional[str] = None,
) -> None:
    """
    Generic verification for a scalar spec (string-based value).
    Creates:
      - value_provided (critical custom leaf)
      - sources_provided (critical custom leaf)
      - match_expected (critical leaf, simple verification against the answer)
      - supported_by_sources (critical leaf, URL-grounded verification)
    """
    node = evaluator.add_parallel(
        id=spec_id,
        desc=spec_claim_text,
        parent=parent_node,
        critical=False,
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_non_empty_str(answer_value),
        id=f"{spec_id}_value_provided",
        desc=f"The answer provides a value for {_readable_name(spec_id)}",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_sources(sources),
        id=f"{spec_id}_sources_provided",
        desc=f"Official sources are cited for {_readable_name(spec_id)}",
        parent=node,
        critical=True,
    )

    # Match expected value in the answer
    match_node = evaluator.add_leaf(
        id=f"{spec_id}_match_expected",
        desc=f"Answer matches the expected value for {_readable_name(spec_id)}",
        parent=node,
        critical=True,
    )
    match_claim = f"In the answer, {spec_claim_text}."
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction=(match_additional or "Allow minor formatting variants (e.g., date formats, casing, punctuation). Focus on semantic equivalence with the expected value."),
    )

    # Support by sources (URL-grounded)
    support_node = evaluator.add_leaf(
        id=f"{spec_id}_supported_by_sources",
        desc=f"'{spec_claim_text}' is supported by the cited official sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=spec_claim_text,
        node=support_node,
        sources=sources or [],
        additional_instruction=(support_additional or "Prefer official WWE 2K / 2K / platforms' official channels. Allow reasonable phrasing variants; verify that the source explicitly supports the claim."),
    )


async def verify_platforms_spec(
    evaluator: Evaluator,
    parent_node,
    spec_id: str,
    spec_claim_text: str,
    extracted_platforms: Optional[List[str]],
    sources: Optional[List[str]],
    expected_platforms: List[str],
) -> None:
    """
    Specialized verification for platforms.
    Creates:
      - value_provided (critical custom leaf)
      - sources_provided (critical custom leaf)
      - match_expected (critical leaf)
      - supported_by_sources (critical leaf)
    """
    node = evaluator.add_parallel(
        id=spec_id,
        desc=spec_claim_text,
        parent=parent_node,
        critical=False,
    )

    evaluator.add_custom_node(
        result=isinstance(extracted_platforms, list) and len(extracted_platforms) > 0,
        id=f"{spec_id}_value_provided",
        desc="The answer provides a list of platforms",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_sources(sources),
        id=f"{spec_id}_sources_provided",
        desc="Official sources are cited for platforms",
        parent=node,
        critical=True,
    )

    # Match expected platforms in the answer
    match_node = evaluator.add_leaf(
        id=f"{spec_id}_match_expected",
        desc="Answer lists platforms equivalent to the expected four",
        parent=node,
        critical=True,
    )
    expected_list_text = "; ".join(expected_platforms)
    match_claim = (
        f"In the answer, the platforms are equivalent to exactly these four: {expected_list_text}."
    )
    match_instruction = (
        "Allow reasonable synonyms and minor naming variants, e.g., 'PS5' ~ 'PlayStation 5'; "
        "'Xbox Series X and Xbox Series S' ~ 'Xbox Series X|S'; "
        "'PC' ~ 'PC via Steam' if the answer otherwise implies Steam availability; "
        "Ensure the answer lists exactly four platforms equivalent to the expected set."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_node,
        additional_instruction=match_instruction,
    )

    # Support by sources (URL-grounded)
    support_node = evaluator.add_leaf(
        id=f"{spec_id}_supported_by_sources",
        desc="Platform availability is supported by cited official sources",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=spec_claim_text,
        node=support_node,
        sources=sources or [],
        additional_instruction="Verify that the official source(s) explicitly list these four platforms. Allow minor naming variants but require clear official confirmation.",
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
    """
    Evaluate an answer for WWE 2K26 comprehensive specifications.
    """
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

    # Extract structured specs
    specs: WWE2K26Specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=WWE2K26Specs,
        extraction_name="wwe2k26_specs",
    )

    # Add ground truth / expected targets for transparency
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED_SPECS,
            "note": "These are the target values to be matched and supported by official sources.",
        },
        gt_type="expected_specs",
    )

    # Build sub-tree root node (parallel aggregation)
    main_node = evaluator.add_parallel(
        id="WWE_2K26_Comprehensive_Specifications",
        desc="Verify comprehensive specifications for WWE 2K26 across release dates, platforms, pricing, features, and content details",
        parent=root,
        critical=False,
    )

    # 1) Standard Edition Release Date
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Standard_Edition_Release_Date",
        spec_claim_text="The Standard Edition release date is March 13, 2026",
        answer_value=specs.standard_edition_release_date,
        sources=specs.standard_edition_release_date_sources,
        expected_value=EXPECTED_SPECS["Standard_Edition_Release_Date"],
        match_additional="Allow date format variants (e.g., 'March 13, 2026' vs '13 March 2026'); treat them as equivalent.",
        support_additional="Verify that the official source explicitly states the Standard Edition releases on March 13, 2026.",
    )

    # 2) Premium Early Access Date
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Premium_Early_Access_Date",
        spec_claim_text="Premium editions (King of Kings Edition, Attitude Era Edition, and Monday Night War Edition) have early access starting March 6, 2026",
        answer_value=specs.premium_early_access_date,
        sources=specs.premium_early_access_date_sources,
        expected_value=EXPECTED_SPECS["Premium_Early_Access_Date"],
        match_additional="Allow date format variants; the key is that early access starts on March 6, 2026.",
        support_additional="Verify official confirmation that premium editions' early access starts on March 6, 2026.",
    )

    # 3) Available Platforms
    await verify_platforms_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Available_Platforms",
        spec_claim_text="The game is available on exactly four platforms: PlayStation 5, Xbox Series X|S, Nintendo Switch 2, and PC via Steam",
        extracted_platforms=specs.available_platforms,
        sources=specs.available_platforms_sources,
        expected_platforms=EXPECTED_SPECS["Available_Platforms"],
    )

    # 4) Standard Edition Price
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Standard_Edition_Price",
        spec_claim_text="The Standard Edition costs $69.99 USD",
        answer_value=specs.standard_edition_price_usd,
        sources=specs.standard_edition_price_usd_sources,
        expected_value=EXPECTED_SPECS["Standard_Edition_Price"],
        match_additional="Allow currency formatting variants like '$69.99 USD', 'USD $69.99', or 'US$ 69.99'. Focus on US price.",
        support_additional="Verify the US Standard Edition price is stated as $69.99 USD by an official source.",
    )

    # 5) Roster Size
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Roster_Size",
        spec_claim_text="The roster contains over 400 playable characters (specifically stated as '400+ Superstars and Legends')",
        answer_value=specs.roster_size_range,
        sources=specs.roster_size_range_sources,
        expected_value=EXPECTED_SPECS["Roster_Size"],
        match_additional="Allow variants like 'over 400', '400+', '400 plus', and phrasing differences such as 'playable characters' vs 'Superstars and Legends'.",
        support_additional="Verify that official source(s) explicitly reference '400+' or equivalent phrasing indicating over 400.",
    )

    # 6) New Match Types Count
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="New_Match_Types_Count",
        spec_claim_text="There are exactly 4 new match types introduced in WWE 2K26",
        answer_value=specs.new_match_types_count,
        sources=specs.new_match_types_count_sources,
        expected_value=EXPECTED_SPECS["New_Match_Types_Count"],
        match_additional="Allow '4' vs 'four'.",
        support_additional="Verify official confirmation of exactly four new match types.",
    )

    # 7) CAW Slots Count
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="CAW_Slots_Count",
        spec_claim_text="The Creation Suite has 200 Create-A-Superstar (CAW) slots",
        answer_value=specs.caw_slots_count,
        sources=specs.caw_slots_count_sources,
        expected_value=EXPECTED_SPECS["CAW_Slots_Count"],
        match_additional="Allow '200' vs '200 slots'.",
        support_additional="Verify an official source states 200 CAW slots.",
    )

    # 8) Image Slots Count
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Image_Slots_Count",
        spec_claim_text="The Creation Suite has 2,000 Create-An-Image slots",
        answer_value=specs.image_slots_count,
        sources=specs.image_slots_count_sources,
        expected_value=EXPECTED_SPECS["Image_Slots_Count"],
        match_additional="Allow numeric formatting variants like '2,000' vs '2000'; focus on the quantity equivalence.",
        support_additional="Verify an official source mentions 2,000 Create-An-Image slots.",
    )

    # 9) MyGM Default Season Length
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="MyGM_Default_Season_Length",
        spec_claim_text="The default MyGM mode season length is 50 weeks",
        answer_value=specs.mygm_default_season_length_weeks,
        sources=specs.mygm_default_season_length_weeks_sources,
        expected_value=EXPECTED_SPECS["MyGM_Default_Season_Length"],
        match_additional="Allow '50 weeks' phrasing variants like 'a 50-week season'.",
        support_additional="Verify official confirmation of a 50-week default season length in MyGM.",
    )

    # 10) MyGM New GMs Count
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="MyGM_New_GMs_Count",
        spec_claim_text="There are 3 new GMs added to MyGM mode",
        answer_value=specs.mygm_new_gms_count,
        sources=specs.mygm_new_gms_count_sources,
        expected_value=EXPECTED_SPECS["MyGM_New_GMs_Count"],
        match_additional="Allow '3' vs 'three'.",
        support_additional="Verify official confirmation that three new GMs are added to MyGM.",
    )

    # 11) Switch 2 Creation Suite Feature
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Switch2_Creation_Suite_Feature",
        spec_claim_text="The Nintendo Switch 2 version has exclusive mouse support for Creation Suite face and body painting",
        answer_value=specs.switch2_creation_suite_feature,
        sources=specs.switch2_creation_suite_feature_sources,
        expected_value=EXPECTED_SPECS["Switch2_Creation_Suite_Feature"],
        match_additional="Allow minor phrasing variations but the core must be 'exclusive mouse support' for face/body painting in Creation Suite.",
        support_additional="Verify an official source states this exclusive Creation Suite feature on Nintendo Switch 2.",
    )

    # 12) Showcase Mode Subject
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Showcase_Mode_Subject",
        spec_claim_text="The Showcase mode is dedicated to CM Punk's career",
        answer_value=specs.showcase_mode_subject,
        sources=specs.showcase_mode_subject_sources,
        expected_value=EXPECTED_SPECS["Showcase_Mode_Subject"],
        match_additional="Allow minor variants like 'CM Punk’s career' vs 'CM Punk'.",
        support_additional="Verify official announcement that Showcase mode focuses on CM Punk's career.",
    )

    # 13) MyRISE Story Title
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="MyRISE_Story_Title",
        spec_claim_text="The MyRISE mode story is titled 'The Comeback'",
        answer_value=specs.myrise_story_title,
        sources=specs.myrise_story_title_sources,
        expected_value=EXPECTED_SPECS["MyRISE_Story_Title"],
        match_additional="Allow quote formatting or casing variants; the title should clearly be 'The Comeback'.",
        support_additional="Verify official confirmation of MyRISE story title 'The Comeback'.",
    )

    # 14) Ringside Pass Seasons Count
    await verify_scalar_spec(
        evaluator=evaluator,
        parent_node=main_node,
        spec_id="Ringside_Pass_Seasons_Count",
        spec_claim_text="There are 6 Ringside Pass Seasons planned throughout the year",
        answer_value=specs.ringside_pass_seasons_count,
        sources=specs.ringside_pass_seasons_count_sources,
        expected_value=EXPECTED_SPECS["Ringside_Pass_Seasons_Count"],
        match_additional="Allow '6' vs 'six'.",
        support_additional="Verify official statement that there are six Ringside Pass Seasons planned during the year.",
    )

    # Return structured evaluation summary
    return evaluator.get_summary()