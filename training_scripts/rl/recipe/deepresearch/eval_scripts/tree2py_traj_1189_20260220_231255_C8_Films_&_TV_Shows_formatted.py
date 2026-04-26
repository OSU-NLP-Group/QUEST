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
TASK_ID = "tv_series_2024_2025"
TASK_DESCRIPTION = """
Identify three distinct TV series that premiered between January 1, 2024, and December 31, 2025, meeting the following specifications:

Series 1: A medical drama series that premiered on the Max streaming platform in January 2025 and stars an actor who previously had a lead role in another medical drama series. Provide the series title, premiere date, lead actor's name, and the previous medical drama series in which they starred.

Series 2: A comedy series that received a nomination for Outstanding Comedy Series at the 76th Emmy Awards (2024 ceremony) and whose creator also stars in the series. Provide the series title, the creator/star's name, the production company, and the broadcast network or streaming platform.

Series 3: A comedy series that set the record for the most Emmy nominations in the comedy category in a single year at the 76th Emmy Awards (2024 ceremony), is available via FX on Hulu, and stars an actor who previously had a main role in the Showtime series Shameless (2011-2021). Provide the series title, the specific number of Emmy nominations received, the lead actor's name, and the production company.

For each series, include reference URLs that verify the information provided.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Series1Info(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    lead_actor_name: Optional[str] = None
    prior_medical_drama_series: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Series2Info(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    creator_star_name: Optional[str] = None
    production_company: Optional[str] = None
    network_or_streaming_platform: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Series3Info(BaseModel):
    title: Optional[str] = None
    premiere_date: Optional[str] = None
    emmy_nominations_number: Optional[str] = None
    lead_actor_name: Optional[str] = None
    production_company: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_series1() -> str:
    return """
    Extract the information for Series 1 from the answer. Series 1 must be a medical drama that premiered on Max in January 2025, with a lead actor who previously had a lead role in another medical drama.

    Return a JSON object with the following fields:
    - title: The series title as written in the answer.
    - premiere_date: The premiere date stated in the answer (string, any format).
    - lead_actor_name: The lead actor's name.
    - prior_medical_drama_series: The name of the previous medical drama series in which the lead actor previously had a lead role.
    - sources: An array of URLs explicitly present in the answer that verify the above facts.

    If any field is missing, set it to null. For sources, return an empty array if none are provided.
    """


def prompt_extract_series2() -> str:
    return """
    Extract the information for Series 2 from the answer. Series 2 must be a comedy that received a nomination for Outstanding Comedy Series at the 76th Emmy Awards (2024 ceremony), and the creator also stars in the series.

    Return a JSON object with the following fields:
    - title: The series title as written in the answer.
    - premiere_date: The premiere date stated in the answer (string, any format).
    - creator_star_name: The name of the creator who also stars.
    - production_company: The production company name.
    - network_or_streaming_platform: The broadcast network or streaming platform name.
    - sources: An array of URLs explicitly present in the answer that verify the above facts.

    If any field is missing, set it to null. For sources, return an empty array if none are provided.
    """


def prompt_extract_series3() -> str:
    return """
    Extract the information for Series 3 from the answer. Series 3 must be a comedy that: (a) set the record for most Emmy nominations in the comedy category in a single year at the 76th Emmy Awards (2024 ceremony), (b) is available via FX on Hulu, and (c) stars an actor who previously had a main role in Showtime’s Shameless (2011–2021).

    Return a JSON object with the following fields:
    - title: The series title as written in the answer.
    - premiere_date: The premiere date stated in the answer (string, any format).
    - emmy_nominations_number: The specific number of Emmy nominations it received (string).
    - lead_actor_name: The lead actor's name.
    - production_company: The production company name.
    - sources: An array of URLs explicitly present in the answer that verify the above facts.

    If any field is missing, set it to null. For sources, return an empty array if none are provided.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _normalize_title(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_series_1(evaluator: Evaluator, parent_node, s1: Series1Info) -> None:
    series_node = evaluator.add_parallel(
        id="series_1",
        desc="Series 1 satisfies the Max/January 2025 medical drama + lead actor prior lead role in another medical drama requirements and provides required fields with sources",
        parent=parent_node,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_non_empty(s1.title),
        id="series_1_title",
        desc="Series 1 title is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s1.premiere_date),
        id="series_1_premiere_date_provided",
        desc="Series 1 premiere date is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s1.lead_actor_name),
        id="series_1_lead_actor_name",
        desc="Series 1 lead actor's name is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s1.prior_medical_drama_series),
        id="series_1_prior_medical_drama_lead_role",  # Will verify the claim below
        desc="A prior medical drama series is named in which the Series 1 lead actor previously had a lead role",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(s1.sources),
        id="series_1_reference_urls",
        desc="Reference URL(s) are provided that verify the Series 1 claims (title, Max premiere, premiere date, medical-drama classification, lead actor, and prior medical-drama lead role)",
        parent=series_node,
        critical=True
    )

    # Verifications grounded by sources
    # Medical drama classification
    node_genre = evaluator.add_leaf(
        id="series_1_genre_medical_drama",
        desc="Series 1 is a medical drama",
        parent=series_node,
        critical=True
    )
    claim_genre = f"The series '{s1.title}' is a medical drama series."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=s1.sources,
        additional_instruction="Use the cited sources to confirm genre classification as a medical drama. Allow minor variations in phrasing like 'medical series' or 'hospital drama'."
    )

    # Platform Max premiere
    node_platform = evaluator.add_leaf(
        id="series_1_platform_max_premiere",
        desc="Series 1 premiered on the Max streaming platform",
        parent=series_node,
        critical=True
    )
    claim_platform = f"The series '{s1.title}' premiered on the Max streaming platform."
    await evaluator.verify(
        claim=claim_platform,
        node=node_platform,
        sources=s1.sources,
        additional_instruction="Confirm that the initial release/premiere was on Max (Warner Bros. Discovery). Accept 'Max' branding even if phrased as 'streaming on Max'."
    )

    # Premiere month/year check
    node_jan25 = evaluator.add_leaf(
        id="series_1_premiere_in_january_2025",
        desc="Series 1 premiere date is in January 2025",
        parent=series_node,
        critical=True
    )
    claim_jan25 = f"The series '{s1.title}' premiered in January 2025."
    await evaluator.verify(
        claim=claim_jan25,
        node=node_jan25,
        sources=s1.sources,
        additional_instruction="Verify the premiere date falls within January 2025. Accept typical date formats and regional variants."
    )

    # Lead actor prior lead role in another medical drama
    node_prior_lead = evaluator.add_leaf(
        id="series_1_prior_medical_drama_lead_role_verified",
        desc="Series 1 lead actor previously had a lead role in another medical drama series (as named)",
        parent=series_node,
        critical=True
    )
    claim_prior_lead = f"The lead actor {s1.lead_actor_name} previously had a lead role in the medical drama series '{s1.prior_medical_drama_series}'."
    await evaluator.verify(
        claim=claim_prior_lead,
        node=node_prior_lead,
        sources=s1.sources,
        additional_instruction="Confirm that the named actor held a lead role (main starring role) in the specified medical drama series."
    )


async def verify_series_2(evaluator: Evaluator, parent_node, s2: Series2Info) -> None:
    series_node = evaluator.add_parallel(
        id="series_2",
        desc="Series 2 satisfies the Emmy nomination + comedy + creator-stars requirements and provides required fields with sources",
        parent=parent_node,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_non_empty(s2.title),
        id="series_2_title",
        desc="Series 2 title is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s2.premiere_date),
        id="series_2_premiere_date_provided",
        desc="Series 2 premiere date is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s2.creator_star_name),
        id="series_2_creator_star_name",
        desc="Series 2 creator/star's name is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s2.production_company),
        id="series_2_production_company",
        desc="Series 2 production company is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s2.network_or_streaming_platform),
        id="series_2_network_or_streaming_platform",
        desc="Series 2 broadcast network or streaming platform is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(s2.sources),
        id="series_2_reference_urls",
        desc="Reference URL(s) are provided that verify the Series 2 claims (premiere-window compliance, comedy classification, Emmy nomination, creator/star identity, production company, and network/platform)",
        parent=series_node,
        critical=True
    )

    # Verifications grounded by sources
    # Premiere within 2024-2025 window
    node_window = evaluator.add_leaf(
        id="series_2_premiere_within_2024_2025_window",
        desc="Series 2 premiered between January 1, 2024 and December 31, 2025 (with verifiable evidence)",
        parent=series_node,
        critical=True
    )
    claim_window = f"The series '{s2.title}' premiered between January 1, 2024, and December 31, 2025."
    await evaluator.verify(
        claim=claim_window,
        node=node_window,
        sources=s2.sources,
        additional_instruction="Confirm the date of the first public release or broadcast is within the specified window."
    )

    # Genre comedy
    node_genre = evaluator.add_leaf(
        id="series_2_genre_comedy",
        desc="Series 2 is a comedy series",
        parent=series_node,
        critical=True
    )
    claim_genre = f"The series '{s2.title}' is a comedy series."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=s2.sources,
        additional_instruction="Verify that the series is categorized or described as a comedy."
    )

    # Emmy nomination Outstanding Comedy Series at 76th
    node_emmy_nom = evaluator.add_leaf(
        id="series_2_emmy_nomination_outstanding_comedy_76th",
        desc="Series 2 received a nomination for Outstanding Comedy Series at the 76th Emmy Awards (2024 ceremony)",
        parent=series_node,
        critical=True
    )
    claim_emmy_nom = f"The series '{s2.title}' received a nomination for Outstanding Comedy Series at the 76th Emmy Awards (2024 ceremony)."
    await evaluator.verify(
        claim=claim_emmy_nom,
        node=node_emmy_nom,
        sources=s2.sources,
        additional_instruction="Confirm the official nomination listing or credible coverage references the Outstanding Comedy Series nomination for the 76th Emmys."
    )

    # Creator also stars
    node_creator_stars = evaluator.add_leaf(
        id="series_2_creator_also_stars_confirmed",
        desc="It is verified that the Series 2 creator also stars in the series",
        parent=series_node,
        critical=True
    )
    claim_creator_stars = f"The series '{s2.title}' is created by {s2.creator_star_name}, who also stars in the series."
    await evaluator.verify(
        claim=claim_creator_stars,
        node=node_creator_stars,
        sources=s2.sources,
        additional_instruction="Verify both the creator credit and on-screen starring role for the same person."
    )


async def verify_series_3(evaluator: Evaluator, parent_node, s3: Series3Info) -> None:
    series_node = evaluator.add_parallel(
        id="series_3",
        desc="Series 3 satisfies the Emmy nominations record + FX on Hulu + Shameless actor requirements and provides required fields with sources",
        parent=parent_node,
        critical=False
    )

    # Existence checks
    evaluator.add_custom_node(
        result=_non_empty(s3.title),
        id="series_3_title",
        desc="Series 3 title is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s3.premiere_date),
        id="series_3_premiere_date_provided",
        desc="Series 3 premiere date is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s3.emmy_nominations_number),
        id="series_3_emmy_nominations_number",
        desc="The specific number of Emmy nominations received (as referenced in the record claim) is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s3.lead_actor_name),
        id="series_3_lead_actor_name",
        desc="Series 3 lead actor's name is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(s3.production_company),
        id="series_3_production_company",
        desc="Series 3 production company is provided",
        parent=series_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(s3.sources),
        id="series_3_reference_urls",
        desc="Reference URL(s) are provided that verify the Series 3 claims (premiere-window compliance, comedy classification, FX on Hulu availability, Emmy record claim, nomination count, lead actor, Shameless main-role claim, and production company)",
        parent=series_node,
        critical=True
    )

    # Verifications grounded by sources
    # Premiere within 2024-2025 window
    node_window = evaluator.add_leaf(
        id="series_3_premiere_within_2024_2025_window",
        desc="Series 3 premiered between January 1, 2024 and December 31, 2025 (with verifiable evidence)",
        parent=series_node,
        critical=True
    )
    claim_window = f"The series '{s3.title}' premiered between January 1, 2024, and December 31, 2025."
    await evaluator.verify(
        claim=claim_window,
        node=node_window,
        sources=s3.sources,
        additional_instruction="Confirm the date of the first public release or broadcast is within the specified window."
    )

    # Genre comedy
    node_genre = evaluator.add_leaf(
        id="series_3_genre_comedy",
        desc="Series 3 is a comedy series",
        parent=series_node,
        critical=True
    )
    claim_genre = f"The series '{s3.title}' is a comedy series."
    await evaluator.verify(
        claim=claim_genre,
        node=node_genre,
        sources=s3.sources,
        additional_instruction="Verify that the series is categorized or described as a comedy."
    )

    # FX on Hulu availability
    node_fx_hulu = evaluator.add_leaf(
        id="series_3_platform_fx_on_hulu",
        desc="Series 3 is available via FX on Hulu",
        parent=series_node,
        critical=True
    )
    claim_fx_hulu = f"The series '{s3.title}' is available via FX on Hulu."
    await evaluator.verify(
        claim=claim_fx_hulu,
        node=node_fx_hulu,
        sources=s3.sources,
        additional_instruction="Confirm distribution/availability branding as 'FX on Hulu'. Accept phrasing like 'streaming on Hulu under FX on Hulu'."
    )

    # Emmy record claim for most comedy nominations at the 76th Emmys
    node_emmy_record = evaluator.add_leaf(
        id="series_3_emmy_record_most_comedy_noms_76th",
        desc="Series 3 set the record for most Emmy nominations in the comedy category in a single year at the 76th Emmy Awards (2024 ceremony)",
        parent=series_node,
        critical=True
    )
    claim_emmy_record = f"The series '{s3.title}' set the record for the most Emmy nominations in the comedy category in a single year at the 76th Emmy Awards (2024 ceremony)."
    await evaluator.verify(
        claim=claim_emmy_record,
        node=node_emmy_record,
        sources=s3.sources,
        additional_instruction="Verify credible sources state this record for the 76th Emmys (2024)."
    )

    # Specific nominations number
    node_emmy_num_verify = evaluator.add_leaf(
        id="series_3_emmy_nominations_number_verified",
        desc="The specific number of Emmy nominations received is accurate",
        parent=series_node,
        critical=True
    )
    claim_emmy_num = f"The series '{s3.title}' received {s3.emmy_nominations_number} Emmy nominations."
    await evaluator.verify(
        claim=claim_emmy_num,
        node=node_emmy_num_verify,
        sources=s3.sources,
        additional_instruction="Confirm the exact nomination count as stated."
    )

    # Lead actor previously had a main role in Shameless (2011–2021)
    node_shameless = evaluator.add_leaf(
        id="series_3_lead_actor_main_role_in_shameless",
        desc="It is verified that the Series 3 lead actor previously had a main role in Shameless (2011–2021)",
        parent=series_node,
        critical=True
    )
    claim_shameless = f"The lead actor {s3.lead_actor_name} previously had a main role in the Showtime series 'Shameless' (2011–2021)."
    await evaluator.verify(
        claim=claim_shameless,
        node=node_shameless,
        sources=s3.sources,
        additional_instruction="Confirm that the named actor is listed among the main cast for Shameless during its 2011–2021 run."
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
    Evaluate an answer for the TV series identification task (2024–2025).
    """
    # Initialize evaluator (root is non-critical by design to allow partial scoring; critical children gate scoring)
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

    # Extract structured info for the three series
    s1 = await evaluator.extract(
        prompt=prompt_extract_series1(),
        template_class=Series1Info,
        extraction_name="series_1_extraction"
    )
    s2 = await evaluator.extract(
        prompt=prompt_extract_series2(),
        template_class=Series2Info,
        extraction_name="series_2_extraction"
    )
    s3 = await evaluator.extract(
        prompt=prompt_extract_series3(),
        template_class=Series3Info,
        extraction_name="series_3_extraction"
    )

    # Root-level critical checks
    # Exactly three series entries provided (we require all three titles present)
    exactly_three = _non_empty(s1.title) and _non_empty(s2.title) and _non_empty(s3.title)
    evaluator.add_custom_node(
        result=exactly_three,
        id="three_series_provided",
        desc="Exactly three series entries (Series 1, Series 2, Series 3) are provided",
        parent=root,
        critical=True
    )

    # Distinct series titles across the three
    titles = [_normalize_title(s1.title), _normalize_title(s2.title), _normalize_title(s3.title)]
    distinct = len(set([t for t in titles if t])) == 3
    evaluator.add_custom_node(
        result=distinct,
        id="series_titles_distinct",
        desc="The three series are distinct (no duplicate series titles across Series 1–3)",
        parent=root,
        critical=True
    )

    # Per-series verification subtrees
    await verify_series_1(evaluator, root, s1)
    await verify_series_2(evaluator, root, s2)
    await verify_series_3(evaluator, root, s3)

    # Return final summary
    return evaluator.get_summary()