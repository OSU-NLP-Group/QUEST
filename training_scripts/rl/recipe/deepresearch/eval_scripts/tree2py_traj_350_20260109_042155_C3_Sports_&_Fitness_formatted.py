import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "marathon_world_record"
TASK_DESCRIPTION = """
Identify the current men's marathon world record holder and verify that their record performance meets all official World Athletics eligibility requirements. Your answer must include: (1) The name of the current world record holder and their record time; (2) The specific race (name and location) where the record was set and the date; (3) Verification that the marathon course meets the official World Athletics technical standards, including: the standard marathon distance requirement, the maximum allowable net elevation drop between start and finish, confirmation that the Chicago Marathon course complies with this elevation requirement, the maximum allowable straight-line separation between start and finish points, and confirmation that the Chicago Marathon course complies with this separation requirement; (4) Confirmation that the record has been officially ratified by World Athletics, including the ratification date. For each piece of information, provide supporting reference URLs from authoritative sources such as World Athletics, AIMS, or official race websites.
"""

# Ground truth reference (for matching checks)
EXPECTED_HOLDER_NAME = "Kelvin Kiptum"
EXPECTED_RECORD_TIME = "2:00:35"
EXPECTED_RACE_NAME = "Chicago Marathon (Bank of America Chicago Marathon)"
EXPECTED_RACE_DATE = "October 8, 2023"
EXPECTED_RATIFICATION_DATE = "February 6, 2024"


class HolderInfo(BaseModel):
    name: Optional[str] = None
    time: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class RaceInfo(BaseModel):
    race_name: Optional[str] = None
    location: Optional[str] = None
    date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class StandardsInfo(BaseModel):
    marathon_distance_standard: Optional[str] = None
    marathon_distance_urls: List[str] = Field(default_factory=list)

    measurer_requirement_text: Optional[str] = None
    measurer_requirement_urls: List[str] = Field(default_factory=list)

    elevation_drop_limit_text: Optional[str] = None
    elevation_drop_limit_urls: List[str] = Field(default_factory=list)

    chicago_elevation_compliance_text: Optional[str] = None
    chicago_elevation_compliance_urls: List[str] = Field(default_factory=list)

    start_finish_separation_limit_text: Optional[str] = None
    start_finish_separation_limit_urls: List[str] = Field(default_factory=list)

    chicago_separation_compliance_text: Optional[str] = None
    chicago_separation_compliance_urls: List[str] = Field(default_factory=list)


class RatificationInfo(BaseModel):
    ratified_status: Optional[str] = None
    ratification_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MarathonRecordExtraction(BaseModel):
    holder: Optional[HolderInfo] = None
    race: Optional[RaceInfo] = None
    standards: Optional[StandardsInfo] = None
    ratification: Optional[RatificationInfo] = None


def prompt_extract_all() -> str:
    return """
    Extract the complete set of information the answer provides about the current men's marathon world record and compliance with World Athletics requirements. Return a JSON object with the following structure:

    holder:
      - name: The name of the current men's marathon world record holder, exactly as stated in the answer text.
      - time: The record time exactly as stated (e.g., "2:00:35").
      - urls: All reference URLs cited in the answer that support the holder and time. Only include actual URLs present in the answer.

    race:
      - race_name: The specific race name exactly as stated (e.g., "Chicago Marathon" or "Bank of America Chicago Marathon").
      - location: The location of the race exactly as stated in the answer (e.g., "Chicago, Illinois, USA" or similar).
      - date: The date when the record was set exactly as stated (e.g., "October 8, 2023").
      - urls: All reference URLs cited in the answer that support the race name/location/date details.

    standards:
      - marathon_distance_standard: The value/text the answer states for the official marathon distance standard (e.g., "42.195 kilometers").
      - marathon_distance_urls: All URLs cited for the marathon distance standard.
      - measurer_requirement_text: The text the answer states about approved course measurers (e.g., "AIMS/World Athletics approved 'A' or 'B' grade measurers").
      - measurer_requirement_urls: All URLs cited for measurer requirement.
      - elevation_drop_limit_text: The text the answer states for elevation drop limit (e.g., "not exceed 1 meter per kilometer").
      - elevation_drop_limit_urls: All URLs cited for elevation drop limit standard.
      - chicago_elevation_compliance_text: The text the answer states confirming Chicago Marathon elevation-drop compliance.
      - chicago_elevation_compliance_urls: All URLs cited to support Chicago elevation-drop compliance.
      - start_finish_separation_limit_text: The text the answer states for start-finish separation limit (e.g., "no more than 50% of race distance").
      - start_finish_separation_limit_urls: All URLs cited for separation limit standard.
      - chicago_separation_compliance_text: The text the answer states confirming Chicago Marathon start-finish separation compliance.
      - chicago_separation_compliance_urls: All URLs cited to support Chicago separation compliance.

    ratification:
      - ratified_status: The answer’s statement of ratification status (e.g., "officially ratified by World Athletics").
      - ratification_date: The ratification date exactly as stated (e.g., "February 6, 2024").
      - urls: All URLs cited for ratification status/date.

    Rules:
    - Extract only what is explicitly present in the answer; do not invent or infer any values or URLs.
    - For any missing field, return null. For any missing URL list, return an empty array.
    - Include only valid HTTP/HTTPS URLs. If a URL is missing protocol, prepend http://.
    - Keep strings as they appear in the answer (including formatting).
    """


async def build_record_holder_and_time(
    evaluator: Evaluator,
    parent_node,
    extracted: MarathonRecordExtraction
) -> None:
    group = evaluator.add_parallel(
        id="record_holder_and_time",
        desc="Identify the current world record holder and record time, supported by an authoritative reference URL.",
        parent=parent_node,
        critical=True
    )

    holder = extracted.holder or HolderInfo()

    # Leaf: Holder Name must match Kelvin Kiptum (answer correctness check)
    holder_name_leaf = evaluator.add_leaf(
        id="holder_name",
        desc="Identify Kelvin Kiptum as the current men's marathon world record holder.",
        parent=group,
        critical=True
    )
    claim_holder_match = f"The name '{holder.name or ''}' and '{EXPECTED_HOLDER_NAME}' refer to the same person."
    await evaluator.verify(
        claim=claim_holder_match,
        node=holder_name_leaf,
        additional_instruction="Treat minor spelling or spacing variations and letter case as acceptable. Confirm if they refer to the same person."
    )

    # Leaf: Record time must match 2:00:35 (answer correctness check)
    record_time_leaf = evaluator.add_leaf(
        id="record_time",
        desc="State the current men's marathon world record time as 2:00:35.",
        parent=group,
        critical=True
    )
    claim_time_match = f"The record time stated '{holder.time or ''}' equals '2:00:35' allowing minor formatting differences."
    await evaluator.verify(
        claim=claim_time_match,
        node=record_time_leaf,
        additional_instruction="Allow minor formatting differences (e.g., leading zeros or colon spacing). The intended value must be 2:00:35."
    )

    # Leaf: Reference URL(s) must confirm holder and time (evidence verification)
    holder_time_ref_leaf = evaluator.add_leaf(
        id="record_holder_time_reference_url",
        desc="Provide an authoritative reference URL (e.g., World Athletics) confirming the current record holder and time.",
        parent=group,
        critical=True
    )
    claim_holder_time_supported = f"{EXPECTED_HOLDER_NAME} holds the current men's marathon world record in a time of {EXPECTED_RECORD_TIME}."
    await evaluator.verify(
        claim=claim_holder_time_supported,
        node=holder_time_ref_leaf,
        sources=holder.urls,
        additional_instruction="Use the provided URLs to confirm both the holder name and the record time. Prefer World Athletics, AIMS, or official race websites."
    )


async def build_record_race_details(
    evaluator: Evaluator,
    parent_node,
    extracted: MarathonRecordExtraction
) -> None:
    group = evaluator.add_parallel(
        id="record_race_details",
        desc="Provide the specific race (name and location) and date where the record was set, supported by an authoritative reference URL.",
        parent=parent_node,
        critical=True
    )

    race = extracted.race or RaceInfo()

    # Leaf: Race Name should be Chicago Marathon (Bank of America Chicago Marathon)
    race_name_leaf = evaluator.add_leaf(
        id="race_name",
        desc="State that the record was set at the Chicago Marathon (Bank of America Chicago Marathon).",
        parent=group,
        critical=True
    )
    claim_race_name_match = f"The stated race name '{race.race_name or ''}' refers to the Chicago Marathon (also known as the Bank of America Chicago Marathon)."
    await evaluator.verify(
        claim=claim_race_name_match,
        node=race_name_leaf,
        additional_instruction="Accept synonymous naming such as 'Chicago Marathon' or 'Bank of America Chicago Marathon'."
    )

    # Leaf: Race Location (verify the location stated in the answer using URLs)
    race_location_leaf = evaluator.add_leaf(
        id="race_location",
        desc="Provide the correct location of the race where the record was set (as stated by authoritative sources).",
        parent=group,
        critical=True
    )
    claim_location_supported = f"The race took place in {race.location or ''}."
    await evaluator.verify(
        claim=claim_location_supported,
        node=race_location_leaf,
        sources=race.urls,
        additional_instruction="Verify the location using the provided URLs. Accept reasonable variants like 'Chicago, IL' or 'Chicago, Illinois, USA'."
    )

    # Leaf: Race Date must be October 8, 2023 (answer correctness check)
    race_date_leaf = evaluator.add_leaf(
        id="race_date",
        desc="State that the record was set on October 8, 2023.",
        parent=group,
        critical=True
    )
    claim_date_match = f"The stated record date '{race.date or ''}' equals '{EXPECTED_RACE_DATE}' allowing minor formatting differences."
    await evaluator.verify(
        claim=claim_date_match,
        node=race_date_leaf,
        additional_instruction="Allow minor formatting differences (e.g., '8 October 2023'). The intended date must be October 8, 2023."
    )

    # Leaf: Reference URL(s) must confirm race name/location/date
    race_ref_leaf = evaluator.add_leaf(
        id="race_details_reference_url",
        desc="Provide an authoritative reference URL confirming the race name/location and date for the record performance.",
        parent=group,
        critical=True
    )
    claim_race_supported = "Kelvin Kiptum set the world record at the Chicago Marathon in Chicago, USA on October 8, 2023."
    await evaluator.verify(
        claim=claim_race_supported,
        node=race_ref_leaf,
        sources=race.urls,
        additional_instruction="Use the provided URLs to confirm race name, location, and date together. Prefer official race sites or World Athletics."
    )


async def build_world_athletics_technical_standards(
    evaluator: Evaluator,
    parent_node,
    extracted: MarathonRecordExtraction
) -> None:
    group = evaluator.add_parallel(
        id="world_athletics_technical_standards",
        desc="Verify the stated World Athletics technical standards and Chicago Marathon course eligibility points, each supported by authoritative URLs.",
        parent=parent_node,
        critical=True
    )

    std = extracted.standards or StandardsInfo()

    # Marathon distance standard exact value check (answer correctness) + source support
    distance_leaf = evaluator.add_leaf(
        id="marathon_distance_standard",
        desc="State the official marathon distance standard as 42.195 kilometers.",
        parent=group,
        critical=True
    )
    claim_distance_text_match = f"The stated marathon distance '{std.marathon_distance_standard or ''}' equals '42.195 kilometers' allowing minor formatting differences."
    await evaluator.verify(
        claim=claim_distance_text_match,
        node=distance_leaf,
        additional_instruction="Accept variants like '42.195 km'. The intended value is 42.195 kilometers."
    )

    distance_ref_leaf = evaluator.add_leaf(
        id="marathon_distance_reference_url",
        desc="Provide an authoritative reference URL documenting the official marathon distance standard.",
        parent=group,
        critical=True
    )
    claim_distance_supported = "The official marathon distance is 42.195 kilometers."
    await evaluator.verify(
        claim=claim_distance_supported,
        node=distance_ref_leaf,
        sources=std.marathon_distance_urls,
        additional_instruction="Use authoritative rules resources (World Athletics, AIMS) to confirm 42.195 km."
    )

    # Approved course measurement (A/B grade measurers)
    measurer_leaf = evaluator.add_leaf(
        id="approved_course_measurement",
        desc="Confirm the course must be measured by AIMS/World Athletics approved 'A' or 'B' grade measurers (as a stated eligibility/standard requirement).",
        parent=group,
        critical=True
    )
    claim_measurer_text_match = f"The statement '{std.measurer_requirement_text or ''}' asserts that approved AIMS/World Athletics Grade A or Grade B measurers are required."
    await evaluator.verify(
        claim=claim_measurer_text_match,
        node=measurer_leaf,
        additional_instruction="Focus on whether the text asserts the A/B grade measurer requirement for record eligibility."
    )

    measurer_ref_leaf = evaluator.add_leaf(
        id="approved_course_measurement_reference_url",
        desc="Provide an authoritative reference URL documenting the A/B grade measurer requirement for record-eligible courses.",
        parent=group,
        critical=True
    )
    claim_measurer_supported = "World Athletics/AIMS require courses to be measured by approved Grade A or Grade B measurers for record eligibility."
    await evaluator.verify(
        claim=claim_measurer_supported,
        node=measurer_ref_leaf,
        sources=std.measurer_requirement_urls,
        additional_instruction="Use rules or measurement handbooks from World Athletics/AIMS."
    )

    # Elevation drop limit standard
    elevation_limit_leaf = evaluator.add_leaf(
        id="elevation_drop_limit",
        desc="State that net elevation drop must not exceed 1 meter per kilometer (42.195 meters total for a marathon).",
        parent=group,
        critical=True
    )
    claim_elev_text_match = f"The statement '{std.elevation_drop_limit_text or ''}' asserts that the net elevation drop is limited to 1 meter per kilometer (≤ 42.195 m for a marathon)."
    await evaluator.verify(
        claim=claim_elev_text_match,
        node=elevation_limit_leaf,
        additional_instruction="Accept equivalent phrasings such as 'no more than 1 m/km' or '≤ 42.195 m total'."
    )

    elevation_ref_leaf = evaluator.add_leaf(
        id="elevation_drop_limit_reference_url",
        desc="Provide an authoritative reference URL documenting the elevation drop limit requirement.",
        parent=group,
        critical=True
    )
    claim_elev_supported = "For marathon records, the allowable net elevation drop must not exceed 1 meter per kilometer."
    await evaluator.verify(
        claim=claim_elev_supported,
        node=elevation_ref_leaf,
        sources=std.elevation_drop_limit_urls,
        additional_instruction="Use authoritative rules resources from World Athletics or AIMS."
    )

    # Chicago elevation compliance
    chicago_elev_leaf = evaluator.add_leaf(
        id="chicago_elevation_compliance",
        desc="Confirm that the Chicago Marathon course meets the elevation drop requirement for world record eligibility.",
        parent=group,
        critical=True
    )
    claim_chicago_elev_supported = "The Chicago Marathon course net elevation drop meets the required ≤ 1 m per km limit."
    await evaluator.verify(
        claim=claim_chicago_elev_supported,
        node=chicago_elev_leaf,
        sources=std.chicago_elevation_compliance_urls,
        additional_instruction="Use official course certification, measurement reports, or authoritative race resources to confirm compliance."
    )

    chicago_elev_ref_leaf = evaluator.add_leaf(
        id="chicago_elevation_compliance_reference_url",
        desc="Provide an authoritative reference URL supporting Chicago Marathon elevation-drop compliance.",
        parent=group,
        critical=True
    )
    claim_chicago_elev_ref_supported = "Authoritative resources confirm that the Chicago Marathon satisfies the elevation drop requirement."
    await evaluator.verify(
        claim=claim_chicago_elev_ref_supported,
        node=chicago_elev_ref_leaf,
        sources=std.chicago_elevation_compliance_urls,
        additional_instruction="The provided URLs must substantively confirm elevation compliance."
    )

    # Start-finish separation limit
    separation_limit_leaf = evaluator.add_leaf(
        id="start_finish_separation_limit",
        desc="State that the straight-line start-finish separation must be no more than 50% of race distance (21.0975 km for a marathon).",
        parent=group,
        critical=True
    )
    claim_sep_text_match = f"The statement '{std.start_finish_separation_limit_text or ''}' asserts that start-finish straight-line separation is limited to ≤ 50% of race distance (≤ 21.0975 km for a marathon)."
    await evaluator.verify(
        claim=claim_sep_text_match,
        node=separation_limit_leaf,
        additional_instruction="Accept equivalent numerical phrasing. The intended standard is ≤ 50% of race distance."
    )

    separation_ref_leaf = evaluator.add_leaf(
        id="start_finish_separation_limit_reference_url",
        desc="Provide an authoritative reference URL documenting the start-finish separation requirement.",
        parent=group,
        critical=True
    )
    claim_sep_supported = "World Athletics imposes a start-finish straight-line separation limit of ≤ 50% of race distance."
    await evaluator.verify(
        claim=claim_sep_supported,
        node=separation_ref_leaf,
        sources=std.start_finish_separation_limit_urls,
        additional_instruction="Use World Athletics/AIMS rules documentation."
    )

    # Chicago separation compliance
    chicago_sep_leaf = evaluator.add_leaf(
        id="chicago_separation_compliance",
        desc="Confirm that the Chicago Marathon course meets the start-finish separation requirement for world record eligibility.",
        parent=group,
        critical=True
    )
    claim_chicago_sep_supported = "The Chicago Marathon course meets the start-finish separation limit (≤ 50% of race distance)."
    await evaluator.verify(
        claim=claim_chicago_sep_supported,
        node=chicago_sep_leaf,
        sources=std.chicago_separation_compliance_urls,
        additional_instruction="Use authoritative sources (course map, certification, official race info) to confirm compliance."
    )

    chicago_sep_ref_leaf = evaluator.add_leaf(
        id="chicago_separation_compliance_reference_url",
        desc="Provide an authoritative reference URL supporting Chicago Marathon start-finish separation compliance.",
        parent=group,
        critical=True
    )
    claim_chicago_sep_ref_supported = "Authoritative resources confirm that the Chicago Marathon satisfies the start-finish separation requirement."
    await evaluator.verify(
        claim=claim_chicago_sep_ref_supported,
        node=chicago_sep_ref_leaf,
        sources=std.chicago_separation_compliance_urls,
        additional_instruction="The provided URLs must substantively confirm separation compliance."
    )


async def build_official_ratification(
    evaluator: Evaluator,
    parent_node,
    extracted: MarathonRecordExtraction
) -> None:
    group = evaluator.add_parallel(
        id="official_ratification",
        desc="Confirm that the record has been officially ratified by World Athletics, including the ratification date, supported by an authoritative reference URL.",
        parent=parent_node,
        critical=True
    )

    rat = extracted.ratification or RatificationInfo()

    # Ratification status verification
    rat_status_leaf = evaluator.add_leaf(
        id="ratification_status",
        desc="Confirm that the men's marathon world record performance has been officially ratified by World Athletics.",
        parent=group,
        critical=True
    )
    claim_ratified_supported = "Kelvin Kiptum's men's marathon world record has been officially ratified by World Athletics."
    await evaluator.verify(
        claim=claim_ratified_supported,
        node=rat_status_leaf,
        sources=rat.urls,
        additional_instruction="Use authoritative World Athletics publications or official communications."
    )

    # Ratification date check (answer correctness) + source support via separate leaf
    rat_date_leaf = evaluator.add_leaf(
        id="ratification_date",
        desc="State the World Athletics ratification date as February 6, 2024.",
        parent=group,
        critical=True
    )
    claim_rat_date_match = f"The stated ratification date '{rat.ratification_date or ''}' equals '{EXPECTED_RATIFICATION_DATE}' allowing minor formatting differences."
    await evaluator.verify(
        claim=claim_rat_date_match,
        node=rat_date_leaf,
        additional_instruction="Accept minor formatting differences (e.g., '6 February 2024'). The intended date must be February 6, 2024."
    )

    rat_ref_leaf = evaluator.add_leaf(
        id="ratification_reference_url",
        desc="Provide an authoritative reference URL confirming official ratification and the ratification date.",
        parent=group,
        critical=True
    )
    claim_rat_ref_supported = f"World Athletics ratified the record on {EXPECTED_RATIFICATION_DATE}."
    await evaluator.verify(
        claim=claim_rat_ref_supported,
        node=rat_ref_leaf,
        sources=rat.urls,
        additional_instruction="Use the provided URLs to confirm both ratification status and the ratification date."
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=MarathonRecordExtraction,
        extraction_name="marathon_record_extraction"
    )

    evaluator.add_ground_truth({
        "expected_holder": EXPECTED_HOLDER_NAME,
        "expected_record_time": EXPECTED_RECORD_TIME,
        "expected_race_name": EXPECTED_RACE_NAME,
        "expected_race_date": EXPECTED_RACE_DATE,
        "expected_ratification_date": EXPECTED_RATIFICATION_DATE
    }, gt_type="ground_truth")

    world_node = evaluator.add_parallel(
        id="world_record_verification",
        desc="Verify that the current men's marathon world record holder and the record performance meet all stated World Athletics eligibility/verification requirements, with authoritative reference URLs for each required piece of information.",
        parent=root,
        critical=True
    )

    await build_record_holder_and_time(evaluator, world_node, extracted)
    await build_record_race_details(evaluator, world_node, extracted)
    await build_world_athletics_technical_standards(evaluator, world_node, extracted)
    await build_official_ratification(evaluator, world_node, extracted)

    return evaluator.get_summary()