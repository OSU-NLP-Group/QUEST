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
TASK_ID = "neuro_conf_2026"
TASK_DESCRIPTION = (
    "I am a neuroscience graduate student planning to present my research at a conference in 2026. "
    "I need to find a neuroscience conference in North America (United States or Canada) that accepts poster presentations. "
    "Please provide the following information:\n\n"
    "1. The full official name of the conference\n"
    "2. The specific city and state/province where it will be held\n"
    "3. The exact dates of the conference in 2026\n"
    "4. The maximum word count allowed for poster abstracts\n"
    "5. The required or maximum poster dimensions (in inches or centimeters)\n"
    "6. The conference's policy regarding citation of preprints (such as bioRxiv preprints) in abstracts or posters\n"
    "7. A valid URL to the conference website or official announcement page\n\n"
    "Please ensure all information is from official conference sources."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceInfo(BaseModel):
    """Model for the extracted conference information."""
    conference_name: Optional[str] = None
    location_city: Optional[str] = None
    location_state_or_province: Optional[str] = None
    location_country: Optional[str] = None  # optional; allow null
    conference_dates_2026: Optional[str] = None
    abstract_word_or_char_limit: Optional[str] = None  # keep as string (e.g., '250 words', '1500 characters')
    poster_dimensions: Optional[str] = None  # keep as string (e.g., '36 x 48 inches', '90 cm x 120 cm')
    preprint_citation_policy: Optional[str] = None  # string description if provided
    preprint_policy_not_available_noted: Optional[bool] = None  # True if answer explicitly notes no policy on official sources
    official_urls: List[str] = Field(default_factory=list)  # list of official conference URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conference_info() -> str:
    return (
        "Extract the details for a single neuroscience conference mentioned in the answer. "
        "Return the following fields:\n"
        "1. conference_name: The full official name of the conference.\n"
        "2. location_city: The specific city.\n"
        "3. location_state_or_province: The specific state or province.\n"
        "4. location_country: The country if explicitly provided (United States or Canada); if not provided, return null.\n"
        "5. conference_dates_2026: The exact dates of the conference (as written in the answer, e.g., 'June 10–14, 2026').\n"
        "6. abstract_word_or_char_limit: The maximum abstract length for posters as written (e.g., '250 words' or '1500 characters').\n"
        "7. poster_dimensions: The required or maximum poster dimensions (e.g., '36 x 48 inches' or '90 cm x 120 cm').\n"
        "8. preprint_citation_policy: The policy on citing preprints in abstracts/posters if explicitly stated by official sources in the answer.\n"
        "9. preprint_policy_not_available_noted: Boolean (true/false). True if the answer explicitly notes that official conference sources do not state any policy regarding preprint citations; false otherwise.\n"
        "10. official_urls: All official conference URLs explicitly provided in the answer (website, call for abstracts page, official announcement). "
        "Include only URLs mentioned in the answer; do not infer.\n\n"
        "If any required field is missing from the answer, return null for that field. For URLs, return an empty list if none are provided."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip() != "")


def _urls_available(urls: List[str]) -> bool:
    return bool(urls and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls))


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, info: ConferenceInfo) -> None:
    """
    Build the verification tree for the neuroscience conference task and run all checks.
    """

    # Top-level task node (non-critical to allow nuanced scoring)
    task_node = evaluator.add_parallel(
        id="Conference_Task",
        desc="Find an eligible 2026 neuroscience conference in the US/Canada that accepts posters and provide the required details from official sources",
        parent=evaluator.root,
        critical=False
    )

    # --------------------------- Sources & URL existence ------------------------
    sources_node = evaluator.add_parallel(
        id="Sources_and_Verifiability",
        desc="Provide official source URL(s) and ensure claims are supported by them",
        parent=task_node,
        critical=True
    )

    official_url_exists_node = evaluator.add_custom_node(
        result=_urls_available(info.official_urls),
        id="Official_Conference_URL",
        desc="A valid URL to the official conference website or official announcement page is provided",
        parent=sources_node,
        critical=True
    )

    # --------------------------- Conference eligibility -------------------------
    eligibility_node = evaluator.add_parallel(
        id="Conference_Eligibility",
        desc="The selected conference meets the domain and submission-type constraints",
        parent=task_node,
        critical=True
    )

    # Neuroscience domain check (from official sources)
    neuro_leaf = evaluator.add_leaf(
        id="Neuroscience_Conference",
        desc="The conference is explicitly identified as a neuroscience conference/meeting by an official source",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="This official page explicitly identifies the event as a neuroscience conference or meeting.",
        node=neuro_leaf,
        sources=info.official_urls,
        additional_instruction=(
            "Look for explicit cues such as 'neuroscience', 'Society for Neuroscience', "
            "'cognitive neuroscience', 'neural', 'brain research', or similar. "
            "General medical conferences are NOT sufficient unless they explicitly have a neuroscience focus."
        ),
        extra_prerequisites=[official_url_exists_node]
    )

    # Accepts poster presentations (from official sources)
    posters_leaf = evaluator.add_leaf(
        id="Accepts_Poster_Presentations",
        desc="An official source indicates the conference accepts poster presentations (e.g., call for posters/poster session/poster submissions)",
        parent=eligibility_node,
        critical=True
    )
    await evaluator.verify(
        claim="This official page indicates that the conference accepts poster presentations (e.g., poster submissions, poster sessions, call for posters).",
        node=posters_leaf,
        sources=info.official_urls,
        additional_instruction=(
            "Identify terms such as 'posters', 'poster session', 'call for posters', 'poster submissions', "
            "'abstract submission for posters', or similar language confirming poster acceptance."
        ),
        extra_prerequisites=[official_url_exists_node]
    )

    # --------------------------- Conference basic info --------------------------
    basic_info_node = evaluator.add_parallel(
        id="Conference_Basic_Info",
        desc="Provide the required identifying information for the conference",
        parent=task_node,
        critical=True
    )

    # Conference name provided
    name_exists_node = evaluator.add_custom_node(
        result=_has_text(info.conference_name),
        id="Conference_Name",
        desc="The full official name of the conference is provided",
        parent=basic_info_node,
        critical=True
    )

    # Conference Location: city + state/province provided AND in US/Canada
    location_group_node = evaluator.add_sequential(
        id="Conference_Location_US_or_Canada",
        desc="The specific city and state/province are provided, and the location is in the United States or Canada",
        parent=basic_info_node,
        critical=True
    )

    location_provided_node = evaluator.add_custom_node(
        result=_has_text(info.location_city) and _has_text(info.location_state_or_province),
        id="location_provided",
        desc="Location is provided (city and state/province are present)",
        parent=location_group_node,
        critical=True
    )

    location_in_na_leaf = evaluator.add_leaf(
        id="location_in_us_or_canada",
        desc="The provided location is in the United States or Canada",
        parent=location_group_node,
        critical=True
    )
    loc_city = info.location_city or ""
    loc_region = info.location_state_or_province or ""
    await evaluator.verify(
        claim=f"The location '{loc_city}, {loc_region}' is in the United States or Canada.",
        node=location_in_na_leaf,
        additional_instruction=(
            "Use general geographic knowledge: US states (e.g., CA, NY, TX, etc.) and Canadian provinces/territories "
            "(e.g., ON, BC, QC, AB, etc.). Minor formatting/casing differences are acceptable."
        ),
    )

    # Conference Dates 2026: provided and occur in 2026
    dates_group_node = evaluator.add_sequential(
        id="Conference_Dates_2026",
        desc="The exact conference dates are provided and they occur in 2026",
        parent=basic_info_node,
        critical=True
    )

    dates_provided_node = evaluator.add_custom_node(
        result=_has_text(info.conference_dates_2026),
        id="dates_provided",
        desc="Conference dates are provided",
        parent=dates_group_node,
        critical=True
    )

    dates_in_2026_leaf = evaluator.add_leaf(
        id="dates_in_2026",
        desc="Conference dates occur in the year 2026",
        parent=dates_group_node,
        critical=True
    )
    dates_str = info.conference_dates_2026 or ""
    await evaluator.verify(
        claim=f"The provided dates '{dates_str}' occur in the year 2026.",
        node=dates_in_2026_leaf,
        additional_instruction=(
            "Allow ranges and various formats (e.g., 'June 10–14, 2026', '10-14 June 2026'). "
            "If multiple date parts are included, ensure the year is 2026."
        ),
    )

    # ----------------------- Poster & Abstract Requirements ---------------------
    # Parent set to non-critical to allow partial credit and a non-critical preprint policy
    reqs_node = evaluator.add_parallel(
        id="Poster_and_Abstract_Requirements",
        desc="Provide required poster/abstract requirement details",
        parent=task_node,
        critical=False
    )

    # Abstract limit: provided and supported by official sources
    abstract_group_node = evaluator.add_sequential(
        id="Abstract_Word_or_Character_Limit",
        desc="The maximum abstract length for poster submissions is specified (word count and/or character limit, as given by the conference)",
        parent=reqs_node,
        critical=True  # Essential requirement within this category
    )

    abstract_exists_node = evaluator.add_custom_node(
        result=_has_text(info.abstract_word_or_char_limit),
        id="abstract_limit_provided",
        desc="Abstract length limit is provided",
        parent=abstract_group_node,
        critical=True
    )

    abstract_supported_leaf = evaluator.add_leaf(
        id="abstract_limit_supported_by_sources",
        desc="The official source states the abstract length limit as provided",
        parent=abstract_group_node,
        critical=True
    )
    abstract_limit_str = info.abstract_word_or_char_limit or ""
    await evaluator.verify(
        claim=f"The official page states the maximum abstract length is '{abstract_limit_str}'.",
        node=abstract_supported_leaf,
        sources=info.official_urls,
        additional_instruction=(
            "Match word or character limits as written (e.g., '250 words', '1500 characters'). "
            "Allow minor punctuation variations. If multiple limits are shown, ensure the poster abstract limit matches the provided value."
        ),
        extra_prerequisites=[official_url_exists_node, abstract_exists_node]
    )

    # Poster dimensions: provided and supported by official sources
    poster_group_node = evaluator.add_sequential(
        id="Poster_Dimensions",
        desc="The required or maximum poster dimensions are specified (width and height) in inches or centimeters",
        parent=reqs_node,
        critical=True  # Essential requirement within this category
    )

    poster_dims_exists_node = evaluator.add_custom_node(
        result=_has_text(info.poster_dimensions),
        id="poster_dimensions_provided",
        desc="Poster dimensions are provided",
        parent=poster_group_node,
        critical=True
    )

    poster_dims_supported_leaf = evaluator.add_leaf(
        id="poster_dimensions_supported_by_sources",
        desc="The official source states the poster dimensions as provided",
        parent=poster_group_node,
        critical=True
    )
    dims_str = info.poster_dimensions or ""
    await evaluator.verify(
        claim=f"The official page states the required or maximum poster dimensions are '{dims_str}'.",
        node=poster_dims_supported_leaf,
        sources=info.official_urls,
        additional_instruction=(
            "Match dimensions (width × height) in inches or centimeters; allow minor formatting differences like 'x' vs '×'."
        ),
        extra_prerequisites=[official_url_exists_node, poster_dims_exists_node]
    )

    # Preprint policy: non-critical; either supported by sources or explicitly noted as unavailable
    preprint_node = evaluator.add_parallel(
        id="Preprint_Citation_Policy",
        desc="The conference policy on citing preprints in abstracts/posters is stated; if not available on official sources, this is explicitly noted",
        parent=reqs_node,
        critical=False
    )

    preprint_supported_leaf = evaluator.add_leaf(
        id="preprint_policy_supported_by_sources",
        desc="Official sources state the preprint citation policy as provided in the answer",
        parent=preprint_node,
        critical=False
    )
    preprint_policy_str = info.preprint_citation_policy or ""
    await evaluator.verify(
        claim=(
            f"The official page states the conference's policy on citing preprints (e.g., bioRxiv) and it matches: '{preprint_policy_str}'."
            if _has_text(preprint_policy_str)
            else "The official page states the conference's policy on citing preprints (e.g., bioRxiv)."
        ),
        node=preprint_supported_leaf,
        sources=info.official_urls,
        additional_instruction=(
            "Look for explicit language about preprints, preprint servers (e.g., bioRxiv), citation policies, or restrictions. "
            "If the answer did not provide an explicit policy text, check whether any policy is stated on the official page."
        ),
        extra_prerequisites=[official_url_exists_node]
    )

    preprint_noted_leaf = evaluator.add_leaf(
        id="preprint_policy_unavailable_noted",
        desc="The answer explicitly notes that no preprint citation policy is available on official sources",
        parent=preprint_node,
        critical=False
    )
    await evaluator.verify(
        claim=(
            "The answer explicitly states that the official conference sources do not provide any policy regarding citation of preprints."
            if info.preprint_policy_not_available_noted is True
            else "The answer does NOT state that the official sources lack a preprint citation policy."
        ),
        node=preprint_noted_leaf,
        additional_instruction=(
            "Verify this claim based solely on the provided answer text."
        ),
    )

    # ----------------------- Official source support group ----------------------
    # Critical group verifying that official sources support the key reported info
    support_group = evaluator.add_parallel(
        id="Official_Source_Support",
        desc="Official sources support the reported required information (name, location, dates, abstract limit, poster dimensions)",
        parent=sources_node,
        critical=True
    )

    # Name supported
    name_supported_leaf = evaluator.add_leaf(
        id="name_supported_by_sources",
        desc="Official page shows the full official conference name as provided",
        parent=support_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page shows the conference name '{info.conference_name or ''}'.",
        node=name_supported_leaf,
        sources=info.official_urls,
        additional_instruction="Allow minor formatting/punctuation/casing variations.",
        extra_prerequisites=[official_url_exists_node, name_exists_node]
    )

    # Location supported
    location_supported_leaf = evaluator.add_leaf(
        id="location_supported_by_sources",
        desc="Official page shows the specific city and state/province as provided",
        parent=support_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page shows the conference location as '{loc_city}, {loc_region}'.",
        node=location_supported_leaf,
        sources=info.official_urls,
        additional_instruction="Allow minor formatting differences, e.g., commas or abbreviations.",
        extra_prerequisites=[official_url_exists_node, location_provided_node]
    )

    # Dates supported
    dates_supported_leaf = evaluator.add_leaf(
        id="dates_supported_by_sources",
        desc="Official page shows the exact conference dates as provided",
        parent=support_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page shows the conference dates '{dates_str}'.",
        node=dates_supported_leaf,
        sources=info.official_urls,
        additional_instruction="Allow minor punctuation differences and en-dash vs hyphen.",
        extra_prerequisites=[official_url_exists_node, dates_provided_node]
    )

    # Abstract limit supported (second check under sources group)
    abstract_supported_leaf_2 = evaluator.add_leaf(
        id="abstract_limit_supported_by_sources_2",
        desc="Official page shows the maximum abstract length as provided",
        parent=support_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page states the maximum abstract length is '{abstract_limit_str}'.",
        node=abstract_supported_leaf_2,
        sources=info.official_urls,
        additional_instruction="Match word/character limits; allow minor formatting differences.",
        extra_prerequisites=[official_url_exists_node, abstract_exists_node]
    )

    # Poster dimensions supported (second check under sources group)
    poster_supported_leaf_2 = evaluator.add_leaf(
        id="poster_dimensions_supported_by_sources_2",
        desc="Official page shows the poster dimensions as provided",
        parent=support_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official page states the poster dimensions are '{dims_str}'.",
        node=poster_supported_leaf_2,
        sources=info.official_urls,
        additional_instruction="Match width × height; allow 'x' vs '×' and minor punctuation differences.",
        extra_prerequisites=[official_url_exists_node, poster_dims_exists_node]
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the neuroscience conference (2026) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured conference info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceInfo,
        extraction_name="conference_info"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return structured evaluation summary
    return evaluator.get_summary()