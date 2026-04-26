import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "largest_districts_lang_req"
TASK_DESCRIPTION = (
    "Identify four public school districts from among the top 10 largest U.S. school districts by student "
    "enrollment (based on 2023-2024 school year data). Each district must be located in a different U.S. state. "
    "For each district, the graduation requirements must include a specific number of world language, foreign "
    "language, or Language Other Than English (LOTE) credits, and must explicitly state that these language "
    "credits must be earned in the SAME language (not allowing students to mix different languages to fulfill "
    "the requirement). For each of the four districts, provide the following information: (1) The official "
    "district name, (2) The state where the district is located, (3) The district's student enrollment number "
    "for the 2023-2024 school year (or more recent data), (4) A URL to a source that confirms the enrollment "
    "number, (5) The total number of credits required for high school graduation, (6) The specific number of "
    "world/foreign language/LOTE credits required for graduation, (7) Evidence or confirmation that the language "
    "requirement explicitly specifies that credits must be earned in the same language, and (8) A URL to the "
    "official district or state education department page detailing the graduation requirements."
)


# --------------------------------------------------------------------------- #
# Pydantic models for extraction                                              #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    enrollment_year: Optional[str] = None
    enrollment_source_url: Optional[str] = None
    total_graduation_credits: Optional[str] = None
    language_credits_required: Optional[str] = None
    same_language_evidence_text: Optional[str] = None
    graduation_requirements_url: Optional[str] = None
    top10_source_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
Extract up to the first 6 district entries from the answer. For each district, return the following fields as a JSON array named `districts`:

Required fields per district entry:
- district_name: The official district name (string).
- state: The U.S. state where the district is located (string, full state name or postal abbreviation from the answer).
- enrollment: The reported student enrollment number for the 2023–2024 school year or a more recent school year (string; keep formatting as in the answer, e.g., with commas).
- enrollment_year: The explicit school year or year label associated with the enrollment figure if provided (e.g., "2023–2024", "Fall 2024", or similar). Use null if not specified.
- enrollment_source_url: A URL that confirms the enrollment number (URL string). If none provided, set to null.

Graduation requirement fields:
- total_graduation_credits: The total number of credits (or equivalent units) required for high school graduation, as stated in the answer (string; do not coerce to numbers).
- language_credits_required: The specific number of world/foreign language/LOTE credits (or equivalent units/years) required for graduation (string).
- same_language_evidence_text: Any exact phrase or short snippet from the answer indicating that the language credits must be earned in the SAME language (e.g., “in the same language”, “two years of the same world language”, “sequential in a single language”). If not present in the answer, return null.
- graduation_requirements_url: A URL to the official district or state education department page detailing the graduation requirements (URL string). If none provided, set to null.

Top-10 ranking support (optional but helpful):
- top10_source_urls: An array of URL(s) that explicitly support that this district is among the top 10 largest U.S. public school districts by enrollment for 2023–2024 (or newer). Extract any such URLs if present in the answer; otherwise return an empty list.

General rules:
- Only extract information explicitly present in the answer text. Do not invent data.
- For missing fields, use null (or [] for the URL list).
- Preserve strings as-is (e.g., “2 credits”, “2.0 credits”, “2 years”, “24 credits”, etc.).
- Ensure all URLs are full valid URLs.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _first_k_valid(items: List[DistrictItem], k: int) -> List[DistrictItem]:
    valid = [d for d in items if _is_nonempty_str(d.district_name)]
    return valid[:k]


# --------------------------------------------------------------------------- #
# District verification                                                       #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    parent_node,
    district: DistrictItem,
    index: int,
) -> Dict[str, Any]:
    """
    Build verification nodes and run checks for a single district.
    Returns a dictionary with useful references (e.g., URLs) for global checks.
    """
    # Create container node for this district (parallel; non-critical under root)
    dnode = evaluator.add_parallel(
        id=f"District_{index+1}",
        desc=f"District {index+1}: required attributes and sources.",
        parent=parent_node,
        critical=False,
    )

    # 1) Presence checks (critical)
    name_present = evaluator.add_custom_node(
        result=_is_nonempty_str(district.district_name),
        id=f"district_{index+1}_official_name_provided",
        desc="Provides the official district name.",
        parent=dnode,
        critical=True,
    )

    state_present = evaluator.add_custom_node(
        result=_is_nonempty_str(district.state),
        id=f"district_{index+1}_state_provided",
        desc="Provides the U.S. state where the district is located.",
        parent=dnode,
        critical=True,
    )

    enroll_url_present = evaluator.add_custom_node(
        result=_is_nonempty_str(district.enrollment_source_url),
        id=f"district_{index+1}_enrollment_source_url_provided",
        desc="Provides a URL that confirms the enrollment number.",
        parent=dnode,
        critical=True,
    )

    grad_url_present = evaluator.add_custom_node(
        result=_is_nonempty_str(district.graduation_requirements_url),
        id=f"district_{index+1}_graduation_requirements_url_provided",
        desc="Provides a URL to the official district or official state education department page detailing the graduation requirements.",
        parent=dnode,
        critical=True,
    )

    # 2) Top-10 ranking claim (critical)
    top10_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_is_top10_2023_2024_or_later",
        desc="The selected district is among the top 10 largest U.S. public school districts by student enrollment based on 2023-2024 (or newer) data.",
        parent=dnode,
        critical=True,
    )
    # Prefer explicit top-10 sources; if none, optionally include the enrollment source as weak fallback
    top10_sources: List[str] = []
    if district.top10_source_urls:
        top10_sources.extend([u for u in district.top10_source_urls if _is_nonempty_str(u)])
    if not top10_sources and _is_nonempty_str(district.enrollment_source_url):
        top10_sources.append(district.enrollment_source_url)  # may or may not explicitly state ranking

    top10_claim = (
        f"The school district '{district.district_name or 'UNKNOWN DISTRICT'}' is among the top 10 largest U.S. "
        f"public school districts by student enrollment for the 2023–2024 school year or newer."
    )
    await evaluator.verify(
        claim=top10_claim,
        node=top10_leaf,
        sources=top10_sources if len(top10_sources) > 0 else None,
        additional_instruction=(
            "Only mark as Supported if the provided webpage(s) explicitly state the district is among the top 10 largest "
            "U.S. public school districts by student enrollment, and the timeframe is 2023–2024 or newer. "
            "If no source URL is provided, or if the provided page(s) do not explicitly support this top-10 claim and timeframe, "
            "conclude Not Supported."
        ),
    )

    # 3) Enrollment number (critical): verify number and that it is 2023–2024 or later
    enrollment_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_enrollment_number_verified",
        desc="Provides the district student enrollment number for the 2023-2024 school year or more recent data.",
        parent=dnode,
        critical=True,
    )
    enrollment_claim = (
        f"The student enrollment for the school district '{district.district_name or 'UNKNOWN DISTRICT'}' is "
        f"'{district.enrollment or 'UNKNOWN'}' for the 2023–2024 school year or a later school year."
    )
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=district.enrollment_source_url if _is_nonempty_str(district.enrollment_source_url) else None,
        additional_instruction=(
            "Check the page for the enrollment figure and confirm the associated school year. "
            "Accept minor formatting differences (e.g., commas/spaces). "
            "Accept a more recent year than 2023–2024. "
            "If the page only shows a year earlier than 2023–2024 or a clearly different student count, mark Not Supported."
        ),
    )

    # 4) Total graduation credits (critical): verify with graduation requirements page
    total_grad_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_total_graduation_credits_provided",
        desc="States the total number of credits required for high school graduation.",
        parent=dnode,
        critical=True,
    )
    total_grad_claim = (
        f"The total number of credits (or equivalent graduation units) required for high school graduation "
        f"in the district '{district.district_name or 'UNKNOWN DISTRICT'}' is '{district.total_graduation_credits or 'UNKNOWN'}'."
    )
    await evaluator.verify(
        claim=total_grad_claim,
        node=total_grad_leaf,
        sources=district.graduation_requirements_url if _is_nonempty_str(district.graduation_requirements_url) else None,
        additional_instruction=(
            "Verify the stated total against the official graduation requirements page. "
            "Treat 'credits', 'Carnegie units', or clearly equivalent 'units' as acceptable synonyms if used on the official page."
        ),
    )

    # 5) World language credits (critical): verify number with graduation requirements page
    lang_credits_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_world_language_credits_provided",
        desc="States the specific number of world/foreign language/LOTE credits required for graduation.",
        parent=dnode,
        critical=True,
    )
    lang_credits_claim = (
        f"The graduation requirements specify '{district.language_credits_required or 'UNKNOWN'}' "
        f"credits (or clearly equivalent units/years) in world/foreign language/LOTE."
    )
    await evaluator.verify(
        claim=lang_credits_claim,
        node=lang_credits_leaf,
        sources=district.graduation_requirements_url if _is_nonempty_str(district.graduation_requirements_url) else None,
        additional_instruction=(
            "Verify the number of world/foreign language/LOTE credits (or equivalent requirement such as 'years' or course levels) "
            "on the official page. Accept synonyms and common phrasings (e.g., 'world language', 'foreign language', 'LOTE')."
        ),
    )

    # 6) Same-language explicitly required (critical): must state that credits are earned in the SAME language
    same_lang_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_same_language_explicitly_required",
        desc="Provides evidence that the graduation requirements explicitly require the language credits to be earned in the SAME language (no mixing languages to meet the requirement).",
        parent=dnode,
        critical=True,
    )
    snippet_part = (
        f" The answer cites: '{district.same_language_evidence_text}'." if _is_nonempty_str(district.same_language_evidence_text) else ""
    )
    same_lang_claim = (
        "The graduation requirements explicitly require that the world/foreign language/LOTE credits be earned in the same language "
        "(students cannot mix different languages to satisfy this requirement)." + snippet_part
    )
    await evaluator.verify(
        claim=same_lang_claim,
        node=same_lang_leaf,
        sources=district.graduation_requirements_url if _is_nonempty_str(district.graduation_requirements_url) else None,
        additional_instruction=(
            "Look for explicit wording such as 'in the same language', 'two (or more) credits/years in the same world language', "
            "'sequential courses in one language', 'Level I and II of the same language', etc. "
            "Generic language or options that allow mixing different languages should be marked Not Supported."
        ),
    )

    return {
        "district_node": dnode,
        "name_present_node": name_present,
        "state_present_node": state_present,
        "enroll_url_present_node": enroll_url_present,
        "grad_url_present_node": grad_url_present,
        "enrollment_url": district.enrollment_source_url if _is_nonempty_str(district.enrollment_source_url) else None,
        "grad_url": district.graduation_requirements_url if _is_nonempty_str(district.graduation_requirements_url) else None,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point for evaluating an answer to the 'Four Large School Districts with Language Requirements' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root remains non-critical to allow partial credit aggregation
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

    # -------------------------- Extraction -------------------------------- #
    extracted: DistrictsExtraction = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Select the first 4 valid district entries (filtering extras per policy)
    selected: List[DistrictItem] = _first_k_valid(extracted.districts, 4)
    while len(selected) < 4:
        selected.append(DistrictItem())  # pad if fewer provided

    # ----------------------- Global Requirements -------------------------- #
    global_node = evaluator.add_parallel(
        id="Global_Requirements",
        desc="Global constraints that apply across the full set of four districts.",
        parent=root,
        critical=True,  # Critical global constraints
    )

    # Exactly four (apply after filtering policy: requires at least 4 valid entries in the answer)
    exactly_four = evaluator.add_custom_node(
        result=(len([d for d in extracted.districts if _is_nonempty_str(d.district_name)]) >= 4),
        id="Exactly_Four_Districts_Provided",
        desc="The answer identifies exactly four districts (four distinct district entries).",
        parent=global_node,
        critical=True,
    )

    # All four states are distinct (check over the evaluated set of 4)
    states_list = [d.state.strip() for d in selected if _is_nonempty_str(d.state)]
    all_four_distinct = evaluator.add_custom_node(
        result=(len(states_list) == 4 and len(set(s.upper() for s in states_list)) == 4),
        id="All_Four_States_Are_Distinct",
        desc="The four districts are located in four different U.S. states (no duplicates).",
        parent=global_node,
        critical=True,
    )

    # Container to hold officialness checks for provided URLs
    official_urls_node = evaluator.add_parallel(
        id="All_Provided_Reference_URLs_Are_Official",
        desc="All URLs provided in the answer point to official school district websites or official state education department websites.",
        parent=global_node,
        critical=True,
    )

    # --------------------- Per-District Verification ---------------------- #
    # Create a container for district verifications (optional)
    districts_container = evaluator.add_parallel(
        id="Districts",
        desc="Per-district verification (four districts).",
        parent=root,
        critical=False,
    )

    # Verify each of the 4 districts (in parallel)
    district_tasks = [
        verify_one_district(evaluator, districts_container, selected[i], i) for i in range(4)
    ]
    district_results: List[Dict[str, Any]] = await asyncio.gather(*district_tasks)

    # ---------------- Officialness checks for provided URLs --------------- #
    # We only check URLs that are actually provided (missing URLs are handled by district-level presence checks).
    official_claims_and_sources = []
    for i, res in enumerate(district_results, start=1):
        name = selected[i - 1].district_name or f"District {i}"
        # Enrollment source URL
        if res.get("enrollment_url"):
            n = evaluator.add_leaf(
                id=f"district_{i}_official_enrollment_url",
                desc=f"Enrollment source URL for District {i} is official.",
                parent=official_urls_node,
                critical=True,
            )
            claim = (
                "This URL is an official page controlled by either a U.S. public school district "
                "or a U.S. state education department (not a third‑party or media site)."
            )
            add_ins = (
                "Assess officialness from the domain, branding, and on-page identifiers. "
                "Official examples include district-controlled domains (e.g., 'lausd.org', 'houstonisd.org', "
                "state K‑12/education subdomains) and .gov state education sites. "
                "Pages like BoardDocs/Finalsite/etc. can count as official if clearly used by and branded for the district. "
                "Non-official examples: Wikipedia, newspapers, blogs, US News, GreatSchools, random PDFs not hosted by the district/state."
            )
            official_claims_and_sources.append((claim, res["enrollment_url"], n, add_ins))

        # Graduation requirements URL
        if res.get("grad_url"):
            n = evaluator.add_leaf(
                id=f"district_{i}_official_grad_requirements_url",
                desc=f"Graduation requirements URL for District {i} is official.",
                parent=official_urls_node,
                critical=True,
            )
            claim = (
                "This URL is an official page controlled by either a U.S. public school district "
                "or a U.S. state education department (not a third‑party or media site)."
            )
            add_ins = (
                "Assess officialness from the domain, branding, and on-page identifiers. "
                "Official examples include district-controlled domains (e.g., 'ccsd.net', 'dallasisd.org', "
                "state K‑12/education subdomains) and .gov state education sites. "
                "Pages like BoardDocs/Finalsite/etc. can count as official if clearly used by and branded for the district. "
                "Non-official examples: Wikipedia, newspapers, blogs, US News, GreatSchools, random PDFs not hosted by the district/state."
            )
            official_claims_and_sources.append((claim, res["grad_url"], n, add_ins))

    if official_claims_and_sources:
        await evaluator.batch_verify(official_claims_and_sources)

    # -------------------------- Return summary ---------------------------- #
    return evaluator.get_summary()