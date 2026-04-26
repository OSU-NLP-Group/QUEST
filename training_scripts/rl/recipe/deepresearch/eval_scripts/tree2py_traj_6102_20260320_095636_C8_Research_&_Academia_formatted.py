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
TASK_ID = "astro_postdoc_4"
TASK_DESCRIPTION = (
    "Identify 4 postdoctoral fellowship programs in astronomy/astrophysics that meet all specified criteria regarding "
    "location, stipend, duration, research focus, and eligibility."
)

ALLOWED_STATES_FULL = {"Massachusetts", "California", "Washington", "Illinois", "New York", "Maryland"}
ALLOWED_STATES_ABBR = {"MA", "CA", "WA", "IL", "NY", "MD"}
ALLOWED_STATES_LIST_STR = "Massachusetts (MA), California (CA), Washington (WA), Illinois (IL), New York (NY), or Maryland (MD)"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FellowshipLocation(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer full name or USPS abbreviation
    portable: Optional[bool] = None  # True if portable across U.S. institutions


class FellowshipInfo(BaseModel):
    name: Optional[str] = None
    institution: Optional[str] = None  # If portable, the answer may note "portable"
    location: Optional[FellowshipLocation] = None
    stipend: Optional[str] = None  # Keep as string to maximize compatibility (e.g., "$80k-$90k plus benefits")
    duration_years: Optional[str] = None  # e.g., "3 years", "2-3 years", "2 years (renewable)"
    research_areas: List[str] = Field(default_factory=list)  # e.g., ["astrophysics", "cosmology"]
    eligibility_phd_timing: Optional[str] = None  # e.g., "within 3 years of PhD"
    urls: List[str] = Field(default_factory=list)  # Official program page or reputable listing URLs


class FellowshipsExtraction(BaseModel):
    items: List[FellowshipInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fellowships() -> str:
    return f"""
Extract up to the first 6 distinct postdoctoral fellowship programs mentioned in the answer that are related to astronomy, astrophysics, cosmology, planetary science, or closely related fields. For each fellowship, extract:

- name: The fellowship program name, exactly as written in the answer.
- institution: The host institution name as given in the answer. If the fellowship is portable (i.e., not tied to a single host), keep the institution field as given (can be null) and rely on the 'portable' flag in 'location'.
- location: An object with:
  - city: City provided in the answer (null if not specified or not applicable).
  - state: State provided in the answer as full name or 2-letter abbreviation (null if not specified or not applicable).
  - portable: true if the answer says the fellowship is portable (e.g., can be hosted at multiple or any U.S. institutions), otherwise false or null.
- stipend: The annual stipend/salary amount string as stated in the answer (e.g., "$80,000", "$75k-$85k + benefits"). Do NOT convert to a number; copy text.
- duration_years: The appointment duration in years as a string exactly as stated (e.g., "3 years", "2-3 years", "2 years renewable").
- research_areas: A list of the primary research focus areas mentioned (e.g., ["astronomy", "astrophysics", "cosmology"]).
- eligibility_phd_timing: The eligibility requirement related to time since PhD (e.g., "within 3 years of PhD", "PhD by start date", "no more than 3 years since PhD").
- urls: All reference URLs in the answer that correspond to the official fellowship page or a reputable job/announcement page (e.g., institutional site, AAS Job Register). Include full URLs. If a URL in the answer is missing a protocol, prepend "http://".

Rules:
- Extract only what is explicitly in the answer.
- If an item is missing, set it to null (or empty list for arrays).
- Preserve strings as-is (do not normalize numbers or abbreviations).
- The 'portable' flag should be true ONLY if the answer clearly indicates the fellowship is portable across institutions.

Return a JSON object with:
{{
  "items": [FellowshipInfo, ...]
}}
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _state_is_allowed(state: Optional[str]) -> bool:
    if not state:
        return False
    s = state.strip()
    if not s:
        return False
    if s in ALLOWED_STATES_FULL or s in ALLOWED_STATES_ABBR:
        return True
    # Try to normalize capitalization of full names
    title_state = s.title()
    if title_state in ALLOWED_STATES_FULL:
        return True
    # Uppercase for abbreviation
    upper_state = s.upper()
    if upper_state in ALLOWED_STATES_ABBR:
        return True
    return False


def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Keep as-is; framework normalizes. Filter obviously empty.
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _portable_text(loc: Optional[FellowshipLocation]) -> bool:
    return bool(loc and loc.portable)


# --------------------------------------------------------------------------- #
# Verification logic for each fellowship                                      #
# --------------------------------------------------------------------------- #
async def verify_single_fellowship(
    evaluator: Evaluator,
    parent_node,
    fellowship: FellowshipInfo,
    index: int
) -> None:
    """
    Build and run verification for one fellowship entry.
    Follows the rubric's parallel child leaves and treats each as critical.
    """
    fn = evaluator.add_parallel(
        id=f"Fellowship_{index+1}",
        desc=f"Postdoctoral fellowship #{index+1} verification",
        parent=parent_node,
        critical=False,  # The set of 4 fellowships is evaluated in parallel; each leaf below is critical
    )

    urls = _safe_urls(fellowship.urls)
    loc = fellowship.location or FellowshipLocation()

    # 1) URL presence (helper existence gate) – Additional to rubric to avoid source-less verification
    url_present_node = evaluator.add_custom_node(
        result=len(urls) > 0,
        id=f"Fellowship_{index+1}_URL_present",
        desc=f"Fellowship #{index+1}: At least one reference URL is provided in the answer",
        parent=fn,
        critical=True
    )

    # 1) URL validity/support (rubric leaf)
    url_valid_node = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_URL",
        desc=f"Valid reference URL for fellowship #{index+1} from an official/reputable source",
        parent=fn,
        critical=True
    )
    url_claim_bits = []
    if fellowship.name:
        url_claim_bits.append(f"the fellowship program named '{fellowship.name}'")
    if fellowship.institution:
        url_claim_bits.append(f"hosted by '{fellowship.institution}'")
    if not url_claim_bits:
        url_claim_bits.append("a specific postdoctoral fellowship program relevant to astronomy/astrophysics")
    url_claim = "This webpage is the official program page or a reputable announcement/listing for " + " ".join(url_claim_bits) + "."
    await evaluator.verify(
        claim=url_claim,
        node=url_valid_node,
        sources=urls,
        additional_instruction="Treat official institutional pages (e.g., .edu, .gov, .org) and reputable listings such as the AAS Job Register as valid sources."
    )

    # 2) Fellowship name provided (rubric leaf: provided)
    name_provided_node = evaluator.add_custom_node(
        result=bool(fellowship.name and fellowship.name.strip()),
        id=f"Fellowship_{index+1}_Name",
        desc=f"Fellowship #{index+1}: The fellowship program name is provided in the answer",
        parent=fn,
        critical=True
    )

    # 3) Institution provided or portable noted (rubric leaf: provided)
    institution_ok = bool((fellowship.institution and fellowship.institution.strip()) or _portable_text(loc))
    institution_node = evaluator.add_custom_node(
        result=institution_ok,
        id=f"Fellowship_{index+1}_Institution",
        desc=f"Fellowship #{index+1}: Host institution is provided, or the fellowship is noted as portable",
        parent=fn,
        critical=True
    )

    # 4) Location requirement – rubric states: provided AND state in allowed list.
    # We split into: (a) answer provided check (helper) and (b) supported-by-URL check (rubric leaf).
    loc_provided = (_portable_text(loc)) or (bool(loc.city and loc.city.strip()) and bool(loc.state and loc.state.strip()))
    evaluator.add_custom_node(
        result=loc_provided,
        id=f"Fellowship_{index+1}_Location_Provided",
        desc=f"Fellowship #{index+1}: Location (city and state) provided in the answer, or fellowship is portable",
        parent=fn,
        critical=True
    )
    location_leaf = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_Location",
        desc=f"Fellowship #{index+1}: Institution location is in allowed states (or program is portable to U.S. institutions)",
        parent=fn,
        critical=True
    )
    if _portable_text(loc):
        loc_claim = (
            f"This fellowship is portable across U.S. institutions, and thus can be hosted in one of these allowed states: {ALLOWED_STATES_LIST_STR}."
        )
    else:
        city_txt = loc.city or "the stated city"
        state_txt = loc.state or "the stated state"
        loc_claim = (
            f"The fellowship is hosted in {city_txt}, {state_txt}, and the state '{state_txt}' is one of the allowed states: {ALLOWED_STATES_LIST_STR}."
        )
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=urls,
        additional_instruction="Verify the location information from the page. If the fellowship is explicitly portable to any U.S. institution, treat the state constraint as satisfied."
    )

    # 5) Stipend amount provided (rubric leaf: provided)
    stipend_provided_node = evaluator.add_custom_node(
        result=bool(fellowship.stipend and fellowship.stipend.strip()),
        id=f"Fellowship_{index+1}_Stipend_Amount",
        desc=f"Fellowship #{index+1}: The specific annual stipend/salary amount is provided in the answer",
        parent=fn,
        critical=True
    )

    # 6) Stipend minimum >= $75,000 (rubric leaf: verify with URL)
    stipend_min_node = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_Stipend_Minimum",
        desc=f"Fellowship #{index+1}: Annual stipend/salary is at least $75,000",
        parent=fn,
        critical=True
    )
    stipend_min_claim = "The annual stipend/salary for this fellowship is at least $75,000 (USD)."
    await evaluator.verify(
        claim=stipend_min_claim,
        node=stipend_min_node,
        sources=urls,
        additional_instruction="Check the stipend/salary text. Salary ranges are acceptable if the minimum is ≥ $75,000 or if total guaranteed pay is clearly ≥ $75,000."
    )

    # 7) Duration specified (rubric leaf: provided)
    duration_provided_node = evaluator.add_custom_node(
        result=bool(fellowship.duration_years and fellowship.duration_years.strip()),
        id=f"Fellowship_{index+1}_Duration_Specified",
        desc=f"Fellowship #{index+1}: Appointment duration is specified in the answer",
        parent=fn,
        critical=True
    )

    # 8) Duration minimum >= 2 years (rubric leaf: verify with URL)
    duration_min_node = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_Duration_Minimum",
        desc=f"Fellowship #{index+1}: Appointment duration is at least 2 years",
        parent=fn,
        critical=True
    )
    duration_min_claim = "The initial appointment duration for this fellowship is at least 2 years."
    await evaluator.verify(
        claim=duration_min_claim,
        node=duration_min_node,
        sources=urls,
        additional_instruction="If a range is given (e.g., 2–3 years), treat it as satisfying the ≥2 years requirement. If renewable terms make the minimum ambiguous, verify that at least 2 years are guaranteed."
    )

    # 9) Research area supports relevant fields (rubric leaf: verify with URL)
    research_leaf = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_Research_Area",
        desc=f"Fellowship #{index+1}: Supports research in astronomy/astrophysics/cosmology/planetary science or closely related fields",
        parent=fn,
        critical=True
    )
    research_claim = (
        "This fellowship supports research in astronomy, astrophysics, cosmology, planetary science, or closely related fields "
        "(including subfields such as observational/theoretical astrophysics, instrumentation, exoplanets, astrostatistics, and space physics as appropriate)."
    )
    await evaluator.verify(
        claim=research_claim,
        node=research_leaf,
        sources=urls,
        additional_instruction="Look for explicit statements of supported research areas. Closely related fields should be clearly relevant to astronomy/astrophysics."
    )

    # 10) Eligibility within 3 years of PhD (rubric leaf: verify with URL)
    eligibility_leaf = evaluator.add_leaf(
        id=f"Fellowship_{index+1}_Eligibility",
        desc=f"Fellowship #{index+1}: Open to candidates who completed PhD within 3 years of the start date (or will complete by start)",
        parent=fn,
        critical=True
    )
    eligibility_claim = (
        "Applicants are eligible if they earned their PhD within 3 years before the fellowship start date or will complete their PhD by the start date."
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        sources=urls,
        additional_instruction="Eligibility statements like 'within the last 3 years', 'no more than 3 years since PhD', or 'PhD within X years consistent with 3' satisfy this."
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
    Evaluate an answer for the astronomy/astrophysics postdoc fellowships task.
    """
    # Initialize evaluator (root as parallel). Set root non-critical to avoid child criticality constraints.
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

    # Record allowed states as auxiliary info
    evaluator.add_custom_info(
        {"allowed_states": sorted(list(ALLOWED_STATES_FULL)), "allowed_abbreviations": sorted(list(ALLOWED_STATES_ABBR))},
        info_type="constraints",
        info_name="location_constraints"
    )

    # Extract structured fellowship info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_fellowships(),
        template_class=FellowshipsExtraction,
        extraction_name="fellowships_extraction"
    )

    items = list(extracted.items)[:4] if extracted and extracted.items else []
    while len(items) < 4:
        items.append(FellowshipInfo())

    # Build verification subtrees for each of the four fellowships
    for i in range(4):
        await verify_single_fellowship(evaluator, root, items[i], i)

    # Return final structured summary
    return evaluator.get_summary()