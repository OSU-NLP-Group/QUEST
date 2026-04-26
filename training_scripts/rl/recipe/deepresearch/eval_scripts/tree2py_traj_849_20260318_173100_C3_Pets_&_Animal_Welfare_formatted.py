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
TASK_ID = "co_mountain_lion_victim_and_ca_reg_2025"
TASK_DESCRIPTION = """
On January 1, 2026, a fatal mountain lion attack occurred on a hiking trail in Larimer County, Colorado. The victim was a 46-year-old professional who worked in the equestrian writing field and served as managing director for a horse-related organization.

Your task is to:
(1) Identify the victim's full legal name,
(2) Confirm the victim's age at the time of the attack,
(3) Identify the specific name of the hiking trail where the attack occurred,
(4) Verify the victim's profession as an equestrian writer,
(5) Identify the organization where the victim served as managing director,
(6) Identify the specific California regulation code (Title 14 CCR section) for wildlife rehabilitation that went into effect in 2025, and
(7) Provide the exact effective date when this California regulation became active.

Provide URLs supporting each piece of information.
"""

# Ground truths (expected targets)
EXPECTED_VICTIM_NAME = "Kristen Marie Kovatch"
EXPECTED_AGE = "46"
EXPECTED_TRAIL = "Crosier Mountain Trail"
EXPECTED_ORG = "Retired Racehorse Project"
EXPECTED_ROLE = "Managing Director"
EXPECTED_CA_REG_CODE = "Title 14 CCR §679"
EXPECTED_CA_REG_EFFECTIVE_DATE = "August 13, 2025"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IncidentInfo(BaseModel):
    primary_url: Optional[str] = None

    victim_full_name: Optional[str] = None
    victim_name_urls: List[str] = Field(default_factory=list)

    victim_age: Optional[str] = None
    victim_age_urls: List[str] = Field(default_factory=list)

    trail_name: Optional[str] = None
    trail_urls: List[str] = Field(default_factory=list)


class ProfessionalInfo(BaseModel):
    primary_url: Optional[str] = None

    profession: Optional[str] = None
    equestrian_career_urls: List[str] = Field(default_factory=list)

    organization_name: Optional[str] = None
    role_title: Optional[str] = None
    org_role_urls: List[str] = Field(default_factory=list)


class RegulationInfo(BaseModel):
    primary_url: Optional[str] = None

    regulation_code: Optional[str] = None
    regulation_code_urls: List[str] = Field(default_factory=list)

    effective_date: Optional[str] = None
    effective_date_urls: List[str] = Field(default_factory=list)


class AttackRegulationExtraction(BaseModel):
    incident: IncidentInfo = IncidentInfo()
    professional: ProfessionalInfo = ProfessionalInfo()
    regulation: RegulationInfo = RegulationInfo()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all_fields() -> str:
    return """
    Extract the following information from the provided answer text. Extract exactly what the answer states, without adding or inferring new information. For any item not present, return null for single fields and [] for lists.

    Structure your JSON as:
    {
      "incident": {
        "primary_url": string|null,
        "victim_full_name": string|null,
        "victim_name_urls": string[] (supporting URLs confirming the full name),
        "victim_age": string|null,
        "victim_age_urls": string[] (supporting URLs confirming the age),
        "trail_name": string|null,
        "trail_urls": string[] (supporting URLs confirming the specific trail name)
      },
      "professional": {
        "primary_url": string|null,
        "profession": string|null,  // e.g., "equestrian writer", "freelance writer in horse industry"
        "equestrian_career_urls": string[] (supporting URLs confirming equestrian writer career),
        "organization_name": string|null,  // e.g., "Retired Racehorse Project"
        "role_title": string|null,         // e.g., "Managing Director"
        "org_role_urls": string[] (supporting URLs confirming managing director role at the org)
      },
      "regulation": {
        "primary_url": string|null,
        "regulation_code": string|null,     // e.g., "Title 14 CCR §679", "14 CCR 679", etc.
        "regulation_code_urls": string[] (supporting URLs confirming the code),
        "effective_date": string|null,      // e.g., "August 13, 2025"
        "effective_date_urls": string[] (supporting URLs confirming the effective date)
      }
    }

    Rules:
    - Only extract URLs explicitly present in the answer (plain or markdown). If a URL lacks protocol, prepend http://.
    - Do not fabricate or guess any values.
    - If multiple relevant URLs are present, include them all in the corresponding list.
    - Keep dates as strings exactly as written in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _merge_urls(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for item in args:
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                urls.append(s)
        elif isinstance(item, list):
            for u in item:
                if isinstance(u, str) and u.strip():
                    urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    claim: str,
    node_id: str,
    desc: str,
    parent,
    sources: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    """
    Add a leaf node. If sources exist, run URL-based verification; otherwise mark as failed.
    """
    if sources:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=critical,
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=sources,
            additional_instruction=additional_instruction,
        )
    else:
        # Explicitly mark as failed due to missing sources
        evaluator.add_leaf(
            id=node_id,
            desc=f"{desc} (failed: missing supporting URL(s))",
            parent=parent,
            critical=critical,
            score=0.0,
            status="failed",
        )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_victim_identification_checks(
    evaluator: Evaluator,
    parent,
    ex: AttackRegulationExtraction,
) -> None:
    v = ex.incident

    vic_node = evaluator.add_parallel(
        id="victim_identification",
        desc="Identify and verify all required details about the mountain lion attack victim",
        parent=parent,
        critical=True,
    )

    # 1) Primary incident URL existence + content relevance
    incident_url_present = evaluator.add_custom_node(
        result=(v.primary_url is not None and str(v.primary_url).strip() != ""),
        id="victim_identification_url_present",
        desc="Primary incident URL is provided",
        parent=vic_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        claim="This webpage reports that on January 1, 2026, a fatal mountain lion attack occurred on a hiking trail in Larimer County, Colorado.",
        node_id="victim_identification_url",
        desc="Provide a primary URL reference for the mountain lion attack incident",
        parent=vic_node,
        sources=_merge_urls(v.primary_url),
        additional_instruction="Verify that the page is about the specific incident in Larimer County, Colorado on January 1, 2026, and that the attack was fatal.",
    )

    # 2) Victim full legal name
    name_urls_all = _merge_urls(v.victim_name_urls, v.primary_url)
    name_exists = evaluator.add_custom_node(
        result=(v.victim_full_name is not None and v.victim_full_name.strip() != "" and len(name_urls_all) > 0),
        id="victim_full_name_exists",
        desc="Victim full name and at least one supporting URL are provided",
        parent=vic_node,
        critical=True,
    )

    # First, verify by URLs (source-grounded)
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"The victim's full legal name is {EXPECTED_VICTIM_NAME}.",
        node_id="victim_name_url",
        desc="Provide a valid URL reference confirming the victim's full name",
        parent=vic_node,
        sources=name_urls_all,
        additional_instruction="Allow minor variants (e.g., middle name/initial, case differences). Confirm the full legal name on the page.",
    )

    # Then, check match to expected
    name_match_leaf = evaluator.add_leaf(
        id="victim_full_name",
        desc=f"Provide the victim's complete legal name ({EXPECTED_VICTIM_NAME})",
        parent=vic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted victim name '{v.victim_full_name or ''}' is equivalent to '{EXPECTED_VICTIM_NAME}' (case-insensitive; allow middle name/initial variants).",
        node=name_match_leaf,
        additional_instruction="Treat middle initials vs full middle names and spacing/punctuation as acceptable variants if they clearly refer to the same person.",
    )

    # 3) Victim age
    age_urls_all = _merge_urls(v.victim_age_urls, v.primary_url)
    age_exists = evaluator.add_custom_node(
        result=(v.victim_age is not None and v.victim_age.strip() != "" and len(age_urls_all) > 0),
        id="victim_age_exists",
        desc="Victim age and at least one supporting URL are provided",
        parent=vic_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim="The victim was 46 years old at the time of the attack.",
        node_id="victim_age_url",
        desc="Provide a valid URL reference confirming the victim's age",
        parent=vic_node,
        sources=age_urls_all,
        additional_instruction="Confirm that the page explicitly states the victim was 46 years old.",
    )

    # Then, simple match to expected age
    age_match_leaf = evaluator.add_leaf(
        id="victim_age",
        desc=f"Confirm the victim's age at the time of death ({EXPECTED_AGE} years old)",
        parent=vic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted victim age '{v.victim_age or ''}' is equivalent to '{EXPECTED_AGE}' (e.g., '46', '46 years', '46 years old' all count).",
        node=age_match_leaf,
        additional_instruction="Allow 'years', 'yrs', or 'years old' suffix variations.",
    )

    # 4) Trail location
    trail_urls_all = _merge_urls(v.trail_urls, v.primary_url)
    trail_exists = evaluator.add_custom_node(
        result=(v.trail_name is not None and v.trail_name.strip() != "" and len(trail_urls_all) > 0),
        id="trail_location_exists",
        desc="Trail name and at least one supporting URL are provided",
        parent=vic_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"The fatal mountain lion attack occurred on the {EXPECTED_TRAIL}.",
        node_id="trail_location_url",
        desc="Provide a valid URL reference confirming the trail location",
        parent=vic_node,
        sources=trail_urls_all,
        additional_instruction="Accept reasonable variants (e.g., 'Crosier Mountain Trail' vs 'Crosier Mtn. Trail'). Confirm the trail named on the page.",
    )

    # Then, simple match to expected trail
    trail_match_leaf = evaluator.add_leaf(
        id="trail_location",
        desc=f"Identify the specific trail name where the attack occurred ({EXPECTED_TRAIL})",
        parent=vic_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted trail name '{v.trail_name or ''}' is equivalent to '{EXPECTED_TRAIL}'.",
        node=trail_match_leaf,
        additional_instruction="Allow minor punctuation/spacing differences and common abbreviations if clearly the same trail.",
    )


async def _build_professional_background_checks(
    evaluator: Evaluator,
    parent,
    ex: AttackRegulationExtraction,
) -> None:
    p = ex.professional
    victim_name = ex.incident.victim_full_name or EXPECTED_VICTIM_NAME

    prof_node = evaluator.add_parallel(
        id="professional_background_verification",
        desc="Verify the victim's professional career and organizational affiliation",
        parent=parent,
        critical=True,
    )

    # 1) Primary professional background URL
    prof_url_present = evaluator.add_custom_node(
        result=(p.primary_url is not None and str(p.primary_url).strip() != ""),
        id="professional_background_url_present",
        desc="Primary professional background URL is provided",
        parent=prof_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"This webpage provides professional background about {victim_name}, relevant to equestrian writing and/or organizational roles.",
        node_id="professional_background_url",
        desc="Provide a primary URL reference for the victim's professional background",
        parent=prof_node,
        sources=_merge_urls(p.primary_url),
        additional_instruction="The page should be a credible source about the person's professional profile, such as a bio, organization staff page, or professional portfolio.",
    )

    # 2) Equestrian writer career
    equ_urls_all = _merge_urls(p.equestrian_career_urls, p.primary_url)
    equ_exists = evaluator.add_custom_node(
        result=(p.profession is not None and p.profession.strip() != "" and len(equ_urls_all) > 0),
        id="equestrian_writer_career_exists",
        desc="Profession and at least one supporting URL are provided",
        parent=prof_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim="The victim worked as an equestrian writer (or a freelance writer specializing in horses).",
        node_id="equestrian_career_url",
        desc="Provide a valid URL reference confirming the victim's career as an equestrian writer",
        parent=prof_node,
        sources=equ_urls_all,
        additional_instruction="Allow wording like 'equestrian writer', 'horse industry writer', 'freelance writer covering horses/equestrian topics'.",
    )

    # Then, semantic check on the extracted profession
    equ_match_leaf = evaluator.add_leaf(
        id="equestrian_writer_career",
        desc="Confirm the victim worked as an equestrian writer or freelance writer specializing in horses",
        parent=prof_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted profession '{p.profession or ''}' denotes an equestrian writer or a freelance writer specializing in horses.",
        node=equ_match_leaf,
        additional_instruction="Treat synonyms and close paraphrases as equivalent if they clearly indicate equestrian-focused writing.",
    )

    # 3) Organization role (Managing Director at Retired Racehorse Project)
    org_urls_all = _merge_urls(p.org_role_urls, p.primary_url)
    org_exists = evaluator.add_custom_node(
        result=(
            (p.organization_name is not None and p.organization_name.strip() != "") and
            (p.role_title is not None and p.role_title.strip() != "") and
            len(org_urls_all) > 0
        ),
        id="organization_role_exists",
        desc="Organization name, role title, and at least one supporting URL are provided",
        parent=prof_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"{victim_name} served as Managing Director at the Retired Racehorse Project.",
        node_id="organization_role_url",
        desc="Provide a valid URL reference confirming the victim's role as managing director at the organization",
        parent=prof_node,
        sources=org_urls_all,
        additional_instruction="Confirm the page explicitly states Managing Director (or close equivalent) at Retired Racehorse Project (RRP).",
    )

    # Then, simple match to expected organization and role
    org_role_match_leaf = evaluator.add_leaf(
        id="organization_role",
        desc=f"Identify the horse-related organization where the victim served as managing director ({EXPECTED_ORG})",
        parent=prof_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted organization '{p.organization_name or ''}' is equivalent to '{EXPECTED_ORG}', and the extracted role '{p.role_title or ''}' denotes '{EXPECTED_ROLE}'.",
        node=org_role_match_leaf,
        additional_instruction="Allow minor case variations or abbreviations (e.g., 'RRP') and slight wording variants for 'Managing Director'.",
    )


async def _build_california_regulation_checks(
    evaluator: Evaluator,
    parent,
    ex: AttackRegulationExtraction,
) -> None:
    r = ex.regulation

    reg_node = evaluator.add_parallel(
        id="california_wildlife_regulation",
        desc="Identify the specific California wildlife rehabilitation regulation that became effective in 2025",
        parent=parent,
        critical=True,
    )

    # 1) Primary regulation URL
    reg_url_present = evaluator.add_custom_node(
        result=(r.primary_url is not None and str(r.primary_url).strip() != ""),
        id="california_regulation_url_present",
        desc="Primary California regulation URL is provided",
        parent=reg_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        claim="This webpage concerns California's Wildlife Rehabilitation regulations in Title 14 of the California Code of Regulations.",
        node_id="california_regulation_url",
        desc="Provide a primary URL reference for the California wildlife rehabilitation regulation",
        parent=reg_node,
        sources=_merge_urls(r.primary_url),
        additional_instruction="Prefer an official CDFW or legal reference page. The page should clearly concern CCR Title 14, Wildlife Rehabilitation.",
    )

    # 2) Regulation code (Title 14 CCR §679)
    code_urls_all = _merge_urls(r.regulation_code_urls, r.primary_url)
    code_exists = evaluator.add_custom_node(
        result=(r.regulation_code is not None and r.regulation_code.strip() != "" and len(code_urls_all) > 0),
        id="regulation_code_exists",
        desc="Regulation code and at least one supporting URL are provided",
        parent=reg_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"The California regulation for wildlife rehabilitation is {EXPECTED_CA_REG_CODE}.",
        node_id="regulation_code_url",
        desc="Provide a valid URL reference confirming the regulation code",
        parent=reg_node,
        sources=code_urls_all,
        additional_instruction="Accept common legal formatting variants such as '14 CCR 679', 'Section 679, Title 14 CCR', or using the '§' symbol.",
    )

    # Then, simple match to expected code
    code_match_leaf = evaluator.add_leaf(
        id="regulation_code",
        desc=f"Provide the specific California regulation code for wildlife rehabilitation ({EXPECTED_CA_REG_CODE})",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted regulation code '{r.regulation_code or ''}' is equivalent to '{EXPECTED_CA_REG_CODE}' (allowing minor formatting variants like '14 CCR 679').",
        node=code_match_leaf,
        additional_instruction="Treat 'Title 14 CCR §679', '14 CCR §679', '14 CCR 679', and 'Section 679, Title 14 CCR' as equivalent.",
    )

    # 3) Effective date (August 13, 2025)
    date_urls_all = _merge_urls(r.effective_date_urls, r.primary_url)
    date_exists = evaluator.add_custom_node(
        result=(r.effective_date is not None and r.effective_date.strip() != "" and len(date_urls_all) > 0),
        id="effective_date_exists",
        desc="Effective date and at least one supporting URL are provided",
        parent=reg_node,
        critical=True,
    )

    # Source-grounded first
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"Title 14 CCR §679 became effective on {EXPECTED_CA_REG_EFFECTIVE_DATE}.",
        node_id="effective_date_url",
        desc="Provide a valid URL reference confirming the effective date",
        parent=reg_node,
        sources=date_urls_all,
        additional_instruction="Allow equivalent date formats (e.g., 'Aug. 13, 2025', '8/13/2025') as the same date.",
    )

    # Then, simple match to expected date
    date_match_leaf = evaluator.add_leaf(
        id="effective_date",
        desc=f"Provide the exact date the regulation became effective ({EXPECTED_CA_REG_EFFECTIVE_DATE})",
        parent=reg_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extracted effective date '{r.effective_date or ''}' is equivalent to '{EXPECTED_CA_REG_EFFECTIVE_DATE}' (allowing standard date format variants).",
        node=date_match_leaf,
        additional_instruction="Treat month name abbreviations and numeric formats as equivalent if they denote the same calendar date.",
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
    Evaluate an answer for the Colorado mountain lion victim identification and California regulation task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent high-level sections
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

    # Extract structured information from the answer
    extracted: AttackRegulationExtraction = await evaluator.extract(
        prompt=prompt_extract_all_fields(),
        template_class=AttackRegulationExtraction,
        extraction_name="extracted_fields",
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_victim_name": EXPECTED_VICTIM_NAME,
        "expected_age": EXPECTED_AGE,
        "expected_trail": EXPECTED_TRAIL,
        "expected_profession": "Equestrian writer (or freelance writer specializing in horses)",
        "expected_org": EXPECTED_ORG,
        "expected_role": EXPECTED_ROLE,
        "expected_ca_reg_code": EXPECTED_CA_REG_CODE,
        "expected_ca_reg_effective_date": EXPECTED_CA_REG_EFFECTIVE_DATE,
    }, gt_type="expected_values")

    # Build three critical verification sections
    await _build_victim_identification_checks(evaluator, root, extracted)
    await _build_professional_background_checks(evaluator, root, extracted)
    await _build_california_regulation_checks(evaluator, root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()