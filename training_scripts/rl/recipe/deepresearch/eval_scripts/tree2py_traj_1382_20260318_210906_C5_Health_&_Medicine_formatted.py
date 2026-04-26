import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_level_iv_maternal_care_hospital"
TASK_DESCRIPTION = (
    "Identify a hospital located in California that has received Level IV Maternal Care verification from The Joint "
    "Commission, which represents the highest level of maternal care designation. The hospital must comply with federal "
    "CMS regulations requiring that emergency preparedness plans be reviewed and updated at least every 2 years "
    "(as specified in 42 CFR 482.15). Provide the following information: "
    "1. The name of the hospital, "
    "2. Confirmation of its location in California, "
    "3. Details of its Level IV Maternal Care verification from The Joint Commission (including reference to ACOG collaboration if applicable), "
    "4. Confirmation of its compliance with the CMS emergency preparedness plan review and update requirements (at least every 2 years), "
    "5. Any additional relevant characteristics of the hospital (such as hospital type, Joint Commission accreditation status, or participation in state maternal quality improvement programs). "
    "For each piece of information provided, include supporting URL references from your research."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class HospitalExtraction(BaseModel):
    # Identification
    hospital_name: Optional[str] = None
    hospital_name_support_urls: List[str] = Field(default_factory=list)

    location_state: Optional[str] = None  # Expect something like "California" / "CA" / full address containing CA
    location_support_urls: List[str] = Field(default_factory=list)

    # Level IV Maternal Care (The Joint Commission)
    tjc_level_iv_statement: Optional[str] = None  # Text that states Level IV and TJC for this hospital
    tjc_verification_support_urls: List[str] = Field(default_factory=list)

    level_iv_highest_statement: Optional[str] = None  # Text that says Level IV is the highest level

    acog_collab_statement: Optional[str] = None  # Text referencing collaboration with ACOG
    acog_collab_support_urls: List[str] = Field(default_factory=list)

    # CMS Emergency Preparedness requirement and compliance
    cms_two_year_requirement_statement: Optional[str] = None  # Text that states the 2-year requirement under 42 CFR 482.15
    cms_requirement_support_urls: List[str] = Field(default_factory=list)

    hospital_compliance_statement: Optional[str] = None  # Text that claims the identified hospital complies
    hospital_compliance_support_urls: List[str] = Field(default_factory=list)

    # Additional characteristic
    additional_characteristic: Optional[str] = None  # e.g., hospital type, accreditation status, quality program
    additional_characteristic_support_urls: List[str] = Field(default_factory=list)

    # (Optional) Any other hospital names referenced in the answer (to help consistency checks)
    other_hospital_names_mentioned: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hospital_info() -> str:
    return """
    Extract the following structured information exactly as stated in the answer. Do not invent or infer anything.
    Only extract URLs that are explicitly present in the answer.

    Fields to extract:
    1) hospital_name: The single hospital's name that the answer focuses on (string).
    2) hospital_name_support_urls: URLs that directly support the hospital's name/identity (list of URLs).
    3) location_state: The stated location string demonstrating it is in California (e.g., 'California', 'CA', or an address including California/CA). Return exactly what the answer states (string).
    4) location_support_urls: URLs that support the hospital’s California location (list of URLs).

    5) tjc_level_iv_statement: The exact text from the answer that states the hospital has Level IV Maternal Care verification from The Joint Commission (string). If not explicitly stated, return null.
    6) tjc_verification_support_urls: URLs that support the hospital’s Level IV Maternal Care verification (list of URLs).

    7) level_iv_highest_statement: The exact text that indicates Level IV is the highest level of maternal care (string). If not explicitly stated, return null.

    8) acog_collab_statement: The exact text that references that The Joint Commission's Maternal Levels of Care verification is in collaboration with ACOG (string). If not stated, return null.
    9) acog_collab_support_urls: URLs supporting the ACOG collaboration aspect (these can be program-level pages; list of URLs).

    10) cms_two_year_requirement_statement: The exact text that states CMS (42 CFR 482.15) requires emergency preparedness plans to be reviewed/updated at least every 2 years (string). If not present, return null.
    11) cms_requirement_support_urls: URLs that support the CMS regulation requirement (e.g., eCFR/CMS pages; list of URLs).

    12) hospital_compliance_statement: The exact text that claims the identified hospital complies with the CMS emergency preparedness review/update frequency requirement (string). If not present, return null.
    13) hospital_compliance_support_urls: URLs supporting the hospital-specific compliance claim (list of URLs).

    14) additional_characteristic: One additional relevant characteristic about the same hospital (e.g., hospital type, Joint Commission accreditation status, participation in a state maternal quality improvement program), exactly as stated (string). If none, return null.
    15) additional_characteristic_support_urls: URLs supporting the additional characteristic (list of URLs).

    16) other_hospital_names_mentioned: Any other hospital names mentioned anywhere in the answer (list of strings). If none, return an empty list.

    IMPORTANT:
    - Extract only URLs that actually appear in the answer text (plain or markdown links). Do not fabricate.
    - If any requested field is not present, set it to null (for strings) or [] (for lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_present(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_urls(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def _mentions_california(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    t = text.strip()
    low = t.lower()
    if "california" in low:
        return True
    # Match standalone "CA" (avoid matching "ca" inside other words)
    return re.search(r"\bca\b", t, flags=re.IGNORECASE) is not None


def _mentions_level_iv_and_tjc(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    low = text.lower()
    has_level = ("level iv" in low) or ("level 4" in low)
    has_tjc = ("joint commission" in low) or ("tjc" in low)
    return has_level and has_tjc


def _mentions_highest_level(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    low = text.lower()
    return ("highest" in low) or ("top level" in low) or ("most advanced" in low)


def _mentions_acog(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    return "acog" in text.lower() or "american college of obstetricians" in text.lower()


def _mentions_two_year_requirement(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    low = text.lower()
    has_reg = ("482.15" in low) or ("42 cfr 482.15" in low)
    two_years = ("2 years" in low) or ("two years" in low) or ("every 2 years" in low) or ("at least every 2 years" in low) or ("biennial" in low) or ("biennially" in low)
    return has_reg and two_years


def _mentions_compliance(text: Optional[str]) -> bool:
    if not _is_present(text):
        return False
    low = text.lower()
    return any(kw in low for kw in ["comply", "complies", "compliant", "meets", "in accordance", "adheres", "follows"])


# --------------------------------------------------------------------------- #
# Verification subtree builders                                               #
# --------------------------------------------------------------------------- #
async def build_hospital_identification(evaluator: Evaluator, parent, ex: HospitalExtraction) -> None:
    node = evaluator.add_parallel(
        id="hospital_identification",
        desc="Provide the hospital’s name and confirm it is located in California, with URL support.",
        parent=parent,
        critical=True
    )

    # 1) Hospital name provided (critical presence)
    name_present = _is_present(ex.hospital_name)
    evaluator.add_custom_node(
        result=name_present,
        id="hospital_name_provided",
        desc="Hospital name is provided.",
        parent=node,
        critical=True
    )

    # 2) Location in California provided (critical presence)
    location_ca_present = _mentions_california(ex.location_state)
    evaluator.add_custom_node(
        result=location_ca_present,
        id="hospital_location_in_california_provided",
        desc="Hospital is stated to be located in California.",
        parent=node,
        critical=True
    )

    # 3) Hospital name supported by URL(s)
    name_support_leaf = evaluator.add_leaf(
        id="hospital_name_supported_by_url",
        desc="At least one URL is provided that supports the stated hospital name/identity.",
        parent=node,
        critical=True
    )
    if _has_urls(ex.hospital_name_support_urls):
        claim = f"The webpage clearly identifies the hospital named '{ex.hospital_name}'. Minor name variations (e.g., punctuation, suffix) still refer to the same hospital."
        await evaluator.verify(
            claim=claim,
            node=name_support_leaf,
            sources=ex.hospital_name_support_urls,
            additional_instruction="Pass only if the page explicitly confirms the hospital’s name/identity (official site, directory page, or authoritative profile). Allow minor formatting differences."
        )
    else:
        name_support_leaf.score = 0.0
        name_support_leaf.status = "failed"

    # 4) California location supported by URL(s)
    location_support_leaf = evaluator.add_leaf(
        id="hospital_california_location_supported_by_url",
        desc="At least one URL is provided that supports the hospital’s California location.",
        parent=node,
        critical=True
    )
    if _has_urls(ex.location_support_urls):
        claim = f"The hospital named '{ex.hospital_name}' is located in California (CA), United States."
        await evaluator.verify(
            claim=claim,
            node=location_support_leaf,
            sources=ex.location_support_urls,
            additional_instruction="Accept if the page shows a California address or explicitly states the hospital is in California."
        )
    else:
        location_support_leaf.score = 0.0
        location_support_leaf.status = "failed"

    # 5) Single-hospital consistency (answer-internal consistency)
    consistency_leaf = evaluator.add_leaf(
        id="single_hospital_consistency",
        desc="All subsequent claims (verification, CMS compliance, additional characteristics) apply to the same single hospital named (no switching between hospitals).",
        parent=node,
        critical=True
    )
    hospital_name_for_claim = ex.hospital_name or "the named hospital"
    claim = (
        f"In the provided answer, all details (Level IV verification, CMS emergency preparedness compliance, additional characteristics) "
        f"refer consistently to the same single hospital: '{hospital_name_for_claim}', with no switching to any other hospital."
    )
    await evaluator.verify(
        claim=claim,
        node=consistency_leaf,
        additional_instruction="Judge only based on the answer text (not external links). If multiple hospitals are mentioned or different sections refer to different facilities, mark as Incorrect."
    )


async def build_level_iv_details(evaluator: Evaluator, parent, ex: HospitalExtraction) -> None:
    node = evaluator.add_parallel(
        id="level_iv_verification_details",
        desc="Provide details that the hospital has Level IV Maternal Care verification from The Joint Commission (highest level), with URL support.",
        parent=parent,
        critical=True
    )

    # 1) TJC Level IV verification stated (presence in answer)
    tjc_stated = _mentions_level_iv_and_tjc(ex.tjc_level_iv_statement)
    evaluator.add_custom_node(
        result=tjc_stated,
        id="tjc_level_iv_verification_stated",
        desc="Response states the hospital has Level IV Maternal Care verification from The Joint Commission.",
        parent=node,
        critical=True
    )

    # 2) Level IV described as highest (presence in answer)
    highest_stated = _mentions_highest_level(ex.level_iv_highest_statement)
    evaluator.add_custom_node(
        result=highest_stated,
        id="level_iv_described_as_highest",
        desc="Response indicates Level IV represents the highest level of maternal care designation (or equivalent wording).",
        parent=node,
        critical=True
    )

    # 3) ACOG collaboration referenced (presence in answer)
    acog_ref_present = _mentions_acog(ex.acog_collab_statement)
    evaluator.add_custom_node(
        result=acog_ref_present,
        id="acog_collaboration_referenced",
        desc="Response references that The Joint Commission Maternal Levels of Care verification is in collaboration with ACOG (as applicable per prompt/constraints).",
        parent=node,
        critical=True
    )

    # 4) Verification supported by URL(s)
    verification_support_leaf = evaluator.add_leaf(
        id="verification_supported_by_url",
        desc="At least one URL is provided supporting the hospital’s Level IV Maternal Care verification claim.",
        parent=node,
        critical=True
    )
    if _has_urls(ex.tjc_verification_support_urls):
        hosp = ex.hospital_name or "the hospital"
        claim = f"{hosp} has Level IV Maternal Care verification from The Joint Commission."
        await evaluator.verify(
            claim=claim,
            node=verification_support_leaf,
            sources=ex.tjc_verification_support_urls,
            additional_instruction="Accept if the page (e.g., TJC directory, hospital announcement, or credible news) explicitly confirms Level IV Maternal Care verification for this hospital."
        )
    else:
        verification_support_leaf.score = 0.0
        verification_support_leaf.status = "failed"

    # 5) ACOG collaboration supported by URL(s)
    acog_support_leaf = evaluator.add_leaf(
        id="acog_collaboration_supported_by_url",
        desc="At least one URL is provided supporting the ACOG collaboration aspect of the verification program (program-level evidence acceptable).",
        parent=node,
        critical=True
    )
    if _has_urls(ex.acog_collab_support_urls):
        claim = "The Joint Commission's Maternal Levels of Care verification program is in collaboration with ACOG."
        await evaluator.verify(
            claim=claim,
            node=acog_support_leaf,
            sources=ex.acog_collab_support_urls,
            additional_instruction="Program-level pages from The Joint Commission or ACOG are acceptable as evidence."
        )
    else:
        acog_support_leaf.score = 0.0
        acog_support_leaf.status = "failed"


async def build_cms_requirement_and_compliance(evaluator: Evaluator, parent, ex: HospitalExtraction) -> None:
    node = evaluator.add_parallel(
        id="cms_emergency_preparedness_requirement_and_compliance",
        desc="State the CMS emergency-preparedness plan review/update requirement (≥ every 2 years under 42 CFR 482.15) and confirm the hospital’s compliance, with URL support for both the regulation and the compliance claim.",
        parent=parent,
        critical=True
    )

    # 1) CMS two-year requirement stated (presence)
    cms_req_present = _mentions_two_year_requirement(ex.cms_two_year_requirement_statement)
    evaluator.add_custom_node(
        result=cms_req_present,
        id="cms_two_year_requirement_stated",
        desc="Response states that CMS (42 CFR 482.15) requires hospital emergency preparedness plans to be reviewed and updated at least every 2 years (or equivalent wording).",
        parent=node,
        critical=True
    )

    # 2) CMS requirement supported by URL(s)
    cms_req_support_leaf = evaluator.add_leaf(
        id="cms_requirement_supported_by_url",
        desc="At least one URL is provided supporting the CMS regulation requirement (e.g., eCFR/CMS page).",
        parent=node,
        critical=True
    )
    if _has_urls(ex.cms_requirement_support_urls):
        claim = "CMS regulation 42 CFR 482.15 requires hospital emergency preparedness programs/plans to be reviewed and updated at least every two years."
        await evaluator.verify(
            claim=claim,
            node=cms_req_support_leaf,
            sources=ex.cms_requirement_support_urls,
            additional_instruction="Prefer eCFR, CMS, or other official sources. The evidence must clearly state the biennial (at least every two years) review/update requirement."
        )
    else:
        cms_req_support_leaf.score = 0.0
        cms_req_support_leaf.status = "failed"

    # 3) Hospital compliance claimed (presence)
    compliance_present = _mentions_compliance(ex.hospital_compliance_statement)
    evaluator.add_custom_node(
        result=compliance_present,
        id="hospital_compliance_claimed",
        desc="Response claims the identified hospital complies with the CMS emergency preparedness plan review/update requirement.",
        parent=node,
        critical=True
    )

    # 4) Hospital compliance supported by URL(s)
    compliance_support_leaf = evaluator.add_leaf(
        id="hospital_compliance_supported_by_url",
        desc="At least one URL is provided supporting the hospital-specific compliance claim (e.g., hospital policy/attestation/accreditation compliance statement tied to CMS emergency preparedness).",
        parent=node,
        critical=True
    )
    if _has_urls(ex.hospital_compliance_support_urls):
        hosp = ex.hospital_name or "the hospital"
        claim = f"{hosp} complies with the CMS emergency preparedness plan review/update frequency requirement (at least every two years) under 42 CFR 482.15."
        await evaluator.verify(
            claim=claim,
            node=compliance_support_leaf,
            sources=ex.hospital_compliance_support_urls,
            additional_instruction="Accept if the page provides a hospital policy, attestation, accreditation compliance note, or other credible statement tying the hospital to CMS 42 CFR 482.15 biennial review/update."
        )
    else:
        compliance_support_leaf.score = 0.0
        compliance_support_leaf.status = "failed"


async def build_additional_characteristic(evaluator: Evaluator, parent, ex: HospitalExtraction) -> None:
    node = evaluator.add_parallel(
        id="additional_hospital_characteristic",
        desc="Provide at least one additional relevant characteristic of the hospital beyond name/location/verification/CMS compliance, with URL support.",
        parent=parent,
        critical=True
    )

    # 1) Additional characteristic present (answer presence)
    additional_present = _is_present(ex.additional_characteristic)
    evaluator.add_custom_node(
        result=additional_present,
        id="additional_characteristic_present",
        desc="Response includes at least one additional characteristic (e.g., hospital type, Joint Commission accreditation status, participation in state maternal quality improvement programs) beyond the other required fields.",
        parent=node,
        critical=True
    )

    # 2) Additional characteristic supported by URL(s)
    additional_support_leaf = evaluator.add_leaf(
        id="additional_characteristic_supported_by_url",
        desc="At least one URL is provided supporting the additional characteristic.",
        parent=node,
        critical=True
    )
    if _has_urls(ex.additional_characteristic_support_urls):
        hosp = ex.hospital_name or "the hospital"
        characteristic_text = ex.additional_characteristic or "the stated characteristic"
        claim = f"{hosp} has the following additional characteristic: {characteristic_text}"
        await evaluator.verify(
            claim=claim,
            node=additional_support_leaf,
            sources=ex.additional_characteristic_support_urls,
            additional_instruction="The evidence should clearly support the specific characteristic stated in the answer."
        )
    else:
        additional_support_leaf.score = 0.0
        additional_support_leaf.status = "failed"


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
    Evaluate an answer for the California Level IV Maternal Care hospital verification task.
    """
    # Initialize evaluator with sequential root (to gate later checks on earlier success)
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted: HospitalExtraction = await evaluator.extract(
        prompt=prompt_extract_hospital_info(),
        template_class=HospitalExtraction,
        extraction_name="hospital_info_extraction",
    )

    # Build verification tree per rubric
    await build_hospital_identification(evaluator, root, extracted)
    await build_level_iv_details(evaluator, root, extracted)
    await build_cms_requirement_and_compliance(evaluator, root, extracted)
    await build_additional_characteristic(evaluator, root, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()