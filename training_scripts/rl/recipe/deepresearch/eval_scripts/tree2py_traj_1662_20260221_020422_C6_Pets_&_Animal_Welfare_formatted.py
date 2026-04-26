import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "larimer_animal_welfare_program"
TASK_DESCRIPTION = (
    "You are planning to establish a comprehensive animal welfare and therapy program in Larimer County, Colorado. "
    "The program will include multiple specialized services: training and certifying therapy animals (focusing on dogs), "
    "operating a wildlife rehabilitation facility, maintaining a breeding program for therapy dogs (specializing in Giant Schnauzers "
    "with full health certifications), partnering with local animal shelters, and coordinating with veterinary professionals.\n\n"
    "For this integrated program, research and document the following requirements:\n\n"
    "1. Therapy Animal Certification Track (Pet Partners requirements: handler eligibility, animal requirements, administrative requirements; "
    "provide official reference URLs for each category).\n"
    "2. Wildlife Rehabilitation Licensing Track (Colorado provisional requirements, upgrade to full, renewal deadlines; "
    "provide official regulation URLs for each category).\n"
    "3. Giant Schnauzer Health Testing Track (CHIC certification requirements for hips, thyroid, eye exam, database publication; "
    "provide official GSCA/OFA URLs for each category).\n"
    "4. Local Shelter Partnership Track (NOCO Humane Larimer Campus address, phone, adoption hours (weekday/weekend), services, adoption fee ranges; "
    "provide official NOCO Humane URLs).\n"
    "5. Veterinary Support Requirements (Colorado veterinarian CE: total hours, 2026 delegation/supervision topic, renewal due date, CE standards; "
    "provide official Colorado veterinary board or CVMA URLs).\n\n"
    "For each requirement, provide a clear description and official reference URL(s). All information must be grounded in official sources."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class TherapyURLs(BaseModel):
    handler_urls: List[str] = Field(default_factory=list)
    animal_urls: List[str] = Field(default_factory=list)
    admin_urls: List[str] = Field(default_factory=list)


class WildlifeURLs(BaseModel):
    provisional_urls: List[str] = Field(default_factory=list)
    full_urls: List[str] = Field(default_factory=list)
    renewal_urls: List[str] = Field(default_factory=list)


class GiantURLs(BaseModel):
    hip_urls: List[str] = Field(default_factory=list)
    thyroid_urls: List[str] = Field(default_factory=list)
    eye_urls: List[str] = Field(default_factory=list)
    database_urls: List[str] = Field(default_factory=list)


class ShelterURLs(BaseModel):
    contact_urls: List[str] = Field(default_factory=list)
    hours_urls: List[str] = Field(default_factory=list)
    services_fees_urls: List[str] = Field(default_factory=list)


class VeterinaryURLs(BaseModel):
    ce_urls: List[str] = Field(default_factory=list)


class ProgramURLsExtraction(BaseModel):
    therapy: Optional[TherapyURLs] = None
    wildlife: Optional[WildlifeURLs] = None
    giant: Optional[GiantURLs] = None
    shelter: Optional[ShelterURLs] = None
    veterinary: Optional[VeterinaryURLs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program_urls() -> str:
    return (
        "Extract only the official reference URLs provided in the answer, grouped by category. "
        "Return them under the following nested JSON keys (provide arrays; leave arrays empty if none present in the answer):\n"
        "- therapy.handler_urls: Pet Partners handler eligibility URLs\n"
        "- therapy.animal_urls: Pet Partners animal requirements URLs\n"
        "- therapy.admin_urls: Pet Partners administrative (fees/insurance) URLs\n"
        "- wildlife.provisional_urls: Colorado official regulation/agency URLs for provisional wildlife rehabilitator requirements\n"
        "- wildlife.full_urls: Colorado official regulation/agency URLs for upgrading to full license\n"
        "- wildlife.renewal_urls: Colorado official regulation/agency URLs for renewal deadlines/expiration\n"
        "- giant.hip_urls: GSCA/OFA official URLs for hip dysplasia requirements\n"
        "- giant.thyroid_urls: GSCA/OFA official URLs for thyroid certification requirements\n"
        "- giant.eye_urls: GSCA/OFA official URLs for eye exam requirements\n"
        "- giant.database_urls: GSCA/OFA official URLs for CHIC database publication requirement\n"
        "- shelter.contact_urls: Official NOCO Humane URLs for address and phone\n"
        "- shelter.hours_urls: Official NOCO Humane URLs for adoption hours\n"
        "- shelter.services_fees_urls: Official NOCO Humane URLs for services and adoption fee ranges\n"
        "- veterinary.ce_urls: Official Colorado veterinary board (DPO) or CVMA URLs for CE requirements and standards\n\n"
        "Rules:\n"
        "1) Only include URLs explicitly mentioned in the answer.\n"
        "2) Prefer official domains: petpartners.org; cpw.state.co.us or colorado.gov or sos.state.co.us (regulations); ofa.org or caninehealthinfo.org or giantschnauzerclubofamerica.com; nocohumane.org; dpo.colorado.gov; colovma.org.\n"
        "3) Do not invent URLs. If the answer references an organization without a URL, leave the corresponding list empty."
    )


# --------------------------------------------------------------------------- #
# URL utilities                                                               #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    if not urls:
        return out
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if not s:
            continue
        if not s.startswith(("http://", "https://")):
            s = "http://" + s
        if s not in out:
            out.append(s)
    return out


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _has_official(urls: List[str], checker) -> bool:
    return any(checker(u) for u in urls)


def is_petpartners(url: str) -> bool:
    d = _domain(url)
    return d.endswith("petpartners.org")


def is_colorado_official(url: str) -> bool:
    d = _domain(url)
    return (
        d.endswith("cpw.state.co.us")
        or d.endswith("colorado.gov")
        or d.endswith("sos.state.co.us")
        or d.endswith("dpo.colorado.gov")
    )


def is_ofa_or_gsca(url: str) -> bool:
    d = _domain(url)
    return (
        d.endswith("ofa.org")
        or d.endswith("caninehealthinfo.org")
        or d.endswith("giantschnauzerclubofamerica.com")
    )


def is_noco_humane(url: str) -> bool:
    d = _domain(url)
    return d.endswith("nocohumane.org")


def is_vet_board_or_cvma(url: str) -> bool:
    d = _domain(url)
    return d.endswith("dpo.colorado.gov") or d.endswith("colovma.org")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_therapy_track(evaluator: Evaluator, parent_node, extracted: ProgramURLsExtraction):
    therapy = extracted.therapy or TherapyURLs()
    handler_urls = _normalize_urls(therapy.handler_urls)
    animal_urls = _normalize_urls(therapy.animal_urls)
    admin_urls = _normalize_urls(therapy.admin_urls)

    therapy_node = evaluator.add_parallel(
        id="Therapy_Animal_Certification_Track",
        desc="Document Pet Partners therapy animal certification requirements (handler, animal, and administrative), with official URLs per category.",
        parent=parent_node,
        critical=True,
    )

    # Handler Category
    handler_node = evaluator.add_parallel(
        id="Handler_Eligibility_Category",
        desc="Handler eligibility requirements + official reference URL(s).",
        parent=therapy_node,
        critical=True,
    )
    # Reference URL presence + official domain
    evaluator.add_custom_node(
        result=(len(handler_urls) > 0 and _has_official(handler_urls, is_petpartners)),
        id="Handler_Eligibility_Reference_URL",
        desc="Provide official Pet Partners URL(s) documenting handler eligibility requirements.",
        parent=handler_node,
        critical=True,
    )
    # Minimum age 10
    min_age_node = evaluator.add_leaf(
        id="Handler_Minimum_Age",
        desc="Handler must be at least 10 years of age.",
        parent=handler_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Pet Partners requires that therapy animal handlers be at least 10 years old (minimum age 10).",
        node=min_age_node,
        sources=handler_urls,
        additional_instruction="Verify on Pet Partners official pages. Allow equivalent phrasing like '10 years or older' or 'junior handler age minimum is 10'."
    )
    # Renewal 40-question assessment
    renew_assess_node = evaluator.add_leaf(
        id="Handler_Renewal_Assessment",
        desc="Renewing handlers must complete a 40-question Renewing Pet Partners Handler Knowledge Assessment.",
        parent=handler_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Renewing Pet Partners handlers must complete a 40‑question Handler Knowledge Assessment.",
        node=renew_assess_node,
        sources=handler_urls,
        additional_instruction="Confirm on Pet Partners official renewal/handler documentation that the renewing knowledge assessment has 40 questions."
    )

    # Animal Requirements Category
    animal_node = evaluator.add_parallel(
        id="Animal_Requirements_Category",
        desc="Animal requirements + official reference URL(s).",
        parent=therapy_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(animal_urls) > 0 and _has_official(animal_urls, is_petpartners)),
        id="Animal_Requirements_Reference_URL",
        desc="Provide official Pet Partners URL(s) documenting animal requirements.",
        parent=animal_node,
        critical=True,
    )
    # Eligible species
    species_node = evaluator.add_leaf(
        id="Eligible_Species",
        desc="Eligible species are: dogs, cats, horses, rabbits, guinea pigs, rats, birds, miniature pigs, and llamas/alpacas.",
        parent=animal_node,
        critical=True,
    )
    await evaluator.verify(
        claim=("Pet Partners recognizes the following species as eligible for therapy animal teams: "
               "dogs, cats, horses, rabbits, guinea pigs, rats, birds, miniature pigs, and llamas/alpacas."),
        node=species_node,
        sources=animal_urls,
        additional_instruction="Allow minor list formatting variants or splitting llama/alpaca into separate entries."
    )
    # Animal Health Screening form at renewal
    health_form_node = evaluator.add_leaf(
        id="Animal_Health_Screening_Form",
        desc="Renewal requires an updated Animal Health Screening form.",
        parent=animal_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Pet Partners renewal requires an updated Animal Health Screening form completed by a veterinarian.",
        node=health_form_node,
        sources=animal_urls,
        additional_instruction="Confirm that at renewal an updated veterinary health screening form is required."
    )
    # Team evaluation at renewal
    team_eval_node = evaluator.add_leaf(
        id="Team_Evaluation",
        desc="Renewal requires a team evaluation.",
        parent=animal_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Pet Partners renewal requires a team evaluation (re‑evaluation) of the handler‑animal team.",
        node=team_eval_node,
        sources=animal_urls,
        additional_instruction="Confirm that a team evaluation is required at renewal."
    )

    # Administrative Requirements Category
    admin_node = evaluator.add_parallel(
        id="Administrative_Requirements_Category",
        desc="Administrative requirements + official reference URL(s).",
        parent=therapy_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(admin_urls) > 0 and _has_official(admin_urls, is_petpartners)),
        id="Administrative_Requirements_Reference_URL",
        desc="Provide official Pet Partners URL(s) documenting fees and insurance.",
        parent=admin_node,
        critical=True,
    )
    # Registration fee $70 / 2 years
    fee_node = evaluator.add_leaf(
        id="Registration_Fee",
        desc="Renewal registration fee is $70 for a 2-year period.",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The Pet Partners renewal registration fee is $70 for a two‑year registration period.",
        node=fee_node,
        sources=admin_urls,
        additional_instruction="Confirm fee and 2‑year term on an official Pet Partners page."
    )
    # Insurance coverage
    insurance_node = evaluator.add_leaf(
        id="Insurance_Coverage",
        desc="Pet Partners registration includes commercial general liability insurance coverage.",
        parent=admin_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Active Pet Partners registration includes commercial general liability insurance coverage for volunteer activities.",
        node=insurance_node,
        sources=admin_urls,
        additional_instruction="Look for language about liability insurance included with registration."
    )


async def build_wildlife_track(evaluator: Evaluator, parent_node, extracted: ProgramURLsExtraction):
    wildlife = extracted.wildlife or WildlifeURLs()
    provisional_urls = _normalize_urls(wildlife.provisional_urls)
    full_urls = _normalize_urls(wildlife.full_urls)
    renewal_urls = _normalize_urls(wildlife.renewal_urls)

    wildlife_node = evaluator.add_parallel(
        id="Wildlife_Rehabilitation_Licensing_Track",
        desc="Document Colorado wildlife rehabilitation licensing (provisional requirements, full upgrade requirements, renewal deadlines) with official Colorado regulation URLs per category.",
        parent=parent_node,
        critical=True,
    )

    # Provisional License Category
    prov_node = evaluator.add_parallel(
        id="Provisional_License_Category",
        desc="Provisional Wildlife Rehabilitator License requirements + official reference URL(s).",
        parent=wildlife_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(provisional_urls) > 0 and _has_official(provisional_urls, is_colorado_official)),
        id="Provisional_Reference_URL",
        desc="Provide official Colorado regulation URL(s) documenting provisional requirements.",
        parent=prov_node,
        critical=True,
    )
    # Age 18
    prov_age_node = evaluator.add_leaf(
        id="Provisional_Minimum_Age",
        desc="Provisional applicants must be at least 18 years of age.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado Provisional Wildlife Rehabilitator applicants must be at least 18 years of age.",
        node=prov_age_node,
        sources=provisional_urls,
        additional_instruction="Verify on official Colorado regulations or agency pages (CPW/Colorado.gov)."
    )
    # Sponsor
    prov_sponsor_node = evaluator.add_leaf(
        id="Provisional_Sponsor",
        desc="Provisional rehabilitators must be sponsored by a licensed Colorado Wildlife Rehabilitator for the same species.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado provisional wildlife rehabilitators must be sponsored by a licensed Colorado wildlife rehabilitator for the same species.",
        node=prov_sponsor_node,
        sources=provisional_urls,
        additional_instruction="Look for sponsor requirement in the regulation text."
    )
    # Facility #1404
    prov_facility_node = evaluator.add_leaf(
        id="Provisional_Facility_1404",
        desc="Provisional applicants must possess an on-site holding facility meeting regulation #1404 criteria.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Provisional applicants must have an on‑site holding facility that meets the standards in regulation #1404.",
        node=prov_facility_node,
        sources=provisional_urls,
        additional_instruction="Confirm the facility standards reference (#1404) is required for provisional licensing."
    )
    # Application + DVM letter
    prov_app_dvm_node = evaluator.add_leaf(
        id="Provisional_Application_and_DVM_Letter",
        desc="Provisional applicants must submit a completed written application and a DVM letter.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado provisional wildlife rehabilitator applicants must submit a completed application and a veterinarian (DVM) letter.",
        node=prov_app_dvm_node,
        sources=provisional_urls,
        additional_instruction="Confirm the application and DVM letter documentation requirement."
    )
    # Learning Plan
    prov_learning_plan_node = evaluator.add_leaf(
        id="Provisional_Learning_Plan",
        desc="Provisional applicants must submit a Learning Plan approved and signed by sponsor.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Provisional applicants must submit a sponsor‑approved and signed Learning Plan.",
        node=prov_learning_plan_node,
        sources=provisional_urls,
        additional_instruction="Verify Learning Plan requirement language in official rules."
    )
    # Basic curriculum
    prov_basic_curr_node = evaluator.add_leaf(
        id="Provisional_Basic_Curriculum",
        desc="Provisional Wildlife Rehabilitators must complete Division-approved basic wildlife rehabilitation curriculum prior to second renewal.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Provisional wildlife rehabilitators must complete a Division‑approved basic wildlife rehabilitation curriculum prior to their second renewal.",
        node=prov_basic_curr_node,
        sources=provisional_urls,
        additional_instruction="Locate curriculum requirement in the regulation or official CPW guidance."
    )
    # Facility inspection
    prov_inspect_node = evaluator.add_leaf(
        id="Provisional_Facility_Inspection",
        desc="Facilities must be inspected by Colorado Division of Wildlife prior to license issuance.",
        parent=prov_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Facilities must be inspected by the Colorado Division of Wildlife (CPW) prior to issuance of a wildlife rehabilitator license.",
        node=prov_inspect_node,
        sources=provisional_urls,
        additional_instruction="Confirm inspection requirement."
    )

    # Full License Upgrade Category
    full_node = evaluator.add_parallel(
        id="Full_License_Upgrade_Category",
        desc="Additional requirements to upgrade from provisional to full Wildlife Rehabilitator License + official reference URL(s).",
        parent=wildlife_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(full_urls) > 0 and _has_official(full_urls, is_colorado_official)),
        id="Full_License_Reference_URL",
        desc="Provide official Colorado regulation URL(s) documenting full license upgrade requirements.",
        parent=full_node,
        critical=True,
    )
    # Age 18
    full_age_node = evaluator.add_leaf(
        id="Full_License_Minimum_Age",
        desc="Full Wildlife Rehabilitator License applicants must be at least 18 years of age.",
        parent=full_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Applicants for a full Colorado Wildlife Rehabilitator License must be at least 18 years of age.",
        node=full_age_node,
        sources=full_urls,
        additional_instruction="Confirm age requirement for full license."
    )
    # Minimum 1 year experience as provisional
    full_exp_node = evaluator.add_leaf(
        id="Full_License_Min_Experience",
        desc="Full Wildlife Rehabilitator License requires minimum 1 year experience as Provisional Wildlife Rehabilitator.",
        parent=full_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Upgrading to a full Colorado Wildlife Rehabilitator License requires at least one year of experience as a Provisional Wildlife Rehabilitator.",
        node=full_exp_node,
        sources=full_urls,
        additional_instruction="Verify minimum experience duration in official rules."
    )
    # Completed Learning Plan with dates
    full_lp_dates_node = evaluator.add_leaf(
        id="Full_License_Learning_Plan_With_Dates",
        desc="Full license applicants must submit completed Learning Plan with dates signed by sponsor.",
        parent=full_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Full license applicants must submit a completed Learning Plan with dates, signed by the sponsor.",
        node=full_lp_dates_node,
        sources=full_urls,
        additional_instruction="Look for the requirement to submit a completed, dated and signed Learning Plan."
    )

    # Renewal Deadlines Category
    renewal_node = evaluator.add_parallel(
        id="Renewal_Deadlines_Category",
        desc="Wildlife rehabilitator renewal deadlines/expiration + official reference URL(s).",
        parent=wildlife_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(renewal_urls) > 0 and _has_official(renewal_urls, is_colorado_official)),
        id="Renewal_Reference_URL",
        desc="Provide official Colorado regulation URL(s) documenting renewal deadlines/expiration.",
        parent=renewal_node,
        critical=True,
    )
    # Submit by Jan 31
    renewal_submit_node = evaluator.add_leaf(
        id="Renewal_Submission_Deadline",
        desc="Wildlife Rehabilitator license renewal materials must be submitted by January 31.",
        parent=renewal_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado wildlife rehabilitator renewal materials must be submitted by January 31.",
        node=renewal_submit_node,
        sources=renewal_urls,
        additional_instruction="Confirm the annual submission deadline on official rules or guidance."
    )
    # Expire March 31
    renewal_expire_node = evaluator.add_leaf(
        id="Annual_Expiration_Date",
        desc="Wildlife Rehabilitator licenses expire March 31 annually.",
        parent=renewal_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado wildlife rehabilitator licenses expire annually on March 31.",
        node=renewal_expire_node,
        sources=renewal_urls,
        additional_instruction="Confirm the annual expiration date."
    )


async def build_giant_track(evaluator: Evaluator, parent_node, extracted: ProgramURLsExtraction):
    giant = extracted.giant or GiantURLs()
    hip_urls = _normalize_urls(giant.hip_urls)
    thyroid_urls = _normalize_urls(giant.thyroid_urls)
    eye_urls = _normalize_urls(giant.eye_urls)
    db_urls = _normalize_urls(giant.database_urls)

    giant_node = evaluator.add_parallel(
        id="Giant_Schnauzer_Health_Testing_Track",
        desc="Document CHIC health testing requirements for Giant Schnauzers, with official GSCA or OFA URLs per testing category.",
        parent=parent_node,
        critical=True,
    )

    # Hip category
    hip_node = evaluator.add_parallel(
        id="Hip_Dysplasia_Category",
        desc="Hip dysplasia screening requirement + official reference URL(s).",
        parent=giant_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(hip_urls) > 0 and _has_official(hip_urls, is_ofa_or_gsca)),
        id="Hip_Reference_URL",
        desc="Provide official GSCA or OFA URL(s) documenting hip screening requirements.",
        parent=hip_node,
        critical=True,
    )
    hip_methods_node = evaluator.add_leaf(
        id="Hip_Acceptable_Methods",
        desc="Hip dysplasia screening must be via OFA, PennHIP, GDC, or OVC for CHIC certification.",
        parent=hip_node,
        critical=True,
    )
    await evaluator.verify(
        claim="For Giant Schnauzer CHIC, acceptable hip dysplasia screening methods include OFA, PennHIP, GDC, or OVC.",
        node=hip_methods_node,
        sources=hip_urls,
        additional_instruction="Verify on OFA/CHIC or GSCA reference pages; allow minor naming variants for program names."
    )

    # Thyroid category
    thyroid_node = evaluator.add_parallel(
        id="Thyroid_Category",
        desc="Thyroid certification requirement + official reference URL(s).",
        parent=giant_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(thyroid_urls) > 0 and _has_official(thyroid_urls, is_ofa_or_gsca)),
        id="Thyroid_Reference_URL",
        desc="Provide official GSCA or OFA URL(s) documenting thyroid requirements.",
        parent=thyroid_node,
        critical=True,
    )
    thyroid_req_node = evaluator.add_leaf(
        id="Thyroid_OFA_Protocol",
        desc="Thyroid certification must be via OFA Protocol for CHIC certification.",
        parent=thyroid_node,
        critical=True,
    )
    await evaluator.verify(
        claim="For Giant Schnauzer CHIC, thyroid certification must follow the OFA Thyroid Certification Protocol.",
        node=thyroid_req_node,
        sources=thyroid_urls,
        additional_instruction="Confirm requirement for OFA thyroid protocol for CHIC."
    )

    # Eye exam category
    eye_node = evaluator.add_parallel(
        id="Eye_Exam_Category",
        desc="Eye examination requirement + official reference URL(s).",
        parent=giant_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(eye_urls) > 0 and _has_official(eye_urls, is_ofa_or_gsca)),
        id="Eye_Reference_URL",
        desc="Provide official GSCA or OFA URL(s) documenting eye exam requirements.",
        parent=eye_node,
        critical=True,
    )
    eye_req_node = evaluator.add_leaf(
        id="CERF_Eye_Exam",
        desc="Eye examination must be a CERF eye examination for CHIC certification.",
        parent=eye_node,
        critical=True,
    )
    await evaluator.verify(
        claim="For Giant Schnauzer CHIC, the required eye examination is a CERF eye exam (historically; current CAER equivalents may be noted).",
        node=eye_req_node,
        sources=eye_urls,
        additional_instruction="Allow historical CERF/modern CAER equivalence when the page indicates continuity."
    )

    # Database publication category
    db_node = evaluator.add_parallel(
        id="Database_Publication_Category",
        desc="Database publication requirement + official reference URL(s).",
        parent=giant_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(db_urls) > 0 and _has_official(db_urls, is_ofa_or_gsca)),
        id="Database_Reference_URL",
        desc="Provide official GSCA or OFA URL(s) documenting the database publication requirement.",
        parent=db_node,
        critical=True,
    )
    db_pub_node = evaluator.add_leaf(
        id="OFA_Publication",
        desc="All Giant Schnauzer CHIC test results must be publicly available in the OFA database.",
        parent=db_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Giant Schnauzer CHIC test results must be publicly available in the OFA database for CHIC certification.",
        node=db_pub_node,
        sources=db_urls,
        additional_instruction="Confirm CHIC requires public availability of results in OFA database."
    )


async def build_shelter_track(evaluator: Evaluator, parent_node, extracted: ProgramURLsExtraction):
    shelter = extracted.shelter or ShelterURLs()
    contact_urls = _normalize_urls(shelter.contact_urls)
    hours_urls = _normalize_urls(shelter.hours_urls)
    services_urls = _normalize_urls(shelter.services_fees_urls)

    shelter_node = evaluator.add_parallel(
        id="Local_Shelter_Partnership_Track",
        desc="Document NOCO Humane Larimer Campus details (address, phone, hours, services offered, and dog/cat adoption fee ranges) with official NOCO Humane URLs.",
        parent=parent_node,
        critical=True,
    )

    # Address and Phone Category
    contact_node = evaluator.add_parallel(
        id="Address_and_Phone_Category",
        desc="Address and phone + official reference URL(s).",
        parent=shelter_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(contact_urls) > 0 and _has_official(contact_urls, is_noco_humane)),
        id="Contact_Reference_URL",
        desc="Provide official NOCO Humane URL(s) documenting address and phone.",
        parent=contact_node,
        critical=True,
    )
    # Address
    address_node = evaluator.add_leaf(
        id="Physical_Address",
        desc="NOCO Humane Larimer Campus is located at 3501 E 71st Street, Loveland, CO 80538.",
        parent=contact_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The NOCO Humane Larimer Campus physical address is 3501 E 71st Street, Loveland, CO 80538.",
        node=address_node,
        sources=contact_urls,
        additional_instruction="Verify address exactly or with minor formatting differences (e.g., 'E.' vs 'East')."
    )
    # Phone
    phone_node = evaluator.add_leaf(
        id="Phone_Number",
        desc="NOCO Humane Larimer Campus phone number is 970-226-3647.",
        parent=contact_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The NOCO Humane Larimer Campus phone number is 970-226-3647.",
        node=phone_node,
        sources=contact_urls,
        additional_instruction="Allow reasonable formatting variants like (970) 226‑3647."
    )

    # Adoption Hours Category
    hours_node = evaluator.add_parallel(
        id="Adoption_Hours_Category",
        desc="Adoption hours (weekday vs weekend) + official reference URL(s).",
        parent=shelter_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(hours_urls) > 0 and _has_official(hours_urls, is_noco_humane)),
        id="Hours_Reference_URL",
        desc="Provide official NOCO Humane URL(s) documenting adoption hours.",
        parent=hours_node,
        critical=True,
    )
    weekday_node = evaluator.add_leaf(
        id="Weekday_Hours",
        desc="Adoption hours Monday–Friday are 12:00pm to 6:00pm.",
        parent=hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NOCO Humane Larimer Campus adoption hours Monday through Friday are 12:00 pm to 6:00 pm.",
        node=weekday_node,
        sources=hours_urls,
        additional_instruction="Verify weekday adoption hours; minor formatting differences acceptable."
    )
    weekend_node = evaluator.add_leaf(
        id="Weekend_Hours",
        desc="Adoption hours Saturday–Sunday are 10:00am to 5:00pm.",
        parent=hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NOCO Humane Larimer Campus adoption hours Saturday and Sunday are 10:00 am to 5:00 pm.",
        node=weekend_node,
        sources=hours_urls,
        additional_instruction="Verify weekend adoption hours."
    )

    # Services and Fees Category
    services_node = evaluator.add_parallel(
        id="Services_and_Fees_Category",
        desc="Services offered + adoption fee ranges (dogs and cats separately) + official reference URL(s).",
        parent=shelter_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(services_urls) > 0 and _has_official(services_urls, is_noco_humane)),
        id="Services_and_Fees_Reference_URL",
        desc="Provide official NOCO Humane URL(s) documenting services and adoption fee ranges.",
        parent=services_node,
        critical=True,
    )
    # Services offered described (coarse check)
    services_desc_node = evaluator.add_leaf(
        id="Services_Offered_Described",
        desc="Provide a description/list of services offered at NOCO Humane Larimer Campus.",
        parent=services_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The official NOCO Humane Larimer Campus page lists services offered at the campus, including adoption services.",
        node=services_desc_node,
        sources=services_urls,
        additional_instruction="It's sufficient to confirm that the page describes services offered (e.g., adoptions, lost & found, intake, etc.)."
    )
    # Dog adoption fee range
    dog_fee_node = evaluator.add_leaf(
        id="Dog_Adoption_Fee_Range",
        desc="NOCO Humane dog adoption fees range from $75 to $750.",
        parent=services_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NOCO Humane dog adoption fees range from $75 to $750.",
        node=dog_fee_node,
        sources=services_urls,
        additional_instruction="Verify fee range for dogs; minor text formatting differences acceptable."
    )
    # Cat adoption fee range
    cat_fee_node = evaluator.add_leaf(
        id="Cat_Adoption_Fee_Range",
        desc="NOCO Humane cat adoption fees range from $25 to $150.",
        parent=services_node,
        critical=True,
    )
    await evaluator.verify(
        claim="NOCO Humane cat adoption fees range from $25 to $150.",
        node=cat_fee_node,
        sources=services_urls,
        additional_instruction="Verify fee range for cats."
    )


async def build_veterinary_track(evaluator: Evaluator, parent_node, extracted: ProgramURLsExtraction):
    vet = extracted.veterinary or VeterinaryURLs()
    ce_urls = _normalize_urls(vet.ce_urls)

    vet_node = evaluator.add_parallel(
        id="Veterinary_Support_Requirements",
        desc="Document Colorado veterinarian CE requirements (hours, 2026 delegation/supervision topic, renewal due date, and CE standards) with official Colorado veterinary board or CVMA URLs.",
        parent=parent_node,
        critical=True,
    )

    ce_cat_node = evaluator.add_parallel(
        id="CE_and_Renewal_Requirements_Category",
        desc="All required CE/renewal constraints + official reference URL(s).",
        parent=vet_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=(len(ce_urls) > 0 and _has_official(ce_urls, is_vet_board_or_cvma)),
        id="Veterinary_CE_Reference_URL",
        desc="Provide official Colorado veterinary board or CVMA URL(s) documenting the CE/renewal requirements and CE standards.",
        parent=ce_cat_node,
        critical=True,
    )
    # Total CE hours
    ce_total_node = evaluator.add_leaf(
        id="Total_CE_Hours",
        desc="Colorado veterinarians must complete 32 hours of CE per 2-year renewal period.",
        parent=ce_cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado veterinarians must complete 32 hours of continuing education every two‑year renewal period.",
        node=ce_total_node,
        sources=ce_urls,
        additional_instruction="Confirm on DPO/Board or CVMA pages for Colorado."
    )
    # Delegation and supervision topic (2026)
    ce_delegation_node = evaluator.add_leaf(
        id="Delegation_and_Supervision_Topic_2026",
        desc="For 2026 renewal, Colorado veterinarians must complete 2 hours of CE on delegation and supervision topics.",
        parent=ce_cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim="For the 2026 renewal, Colorado veterinarians must complete 2 hours of CE on delegation and supervision topics.",
        node=ce_delegation_node,
        sources=ce_urls,
        additional_instruction="Look for special topic requirement for the 2026 renewal."
    )
    # Renewal due date
    renewal_due_node = evaluator.add_leaf(
        id="Renewal_Due_Date",
        desc="Colorado veterinary license renewal is due October 31 of even-numbered years.",
        parent=ce_cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado veterinary license renewal is due on October 31 of even‑numbered years.",
        node=renewal_due_node,
        sources=ce_urls,
        additional_instruction="Confirm the due date cadence."
    )
    # CE course standards
    ce_standards_node = evaluator.add_leaf(
        id="CE_Course_Standards",
        desc="Colorado veterinary CE must be Board-approved or RACE-approved.",
        parent=ce_cat_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Colorado veterinary CE must be Board‑approved or AAVSB RACE‑approved.",
        node=ce_standards_node,
        sources=ce_urls,
        additional_instruction="Verify course accreditation/approval standards in Colorado."
    )


# --------------------------------------------------------------------------- #
# Root builder                                                                #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ProgramURLsExtraction):
    program_node = evaluator.add_parallel(
        id="Comprehensive_Animal_Welfare_Program_Requirements",
        desc="Verify all required tracks and required citations for the integrated Larimer County animal welfare/therapy program.",
        parent=evaluator.root,
        critical=True,
    )

    await build_therapy_track(evaluator, program_node, extracted)
    await build_wildlife_track(evaluator, program_node, extracted)
    await build_giant_track(evaluator, program_node, extracted)
    await build_shelter_track(evaluator, program_node, extracted)
    await build_veterinary_track(evaluator, program_node, extracted)


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
        default_model=model,
    )

    # Extract grouped official URLs from the answer
    extracted_urls = await evaluator.extract(
        prompt=prompt_extract_program_urls(),
        template_class=ProgramURLsExtraction,
        extraction_name="program_official_urls",
    )

    # Build verification tree and execute checks
    await build_verification_tree(evaluator, extracted_urls)

    # Return structured summary
    return evaluator.get_summary()