import asyncio
import logging
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oh_columbus_gov_services"
TASK_DESCRIPTION = """
I am a resident of Columbus, Ohio, and need to handle several government service tasks. Please provide comprehensive information for each of the following five services:

1. Passport Application: Identify a facility in Columbus or Franklin County where I can apply for a first-time U.S. passport in person. Include the facility name, complete street address, and a link to an official government website with more information.

2. Apostille Service: Identify the Ohio state office that provides apostille certification for documents that will be used internationally. Include the office name, complete street address in Columbus, the fee per document, and a link to an official government website with more information.

3. Small Claims Court: Identify the court in Franklin County where I can file a small claims case for a dispute involving less than $6,000. Include the court name, complete street address in Columbus, and a link to an official government website with more information.

4. Federal Court: Identify the U.S. District Court facility in Columbus where federal civil cases are filed. Include the courthouse name, complete street address, and a link to an official government website with more information.

5. TSA PreCheck: Provide information on how to enroll in TSA PreCheck in the Columbus, Ohio area. Include where enrollment can be completed, the fee for a 5-year membership, and a link to an official government website with more information.

For each service, please ensure all addresses include the street address, city, state, and zip code, and that all reference links point to official government sources (.gov domains).
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PassportFacility(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    reference_url: Optional[str] = None


class ApostilleService(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    fee_per_document: Optional[str] = None
    reference_url: Optional[str] = None


class SmallClaimsCourt(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    reference_url: Optional[str] = None


class FederalCourthouse(BaseModel):
    name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    reference_url: Optional[str] = None


class TSAPreCheck(BaseModel):
    enrollment_location_name: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    fee_5yr: Optional[str] = None
    reference_url: Optional[str] = None


class GovServicesExtraction(BaseModel):
    passport: Optional[PassportFacility] = None
    apostille: Optional[ApostilleService] = None
    small_claims: Optional[SmallClaimsCourt] = None
    federal_court: Optional[FederalCourthouse] = None
    tsa_precheck: Optional[TSAPreCheck] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_services() -> str:
    return """
Extract the information the answer provides for each of the five requested government services. Only extract what is explicitly present in the answer text; do not invent anything. For each item, split the address into street, city, state, and zip code if possible. If a field is missing in the answer, set it to null. For URLs, extract exactly the URL(s) shown in the answer (plain link or markdown link). Do not infer or construct URLs.

Return a JSON object with this exact structure and field names:

{
  "passport": {
    "name": string | null,
    "street": string | null,
    "city": string | null,
    "state": string | null,
    "zip_code": string | null,
    "reference_url": string | null
  },
  "apostille": {
    "name": string | null,
    "street": string | null,
    "city": string | null,
    "state": string | null,
    "zip_code": string | null,
    "fee_per_document": string | null,
    "reference_url": string | null
  },
  "small_claims": {
    "name": string | null,
    "street": string | null,
    "city": string | null,
    "state": string | null,
    "zip_code": string | null,
    "reference_url": string | null
  },
  "federal_court": {
    "name": string | null,
    "street": string | null,
    "city": string | null,
    "state": string | null,
    "zip_code": string | null,
    "reference_url": string | null
  },
  "tsa_precheck": {
    "enrollment_location_name": string | null,
    "street": string | null,
    "city": string | null,
    "state": string | null,
    "zip_code": string | null,
    "fee_5yr": string | null,
    "reference_url": string | null
  }
}

Guidelines:
- For addresses, keep the fields exactly as written in the answer.
- For fees (fee_per_document, fee_5yr), keep the exact string from the answer (e.g., "$5", "$78", "USD 78").
- For URLs, extract only those explicitly shown; include the protocol. If the answer shows multiple URLs for an item, select the most official one if any, otherwise the first one mentioned for that item.
""".strip()


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _address_complete(street: Optional[str], city: Optional[str], state: Optional[str], zip_code: Optional[str]) -> bool:
    return all(_nonempty(x) for x in [street, city, state, zip_code])


def _is_gov_url(url: Optional[str]) -> bool:
    if not _nonempty(url):
        return False
    try:
        host = urlparse(str(url)).netloc.lower()
        # Accept subdomains that end with .gov
        return host.endswith(".gov")
    except Exception:
        return False


def _full_address(street: Optional[str], city: Optional[str], state: Optional[str], zip_code: Optional[str]) -> str:
    street_s = street or ""
    city_s = city or ""
    state_s = state or ""
    zip_s = zip_code or ""
    return f"{street_s}, {city_s}, {state_s} {zip_s}".strip().strip(",")


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_passport(evaluator: Evaluator, parent, data: Optional[PassportFacility]) -> None:
    node = evaluator.add_sequential(
        id="passport_facility",
        desc="Passport acceptance facility in Columbus or Franklin County with address and official source",
        parent=parent,
        critical=False
    )
    name = data.name if data else None
    street = data.street if data else None
    city = data.city if data else None
    state = data.state if data else None
    zip_code = data.zip_code if data else None
    url = data.reference_url if data else None

    # Required fields (name, complete address, url)
    evaluator.add_custom_node(
        result=_nonempty(name) and _address_complete(street, city, state, zip_code) and _nonempty(url),
        id="passport_required_info",
        desc="Passport facility has name, complete address (street/city/state/zip), and a reference URL",
        parent=node,
        critical=True
    )

    # .gov domain required
    evaluator.add_custom_node(
        result=_is_gov_url(url),
        id="passport_gov_url",
        desc="Passport facility reference URL is a .gov official source",
        parent=node,
        critical=True
    )

    # Content checks (parallel, non-critical to allow partial credit across checks)
    content = evaluator.add_parallel(
        id="passport_content_checks",
        desc="Passport facility content checks",
        parent=node,
        critical=False
    )

    # Verify facility is a passport acceptance facility allowing first-time in-person applications
    leaf_accept = evaluator.add_leaf(
        id="passport_is_acceptance_facility",
        desc="Source indicates this is a passport acceptance facility (accepts first-time in-person applications)",
        parent=content,
        critical=False
    )
    claim_accept = (
        f"The provided official page indicates that '{name}' is a passport acceptance facility where first-time U.S. "
        f"passport applications can be submitted in person."
    )
    await evaluator.verify(
        claim=claim_accept,
        node=leaf_accept,
        sources=url,
        additional_instruction="Look for terms like 'Passport Acceptance Facility', 'accepts passport applications', or instructions for in-person first-time applications."
    )

    # Verify address matches
    leaf_addr = evaluator.add_leaf(
        id="passport_address_matches",
        desc="The facility address matches the address on the official page",
        parent=content,
        critical=False
    )
    full_addr = _full_address(street, city, state, zip_code)
    claim_addr = f"The address of '{name}' is '{full_addr}'."
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=url,
        additional_instruction="Allow minor formatting variations (e.g., abbreviations like St. vs Street, ZIP+4). Match on substance."
    )


async def verify_apostille(evaluator: Evaluator, parent, data: Optional[ApostilleService]) -> None:
    node = evaluator.add_sequential(
        id="apostille_service",
        desc="Ohio state apostille certification office with address, fee, and official source",
        parent=parent,
        critical=False
    )
    name = data.name if data else None
    street = data.street if data else None
    city = data.city if data else None
    state = data.state if data else None
    zip_code = data.zip_code if data else None
    fee = data.fee_per_document if data else None
    url = data.reference_url if data else None

    evaluator.add_custom_node(
        result=_nonempty(name) and _address_complete(street, city, state, zip_code) and _nonempty(fee) and _nonempty(url),
        id="apostille_required_info",
        desc="Apostille office has name, complete address (street/city/state/zip), fee per document, and a reference URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_gov_url(url),
        id="apostille_gov_url",
        desc="Apostille reference URL is a .gov official source",
        parent=node,
        critical=True
    )

    content = evaluator.add_parallel(
        id="apostille_content_checks",
        desc="Apostille content checks",
        parent=node,
        critical=False
    )

    leaf_office = evaluator.add_leaf(
        id="apostille_office_correct",
        desc="Source confirms it is the Ohio state office providing apostille certification",
        parent=content,
        critical=False
    )
    claim_office = (
        "This official page is for the Ohio state government office responsible for apostille (authentication) "
        "certification of documents for international use."
    )
    await evaluator.verify(
        claim=claim_office,
        node=leaf_office,
        sources=url,
        additional_instruction="Look for 'apostille', 'authentication', or certification of documents for use in foreign countries."
    )

    leaf_addr = evaluator.add_leaf(
        id="apostille_address_matches",
        desc="The apostille office address matches the address on the official page",
        parent=content,
        critical=False
    )
    full_addr = _full_address(street, city, state, zip_code)
    claim_addr = f"The address of '{name}' is '{full_addr}'."
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=url,
        additional_instruction="Allow minor formatting differences; verify the same physical location."
    )

    leaf_fee = evaluator.add_leaf(
        id="apostille_fee_matches",
        desc="The apostille fee per document matches the official page",
        parent=content,
        critical=False
    )
    claim_fee = f"The fee per document for an apostille/authentication is {fee}."
    await evaluator.verify(
        claim=claim_fee,
        node=leaf_fee,
        sources=url,
        additional_instruction="Check fee schedule or payment information; accept minor format variants like $5 vs 5 USD."
    )


async def verify_small_claims(evaluator: Evaluator, parent, data: Optional[SmallClaimsCourt]) -> None:
    node = evaluator.add_sequential(
        id="small_claims_court",
        desc="Franklin County small claims filing court with address and official source",
        parent=parent,
        critical=False
    )
    name = data.name if data else None
    street = data.street if data else None
    city = data.city if data else None
    state = data.state if data else None
    zip_code = data.zip_code if data else None
    url = data.reference_url if data else None

    evaluator.add_custom_node(
        result=_nonempty(name) and _address_complete(street, city, state, zip_code) and _nonempty(url),
        id="small_claims_required_info",
        desc="Small claims court has name, complete address (street/city/state/zip), and a reference URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_gov_url(url),
        id="small_claims_gov_url",
        desc="Small claims reference URL is a .gov official source",
        parent=node,
        critical=True
    )

    content = evaluator.add_parallel(
        id="small_claims_content_checks",
        desc="Small claims content checks",
        parent=node,
        critical=False
    )

    leaf_role = evaluator.add_leaf(
        id="small_claims_role_confirm",
        desc="Source confirms this is the Franklin County court where small claims can be filed",
        parent=content,
        critical=False
    )
    claim_role = (
        "This official page indicates this is the Franklin County, Ohio court where small claims cases can be filed."
    )
    await evaluator.verify(
        claim=claim_role,
        node=leaf_role,
        sources=url,
        additional_instruction="Look for 'Small Claims' filing information for Franklin County residents."
    )

    leaf_limit = evaluator.add_leaf(
        id="small_claims_limit_6000",
        desc="Source confirms small claims jurisdiction is under or up to $6,000",
        parent=content,
        critical=False
    )
    claim_limit = "This court's small claims jurisdiction covers claims for $6,000 or less (i.e., under or up to $6,000)."
    await evaluator.verify(
        claim=claim_limit,
        node=leaf_limit,
        sources=url,
        additional_instruction="Accept phrasing like 'claims of $6,000 or less' or '$6,000 maximum'."
    )

    leaf_addr = evaluator.add_leaf(
        id="small_claims_address_matches",
        desc="The court address matches the address on the official page",
        parent=content,
        critical=False
    )
    full_addr = _full_address(street, city, state, zip_code)
    claim_addr = f"The address of '{name}' is '{full_addr}'."
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=url,
        additional_instruction="Allow minor formatting differences and ZIP+4."
    )


async def verify_federal_court(evaluator: Evaluator, parent, data: Optional[FederalCourthouse]) -> None:
    node = evaluator.add_sequential(
        id="federal_district_courthouse",
        desc="U.S. District Court facility in Columbus for federal civil filings, with address and official source",
        parent=parent,
        critical=False
    )
    name = data.name if data else None
    street = data.street if data else None
    city = data.city if data else None
    state = data.state if data else None
    zip_code = data.zip_code if data else None
    url = data.reference_url if data else None

    evaluator.add_custom_node(
        result=_nonempty(name) and _address_complete(street, city, state, zip_code) and _nonempty(url),
        id="federal_required_info",
        desc="Federal court has name, complete address (street/city/state/zip), and a reference URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_gov_url(url),
        id="federal_gov_url",
        desc="Federal court reference URL is a .gov official source",
        parent=node,
        critical=True
    )

    content = evaluator.add_parallel(
        id="federal_content_checks",
        desc="Federal court content checks",
        parent=node,
        critical=False
    )

    leaf_role = evaluator.add_leaf(
        id="federal_is_district_court",
        desc="Source confirms this is the U.S. District Court facility in Columbus handling civil filings",
        parent=content,
        critical=False
    )
    claim_role = (
        "This official page indicates this is the U.S. District Court (federal) facility located in Columbus, Ohio, "
        "where civil cases are filed."
    )
    await evaluator.verify(
        claim=claim_role,
        node=leaf_role,
        sources=url,
        additional_instruction="Look for Southern District of Ohio or similar and civil case filing info for the Columbus location."
    )

    leaf_addr = evaluator.add_leaf(
        id="federal_address_matches",
        desc="The courthouse address matches the address on the official page",
        parent=content,
        critical=False
    )
    full_addr = _full_address(street, city, state, zip_code)
    claim_addr = f"The address of '{name}' is '{full_addr}'."
    await evaluator.verify(
        claim=claim_addr,
        node=leaf_addr,
        sources=url,
        additional_instruction="Allow minor formatting differences; match substance."
    )


async def verify_tsa_precheck(evaluator: Evaluator, parent, data: Optional[TSAPreCheck]) -> None:
    node = evaluator.add_sequential(
        id="tsa_precheck_info",
        desc="TSA PreCheck enrollment info in Columbus area with location, fee, and official source",
        parent=parent,
        critical=False
    )
    loc_name = data.enrollment_location_name if data else None
    street = data.street if data else None
    city = data.city if data else None
    state = data.state if data else None
    zip_code = data.zip_code if data else None
    fee = data.fee_5yr if data else None
    url = data.reference_url if data else None

    evaluator.add_custom_node(
        result=_nonempty(loc_name) and _address_complete(street, city, state, zip_code) and _nonempty(fee) and _nonempty(url),
        id="tsa_required_info",
        desc="TSA PreCheck has enrollment location (name + complete address), 5-year fee, and a reference URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_gov_url(url),
        id="tsa_gov_url",
        desc="TSA PreCheck reference URL is a .gov official source",
        parent=node,
        critical=True
    )

    content = evaluator.add_parallel(
        id="tsa_content_checks",
        desc="TSA PreCheck content checks",
        parent=node,
        critical=False
    )

    # Fee verification
    leaf_fee = evaluator.add_leaf(
        id="tsa_fee_matches",
        desc="The 5-year TSA PreCheck fee matches the official page",
        parent=content,
        critical=False
    )
    claim_fee = f"The fee for a 5-year TSA PreCheck membership is {fee}."
    await evaluator.verify(
        claim=claim_fee,
        node=leaf_fee,
        sources=url,
        additional_instruction="Check the official TSA or DHS enrollment page for current pricing; allow minor format variants (e.g., $78 vs 78 USD)."
    )

    # Enrollment location verification
    leaf_loc = evaluator.add_leaf(
        id="tsa_location_matches",
        desc="The enrollment location and address appear on the official page",
        parent=content,
        critical=False
    )
    full_addr = _full_address(street, city, state, zip_code)
    claim_loc = f"Enrollment can be completed at '{loc_name}' located at '{full_addr}'."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=url,
        additional_instruction="Verify that this specific enrollment center/location and address are listed or viewable on the official page."
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
    # Initialize evaluator (root is non-critical aggregator)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_services(),
        template_class=GovServicesExtraction,
        extraction_name="gov_services_extraction"
    )

    # Build top-level rubric node (parallel across 5 services)
    services_root = evaluator.add_parallel(
        id="Government_Services_Information",
        desc="Provide complete and accurate information for five different government service facilities in Columbus, Ohio",
        parent=root,
        critical=False
    )

    # Create child nodes for each of the five services and verify
    # We evaluate all five in parallel to speed up
    tasks: List[asyncio.Task] = []

    # Passport
    passport_parent = evaluator.add_sequential(
        id="Passport_Acceptance_Facility",
        desc="Passport acceptance facility verification",
        parent=services_root,
        critical=False
    )
    tasks.append(asyncio.create_task(verify_passport(evaluator, passport_parent, extracted.passport)))

    # Apostille
    apostille_parent = evaluator.add_sequential(
        id="Apostille_Service_Provider",
        desc="Ohio apostille service provider verification",
        parent=services_root,
        critical=False
    )
    tasks.append(asyncio.create_task(verify_apostille(evaluator, apostille_parent, extracted.apostille)))

    # Small Claims
    small_claims_parent = evaluator.add_sequential(
        id="Small_Claims_Court_Facility",
        desc="Franklin County small claims court verification",
        parent=services_root,
        critical=False
    )
    tasks.append(asyncio.create_task(verify_small_claims(evaluator, small_claims_parent, extracted.small_claims)))

    # Federal Court
    federal_parent = evaluator.add_sequential(
        id="Federal_District_Courthouse",
        desc="U.S. District Court (Columbus) verification",
        parent=services_root,
        critical=False
    )
    tasks.append(asyncio.create_task(verify_federal_court(evaluator, federal_parent, extracted.federal_court)))

    # TSA PreCheck
    tsa_parent = evaluator.add_sequential(
        id="TSA_PreCheck_Enrollment_Location",
        desc="TSA PreCheck enrollment verification",
        parent=services_root,
        critical=False
    )
    tasks.append(asyncio.create_task(verify_tsa_precheck(evaluator, tsa_parent, extracted.tsa_precheck)))

    await asyncio.gather(*tasks, return_exceptions=True)

    # Add a small custom info note about the .gov policy we enforced
    evaluator.add_custom_info(
        info={"requirement": "All reference URLs must be .gov", "enforced_by": "domain_check"},
        info_type="policy",
        info_name="gov_link_policy"
    )

    return evaluator.get_summary()