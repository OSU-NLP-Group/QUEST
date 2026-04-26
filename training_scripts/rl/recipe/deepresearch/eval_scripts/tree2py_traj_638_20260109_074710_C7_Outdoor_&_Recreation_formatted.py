import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "tx_hill_parks"
TASK_DESCRIPTION = (
    "Identify three Texas Hill Country state parks that each offer swimming areas, hiking trails, camping facilities, "
    "and picnic areas. For each of the three parks, provide the following information: (1) The official park name as "
    "listed by Texas Parks & Wildlife Department, (2) The complete physical address (street address, city, state, and "
    "ZIP code), (3) The park's direct phone number, and (4) The daily entrance fee for adults aged 13 and older. "
    "All information must be sourced from official Texas Parks & Wildlife Department resources."
)


# ----------------------------- Data Models --------------------------------- #
class ParkInfo(BaseModel):
    name: Optional[str] = None
    address_street: Optional[str] = None
    address_city: Optional[str] = None
    address_state: Optional[str] = None
    address_zip: Optional[str] = None
    full_address: Optional[str] = None
    phone: Optional[str] = None
    adult_fee: Optional[str] = None
    tpwd_primary_url: Optional[str] = None
    tpwd_urls: List[str] = Field(default_factory=list)


class ParksExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_parks() -> str:
    return (
        "Extract up to the first three parks presented in the answer that claim to be Texas Hill Country state parks. "
        "For each park, extract the following fields exactly as stated in the answer:\n"
        "1) name: The official park name as presented (intended to be the TPWD official name)\n"
        "2) address_street: Street address line\n"
        "3) address_city: City name\n"
        "4) address_state: State abbreviation (e.g., TX)\n"
        "5) address_zip: ZIP code\n"
        "6) full_address: If the answer provides the address as a single line, include that full line\n"
        "7) phone: The park's direct phone number\n"
        "8) adult_fee: The daily entrance fee for adults aged 13 and older\n"
        "9) tpwd_primary_url: The primary official TPWD park page URL for this park, if provided (e.g., https://tpwd.texas.gov/state-parks/<park>)\n"
        "10) tpwd_urls: All official TPWD URLs cited for this park in the answer (only include URLs on the tpwd.texas.gov domain; ignore other domains)\n\n"
        "Rules:\n"
        "- Only include URLs explicitly present in the answer. Do not invent or infer URLs.\n"
        "- If a URL lacks protocol, prepend http://.\n"
        "- If any field is missing for a park, set it to null. For tpwd_urls, return an empty array if none are present.\n"
        "- Return exactly a JSON object with 'parks' being an array of up to three ParkInfo objects following the schema."
    )


# ----------------------------- Helper Utils -------------------------------- #
def _compose_full_address(park: ParkInfo) -> Optional[str]:
    if park.full_address and park.full_address.strip():
        return park.full_address.strip()
    parts = []
    if park.address_street:
        parts.append(park.address_street.strip())
    city_state_zip = []
    if park.address_city:
        city_state_zip.append(park.address_city.strip())
    state = park.address_state.strip() if park.address_state else "TX"
    if state:
        city_state_zip.append(state)
    if park.address_zip:
        city_state_zip.append(park.address_zip.strip())
    if parts or city_state_zip:
        city_line = ", ".join(city_state_zip[:-1]) if len(city_state_zip) > 1 else (city_state_zip[0] if city_state_zip else "")
        last_zip = city_state_zip[-1] if len(city_state_zip) >= 1 else ""
        if city_line:
            parts.append(f"{city_line} {last_zip}".strip())
        else:
            if last_zip:
                parts.append(last_zip)
        return ", ".join([p for p in parts if p])
    return None


def _tpwd_urls_for_park(park: ParkInfo) -> List[str]:
    urls = list(park.tpwd_urls or [])
    if park.tpwd_primary_url:
        if park.tpwd_primary_url not in urls:
            urls.insert(0, park.tpwd_primary_url)
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _is_tpwd_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.endswith("tpwd.texas.gov")
    except Exception:
        return False


def _all_tpwd_domains(urls: List[str]) -> bool:
    if not urls:
        return False
    return all(_is_tpwd_domain(u) for u in urls)


# --------------------------- Verification Logic ---------------------------- #
async def verify_park(
    evaluator: Evaluator,
    root_node,
    park: ParkInfo,
    index_one_based: int,
) -> None:
    """
    Build and verify the subtree for one park.
    """
    parent_node = evaluator.add_parallel(
        id=f"Park_{index_one_based}",
        desc=f"Park #{index_one_based} and its required constraints/attributes",
        parent=root_node,
        critical=False,
    )

    urls = _tpwd_urls_for_park(park)
    minimal_info_ok = (park.name is not None and park.name.strip() != "" and len(urls) > 0)

    evaluator.add_custom_node(
        result=minimal_info_ok,
        id=f"park_{index_one_based}_required_fields",
        desc=f"Park #{index_one_based} has at least a name and one TPWD URL to verify",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_all_tpwd_domains(urls),
        id=f"park_{index_one_based}_tpwd_domain_check",
        desc=f"Park #{index_one_based}: all provided URLs are on tpwd.texas.gov (official TPWD)",
        parent=parent_node,
        critical=True
    )

    # 1) Is listed by TPWD as a Texas state park
    listed_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Is_State_Park_Listed_By_TPWD",
        desc=f"The park is a Texas state park listed by TPWD",
        parent=parent_node,
        critical=True
    )
    claim_listed = (
        f"The park '{park.name or ''}' is listed as a Texas state park on the Texas Parks & Wildlife Department website."
    )
    await evaluator.verify(
        claim=claim_listed,
        node=listed_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the page is an official listing on the tpwd.texas.gov domain and clearly identifies the site as "
            "a Texas State Park managed by Texas Parks & Wildlife Department."
        ),
    )

    # 2) Located in Texas Hill Country region
    location_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Location",
        desc="The park is located in the Texas Hill Country region",
        parent=parent_node,
        critical=True
    )
    claim_location = "This park is located in the Texas Hill Country region of Texas."
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=urls,
        additional_instruction=(
            "Only accept if the TPWD page(s) explicitly indicate the park belongs to the 'Hill Country' region or a TPWD region list "
            "includes this park under 'Hill Country'. If not explicitly stated on the provided TPWD sources, judge as not supported."
        ),
    )

    # 3) Amenities: swimming
    swimming_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Swimming",
        desc="The park offers swimming areas (river or lake access)",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This park offers swimming areas or water access where swimming is allowed.",
        node=swimming_leaf,
        sources=urls,
        additional_instruction=(
            "Look for 'swimming' amenity or clear language on the TPWD page indicating swimming is permitted in a lake/river or a designated swimming area."
        ),
    )

    # 4) Amenities: hiking
    hiking_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Hiking",
        desc="The park has hiking trails",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This park has hiking trails.",
        node=hiking_leaf,
        sources=urls,
        additional_instruction="Look for 'hiking' or 'trails' on the TPWD page.",
    )

    # 5) Amenities: camping
    camping_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Camping",
        desc="The park has camping facilities",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This park offers camping facilities or campsites.",
        node=camping_leaf,
        sources=urls,
        additional_instruction="Look for 'camping', 'campsites', or 'campgrounds' on the TPWD page.",
    )

    # 6) Amenities: picnic
    picnic_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Picnic",
        desc="The park has picnic areas",
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim="This park has picnic areas or designated picnicking sites.",
        node=picnic_leaf,
        sources=urls,
        additional_instruction="Look for 'picnic' or 'picnicking' amenities on the TPWD page.",
    )

    # 7) Official park name
    name_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Name",
        desc="Provide the official park name as listed by TPWD",
        parent=parent_node,
        critical=True
    )
    claim_name = f"The official park name on the TPWD page is '{park.name or ''}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=urls,
        additional_instruction=(
            "Match the main heading or official name shown on the TPWD page. Allow minor punctuation or casing variations as equivalent."
        ),
    )

    # 8) Physical address (complete)
    address_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Address",
        desc="Provide the complete physical address (street address, city, state, ZIP code)",
        parent=parent_node,
        critical=True
    )
    full_addr = _compose_full_address(park) or ""
    claim_address = f"The park's physical address is '{full_addr}'."
    await evaluator.verify(
        claim=claim_address,
        node=address_leaf,
        sources=urls,
        additional_instruction=(
            "Verify the street address, city, state (TX), and ZIP code match the TPWD page. Allow minor punctuation or abbreviation differences "
            "but the substantive address components must be equivalent."
        ),
    )

    # 9) Phone number
    phone_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Phone",
        desc="Provide the park's direct phone number",
        parent=parent_node,
        critical=True
    )
    claim_phone = f"The park's direct phone number is '{(park.phone or '').strip()}'."
    await evaluator.verify(
        claim=claim_phone,
        node=phone_leaf,
        sources=urls,
        additional_instruction=(
            "Check the phone number on the TPWD page. Allow formatting variations (hyphens, parentheses, spaces) but the digits must match."
        ),
    )

    # 10) Adult daily entrance fee (13+)
    fee_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_Fee",
        desc="Provide the daily entrance fee for adults aged 13 and older",
        parent=parent_node,
        critical=True
    )
    claim_fee = f"The daily entrance fee for adults (age 13+) is '{(park.adult_fee or '').strip()}'."
    await evaluator.verify(
        claim=claim_fee,
        node=fee_leaf,
        sources=urls,
        additional_instruction=(
            "Verify entrance fees shown on the TPWD page. Look for 'Adult' or 'Age 13 and older' fee category. Minor rounding differences may be acceptable."
        ),
    )

    # 11) TPWD Source Verification (all details from official TPWD)
    tpwd_src_leaf = evaluator.add_leaf(
        id=f"Park_{index_one_based}_TPWD_Source_Verification",
        desc="All provided details for the park are verifiable via official TPWD resources",
        parent=parent_node,
        critical=True
    )
    # Use simple verify to explicitly evaluate domain list; additional instruction enforces domain rule.
    domain_list_str = ", ".join(urls) if urls else ""
    claim_tpwd_src = f"All provided URLs for this park are official Texas Parks & Wildlife Department pages: {domain_list_str}"
    await evaluator.verify(
        claim=claim_tpwd_src,
        node=tpwd_src_leaf,
        sources=None,
        additional_instruction=(
            "Judge whether every provided URL belongs to the official TPWD domain (tpwd.texas.gov). "
            "If any URL is not on tpwd.texas.gov, this claim is incorrect."
        ),
    )


# --------------------------- Main Evaluation -------------------------------- #
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Keep only the first three parks for evaluation
    parks = list(extracted.parks[:3])
    # Pad to exactly 3 items to build a consistent tree (placeholders will immediately fail minimal checks)
    while len(parks) < 3:
        parks.append(ParkInfo())

    # Check distinctness: exactly 3 provided with non-empty, distinct names
    provided_names = [p.name.strip() for p in parks if p.name and p.name.strip()]
    distinct_ok = (len(provided_names) == 3 and len(set(provided_names)) == 3)

    evaluator.add_custom_node(
        result=distinct_ok,
        id="Three_Distinct_Parks_Provided",
        desc="Exactly three parks are provided and they are distinct from one another",
        parent=root,
        critical=True
    )

    # Build park subtrees and verify
    for idx, park in enumerate(parks, start=1):
        await verify_park(evaluator, root, park, idx)

    return evaluator.get_summary()