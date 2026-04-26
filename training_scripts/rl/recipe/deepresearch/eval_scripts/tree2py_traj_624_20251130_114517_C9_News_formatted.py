import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "congress_directory_2025"
TASK_DESCRIPTION = """
You are preparing a comprehensive Congressional Contact Directory for a constituent services organization. The directory must include complete and accurate official contact information for current members of Congress, verified from official government sources.

Compile detailed contact information for the following three current members of the United States Congress:
1. Senator Mark Kelly (Arizona)
2. Senator Ruben Gallego (Arizona)
3. Representative Maxine Waters (California, 43rd District)

For each member, you must provide:
- Confirmation of their current congressional status
- Their Washington DC office location with complete address (including building name, suite/room number, city, state, and zip code) and phone number
- All state or district office locations with complete addresses (street address with suite number, city, state, and zip code) and phone numbers
- Source URL from their official .senate.gov or .house.gov website for verification

Additionally, your research assistant has flagged "Lara Trump" as a potential entry due to recent news coverage. Explain why Lara Trump should NOT be included in this Congressional Contact Directory, specifically addressing:
- Whether she holds or has held a congressional office
- What position she actually held
- Why that position does not qualify for inclusion in a government congressional directory
- The key distinction between her role and actual congressional offices

All information must be current as of November 2025 and verifiable from official government websites or credible news sources.
"""

SENATE_BUILDINGS = ["hart", "dirksen", "russell"]  # Senate office buildings (keywords)
HOUSE_BUILDINGS = ["rayburn", "cannon", "longworth"]  # House office buildings (keywords)


class DCOffice(BaseModel):
    building_name: Optional[str] = None
    room: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None
    address_text: Optional[str] = None


class LocalOffice(BaseModel):
    street: Optional[str] = None
    suite: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None
    address_text: Optional[str] = None


class MemberContact(BaseModel):
    name: Optional[str] = None
    current_status: Optional[str] = None
    dc_office: Optional[DCOffice] = None
    state_or_district_offices: List[LocalOffice] = Field(default_factory=list)
    official_sources: List[str] = Field(default_factory=list)


class LaraTrumpInfo(BaseModel):
    holds_congressional_office: Optional[str] = None
    actual_position: Optional[str] = None
    why_not_qualify: Optional[str] = None
    key_distinction: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DirectoryExtraction(BaseModel):
    recency_statement: Optional[str] = None
    kelly: Optional[MemberContact] = None
    gallego: Optional[MemberContact] = None
    waters: Optional[MemberContact] = None
    lara_trump: Optional[LaraTrumpInfo] = None


def prompt_extract_directory() -> str:
    return """
Extract the structured information from the answer for the specified congressional directory.

Include the following fields:

1) recency_statement: The exact sentence or phrase in the answer that asserts the information is current as of November 2025. If none, return null.

2) kelly: MemberContact object for Senator Mark Kelly (AZ)
   - name: The member's full name as stated in the answer.
   - current_status: A sentence stating the current congressional status (e.g., "Mark Kelly is a current U.S. Senator from Arizona").
   - dc_office: DCOffice object with building_name, room (suite/office number), city, state, zip, phone, and address_text (full DC address as presented).
   - state_or_district_offices: Array of LocalOffice objects, one per Arizona office; each includes street, suite (if applicable), city, state, zip, phone, and address_text (full text as presented).
   - official_sources: All URLs cited in the answer for Kelly. Include only the URLs explicitly present. Do not invent URLs.

3) gallego: MemberContact object for Senator Ruben Gallego (AZ), same fields as above.

4) waters: MemberContact object for Representative Maxine Waters (CA-43), same fields as above. Her DC building must be a House office building (Rayburn, Cannon, or Longworth), but you must extract whatever is in the answer without inference.

5) lara_trump: LaraTrumpInfo object
   - holds_congressional_office: Extract the statement indicating whether Lara Trump holds or has ever held a congressional office (e.g., "No").
   - actual_position: Extract the position she actually held (e.g., "Co-chair of the Republican National Committee").
   - why_not_qualify: Extract the explanation of why that position is not a congressional office and should not be in a government congressional directory.
   - key_distinction: Extract the explanation distinguishing party roles from elected congressional offices.
   - sources: All URLs cited to support the above statements. Include only URLs explicitly present.

General rules:
- Extract only what is explicitly stated in the answer.
- For URLs, include them exactly as shown (plain or markdown). Do not infer or add new URLs.
- If any subfield is missing in the answer, set it to null (single field) or [] for arrays.
"""


def filter_official_urls(urls: List[str], required_domain_keywords: List[str]) -> List[str]:
    if not urls:
        return []
    filtered = []
    for u in urls:
        low = u.lower()
        if any(k in low for k in required_domain_keywords):
            filtered.append(u)
    return filtered


def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def is_dc_phone_202(phone: Optional[str]) -> bool:
    if not is_nonempty(phone):
        return False
    digits = re.sub(r"[^\d]", "", phone)
    return digits.startswith("202") or digits.startswith("1202")


def is_valid_zip(zip_code: Optional[str]) -> bool:
    if not is_nonempty(zip_code):
        return False
    return bool(re.fullmatch(r"\d{5}(-\d{4})?", zip_code.strip()))


def dc_address_complete(dc: Optional[DCOffice]) -> bool:
    if dc is None:
        return False
    return all([
        is_nonempty(dc.building_name),
        is_nonempty(dc.room),
        is_nonempty(dc.city),
        is_nonempty(dc.state),
        is_valid_zip(dc.zip)
    ])


def building_name_matches(dc: Optional[DCOffice], allowed_keywords: List[str]) -> bool:
    if dc is None or not is_nonempty(dc.building_name):
        return False
    bn = dc.building_name.lower()
    return any(k in bn for k in allowed_keywords)


def local_office_entry_complete(off: LocalOffice) -> bool:
    return all([
        is_nonempty(off.street),
        is_nonempty(off.city),
        is_nonempty(off.state),
        is_valid_zip(off.zip),
        is_nonempty(off.phone),
    ])


def all_local_offices_complete(offices: List[LocalOffice]) -> bool:
    if not offices:
        return False
    return all(local_office_entry_complete(o) for o in offices)


def format_dc_summary(dc: Optional[DCOffice]) -> str:
    if dc is None:
        return "DC office: [not provided]"
    parts = []
    if is_nonempty(dc.building_name):
        parts.append(dc.building_name)
    if is_nonempty(dc.room):
        parts.append(f"Room {dc.room}")
    if is_nonempty(dc.city) or is_nonempty(dc.state) or is_nonempty(dc.zip):
        city_state_zip = " ".join([p for p in [dc.city, dc.state, dc.zip] if is_nonempty(p)])
        parts.append(city_state_zip)
    if is_nonempty(dc.phone):
        parts.append(f"Phone: {dc.phone}")
    return "; ".join(parts)


def format_local_offices(offices: List[LocalOffice]) -> str:
    if not offices:
        return "[no state/district offices provided]"
    formatted = []
    for i, o in enumerate(offices, start=1):
        addr_parts = []
        if is_nonempty(o.street):
            sline = o.street
            if is_nonempty(o.suite):
                sline = f"{sline}, {o.suite}"
            addr_parts.append(sline)
        city_state_zip = " ".join([p for p in [o.city, o.state, o.zip] if is_nonempty(p)])
        if city_state_zip:
            addr_parts.append(city_state_zip)
        if is_nonempty(o.phone):
            addr_parts.append(f"Phone: {o.phone}")
        formatted.append(f"{i}) " + "; ".join(addr_parts) if addr_parts else f"{i}) [incomplete]")
    return " | ".join(formatted)


async def verify_member_block(
    evaluator: Evaluator,
    parent_top_node,
    member: Optional[MemberContact],
    block_id: str,
    block_desc: str,
    current_status_id: str,
    current_status_desc: str,
    dc_block_id: str,
    dc_address_complete_id: str,
    dc_address_complete_desc: str,
    dc_building_constraint_id: str,
    dc_building_constraint_desc: str,
    dc_phone_202_id: str,
    dc_phone_202_desc: str,
    local_block_id: str,
    local_all_listed_id: str,
    local_all_listed_desc: str,
    local_entries_complete_id: str,
    local_entries_complete_desc: str,
    official_source_id: str,
    official_source_desc: str,
    expected_status_claim: str,
    official_domain_keywords: List[str],
    allowed_building_keywords: List[str],
    office_label: str,
) -> None:
    node_main = evaluator.add_parallel(
        id=block_id,
        desc=block_desc,
        parent=parent_top_node,
        critical=True
    )

    # Current status leaf
    status_leaf = evaluator.add_leaf(
        id=current_status_id,
        desc=current_status_desc,
        parent=node_main,
        critical=True
    )
    sources_for_status = filter_official_urls(member.official_sources if member else [], official_domain_keywords)
    await evaluator.verify(
        claim=expected_status_claim,
        node=status_leaf,
        sources=sources_for_status,
        additional_instruction=f"Only consider supported if the source is an official {'.'.join(official_domain_keywords)} domain page that clearly states this current status."
    )

    # DC office block (structure checks)
    dc_block = evaluator.add_parallel(
        id=dc_block_id,
        desc=f"Provides Washington, DC office location (complete address) and phone number.",
        parent=node_main,
        critical=True
    )
    # Completeness check (custom)
    dc_complete_res = dc_address_complete(member.dc_office if member else None)
    evaluator.add_custom_node(
        result=dc_complete_res,
        id=dc_address_complete_id,
        desc=dc_address_complete_desc,
        parent=dc_block,
        critical=True
    )
    # Building constraint (custom)
    building_ok = building_name_matches(member.dc_office if member else None, allowed_building_keywords)
    evaluator.add_custom_node(
        result=building_ok,
        id=dc_building_constraint_id,
        desc=dc_building_constraint_desc,
        parent=dc_block,
        critical=True
    )
    # Phone 202 area code (custom)
    phone_ok = is_dc_phone_202(member.dc_office.phone if member and member.dc_office else None)
    evaluator.add_custom_node(
        result=phone_ok,
        id=dc_phone_202_id,
        desc=dc_phone_202_desc,
        parent=dc_block,
        critical=True
    )

    # Official Source URL supporting provided details
    official_leaf = evaluator.add_leaf(
        id=official_source_id,
        desc=official_source_desc,
        parent=node_main,
        critical=True
    )
    official_sources = filter_official_urls(member.official_sources if member else [], official_domain_keywords)
    dc_summary = format_dc_summary(member.dc_office if member else None)
    local_summary = format_local_offices(member.state_or_district_offices if member else [])
    claim_official_support = (
        f"An official page on a {'.'.join(official_domain_keywords)} domain lists and supports the following for {member.name if member and member.name else office_label}: "
        f"DC Office: {dc_summary}. State/District Offices: {local_summary}."
    )
    await evaluator.verify(
        claim=claim_official_support,
        node=official_leaf,
        sources=official_sources if official_sources else None,
        additional_instruction=(
            f"If no official {'.'.join(official_domain_keywords)} URL is provided, mark as not supported. "
            "Verify that the DC address and each listed local office (addresses and phone numbers) appear on the cited official page(s)."
        )
    )

    # Local offices block
    local_block = evaluator.add_parallel(
        id=local_block_id,
        desc=f"Provides all {office_label} office locations with complete addresses and phone numbers.",
        parent=node_main,
        critical=True
    )

    # All offices listed leaf (verify by official sources; depends on official_leaf automatically as critical sibling in ancestor)
    all_listed_leaf = evaluator.add_leaf(
        id=local_all_listed_id,
        desc=local_all_listed_desc,
        parent=local_block,
        critical=True
    )
    claim_all_listed = (
        f"The official page lists the following {office_label} offices, and the answer includes all of them (not a partial subset): "
        f"{local_summary}."
    )
    await evaluator.verify(
        claim=claim_all_listed,
        node=all_listed_leaf,
        sources=official_sources if official_sources else None,
        additional_instruction=(
            f"Confirm that all {office_label} offices shown on the official page are included. "
            "If any office on the official page is missing from the answer, mark as not supported. "
            f"If no official {'.'.join(official_domain_keywords)} source is provided, mark as not supported."
        )
    )

    # Entries completeness (custom)
    local_complete = all_local_offices_complete(member.state_or_district_offices if member else [])
    evaluator.add_custom_node(
        result=local_complete,
        id=local_entries_complete_id,
        desc=local_entries_complete_desc,
        parent=local_block,
        critical=True
    )


async def verify_lara_trump_block(
    evaluator: Evaluator,
    parent_top_node,
    lara: Optional[LaraTrumpInfo]
) -> None:
    node_main = evaluator.add_parallel(
        id="Lara_Trump_Exclusion",
        desc="Explains why Lara Trump should not be included in the congressional contact directory.",
        parent=parent_top_node,
        critical=True
    )

    # Congressional status
    status_leaf = evaluator.add_leaf(
        id="Trump_Congressional_Status",
        desc="States whether Lara Trump holds/has held a congressional office.",
        parent=node_main,
        critical=True
    )
    holds_text = lara.holds_congressional_office if lara else None
    claim_status = (
        "Lara Trump has never held a United States congressional office (she has not served as a U.S. Senator or U.S. Representative)."
    )
    await evaluator.verify(
        claim=claim_status,
        node=status_leaf,
        sources=None,
        additional_instruction="Judge based on the answer’s explanation and common facts; do not rely on your own memory beyond the answer unless clearly stated."
    )

    # Actual position
    actual_leaf = evaluator.add_leaf(
        id="Trump_Actual_Position",
        desc="Identifies what position she actually held (non-congressional role).",
        parent=node_main,
        critical=True
    )
    actual_pos = lara.actual_position if lara and is_nonempty(lara.actual_position) else "[not provided]"
    claim_actual = f"Lara Trump’s actual position was: {actual_pos}."
    await evaluator.verify(
        claim=claim_actual,
        node=actual_leaf,
        sources=lara.sources if lara and lara.sources else None,
        additional_instruction="Verify that credible source(s) support this stated position."
    )

    # Why not qualifying
    why_leaf = evaluator.add_leaf(
        id="Why_Position_Not_Qualifying",
        desc="Explains why that position does not qualify for inclusion in a government congressional directory.",
        parent=node_main,
        critical=True
    )
    why_text = lara.why_not_qualify if lara and is_nonempty(lara.why_not_qualify) else "[not provided]"
    claim_why = (
        f"The explanation correctly states that Lara Trump’s position ({actual_pos}) is not an elected congressional office and therefore does not belong in a government congressional directory. Explanation: {why_text}."
    )
    await evaluator.verify(
        claim=claim_why,
        node=why_leaf,
        sources=None,
        additional_instruction="Confirm the explanation clearly ties the role to a party/organization rather than elected government office."
    )

    # Key distinction
    distinction_leaf = evaluator.add_leaf(
        id="Key_Distinction_Explained",
        desc="Clearly distinguishes political party/organizational roles from elected congressional offices (Senate/House).",
        parent=node_main,
        critical=True
    )
    distinction_text = lara.key_distinction if lara and is_nonempty(lara.key_distinction) else "[not provided]"
    claim_distinction = (
        f"The answer clearly distinguishes party/organizational roles from elected congressional offices. Distinction: {distinction_text}."
    )
    await evaluator.verify(
        claim=claim_distinction,
        node=distinction_leaf,
        sources=None,
        additional_instruction="Confirm clarity of distinction: party/committee roles vs. elected House/Senate offices."
    )

    # Claims verified by credible sources
    sources_leaf = evaluator.add_leaf(
        id="Trump_Claims_Verified_By_Credible_Sources",
        desc="Provides credible source URL(s) supporting the claims about her role and non-congressional status.",
        parent=node_main,
        critical=True
    )
    claim_sources = (
        "The cited sources confirm that Lara Trump’s role was a party/organizational position (e.g., RNC co-chair) and that she did not hold a U.S. congressional office."
    )
    await evaluator.verify(
        claim=claim_sources,
        node=sources_leaf,
        sources=lara.sources if lara and lara.sources else None,
        additional_instruction="Only pass if at least one cited source explicitly supports both the role and non-congressional status."
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

    # Create top-level critical node as per rubric
    top = evaluator.add_parallel(
        id="Congressional_Contact_Information_Compilation",
        desc="Compile official, verifiable contact information for the three specified current members of Congress and explain why Lara Trump should not be included.",
        parent=root,
        critical=True
    )

    # Extract structured data
    extraction = await evaluator.extract(
        prompt=prompt_extract_directory(),
        template_class=DirectoryExtraction,
        extraction_name="directory_extraction"
    )

    # Global recency requirement
    recency_leaf = evaluator.add_leaf(
        id="Global_Recency_Requirement",
        desc="Information is presented/qualified as current as of November 2025.",
        parent=top,
        critical=True
    )
    recency_claim = "The answer explicitly states that the information is current as of November 2025."
    await evaluator.verify(
        claim=recency_claim,
        node=recency_leaf,
        sources=None,
        additional_instruction="Look for explicit phrases like 'current as of November 2025' or 'Updated November 2025' in the answer text."
    )

    # Senator Mark Kelly block
    await verify_member_block(
        evaluator=evaluator,
        parent_top_node=top,
        member=extraction.kelly,
        block_id="Senator_Mark_Kelly_Information",
        block_desc="Required status + office contact information for Senator Mark Kelly (AZ).",
        current_status_id="Kelly_Current_Status",
        current_status_desc="Confirms Mark Kelly’s current congressional status (current U.S. Senator from Arizona).",
        dc_block_id="Kelly_DC_Office",
        dc_address_complete_id="Kelly_DC_Address_Completeness",
        dc_address_complete_desc="DC office address includes building name, suite/room number, city, state, and ZIP code.",
        dc_building_constraint_id="Kelly_DC_Building_Constraint",
        dc_building_constraint_desc="DC office building is consistent with Senate office building constraint (Hart/Dirksen/Russell).",
        dc_phone_202_id="Kelly_DC_Phone_Provided_And_202",
        dc_phone_202_desc="Provides a DC office phone number and it uses the 202 area code.",
        local_block_id="Kelly_State_Offices",
        local_all_listed_id="Kelly_All_State_Offices_Listed",
        local_all_listed_desc="Includes all state office locations listed on the official site (not a partial subset).",
        local_entries_complete_id="Kelly_State_Office_Entries_Complete",
        local_entries_complete_desc="Each listed Arizona office includes full address (street + suite/room if applicable, city, state, ZIP) and a phone number.",
        official_source_id="Kelly_Official_Source_URL",
        official_source_desc="Provides at least one official .senate.gov URL that supports the provided DC and state office contact details.",
        expected_status_claim="Mark Kelly is a current U.S. Senator from Arizona.",
        official_domain_keywords=[".senate.gov"],
        allowed_building_keywords=SENATE_BUILDINGS,
        office_label="Arizona state"
    )

    # Senator Ruben Gallego block
    await verify_member_block(
        evaluator=evaluator,
        parent_top_node=top,
        member=extraction.gallego,
        block_id="Senator_Ruben_Gallego_Information",
        block_desc="Required status + office contact information for Senator Ruben Gallego (AZ).",
        current_status_id="Gallego_Current_Status",
        current_status_desc="Confirms Ruben Gallego’s current congressional status (current U.S. Senator from Arizona).",
        dc_block_id="Gallego_DC_Office",
        dc_address_complete_id="Gallego_DC_Address_Completeness",
        dc_address_complete_desc="DC office address includes building name, suite/room number, city, state, and ZIP code.",
        dc_building_constraint_id="Gallego_DC_Building_Constraint",
        dc_building_constraint_desc="DC office building is consistent with Senate office building constraint (Hart/Dirksen/Russell).",
        dc_phone_202_id="Gallego_DC_Phone_Provided_And_202",
        dc_phone_202_desc="Provides a DC office phone number and it uses the 202 area code.",
        local_block_id="Gallego_State_Offices",
        local_all_listed_id="Gallego_All_State_Offices_Listed",
        local_all_listed_desc="Includes all state office locations listed on the official site (not a partial subset).",
        local_entries_complete_id="Gallego_State_Office_Entries_Complete",
        local_entries_complete_desc="Each listed Arizona office includes full address (street + suite/room if applicable, city, state, ZIP) and a phone number.",
        official_source_id="Gallego_Official_Source_URL",
        official_source_desc="Provides at least one official .senate.gov URL that supports the provided DC and state office contact details.",
        expected_status_claim="Ruben Gallego is a current U.S. Senator from Arizona.",
        official_domain_keywords=[".senate.gov"],
        allowed_building_keywords=SENATE_BUILDINGS,
        office_label="Arizona state"
    )

    # Representative Maxine Waters block
    await verify_member_block(
        evaluator=evaluator,
        parent_top_node=top,
        member=extraction.waters,
        block_id="Representative_Maxine_Waters_Information",
        block_desc="Required status + office contact information for Representative Maxine Waters (CA-43).",
        current_status_id="Waters_Current_Status",
        current_status_desc="Confirms Maxine Waters’s current congressional status (current U.S. Representative for California’s 43rd district).",
        dc_block_id="Waters_DC_Office",
        dc_address_complete_id="Waters_DC_Address_Completeness",
        dc_address_complete_desc="DC office address includes building name, suite/room number, city, state, and ZIP code.",
        dc_building_constraint_id="Waters_DC_Building_Constraint",
        dc_building_constraint_desc="DC office building is consistent with House office building constraint (Rayburn/Cannon/Longworth).",
        dc_phone_202_id="Waters_DC_Phone_Provided_And_202",
        dc_phone_202_desc="Provides a DC office phone number and it uses the 202 area code.",
        local_block_id="Waters_District_Offices",
        local_all_listed_id="Waters_All_District_Offices_Listed",
        local_all_listed_desc="Includes all district office locations listed on the official site (not a partial subset).",
        local_entries_complete_id="Waters_District_Office_Entries_Complete",
        local_entries_complete_desc="Each listed district office includes full address (street + suite/room if applicable, city, state, ZIP) and a phone number.",
        official_source_id="Waters_Official_Source_URL",
        official_source_desc="Provides at least one official .house.gov URL that supports the provided DC and district office contact details.",
        expected_status_claim="Maxine Waters is the current U.S. Representative for California’s 43rd congressional district.",
        official_domain_keywords=[".house.gov"],
        allowed_building_keywords=HOUSE_BUILDINGS,
        office_label="district"
    )

    # Lara Trump exclusion block
    await verify_lara_trump_block(evaluator, top, extraction.lara_trump)

    return evaluator.get_summary()