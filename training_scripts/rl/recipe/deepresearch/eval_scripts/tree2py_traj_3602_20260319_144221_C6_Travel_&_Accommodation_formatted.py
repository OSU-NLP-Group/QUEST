import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "multi_destination_travel_doc_guide_2026"
TASK_DESCRIPTION = (
    "A US citizen is planning a 14-day trip in April 2026 with the following itinerary: "
    "(1) Take a domestic flight within the United States, (2) Visit one national park "
    "(must be selected from the 11 US national parks that charge the nonresident surcharge fee), "
    "(3) Fly internationally to Aruba for a 5-day stay, (4) Continue to Malaysia for a 6-day stay "
    "before returning home. As a travel documentation specialist, compile a comprehensive "
    "documentation guide for this traveler. For each segment of the trip, provide: "
    "For the domestic flight (TSA checkpoint): What types of identification are acceptable at TSA checkpoints as of 2026? "
    "What is the alternative option (including fee and start date) for travelers without acceptable ID? "
    "For the national park visit: Which specific national park from the 11-surcharge parks list should be selected? "
    "What type of annual pass should a US citizen purchase, and what is its cost for 2026? "
    "What is the validity period of this pass? What are the acceptable forms of photo identification that must be shown when using this pass? "
    "What methods are available to obtain this pass, and which one provides immediate availability? "
    "For Aruba entry: What are the passport validity requirements for US citizens? Do US citizens require a visa for tourism stays up to 90 days? "
    "What is the mandatory online card that all travelers must complete, what is its official website, and within what timeframe before departure must it be completed? "
    "What supporting documentation is required? "
    "For Malaysia entry: What are the passport validity requirements (in terms of months beyond exit date)? "
    "Do US citizens require a visa for tourism visits of 90 days or less? "
    "What is the mandatory digital card that all foreign travelers must complete, and within what timeframe prior to arrival must it be submitted? "
    "What is the legal requirement regarding carrying identification while in Malaysia? "
    "All information must be grounded in official or authoritative sources with proper URL references provided for verification."
)

# Ground truth helper info (for reference/logging only)
SURCHARGE_11_PARKS = [
    "Acadia",
    "Bryce Canyon",
    "Everglades",
    "Glacier",
    "Grand Canyon",
    "Grand Teton",
    "Rocky Mountain",
    "Sequoia & Kings Canyon",
    "Yellowstone",
    "Yosemite",
    "Zion",
]


# -----------------------------------------------------------------------------
# Pydantic data models for extraction
# -----------------------------------------------------------------------------
class TSAInfo(BaseModel):
    acceptable_ids: List[str] = Field(default_factory=list)
    acceptable_ids_urls: List[str] = Field(default_factory=list)
    confirmid_fee: Optional[str] = None
    confirmid_start_date: Optional[str] = None
    confirmid_urls: List[str] = Field(default_factory=list)


class ParkInfo(BaseModel):
    selected_park: Optional[str] = None
    other_parks_mentioned: List[str] = Field(default_factory=list)
    surcharge_list_urls: List[str] = Field(default_factory=list)

    pass_type: Optional[str] = None
    pass_cost_2026: Optional[str] = None
    pass_cost_urls: List[str] = Field(default_factory=list)

    pass_validity_desc: Optional[str] = None
    pass_validity_urls: List[str] = Field(default_factory=list)

    pass_id_requirements_desc: Optional[str] = None
    pass_acceptable_id_types: List[str] = Field(default_factory=list)
    pass_id_urls: List[str] = Field(default_factory=list)

    acquisition_methods: List[str] = Field(default_factory=list)
    immediate_method: Optional[str] = None
    acquisition_urls: List[str] = Field(default_factory=list)


class ArubaInfo(BaseModel):
    passport_validity_desc: Optional[str] = None
    passport_urls: List[str] = Field(default_factory=list)

    visa_requirement_desc: Optional[str] = None
    visa_urls: List[str] = Field(default_factory=list)

    ed_card_name: Optional[str] = None
    ed_card_official_site: Optional[str] = None
    ed_card_timeframe_desc: Optional[str] = None
    ed_card_urls: List[str] = Field(default_factory=list)

    supporting_docs_list: List[str] = Field(default_factory=list)
    supporting_docs_urls: List[str] = Field(default_factory=list)


class MalaysiaInfo(BaseModel):
    passport_validity_desc: Optional[str] = None
    passport_urls: List[str] = Field(default_factory=list)

    visa_requirement_desc: Optional[str] = None
    visa_urls: List[str] = Field(default_factory=list)

    mdac_mandatory_desc: Optional[str] = None
    mdac_timeframe_desc: Optional[str] = None
    mdac_urls: List[str] = Field(default_factory=list)

    id_carry_desc: Optional[str] = None
    id_carry_urls: List[str] = Field(default_factory=list)


class TravelDocExtraction(BaseModel):
    tsa: Optional[TSAInfo] = None
    park: Optional[ParkInfo] = None
    aruba: Optional[ArubaInfo] = None
    malaysia: Optional[MalaysiaInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_travel_doc() -> str:
    return """
Extract the structured information from the answer for each segment. Return JSON strictly matching the schema.

For the domestic TSA segment (tsa):
- acceptable_ids: list of the specific identification types stated as acceptable at TSA checkpoints (e.g., "REAL ID-compliant driver's license", "U.S. passport book", "U.S. passport card", "DHS trusted traveler card", etc.).
- acceptable_ids_urls: list of authoritative URLs cited that document TSA acceptable identification (prefer tsa.gov or other U.S. government domains).
- confirmid_fee: the stated fee for ConfirmID (if mentioned).
- confirmid_start_date: the stated start/effective date for ConfirmID (if mentioned).
- confirmid_urls: list of authoritative URLs cited that document TSA ConfirmID.

For the national park segment (park):
- selected_park: exactly one specific park name selected for the visit (as printed in the answer).
- other_parks_mentioned: list of any other parks mentioned besides the selected one (empty if none).
- surcharge_list_urls: list of authoritative URLs cited that document the "11 national parks that charge the Nonresident Surcharge fee".
- pass_type: the stated pass name/type appropriate for a U.S. citizen (e.g., "Resident Annual Pass").
- pass_cost_2026: the stated dollar cost for this pass for year 2026 (e.g., "$80").
- pass_cost_urls: list of authoritative URLs cited for the pass type/cost.
- pass_validity_desc: the stated validity period/wording (e.g., "12 months from the month of purchase, expiring the last day of that month").
- pass_validity_urls: list of authoritative URLs cited that document pass validity.
- pass_id_requirements_desc: stated requirement that valid photo ID proving U.S. citizenship/residency must be shown to use the pass.
- pass_acceptable_id_types: list of acceptable photo ID types for using the pass (e.g., state driver's license/ID, U.S. passport book/card, Permanent Resident Card).
- pass_id_urls: list of authoritative URLs cited that document the pass ID requirements and acceptable ID types.
- acquisition_methods: list of stated methods to obtain the pass (e.g., "digital via Recreation.gov", "in person at a park", "by mail").
- immediate_method: the method stated to provide immediate availability (if any; typically the digital Recreation.gov pass).
- acquisition_urls: list of authoritative URLs cited that document pass acquisition methods.

For the Aruba entry segment (aruba):
- passport_validity_desc: stated passport validity rule for U.S. citizens (e.g., "valid for the duration of stay").
- passport_urls: authoritative URLs cited for Aruba passport validity (e.g., Aruba government/tourism, U.S. Department of State).
- visa_requirement_desc: stated whether visa is required for U.S. citizens up to 90 days (e.g., "no visa required up to 90 days").
- visa_urls: authoritative URLs cited for Aruba visa requirement.
- ed_card_name: the name of the mandatory online card (e.g., "Aruba ED-Card (Embarkation-Disembarkation Card)").
- ed_card_official_site: the official ED-Card website URL (e.g., "https://edcardaruba.aw").
- ed_card_timeframe_desc: stated timeframe before departure to complete the ED-Card (e.g., "within 7 days before departure").
- ed_card_urls: authoritative URLs cited that document ED-Card requirements.
- supporting_docs_list: list of the supporting documents stated as required for Aruba entry.
- supporting_docs_urls: authoritative URLs cited for the supporting documentation requirements.

For the Malaysia entry segment (malaysia):
- passport_validity_desc: stated passport validity months beyond exit date (e.g., "valid for at least 6 months from the date of exit").
- passport_urls: authoritative URLs cited for Malaysia passport validity (e.g., immigration.gov.my, U.S. Department of State).
- visa_requirement_desc: stated whether visa is required for U.S. citizens for 90 days or less (e.g., "no visa required for 90 days or less").
- visa_urls: authoritative URLs cited for Malaysia visa policy.
- mdac_mandatory_desc: stated that MDAC (Malaysia Digital Arrival Card) is mandatory for foreign travelers.
- mdac_timeframe_desc: stated timeframe to submit MDAC (e.g., "within 3 days prior to arrival").
- mdac_urls: authoritative URLs cited for MDAC requirements.
- id_carry_desc: stated legal requirement about carrying identification while in Malaysia.
- id_carry_urls: authoritative URLs cited for the in-country identification-carrying requirement.

Rules:
- Extract only what appears in the answer. If a field is missing, set it to null or empty list as appropriate.
- For all URLs, extract actual URLs as they appear in the answer (plain or markdown link targets), ensuring valid full URLs.
"""


# -----------------------------------------------------------------------------
# Helper: URL-reference verification leaf
# -----------------------------------------------------------------------------
async def verify_url_reference(
    evaluator: Evaluator,
    *,
    node_id: str,
    desc: str,
    parent,
    urls: List[str],
    topic_desc: str,
    critical: bool = True,
):
    """
    Add a leaf node verifying that an official/authoritative URL reference is provided.
    If URLs are present, verify by URLs that the page(s) are official/authoritative for the topic.
    If URLs are missing, fall back to a simple check that the answer provided at least one such URL.
    """
    node = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )

    if urls:
        claim = f"This page is an official or authoritative source that documents: {topic_desc}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=urls,
            additional_instruction="Treat .gov, .mil, official national park, or government/tourism authority sites as authoritative where applicable.",
        )
    else:
        # Fall back: check the answer text itself
        claim = f"The answer includes at least one official or authoritative URL for: {topic_desc}."
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=None,
            additional_instruction="Look for an official URL in the answer text. Accept official government domains (e.g., tsa.gov, state.gov, immigration.gov.my), "
                                  "official national park/NPS or Recreation.gov pages, or official Aruba ED-Card site.",
        )


# -----------------------------------------------------------------------------
# TSA verification subtree
# -----------------------------------------------------------------------------
async def build_tsa_nodes(evaluator: Evaluator, root_parent, tsa: Optional[TSAInfo]):
    parent = evaluator.add_parallel(
        id="Domestic_Flight_TSA_Requirements",
        desc="TSA checkpoint documentation for a domestic US flight as of 2026",
        parent=root_parent,
        critical=False,
    )

    # Acceptable IDs
    ids_leaf = evaluator.add_leaf(
        id="Acceptable_IDs_at_TSA",
        desc="States TSA-acceptable identification types as of 2026, including REAL ID-compliant ID and at least one alternative acceptable ID",
        parent=parent,
        critical=True,
    )
    ids_list = tsa.acceptable_ids if tsa else []
    ids_str = ", ".join(ids_list) if ids_list else "[none stated]"
    await evaluator.verify(
        claim=(
            "The cited TSA page(s) describe acceptable identification at TSA checkpoints, and this includes "
            "REAL ID–compliant state driver's licenses/IDs and a U.S. passport (book or card) as acceptable ID."
        ),
        node=ids_leaf,
        sources=(tsa.acceptable_ids_urls if tsa else []),
        additional_instruction=(
            "Verify that the page lists acceptable IDs; confirm it includes REAL ID–compliant IDs and US passport (book or card). "
            f"The answer listed: {ids_str}. Allow minor wording differences."
        ),
    )

    # Acceptable IDs URL reference
    await verify_url_reference(
        evaluator,
        node_id="TSA_Acceptable_IDs_URL_Reference",
        desc="Provides an official/authoritative URL reference for TSA acceptable identification",
        parent=parent,
        urls=(tsa.acceptable_ids_urls if tsa else []),
        topic_desc="TSA acceptable identification at airport security checkpoints",
        critical=True,
    )

    # ConfirmID group (critical; all children critical)
    confirm_group = evaluator.add_parallel(
        id="Alternative_ConfirmID_Option",
        desc="Describes the alternative option for travelers without acceptable ID (ConfirmID), including fee and start date",
        parent=parent,
        critical=True,
    )

    fee_leaf = evaluator.add_leaf(
        id="ConfirmID_Fee",
        desc="States the ConfirmID fee is $45 for a 10-day travel period",
        parent=confirm_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The ConfirmID fee is $45 for a 10-day travel period.",
        node=fee_leaf,
        sources=(tsa.confirmid_urls if tsa else []),
        additional_instruction="Verify this exact fee and period from the cited official source.",
    )

    start_leaf = evaluator.add_leaf(
        id="ConfirmID_Effective_Date",
        desc="States ConfirmID starts February 1, 2026",
        parent=confirm_group,
        critical=True,
    )
    await evaluator.verify(
        claim="ConfirmID becomes available starting February 1, 2026.",
        node=start_leaf,
        sources=(tsa.confirmid_urls if tsa else []),
        additional_instruction="Verify the effective/start date from the cited official source.",
    )

    await verify_url_reference(
        evaluator,
        node_id="ConfirmID_URL_Reference",
        desc="Provides an official/authoritative URL reference documenting TSA ConfirmID",
        parent=confirm_group,
        urls=(tsa.confirmid_urls if tsa else []),
        topic_desc="TSA ConfirmID program details (fee, validity window, start date)",
        critical=True,
    )


# -----------------------------------------------------------------------------
# National Park verification subtree
# -----------------------------------------------------------------------------
def _has_other_acquisition_methods(acq_methods: List[str], immediate_method: Optional[str]) -> bool:
    methods = [m.strip().lower() for m in (acq_methods or []) if m and m.strip()]
    if not methods:
        return False
    imm = (immediate_method or "").strip().lower()
    # Consider any listed method that is not the immediate one as "other"
    for m in methods:
        if imm and m != imm:
            return True
    # If no explicit immediate method, but 2 or more methods exist, treat as having "other"
    return len(methods) >= 2


async def build_park_nodes(evaluator: Evaluator, root_parent, park: Optional[ParkInfo]):
    parent = evaluator.add_parallel(
        id="National_Park_Entry_Requirements",
        desc="Documentation requirements for visiting one US national park chosen from the 11-park surcharge list",
        parent=root_parent,
        critical=False,
    )

    # Park selection group (critical; all children critical)
    select_group = evaluator.add_parallel(
        id="Park_Selection",
        desc="Selects exactly one park and it is from the 11-park surcharge list",
        parent=parent,
        critical=True,
    )

    # Exactly one park selected (custom boolean check)
    exactly_one = evaluator.add_custom_node(
        result=bool(park and park.selected_park and (not park.other_parks_mentioned)),
        id="Exactly_One_Park_Selected",
        desc="Exactly one specific national park is selected (not multiple)",
        parent=select_group,
        critical=True,
    )

    # Selected park is from the surcharge list (LLM simple verify)
    park_in_list = evaluator.add_leaf(
        id="Park_Is_From_Surcharge_List",
        desc="Selected park is one of the 11 surcharge parks",
        parent=select_group,
        critical=True,
    )
    selected = park.selected_park if park and park.selected_park else "[none]"
    parks_list_str = ", ".join(SURCHARGE_11_PARKS)
    await evaluator.verify(
        claim=f"The selected park '{selected}' is one of the following 11 US national parks that charge the Nonresident Surcharge fee: {parks_list_str}.",
        node=park_in_list,
        additional_instruction="Allow minor name variations and punctuation differences (e.g., ampersand vs 'and').",
    )

    # Provide authoritative URL for the 11-park list
    await verify_url_reference(
        evaluator,
        node_id="Surcharge_List_URL_Reference",
        desc="Provides an official/authoritative URL reference documenting the 11-park surcharge list",
        parent=select_group,
        urls=(park.surcharge_list_urls if park else []),
        topic_desc=f"the '11 national parks that charge the Nonresident Surcharge fee' list (for example, asserting that '{selected}' is on that list)",
        critical=True,
    )

    # Annual pass type and cost (critical)
    pass_group = evaluator.add_parallel(
        id="Annual_Pass_Type_and_Cost",
        desc="Identifies the appropriate annual pass for a US citizen and provides its 2026 cost",
        parent=parent,
        critical=True,
    )

    pass_type_leaf = evaluator.add_leaf(
        id="Pass_Type_Identified",
        desc="Identifies the pass as the (2026) Resident Annual Pass for US citizens/permanent residents",
        parent=pass_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The appropriate pass for a U.S. citizen/permanent resident is the Resident Annual Pass (for 2026).",
        node=pass_type_leaf,
        sources=(park.pass_cost_urls if park else []),
        additional_instruction="Verify that the cited authoritative source uses this pass type naming or equivalent and is applicable to U.S. citizens.",
    )

    pass_cost_leaf = evaluator.add_leaf(
        id="Pass_Cost",
        desc="States the pass cost is $80 for 2026",
        parent=pass_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The Resident Annual Pass costs $80 in 2026.",
        node=pass_cost_leaf,
        sources=(park.pass_cost_urls if park else []),
        additional_instruction="Verify the exact dollar amount for the 2026 price from the authoritative source.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Pass_Cost_URL_Reference",
        desc="Provides an official/authoritative URL reference for the pass type and/or cost",
        parent=pass_group,
        urls=(park.pass_cost_urls if park else []),
        topic_desc="Resident Annual Pass type and $80 cost (2026)",
        critical=True,
    )

    # Pass validity period (critical)
    validity_group = evaluator.add_parallel(
        id="Pass_Validity_Period",
        desc="Provides the validity period of the pass",
        parent=parent,
        critical=True,
    )

    validity_leaf = evaluator.add_leaf(
        id="Validity_Details",
        desc="States validity is 12 months from the month of purchase, expiring the last day of that month",
        parent=validity_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The Resident Annual Pass is valid for 12 months from the month of purchase, expiring on the last day of that month.",
        node=validity_leaf,
        sources=(park.pass_validity_urls if park else []),
        additional_instruction="Verify the validity window exactly as described.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Validity_URL_Reference",
        desc="Provides an official/authoritative URL reference documenting pass validity",
        parent=validity_group,
        urls=(park.pass_validity_urls if park else []),
        topic_desc="Resident Annual Pass validity rules",
        critical=True,
    )

    # ID required for pass use (critical)
    id_req_group = evaluator.add_parallel(
        id="ID_Required_for_Pass_Use",
        desc="States what photo identification must be shown when using the pass",
        parent=parent,
        critical=True,
    )

    id_must_leaf = evaluator.add_leaf(
        id="ID_Must_Prove_US_Status",
        desc="States that valid photo ID proving US citizenship or residency must be shown",
        parent=id_req_group,
        critical=True,
    )
    await evaluator.verify(
        claim="When using the Resident Annual Pass, the pass holder must show valid photo identification proving U.S. citizenship or permanent residency.",
        node=id_must_leaf,
        sources=(park.pass_id_urls if park else []),
        additional_instruction="Verify the explicit requirement to present valid photo ID proving U.S. citizenship or residency.",
    )

    id_types_leaf = evaluator.add_leaf(
        id="Acceptable_ID_Types_For_Pass",
        desc="Lists acceptable ID types for pass use: US state driver's license/ID, US passport book or card, Permanent Resident Card",
        parent=id_req_group,
        critical=True,
    )
    acceptable_types_list = (park.pass_acceptable_id_types if park else [])
    acceptable_types_str = ", ".join(acceptable_types_list) if acceptable_types_list else "[none stated]"
    await evaluator.verify(
        claim=(
            "Acceptable photo ID types for using the pass include U.S. State/Territory driver’s license or state ID, "
            "U.S. passport (book or card), and a Permanent Resident Card."
        ),
        node=id_types_leaf,
        sources=(park.pass_id_urls if park else []),
        additional_instruction=f"Verify that the cited source lists these or equivalent categories. The answer listed: {acceptable_types_str}.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Pass_ID_URL_Reference",
        desc="Provides an official/authoritative URL reference documenting ID requirements for pass use",
        parent=id_req_group,
        urls=(park.pass_id_urls if park else []),
        topic_desc="ID requirements and acceptable photo ID types to use the Resident Annual Pass",
        critical=True,
    )

    # Acquisition methods (parent adjusted to non-critical to allow a non-critical child)
    acq_group = evaluator.add_parallel(
        id="Pass_Acquisition_Methods",
        desc="Explains methods to obtain the pass and identifies which method provides immediate availability",
        parent=parent,
        critical=False,  # Adjusted to allow a non-critical child per framework constraint
    )

    immediate_leaf = evaluator.add_leaf(
        id="Immediate_Availability_Method",
        desc="States that a digital pass via Recreation.gov is available immediately",
        parent=acq_group,
        critical=True,
    )
    await evaluator.verify(
        claim="A digital Resident Annual Pass obtained via Recreation.gov is available immediately after purchase.",
        node=immediate_leaf,
        sources=(park.acquisition_urls if park else []),
        additional_instruction="Verify that the Recreation.gov digital pass is delivered/available immediately.",
    )

    # Non-critical: mentions at least one other method
    other_methods = _has_other_acquisition_methods(park.acquisition_methods if park else [], park.immediate_method if park else None)
    evaluator.add_custom_node(
        result=other_methods,
        id="Other_Acquisition_Methods_Mentioned",
        desc="Mentions at least one additional acquisition method besides the immediately-available digital option",
        parent=acq_group,
        critical=False,
    )

    await verify_url_reference(
        evaluator,
        node_id="Acquisition_URL_Reference",
        desc="Provides an official/authoritative URL reference for pass acquisition methods",
        parent=acq_group,
        urls=(park.acquisition_urls if park else []),
        topic_desc="methods to obtain (purchase/access) the Resident Annual Pass, including digital via Recreation.gov",
        critical=True,
    )


# -----------------------------------------------------------------------------
# Aruba verification subtree
# -----------------------------------------------------------------------------
async def build_aruba_nodes(evaluator: Evaluator, root_parent, aruba: Optional[ArubaInfo]):
    parent = evaluator.add_parallel(
        id="Aruba_Entry_Requirements",
        desc="Documentation requirements for Aruba entry for a US citizen tourist stay (up to 90 days context)",
        parent=root_parent,
        critical=False,
    )

    # Passport validity
    pass_group = evaluator.add_parallel(
        id="Aruba_Passport_Validity_Requirement",
        desc="States Aruba passport validity requirement for US citizens",
        parent=parent,
        critical=True,
    )

    valid_leaf = evaluator.add_leaf(
        id="Aruba_Validity_Requirement",
        desc="States passport must be valid for the entirety of the stay in Aruba",
        parent=pass_group,
        critical=True,
    )
    await evaluator.verify(
        claim="For Aruba, a U.S. citizen's passport must be valid for the duration of the stay.",
        node=valid_leaf,
        sources=(aruba.passport_urls if aruba else []),
        additional_instruction="Verify this validity rule from authoritative sources (Aruba govt/tourism, U.S. State Dept., etc.).",
    )

    await verify_url_reference(
        evaluator,
        node_id="Aruba_Passport_URL_Reference",
        desc="Provides an official/authoritative URL reference for Aruba passport requirements",
        parent=pass_group,
        urls=(aruba.passport_urls if aruba else []),
        topic_desc="Aruba passport validity requirements for U.S. citizens",
        critical=True,
    )

    # Visa requirement
    visa_group = evaluator.add_parallel(
        id="Aruba_Visa_Requirement",
        desc="States whether a visa is required for US citizens for tourism stays up to 90 days",
        parent=parent,
        critical=True,
    )

    visa_free_leaf = evaluator.add_leaf(
        id="Aruba_Visa_Free_Status",
        desc="States US citizens do not require a visa for Aruba tourism/business stays up to 90 days",
        parent=visa_group,
        critical=True,
    )
    await evaluator.verify(
        claim="U.S. citizens do not require a visa for Aruba tourism/business stays up to 90 days.",
        node=visa_free_leaf,
        sources=(aruba.visa_urls if aruba else []),
        additional_instruction="Verify visa-free status and allowable stay length for U.S. citizens.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Aruba_Visa_URL_Reference",
        desc="Provides an official/authoritative URL reference for Aruba visa policy",
        parent=visa_group,
        urls=(aruba.visa_urls if aruba else []),
        topic_desc="Aruba visa policy for U.S. citizens (up to 90 days)",
        critical=True,
    )

    # ED-Card requirement
    ed_group = evaluator.add_parallel(
        id="ED_Card_Requirement",
        desc="States Aruba ED-Card requirements: name, official website, and completion timeframe",
        parent=parent,
        critical=True,
    )

    ed_name_leaf = evaluator.add_leaf(
        id="ED_Card_Name",
        desc="Identifies the mandatory online card as the Aruba ED-Card (Embarkation-Disembarkation Card)",
        parent=ed_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The mandatory online card for Aruba is the Aruba ED-Card (Embarkation-Disembarkation Card).",
        node=ed_name_leaf,
        sources=(aruba.ed_card_urls if aruba else []),
        additional_instruction="Verify that Aruba requires completion of the ED-Card.",
    )

    ed_site_leaf = evaluator.add_leaf(
        id="ED_Card_Website",
        desc="States the official ED-Card website is edcardaruba.aw",
        parent=ed_group,
        critical=True,
    )
    ed_site = aruba.ed_card_official_site if (aruba and aruba.ed_card_official_site) else "https://edcardaruba.aw"
    # Verify using both the site and any references, if available
    ed_urls: List[str] = []
    if aruba:
        if aruba.ed_card_official_site:
            ed_urls.append(aruba.ed_card_official_site)
        ed_urls.extend(aruba.ed_card_urls)
    await evaluator.verify(
        claim="The official Aruba ED-Card website is edcardaruba.aw.",
        node=ed_site_leaf,
        sources=ed_urls if ed_urls else None,
        additional_instruction="If using the ED-Card site directly, confirm it is the official portal. Otherwise, verify via an authoritative Aruba/Tourism page that points to edcardaruba.aw.",
    )

    ed_time_leaf = evaluator.add_leaf(
        id="ED_Card_Timeline",
        desc="States ED-Card must be completed within 7 days before departure",
        parent=ed_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The Aruba ED-Card must be completed within 7 days before departure.",
        node=ed_time_leaf,
        sources=(aruba.ed_card_urls if aruba else []),
        additional_instruction="Verify the submission window/timeline.",
    )

    await verify_url_reference(
        evaluator,
        node_id="ED_Card_URL_Reference",
        desc="Provides an official/authoritative URL reference for ED-Card requirements",
        parent=ed_group,
        urls=(aruba.ed_card_urls if aruba else []),
        topic_desc="Aruba ED-Card requirements, official website, and submission timeframe",
        critical=True,
    )

    # Supporting documentation
    sup_group = evaluator.add_parallel(
        id="Supporting_Documentation",
        desc="Lists supporting documentation required for Aruba entry and provides authoritative citation",
        parent=parent,
        critical=True,
    )

    sup_list_leaf = evaluator.add_leaf(
        id="Supporting_Docs_Listed",
        desc="Lists the required supporting documentation for Aruba entry",
        parent=sup_group,
        critical=True,
    )
    docs_list = (aruba.supporting_docs_list if aruba else [])
    docs_str = ", ".join(docs_list) if docs_list else "[none stated]"
    await evaluator.verify(
        claim=f"The listed supporting documentation for Aruba entry is correct and matches authoritative guidance: {docs_str}.",
        node=sup_list_leaf,
        sources=(aruba.supporting_docs_urls if aruba else []),
        additional_instruction="Accept reasonable synonyms. Verify that the authoritative page lists these requirements.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Supporting_Docs_URL_Reference",
        desc="Provides an official/authoritative URL reference for Aruba supporting documentation requirements",
        parent=sup_group,
        urls=(aruba.supporting_docs_urls if aruba else []),
        topic_desc="supporting documentation requirements for entering Aruba",
        critical=True,
    )


# -----------------------------------------------------------------------------
# Malaysia verification subtree
# -----------------------------------------------------------------------------
async def build_malaysia_nodes(evaluator: Evaluator, root_parent, mys: Optional[MalaysiaInfo]):
    parent = evaluator.add_parallel(
        id="Malaysia_Entry_Requirements",
        desc="Documentation requirements for Malaysia entry and in-country identification-carrying requirement",
        parent=root_parent,
        critical=False,
    )

    # Passport validity
    pass_group = evaluator.add_parallel(
        id="Malaysia_Passport_Validity_Requirement",
        desc="States Malaysia passport validity requirement (months beyond exit date)",
        parent=parent,
        critical=True,
    )

    six_leaf = evaluator.add_leaf(
        id="Six_Month_Validity",
        desc="States passport must be valid for at least 6 months from the date of exit from Malaysia",
        parent=pass_group,
        critical=True,
    )
    await evaluator.verify(
        claim="For Malaysia, a passport must be valid for at least 6 months from the date of exit from Malaysia.",
        node=six_leaf,
        sources=(mys.passport_urls if mys else []),
        additional_instruction="Verify the 6-month validity rule from an authoritative source (e.g., immigration.gov.my, U.S. State Dept.).",
    )

    await verify_url_reference(
        evaluator,
        node_id="Malaysia_Passport_URL_Reference",
        desc="Provides an official/authoritative URL reference for Malaysia passport requirements",
        parent=pass_group,
        urls=(mys.passport_urls if mys else []),
        topic_desc="Malaysia passport validity requirements for foreign visitors/U.S. citizens",
        critical=True,
    )

    # Visa requirement
    visa_group = evaluator.add_parallel(
        id="Malaysia_Visa_Requirement",
        desc="States whether a visa is required for US citizens for tourism visits of 90 days or less",
        parent=parent,
        critical=True,
    )

    visa_free_leaf = evaluator.add_leaf(
        id="Malaysia_Visa_Free_Status",
        desc="States US citizens do not require a visa for Malaysia tourism/business visits of 90 days or less",
        parent=visa_group,
        critical=True,
    )
    await evaluator.verify(
        claim="U.S. citizens do not require a visa for Malaysia tourism/business visits of 90 days or less.",
        node=visa_free_leaf,
        sources=(mys.visa_urls if mys else []),
        additional_instruction="Verify visa-free entry and max stay for U.S. citizens.",
    )

    await verify_url_reference(
        evaluator,
        node_id="Malaysia_Visa_URL_Reference",
        desc="Provides an official/authoritative URL reference for Malaysia visa policy",
        parent=visa_group,
        urls=(mys.visa_urls if mys else []),
        topic_desc="Malaysia visa policy for U.S. citizens (90 days or less)",
        critical=True,
    )

    # MDAC requirement
    mdac_group = evaluator.add_parallel(
        id="MDAC_Requirement",
        desc="States MDAC requirements and submission timeframe",
        parent=parent,
        critical=True,
    )

    mdac_mand_leaf = evaluator.add_leaf(
        id="MDAC_Mandatory",
        desc="States Malaysia Digital Arrival Card (MDAC) is mandatory for foreign travelers",
        parent=mdac_group,
        critical=True,
    )
    await evaluator.verify(
        claim="Malaysia Digital Arrival Card (MDAC) is mandatory for foreign travelers.",
        node=mdac_mand_leaf,
        sources=(mys.mdac_urls if mys else []),
        additional_instruction="Verify the mandatory nature of MDAC for foreigners.",
    )

    mdac_time_leaf = evaluator.add_leaf(
        id="MDAC_Timeline",
        desc="States MDAC must be submitted within 3 days prior to arrival",
        parent=mdac_group,
        critical=True,
    )
    await evaluator.verify(
        claim="The MDAC must be submitted within 3 days prior to arrival in Malaysia.",
        node=mdac_time_leaf,
        sources=(mys.mdac_urls if mys else []),
        additional_instruction="Verify the MDAC submission window/timeline.",
    )

    await verify_url_reference(
        evaluator,
        node_id="MDAC_URL_Reference",
        desc="Provides an official/authoritative URL reference for MDAC requirements",
        parent=mdac_group,
        urls=(mys.mdac_urls if mys else []),
        topic_desc="Malaysia Digital Arrival Card (MDAC) requirements and submission timeframe",
        critical=True,
    )

    # ID carrying requirement
    idcarry_group = evaluator.add_parallel(
        id="ID_Carrying_Requirement",
        desc="States the legal requirement regarding carrying identification while in Malaysia, with an authoritative URL",
        parent=parent,
        critical=True,
    )

    carry_leaf = evaluator.add_leaf(
        id="Carrying_Requirement_Stated",
        desc="States the legal requirement regarding carrying identification while in Malaysia",
        parent=idcarry_group,
        critical=True,
    )
    await evaluator.verify(
        claim="Visitors in Malaysia are legally required to carry identification (e.g., passport) and present it upon request by authorities.",
        node=carry_leaf,
        sources=(mys.id_carry_urls if mys else []),
        additional_instruction="Verify from an authoritative source (e.g., Malaysia govt, U.S. State Dept. advisory).",
    )

    await verify_url_reference(
        evaluator,
        node_id="Carrying_URL_Reference",
        desc="Provides an official/authoritative URL reference for the Malaysia identification-carrying requirement",
        parent=idcarry_group,
        urls=(mys.id_carry_urls if mys else []),
        topic_desc="requirement to carry identification (passport/ID) while in Malaysia",
        critical=True,
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the multi-destination travel documentation task.
    """
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

    # Record the 11-park surcharge list as ground truth info (contextual)
    evaluator.add_ground_truth(
        {
            "eleven_surcharge_parks": SURCHARGE_11_PARKS,
            "note": "Used for validating the selected park membership in the surcharge list.",
        },
        gt_type="reference_list",
    )

    # Extract structured information
    extracted: TravelDocExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_doc(),
        template_class=TravelDocExtraction,
        extraction_name="travel_documentation_extraction",
    )

    # Build four parallel segments
    await build_tsa_nodes(evaluator, root, extracted.tsa or TSAInfo())
    await build_park_nodes(evaluator, root, extracted.park or ParkInfo())
    await build_aruba_nodes(evaluator, root, extracted.aruba or ArubaInfo())
    await build_malaysia_nodes(evaluator, root, extracted.malaysia or MalaysiaInfo())

    # Return summary
    return evaluator.get_summary()