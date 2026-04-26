import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "denver_office_1031_nnn_zoning_ada"
TASK_DESCRIPTION = """
Identify a commercial office building currently available for sale in Denver, Colorado, that would be suitable for a 1031 like-kind exchange investment. The property must operate under or be suitable for a Triple Net (NNN) lease structure where the tenant is responsible for property taxes, building insurance, and common area maintenance (CAM). Additionally, the property must comply with all applicable local zoning regulations for commercial office use and meet the ADA Standards for Accessible Design (2010) requirements.

Provide comprehensive information including:
- Complete property address and basic specifications (building size, year built, occupancy status)
- Current lease structure details confirming NNN arrangement and tenant responsibilities
- Purchase price or asking price
- Zoning classification confirming commercial office use is permitted
- ADA compliance status and accessibility features
- Property's eligibility for 1031 exchange (investment/business use confirmation)
- All relevant URL references to verify the provided information

The property should represent a viable commercial real estate investment opportunity that meets all legal, financial, and operational requirements for a successful 1031 exchange transaction.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AddressModel(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None


class PropertySources(BaseModel):
    property_listing_urls: List[str] = Field(default_factory=list)
    zoning_source_urls: List[str] = Field(default_factory=list)
    ada_source_urls: List[str] = Field(default_factory=list)
    lease_source_urls: List[str] = Field(default_factory=list)
    other_urls: List[str] = Field(default_factory=list)


class PropertyExtraction(BaseModel):
    property_name: Optional[str] = None
    address: Optional[AddressModel] = None

    property_type: Optional[str] = None  # e.g., "Office", "Medical Office", "Office Building"
    sale_status: Optional[str] = None    # e.g., "For Sale", "Available", "Active", etc.

    building_size: Optional[str] = None  # keep as string to allow "±12,000 SF"
    year_built: Optional[str] = None
    occupancy_status: Optional[str] = None  # e.g., "100% occupied", "Vacant"

    asking_price: Optional[str] = None  # allow ranges, "call for pricing", etc.

    lease_structure: Optional[str] = None  # e.g., "NNN", "Triple Net"
    tenant_responsibilities: List[str] = Field(default_factory=list)  # e.g., ["property taxes","insurance","CAM"]

    zoning_classification: Optional[str] = None  # e.g., "C-MX-5"
    zoning_office_permitted: Optional[str] = None  # "yes"/"no"/"unknown" or phrase

    ada_compliance_status: Optional[str] = None  # "compliant", "ADA compliant", "unknown"
    accessibility_features: List[str] = Field(default_factory=list)  # e.g., ["accessible entrance","elevator","accessible parking"]

    exchange_1031_eligibility: Optional[str] = None  # "eligible"/"suitable"/"investment property"/etc.

    sources: PropertySources = Field(default_factory=PropertySources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_property() -> str:
    return """
    You will extract details for exactly ONE specific property referenced in the answer. If multiple properties are mentioned, extract only the first clearly identified commercial office building.
    Return a JSON object that matches the provided schema exactly. Do not invent any information not explicitly present in the answer.

    Required fields and guidance:
    - property_name: The building or listing name if mentioned; else null.
    - address: 
        - street: The full street address (include suite if given).
        - city: City name.
        - state: 2-letter state code (e.g., "CO").
        - postal_code: ZIP/postal code; if missing, set to null.
    - property_type: The building type as stated (e.g., "Office", "Medical Office", "Office Building", "Office Condo").
    - sale_status: A short phrase reflecting availability (e.g., "For Sale", "Available", "Active listing").
    - building_size: Building size exactly as written (e.g., "12,345 SF", "±12k SF").
    - year_built: Year built exactly as written (e.g., "1998", "circa 2001"); if missing, null.
    - occupancy_status: As written (e.g., "100% occupied", "vacant", "single-tenant"); if missing, null.
    - asking_price: As written (e.g., "$3,250,000", "$3.25M", "Call for pricing"); do not normalize units.
    - lease_structure: As written (e.g., "NNN", "Triple Net", "Absolute NNN"); if unspecified, null.
    - tenant_responsibilities: A list of exact phrases the answer attributes to the tenant (aim to normalize the core trio to ["property taxes","building insurance","CAM"] when explicitly present; include any other responsibilities mentioned like "utilities", "maintenance").
    - zoning_classification: Zoning code string (e.g., "C-MX-5"); if missing, null.
    - zoning_office_permitted: Copy the answer’s statement about whether office is permitted under that zoning; "yes"/"no"/"unknown" or a short phrase if stated.
    - ada_compliance_status: The stated status (e.g., "ADA compliant", "complies with 2010 ADA Standards", "unknown"); do not infer.
    - accessibility_features: A list of concrete accessibility features if stated (e.g., "accessible entrance", "elevator", "accessible restrooms", "accessible parking").
    - exchange_1031_eligibility: Copy the answer’s phrasing indicating 1031 suitability or investment/business-use nature ("1031 eligible", "investment property", etc.), or null if not explicitly stated.

    - sources: 
        - property_listing_urls: URLs directly pointing to the listing/details page for this specific property (LoopNet/Crexi/CoStar/brokerage pages, etc.). Must be actual URLs present in the answer.
        - zoning_source_urls: URLs that support the zoning classification and/or permitted uses (official Denver zoning map/code, city documents, credible references).
        - ada_source_urls: URLs supporting ADA compliance or accessibility features if cited.
        - lease_source_urls: URLs that support the lease structure and/or tenant responsibilities.
        - other_urls: Any other URLs cited specifically about this property (avoid generic background links unless explicitly listed).

    Special rules for URLs:
    - Only include URLs explicitly present in the answer. Do NOT invent or infer URLs.
    - Accept plain URLs or markdown links; extract the actual URL string.
    - If a URL is missing protocol, prepend http:// as needed.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def compose_full_address(addr: Optional[AddressModel]) -> str:
    if not addr:
        return ""
    parts = []
    if addr.street:
        parts.append(addr.street.strip())
    city_state_zip = []
    if addr.city:
        city_state_zip.append(addr.city.strip())
    if addr.state:
        city_state_zip.append(addr.state.strip())
    if addr.postal_code:
        city_state_zip.append(addr.postal_code.strip())
    if city_state_zip:
        parts.append(", ".join([city_state_zip[0]] + city_state_zip[1:]))
    return ", ".join(parts)


def ensure_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def merged_sources(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def normalize_features(features: List[str]) -> str:
    return ", ".join([f.strip() for f in features if f and f.strip()])


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, task_node, data: PropertyExtraction):
    # Prepare URL buckets
    prop_urls = ensure_list(data.sources.property_listing_urls if data and data.sources else [])
    zoning_urls = ensure_list(data.sources.zoning_source_urls if data and data.sources else [])
    ada_urls = ensure_list(data.sources.ada_source_urls if data and data.sources else [])
    lease_urls = ensure_list(data.sources.lease_source_urls if data and data.sources else [])

    # URL References (Critical, Parallel) - build first to serve as preconditions for other branches
    url_refs_node = evaluator.add_parallel(
        id="URL_References",
        desc="Provides URLs sufficient to verify key claims about the specific property (not only generic background links).",
        parent=task_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(prop_urls) > 0,
        id="Property_Listing_URL",
        desc="Includes at least one URL to a listing/details page for the specific property supporting core facts (e.g., address, size, price, sale status).",
        parent=url_refs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=len(zoning_urls) > 0,
        id="Zoning_Source_URL",
        desc="Includes at least one URL supporting the zoning claim (official zoning map/code page or other credible zoning reference tied to the stated zoning classification).",
        parent=url_refs_node,
        critical=True
    )

    # Property Identification & Sale Status (Critical, Parallel)
    prop_id_node = evaluator.add_parallel(
        id="Property_Identification_And_Sale_Status",
        desc="Response identifies a specific commercial office building in Denver, CO that is currently available for sale.",
        parent=task_node,
        critical=True
    )

    # Complete Property Address in Denver, CO (Leaf)
    addr_leaf = evaluator.add_leaf(
        id="Complete_Property_Address_In_Denver_CO",
        desc="Provides the complete property address locating it in Denver, CO.",
        parent=prop_id_node,
        critical=True
    )
    full_addr = compose_full_address(data.address) if data else ""
    city = (data.address.city if data and data.address and data.address.city else "")
    state = (data.address.state if data and data.address and data.address.state else "")
    zip_code = (data.address.postal_code if data and data.address and data.address.postal_code else "")
    address_claim_parts = []
    if full_addr:
        address_claim_parts.append(f"The property's street address is '{full_addr}'.")
    # Always enforce Denver, CO
    address_claim_parts.append("The property is located in Denver, CO (Denver, Colorado).")
    if zip_code:
        address_claim_parts.append(f"The ZIP/postal code is '{zip_code}'.")
    address_claim = " ".join(address_claim_parts)
    await evaluator.verify(
        claim=address_claim,
        node=addr_leaf,
        sources=prop_urls,
        additional_instruction="Verify the address and that the city is Denver, CO. Minor formatting differences (commas, abbreviations, suite numbers) are acceptable."
    )

    # Commercial Office Building Type (Leaf)
    type_leaf = evaluator.add_leaf(
        id="Commercial_Office_Building_Type",
        desc="Property is identified as a commercial office building (not residential/industrial/etc.).",
        parent=prop_id_node,
        critical=True
    )
    type_str = data.property_type or ""
    type_claim = "The property is a commercial office building (an 'office' use), including acceptable variants like 'office building', 'medical office', or 'professional office'."
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=prop_urls,
        additional_instruction="Treat 'office', 'office building', 'medical office', 'professional office', 'office condo' as office-type. Do not accept residential or purely industrial types."
    )

    # Currently Available For Sale (Leaf)
    sale_leaf = evaluator.add_leaf(
        id="Currently_Available_For_Sale",
        desc="States and/or evidences that the property is currently listed/available for sale.",
        parent=prop_id_node,
        critical=True
    )
    sale_claim = "This property is currently listed as For Sale (i.e., actively available for purchase)."
    await evaluator.verify(
        claim=sale_claim,
        node=sale_leaf,
        sources=prop_urls,
        additional_instruction="Look for explicit indicators such as 'For Sale', 'Available', 'Active Listing' on the listing page."
    )

    # Basic Specifications (Critical, Parallel)
    specs_node = evaluator.add_parallel(
        id="Basic_Specifications",
        desc="Provides the requested basic specifications: building size, year built, and occupancy status.",
        parent=task_node,
        critical=True
    )

    # Building Size (Leaf)
    size_leaf = evaluator.add_leaf(
        id="Building_Size_Provided",
        desc="Provides building size (e.g., square footage).",
        parent=specs_node,
        critical=True
    )
    size_claim = f"The building size is '{data.building_size or ''}' as stated in the listing."
    await evaluator.verify(
        claim=size_claim,
        node=size_leaf,
        sources=prop_urls,
        additional_instruction="Match the square footage or size information with reasonable tolerance for formatting (e.g., commas, ±, k abbreviations)."
    )

    # Year Built (Leaf)
    year_leaf = evaluator.add_leaf(
        id="Year_Built_Provided",
        desc="Provides year built.",
        parent=specs_node,
        critical=True
    )
    year_claim = f"The building was built in '{data.year_built or ''}'."
    await evaluator.verify(
        claim=year_claim,
        node=year_leaf,
        sources=prop_urls,
        additional_instruction="Check the year built (or similar term). Allow approximate phrasing like 'circa 2001' if consistent."
    )

    # Occupancy Status (Leaf)
    occ_leaf = evaluator.add_leaf(
        id="Occupancy_Status_Provided",
        desc="Provides occupancy status (occupied/vacant/percent occupied).",
        parent=specs_node,
        critical=True
    )
    occ_claim = f"The occupancy status is stated as '{data.occupancy_status or ''}'."
    await evaluator.verify(
        claim=occ_claim,
        node=occ_leaf,
        sources=prop_urls,
        additional_instruction="Verify whether the listing mentions vacant, occupied, single-tenant, multi-tenant, or a % occupied."
    )

    # Asking Price (Leaf)
    price_leaf = evaluator.add_leaf(
        id="Asking_Price_Provided",
        desc="Provides purchase price or asking price.",
        parent=task_node,
        critical=True
    )
    price_claim = f"The asking/list price for the property is '{data.asking_price or ''}'."
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=prop_urls,
        additional_instruction="Accept 'List Price', 'Asking Price', or 'Call for pricing' if explicitly stated in the listing."
    )

    # NNN Lease Requirement (Leaf)
    nnn_leaf = evaluator.add_leaf(
        id="NNN_Lease_Requirement",
        desc="Confirms the property operates under OR is suitable for a Triple Net (NNN) lease structure where the tenant is responsible for property taxes, building insurance, and CAM.",
        parent=task_node,
        critical=True
    )
    nnn_claim = "The property operates under or is appropriate for a Triple Net (NNN) lease in which the tenant is responsible for property taxes, building insurance, and common area maintenance (CAM)."
    await evaluator.verify(
        claim=nnn_claim,
        node=nnn_leaf,
        sources=merged_sources(prop_urls, lease_urls),
        additional_instruction="Evidence of 'NNN', 'Triple Net', or explicit statements that tenant pays taxes, insurance, and CAM is acceptable."
    )

    # Zoning Compliance for Office Use (Critical; split into two critical leaves under a parallel parent)
    zoning_node = evaluator.add_parallel(
        id="Zoning_Compliance_For_Office_Use",
        desc="Provides the zoning classification and confirms commercial office use is permitted under local zoning for the property.",
        parent=task_node,
        critical=True
    )
    # Zoning classification stated for the property
    zoning_class_leaf = evaluator.add_leaf(
        id="Zoning_Classification_Stated",
        desc="The property's zoning classification is correctly stated.",
        parent=zoning_node,
        critical=True
    )
    zoning_class_claim = f"The property's zoning classification is '{data.zoning_classification or ''}' (Denver)."
    await evaluator.verify(
        claim=zoning_class_claim,
        node=zoning_class_leaf,
        sources=merged_sources(prop_urls, zoning_urls),
        additional_instruction="The listing page may mention a zoning code (e.g., C-MX-5) or the official zoning map/code confirms it for this property."
    )
    # Office use permitted under that zoning
    office_perm_leaf = evaluator.add_leaf(
        id="Office_Use_Permitted_Under_Zoning",
        desc="Under the cited zoning classification, commercial office use is permitted.",
        parent=zoning_node,
        critical=True
    )
    office_perm_claim = f"Under Denver zoning classification '{data.zoning_classification or ''}', commercial office use is permitted."
    await evaluator.verify(
        claim=office_perm_claim,
        node=office_perm_leaf,
        sources=zoning_urls,
        additional_instruction="Use the official Denver zoning code, zoning map, or credible city sources to confirm that 'office' use is permitted in the stated zoning district."
    )

    # ADA 2010 Compliance (Critical, Parallel with two leaves)
    ada_node = evaluator.add_parallel(
        id="ADA_2010_Compliance",
        desc="Confirms the property meets (is compliant with) the 2010 ADA Standards for Accessible Design and describes relevant accessibility features.",
        parent=task_node,
        critical=True
    )
    ada_conf_leaf = evaluator.add_leaf(
        id="ADA_2010_Compliance_Confirmed",
        desc="Affirms the property is ADA-compliant under the 2010 ADA Standards (not merely 'unknown' or unspecified).",
        parent=ada_node,
        critical=True
    )
    ada_conf_claim = "The property is compliant with the 2010 ADA Standards for Accessible Design (often phrased as 'ADA compliant')."
    await evaluator.verify(
        claim=ada_conf_claim,
        node=ada_conf_leaf,
        sources=merged_sources(ada_urls, prop_urls),
        additional_instruction="Look for explicit statements like 'ADA compliant' or references to compliance with ADA 2010 standards on the listing or provided ADA sources."
    )
    ada_feat_leaf = evaluator.add_leaf(
        id="Accessibility_Features_Described",
        desc="Describes at least one concrete accessibility feature relevant to ADA access (e.g., accessible entrance, elevator, ramps, accessible restrooms, accessible parking).",
        parent=ada_node,
        critical=True
    )
    features_text = normalize_features(data.accessibility_features if data else [])
    ada_feat_claim = f"At least one of the following ADA accessibility features is present at the property: {features_text or 'accessible entrance, elevator, accessible restroom, accessible parking'}."
    await evaluator.verify(
        claim=ada_feat_claim,
        node=ada_feat_leaf,
        sources=merged_sources(ada_urls, prop_urls),
        additional_instruction="Confirm at least one concrete accessibility feature mentioned is supported by the provided sources."
    )

    # 1031 Exchange Eligibility (Leaf)
    ex1031_leaf = evaluator.add_leaf(
        id="1031_Exchange_Eligibility",
        desc="Confirms the property is held out as investment/business-use real property suitable for a 1031 like-kind exchange (not personal-use property).",
        parent=task_node,
        critical=True
    )
    ex1031_claim = "This property is offered for investment/business use and is suitable for a 1031 like-kind exchange (Section 1031)."
    await evaluator.verify(
        claim=ex1031_claim,
        node=ex1031_leaf,
        sources=prop_urls,
        additional_instruction="Accept explicit mentions like '1031 exchange eligible', 'NNN investment', or clear indicators that it is an investment/commercial offering (not personal-use)."
    )

    # Add some custom info for debugging/summary
    evaluator.add_custom_info(
        info={
            "extracted_address": compose_full_address(data.address) if data else None,
            "property_type": data.property_type if data else None,
            "asking_price": data.asking_price if data else None,
            "zoning_classification": data.zoning_classification if data else None,
            "url_counts": {
                "property_listing_urls": len(prop_urls),
                "zoning_source_urls": len(zoning_urls),
                "ada_source_urls": len(ada_urls),
                "lease_source_urls": len(lease_urls),
            }
        },
        info_type="debug",
        info_name="extraction_debug_summary"
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
    Evaluate an answer for the Denver commercial office 1031/NNN/Zoning/ADA task.
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

    # Extract structured property info
    extracted: PropertyExtraction = await evaluator.extract(
        prompt=prompt_extract_property(),
        template_class=PropertyExtraction,
        extraction_name="property_extraction"
    )

    # Build the critical task root (since evaluator root is always non-critical)
    task_node = evaluator.add_parallel(
        id="Commercial_Office_Building_Denver_1031_NNN_Zoning_ADA",
        desc="Evaluate whether ONE identified commercial office building for sale in Denver meets the 1031/NNN/zoning/ADA requirements and includes the requested information with verifying URLs.",
        parent=root,
        critical=True
    )

    # Construct verification tree and run checks
    await build_verification_tree(evaluator, task_node, extracted)

    # Return standardized summary
    return evaluator.get_summary()