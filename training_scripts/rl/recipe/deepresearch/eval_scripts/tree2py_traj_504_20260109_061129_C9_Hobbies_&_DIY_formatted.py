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
TASK_ID = "nc_woodworking_craft_setup"
TASK_DESCRIPTION = (
    "I'm planning to start a home-based woodworking craft business in Raleigh, North Carolina, "
    "where I'll create handmade wooden furniture and decorative items to sell at local craft fairs and specialty markets. "
    "I'll be setting up a woodworking workshop at my home residence. Please provide a comprehensive compliance and setup guide that includes: "
    "(1) Home Workshop Requirements: What permits, zoning approvals, and safety compliance measures (including OSHA dust collection and fire safety requirements) are needed for a home-based woodworking workshop in Raleigh, NC? "
    "(2) Business Registration: What are the specific steps to legally register this business in North Carolina, including business structure registration, sales tax permits, and any required licenses? "
    "(3) Sales Tax Obligations: What is the total sales tax rate I need to collect in Raleigh, NC, and what are the requirements for obtaining a sales tax permit and resale certificate? "
    "(4) Craft Fair Insurance: What are the typical minimum liability insurance coverage amounts required by craft fairs, and what must be included in a Certificate of Insurance? "
    "(5) Raleigh Craft Fair Venues: Identify at least two specific craft fair or specialty market venues in Raleigh, NC that accept woodworking vendors, including their booth specifications, vendor application requirements, and any specific insurance or documentation requirements. "
    "(6) Product Safety Compliance: What safety compliance requirements apply to handmade wooden products, especially if I plan to make children's toys or furniture? "
    "(7) Vendor Requirements: What are the North Carolina Department of Revenue requirements for vendors selling at specialty markets and craft fairs? "
    "(8) Cost Estimates: What are the typical costs for booth fees at NC craft fairs and annual liability insurance premiums for craft vendors? "
    "For each requirement, provide specific regulatory references, official website URLs where applicable, and exact numerical thresholds or limits (such as square footage limits, coverage amounts, exposure limits, etc.)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClaimWithSources(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HomeWorkshopExtraction(BaseModel):
    home_occupation_permit: ClaimWithSources = Field(default_factory=ClaimWithSources)
    zoning_approval: ClaimWithSources = Field(default_factory=ClaimWithSources)
    accessory_structure_permit_threshold: ClaimWithSources = Field(default_factory=ClaimWithSources)


class BusinessRegistrationExtraction(BaseModel):
    sos_registration: ClaimWithSources = Field(default_factory=ClaimWithSources)
    structure_steps: ClaimWithSources = Field(default_factory=ClaimWithSources)
    required_licenses_statement: ClaimWithSources = Field(default_factory=ClaimWithSources)


class SalesTaxExtraction(BaseModel):
    total_sales_tax_rate_raleigh: ClaimWithSources = Field(default_factory=ClaimWithSources)
    sales_tax_permit_requirements: ClaimWithSources = Field(default_factory=ClaimWithSources)
    resale_certificate_requirements: ClaimWithSources = Field(default_factory=ClaimWithSources)


class WorkshopSafetyExtraction(BaseModel):
    osha_wood_dust_pel: ClaimWithSources = Field(default_factory=ClaimWithSources)
    dust_collection_cfm_range: ClaimWithSources = Field(default_factory=ClaimWithSources)
    fire_extinguisher_thresholds: ClaimWithSources = Field(default_factory=ClaimWithSources)
    sprinkler_system_threshold: ClaimWithSources = Field(default_factory=ClaimWithSources)


class CraftFairInsuranceExtraction(BaseModel):
    typical_minimum_liability_limits: ClaimWithSources = Field(default_factory=ClaimWithSources)
    coi_contents_and_additional_insured: ClaimWithSources = Field(default_factory=ClaimWithSources)


class VenueDetails(BaseModel):
    name: Optional[str] = None
    venue_url: Optional[str] = None
    accepts_woodworking: ClaimWithSources = Field(default_factory=ClaimWithSources)
    booth_specifications: ClaimWithSources = Field(default_factory=ClaimWithSources)
    vendor_application_requirements: ClaimWithSources = Field(default_factory=ClaimWithSources)
    insurance_or_doc_requirements: ClaimWithSources = Field(default_factory=ClaimWithSources)


class VenuesExtraction(BaseModel):
    venues: List[VenueDetails] = Field(default_factory=list)


class ProductSafetyExtraction(BaseModel):
    general_handmade_product_compliance: ClaimWithSources = Field(default_factory=ClaimWithSources)
    childrens_products_cpsc_and_cpc: ClaimWithSources = Field(default_factory=ClaimWithSources)
    numeric_thresholds_when_applicable: ClaimWithSources = Field(default_factory=ClaimWithSources)


class NCDORVendorRequirementsExtraction(BaseModel):
    vendor_certificate_of_registration: ClaimWithSources = Field(default_factory=ClaimWithSources)
    organizer_daily_registration_list: ClaimWithSources = Field(default_factory=ClaimWithSources)


class CostEstimatesExtraction(BaseModel):
    typical_booth_fees_nc_craft_fairs: ClaimWithSources = Field(default_factory=ClaimWithSources)
    typical_annual_insurance_premiums: ClaimWithSources = Field(default_factory=ClaimWithSources)


class MasterExtraction(BaseModel):
    home_workshop: HomeWorkshopExtraction = Field(default_factory=HomeWorkshopExtraction)
    business_registration: BusinessRegistrationExtraction = Field(default_factory=BusinessRegistrationExtraction)
    sales_tax: SalesTaxExtraction = Field(default_factory=SalesTaxExtraction)
    safety: WorkshopSafetyExtraction = Field(default_factory=WorkshopSafetyExtraction)
    insurance: CraftFairInsuranceExtraction = Field(default_factory=CraftFairInsuranceExtraction)
    venues: VenuesExtraction = Field(default_factory=VenuesExtraction)
    product_safety: ProductSafetyExtraction = Field(default_factory=ProductSafetyExtraction)
    ncdor_vendor: NCDORVendorRequirementsExtraction = Field(default_factory=NCDORVendorRequirementsExtraction)
    costs: CostEstimatesExtraction = Field(default_factory=CostEstimatesExtraction)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_master() -> str:
    return """
Extract a structured summary of the answer focusing on the specific claims and the exact URLs cited to support them. Return JSON strictly matching the schema below. For each claim, include any numeric thresholds, limits, dates, dollar amounts, coverage limits, square-footage thresholds, airflow values, PELs, etc., exactly as stated in the answer. Extract only URLs explicitly present in the answer.

Schema (use these exact field names):
{
  "home_workshop": {
    "home_occupation_permit": {"statement": string|null, "urls": [string, ...]},
    "zoning_approval": {"statement": string|null, "urls": [string, ...]},
    "accessory_structure_permit_threshold": {"statement": string|null, "urls": [string, ...]}
  },
  "business_registration": {
    "sos_registration": {"statement": string|null, "urls": [string, ...]},
    "structure_steps": {"statement": string|null, "urls": [string, ...]},
    "required_licenses_statement": {"statement": string|null, "urls": [string, ...]}
  },
  "sales_tax": {
    "total_sales_tax_rate_raleigh": {"statement": string|null, "urls": [string, ...]},
    "sales_tax_permit_requirements": {"statement": string|null, "urls": [string, ...]},
    "resale_certificate_requirements": {"statement": string|null, "urls": [string, ...]}
  },
  "safety": {
    "osha_wood_dust_pel": {"statement": string|null, "urls": [string, ...]},
    "dust_collection_cfm_range": {"statement": string|null, "urls": [string, ...]},
    "fire_extinguisher_thresholds": {"statement": string|null, "urls": [string, ...]},
    "sprinkler_system_threshold": {"statement": string|null, "urls": [string, ...]}
  },
  "insurance": {
    "typical_minimum_liability_limits": {"statement": string|null, "urls": [string, ...]},
    "coi_contents_and_additional_insured": {"statement": string|null, "urls": [string, ...]}
  },
  "venues": {
    "venues": [
      {
        "name": string|null,
        "venue_url": string|null,
        "accepts_woodworking": {"statement": string|null, "urls": [string, ...]},
        "booth_specifications": {"statement": string|null, "urls": [string, ...]},
        "vendor_application_requirements": {"statement": string|null, "urls": [string, ...]},
        "insurance_or_doc_requirements": {"statement": string|null, "urls": [string, ...]}
      }
    ]
  },
  "product_safety": {
    "general_handmade_product_compliance": {"statement": string|null, "urls": [string, ...]},
    "childrens_products_cpsc_and_cpc": {"statement": string|null, "urls": [string, ...]},
    "numeric_thresholds_when_applicable": {"statement": string|null, "urls": [string, ...]}
  },
  "ncdor_vendor": {
    "vendor_certificate_of_registration": {"statement": string|null, "urls": [string, ...]},
    "organizer_daily_registration_list": {"statement": string|null, "urls": [string, ...]}
  },
  "costs": {
    "typical_booth_fees_nc_craft_fairs": {"statement": string|null, "urls": [string, ...]},
    "typical_annual_insurance_premiums": {"statement": string|null, "urls": [string, ...]}
  }
}

Instructions:
- The "statement" should be a concise, faithful rendering of what the answer claims for that item, including any specific numeric thresholds (e.g., mg/m³, CFM, square footage, $ limits).
- "urls" must be the exact URLs explicitly cited in the answer that directly support that specific statement. Do not invent URLs.
- If the answer did not provide a statement or any supporting URLs for an item, set "statement" to null and "urls" to an empty list.
- For venues, include up to the first two venues mentioned (if more are present). If fewer than two venues are given, include whatever venues are present.
"""


# --------------------------------------------------------------------------- #
# Helper: verification for a single claim with sources                        #
# --------------------------------------------------------------------------- #
async def verify_claim_with_urls(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    claim_obj: ClaimWithSources,
    *,
    critical: bool = True,
    require_official: bool = False,
    fallback_claim_hint: Optional[str] = None,
) -> None:
    claim_text = (claim_obj.statement or "").strip()
    urls = claim_obj.urls if claim_obj and claim_obj.urls else []

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=critical,
    )

    add_ins_parts = []
    add_ins_parts.append("Judge this claim strictly against the provided webpage(s). The webpage(s) are the source of truth.")
    add_ins_parts.append("Allow reasonable paraphrasing, but ensure the same substantive facts and any numeric thresholds match.")
    add_ins_parts.append("If the webpage(s) are irrelevant or inaccessible, consider the claim unsupported.")
    if require_official:
        add_ins_parts.append(
            "Prefer authoritative sources such as official government or regulatory domains (e.g., raleighnc.gov, nc.gov, osha.gov, cpsc.gov) or the venue's official site. "
            "If the given URL(s) are not authoritative but still clearly and explicitly support the claim, they may still be acceptable."
        )

    if not claim_text:
        # If missing, create a placeholder and direct the verifier to consider it unsupported
        if fallback_claim_hint:
            claim_text = f"(missing in answer) {fallback_claim_hint}"
        else:
            claim_text = "(missing claim in the answer for this requirement)"
        add_ins_parts.append("No explicit claim text was found in the answer for this requirement; consider this unsupported.")

    if not urls:
        add_ins_parts.append("No URL(s) were provided by the answer for this requirement; consider the claim unsupported.")

    additional_instruction = " ".join(add_ins_parts)

    await evaluator.verify(
        claim=claim_text,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification builders for each rubric section                               #
# --------------------------------------------------------------------------- #
async def build_home_workshop_requirements(
    evaluator: Evaluator,
    parent,
    data: HomeWorkshopExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Home_Workshop_Requirements",
        desc="Permits/zoning approvals and home-workshop compliance in Raleigh, NC, with official references/URLs.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Home_Occupation_Permit",
        "States whether a Raleigh home occupation permit is required and provides an official Raleigh/government reference URL.",
        data.home_occupation_permit,
        critical=True,
        require_official=True,
        fallback_claim_hint="Raleigh requires (or does not require) a home occupation permit/approval for operating a home-based business.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Zoning_Approval_Requirements",
        "Explains Raleigh zoning approval/constraints relevant to operating a woodworking workshop from a residence and provides an official Raleigh/government reference URL.",
        data.zoning_approval,
        critical=True,
        require_official=True,
        fallback_claim_hint="Raleigh zoning rules for home-based woodworking workshops (e.g., allowable use, limitations, approvals).",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Accessory_Structure_Permit_Threshold",
        "If an accessory structure is used, states the permit threshold per constraints (zoning permit under 12 feet in any direction; building permit for larger) with an official reference URL.",
        data.accessory_structure_permit_threshold,
        critical=True,
        require_official=True,
        fallback_claim_hint="Accessory structure permit thresholds for Raleigh (e.g., zoning permit under 12 feet in any direction; building permit for larger).",
    )


async def build_business_registration_nc(
    evaluator: Evaluator,
    parent,
    data: BusinessRegistrationExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Business_Registration_North_Carolina",
        desc="Steps to legally register the business in North Carolina, including business structure registration and licenses, with official references/URLs.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "NC_Secretary_of_State_Registration",
        "Describes registration with the NC Secretary of State (when applicable to the chosen structure) and provides the official SOS registration URL.",
        data.sos_registration,
        critical=True,
        require_official=True,
        fallback_claim_hint="Register with the NC Secretary of State if forming an LLC, corporation, or other entity requiring SOS filing.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Business_Structure_Registration_Steps",
        "Explains that registration steps depend on business structure and describes the needed filings/registrations at a high level (without inventing structure-specific requirements not stated), with citations/URLs.",
        data.structure_steps,
        critical=True,
        require_official=True,
        fallback_claim_hint="Registration steps depend on structure (sole prop, LLC, corp), and may include SOS filings and tax registrations.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Required_Licenses_Statement",
        "States the constraint that NC does not require a statewide business license and notes the constraint about professional privilege licenses ending July 1, 2024 (with cited source/URL).",
        data.required_licenses_statement,
        critical=True,
        require_official=True,
        fallback_claim_hint="NC does not require a statewide business license; professional privilege licenses ended July 1, 2024.",
    )


async def build_sales_tax_obligations(
    evaluator: Evaluator,
    parent,
    data: SalesTaxExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Sales_Tax_Obligations",
        desc="Sales tax rate and sales-tax-related registration/resale documentation requirements for Raleigh, NC, with sources.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Total_Sales_Tax_Rate_Raleigh",
        "Provides the total combined sales tax rate for Raleigh, NC as a single numeric rate and cites an authoritative source/URL (e.g., NC DOR or equivalent authoritative rate source).",
        data.total_sales_tax_rate_raleigh,
        critical=True,
        require_official=True,
        fallback_claim_hint="Total combined sales tax rate for Raleigh, NC (a single percentage figure).",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Sales_Tax_Permit_Requirements",
        "Explains requirements/process to obtain NC sales tax permit (registration before vending, per constraints) with an official NC DOR reference/URL.",
        data.sales_tax_permit_requirements,
        critical=True,
        require_official=True,
        fallback_claim_hint="North Carolina sales tax permit registration is required before making taxable sales.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Resale_Certificate_Requirements",
        "Explains that resale certificates are available after obtaining a sales tax permit and provides requirements/process with an authoritative reference/URL.",
        data.resale_certificate_requirements,
        critical=True,
        require_official=True,
        fallback_claim_hint="Resale certificates can be used after obtaining a NC sales tax permit; include how to issue/retain.",
    )


async def build_workshop_safety_compliance(
    evaluator: Evaluator,
    parent,
    data: WorkshopSafetyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Workshop_Safety_Compliance",
        desc="OSHA dust and fire safety measures for the woodworking workshop, including numeric thresholds from constraints and citations/URLs.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "OSHA_Wood_Dust_PEL",
        "States the OSHA wood dust permissible exposure limit from constraints (5 mg/m³ respirable fraction, 8-hour TWA) and cites an OSHA/official reference URL.",
        data.osha_wood_dust_pel,
        critical=True,
        require_official=True,
        fallback_claim_hint="OSHA PEL for wood dust (e.g., 5 mg/m³ respirable fraction, 8-hour TWA).",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Dust_Collection_CFM_Range",
        "States the dust collection airflow range from constraints (350–600 CFM) and provides a supporting reference/URL.",
        data.dust_collection_cfm_range,
        critical=True,
        require_official=False,
        fallback_claim_hint="Recommended dust collection airflow range (e.g., 350–600 CFM) for woodworking tools.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Fire_Extinguisher_Thresholds",
        "States fire extinguisher rating and coverage threshold from constraints (≥2A per 3,000 sq ft) and provides a code/official reference URL.",
        data.fire_extinguisher_thresholds,
        critical=True,
        require_official=True,
        fallback_claim_hint="Fire extinguisher rating and coverage threshold (e.g., ≥2A per 3,000 sq ft).",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Sprinkler_System_Threshold",
        "States the sprinkler requirement threshold from constraints (required for woodworking operations exceeding 2,500 sq ft in Group F-1 occupancies) and provides a code/official reference URL.",
        data.sprinkler_system_threshold,
        critical=True,
        require_official=True,
        fallback_claim_hint="Sprinklers required for woodworking operations exceeding 2,500 sq ft in Group F-1 occupancies.",
    )


async def build_craft_fair_insurance(
    evaluator: Evaluator,
    parent,
    data: CraftFairInsuranceExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Craft_Fair_Insurance",
        desc="Typical minimum liability insurance requirements for craft fairs and what must be included in a COI, with numeric limits and references.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Typical_Minimum_Liability_Limits",
        "States typical minimum liability coverage amounts from constraints ($1M per occurrence and $2M aggregate) with reference/URL.",
        data.typical_minimum_liability_limits,
        critical=True,
        require_official=False,
        fallback_claim_hint="Typical craft fair vendor insurance minimums: $1,000,000 per occurrence and $2,000,000 aggregate.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "COI_Contents_And_Additional_Insured",
        "States that a Certificate of Insurance must be provided and that event organizers must be named as additional insured (per constraints), with reference/URL.",
        data.coi_contents_and_additional_insured,
        critical=True,
        require_official=False,
        fallback_claim_hint="COI required and event organizer named as additional insured.",
    )


async def build_raleigh_craft_fair_venues(
    evaluator: Evaluator,
    parent,
    venues_data: VenuesExtraction,
) -> None:
    # The rubric marks the top-level venues section as critical, but its children (Venue_1/2) are non-critical.
    # To satisfy framework constraints (critical parent cannot have non-critical children), we set this section to non-critical.
    venues_node = evaluator.add_parallel(
        id="Raleigh_Craft_Fair_Venues",
        desc="At least two specific Raleigh, NC craft fair/specialty market venues that accept woodworking vendors, with booth specs, application requirements, insurance/documentation requirements, and URLs.",
        parent=parent,
        critical=False,
    )

    # Use up to first two venues; if fewer provided, create placeholders to still build tree
    venues_list = venues_data.venues[:2] if venues_data and venues_data.venues else []
    while len(venues_list) < 2:
        venues_list.append(VenueDetails())

    for idx, v in enumerate(venues_list, start=1):
        v_node = evaluator.add_parallel(
            id=f"Venue_{idx}",
            desc=f"{'First' if idx == 1 else 'Second'} Raleigh venue meeting the venue constraints.",
            parent=venues_node,
            critical=False,
        )

        # Venue Identity and URL: use the venue_url primarily
        identity_claim = ClaimWithSources(
            statement=(f"The page {v.venue_url} is the official page for the venue '{v.name}' in or serving Raleigh, NC."
                       if v and (v.venue_url or v.name) else None),
            urls=[v.venue_url] if v and v.venue_url else [],
        )
        await verify_claim_with_urls(
            evaluator, v_node,
            f"Venue_{idx}_Venue_Identity_And_URL",
            "Provides venue name/location and an official/venue URL supporting the information.",
            identity_claim,
            critical=True,
            require_official=False,
            fallback_claim_hint="Official venue page identifying the market/event in Raleigh, NC.",
        )

        await verify_claim_with_urls(
            evaluator, v_node,
            f"Venue_{idx}_Accepts_Woodworking_Vendors",
            "Indicates the venue accepts woodworking vendors and supports this with a venue/official source.",
            v.accepts_woodworking,
            critical=True,
            require_official=False,
            fallback_claim_hint="This venue accepts woodworking or woodcraft vendors.",
        )

        await verify_claim_with_urls(
            evaluator, v_node,
            f"Venue_{idx}_Booth_Specifications",
            "Provides booth specifications (e.g., size and any vendor-provided items if specified by that venue) with a supporting venue/official source.",
            v.booth_specifications,
            critical=True,
            require_official=False,
            fallback_claim_hint="Venue booth specifications such as size (e.g., 10x10) and included/excluded items.",
        )

        await verify_claim_with_urls(
            evaluator, v_node,
            f"Venue_{idx}_Vendor_Application_Requirements",
            "Provides vendor application requirements for that venue (including photos/booth photo if required by that venue) with a supporting venue/official source.",
            v.vendor_application_requirements,
            critical=True,
            require_official=False,
            fallback_claim_hint="Venue vendor application requirements (e.g., photos, jurying, deadlines).",
        )

        await verify_claim_with_urls(
            evaluator, v_node,
            f"Venue_{idx}_Insurance_Or_Documentation_Requirements",
            "Provides any venue-specific insurance/documentation requirements (e.g., COI/additional insured, sales tax registration evidence) with a supporting venue/official source.",
            v.insurance_or_doc_requirements,
            critical=True,
            require_official=False,
            fallback_claim_hint="Venue requires documentation such as COI, additional insured, or sales tax registration evidence.",
        )


async def build_product_safety_compliance(
    evaluator: Evaluator,
    parent,
    data: ProductSafetyExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Product_Safety_Compliance",
        desc="Safety compliance requirements for handmade wooden products, including children's toys/furniture, with regulatory references and numeric thresholds where applicable.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "General_Handmade_Product_Compliance",
        "States that handmade products are subject to the same compliance requirements as mass-produced goods (per constraints) with reference/URL.",
        data.general_handmade_product_compliance,
        critical=True,
        require_official=True,
        fallback_claim_hint="Handmade products must meet the same safety standards as mass-produced goods.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Childrens_Products_CPSC_And_CPC",
        "States that children's products require CPSC compliance and a Children's Product Certificate (CPC) (per constraints) with official reference/URL.",
        data.childrens_products_cpsc_and_cpc,
        critical=True,
        require_official=True,
        fallback_claim_hint="Children's products require CPSC compliance and a Children's Product Certificate (CPC).",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Numeric_Thresholds_When_Applicable",
        "Includes relevant numeric thresholds/limits for applicable children's product rules (when such thresholds exist) and cites authoritative sources for the thresholds.",
        data.numeric_thresholds_when_applicable,
        critical=True,
        require_official=True,
        fallback_claim_hint="Numeric thresholds for children's product safety (e.g., lead limits, small parts rules) with authoritative citations.",
    )


async def build_ncdor_vendor_requirements(
    evaluator: Evaluator,
    parent,
    data: NCDORVendorRequirementsExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="NCDOR_Vendor_Requirements_Specialty_Markets",
        desc="NC Department of Revenue requirements for vendors at specialty markets/craft fairs, with references/URLs.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Vendor_Certificate_of_Registration",
        "States vendors must have a valid Certificate of Registration for sales tax before vending (per constraints) with NC DOR reference/URL.",
        data.vendor_certificate_of_registration,
        critical=True,
        require_official=True,
        fallback_claim_hint="Vendors must have a valid NC Certificate of Registration (sales tax) before vending.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Organizer_Daily_Registration_List",
        "States organizers must maintain a daily registration list of vendors (per constraints) with NC DOR reference/URL.",
        data.organizer_daily_registration_list,
        critical=True,
        require_official=True,
        fallback_claim_hint="Event organizers must maintain a daily vendor registration list per NC DOR requirements.",
    )


async def build_cost_estimates(
    evaluator: Evaluator,
    parent,
    data: CostEstimatesExtraction,
) -> None:
    node = evaluator.add_parallel(
        id="Cost_Estimates",
        desc="Typical costs: booth fees and annual liability insurance premiums, with numeric ranges and sources.",
        parent=parent,
        critical=True,
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Typical_Booth_Fees_NC_Craft_Fairs",
        "Provides a numeric range/typical cost estimate for booth fees at NC craft fairs and cites sources.",
        data.typical_booth_fees_nc_craft_fairs,
        critical=True,
        require_official=False,
        fallback_claim_hint="Typical NC craft fair booth fees range with a numeric range and a supporting source.",
    )

    await verify_claim_with_urls(
        evaluator, node,
        "Typical_Annual_Insurance_Premiums",
        "Provides the annual liability insurance premium range from constraints ($230–$500/year) with supporting source/URL.",
        data.typical_annual_insurance_premiums,
        critical=True,
        require_official=False,
        fallback_claim_hint="Typical annual craft vendor liability insurance premiums (e.g., $230–$500/year).",
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
    Evaluate an answer for the North Carolina home-based woodworking craft business compliance/setup guide.
    Builds a verification tree mirroring the rubric and verifies each extracted claim against its cited URLs.
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

    # 1) Extract all relevant claims and URLs from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_master(),
        template_class=MasterExtraction,
        extraction_name="extracted_compliance_setup_guide",
    )

    # 2) Build the main rubric node (non-critical to allow partial, but critical children still gate)
    setup_node = evaluator.add_parallel(
        id="North_Carolina_Woodworking_Craft_Business_Setup",
        desc="Evaluate whether the response provides the requested compliance and setup guide for a home-based woodworking craft business in Raleigh, NC, including required references/URLs and numeric thresholds where applicable.",
        parent=root,
        critical=False,  # must be non-critical to allow a mix of critical and non-critical descendants
    )

    # 3) Build each rubric section
    await build_home_workshop_requirements(evaluator, setup_node, extracted.home_workshop)
    await build_business_registration_nc(evaluator, setup_node, extracted.business_registration)
    await build_sales_tax_obligations(evaluator, setup_node, extracted.sales_tax)
    await build_workshop_safety_compliance(evaluator, setup_node, extracted.safety)
    await build_craft_fair_insurance(evaluator, setup_node, extracted.insurance)
    await build_raleigh_craft_fair_venues(evaluator, setup_node, extracted.venues)
    await build_product_safety_compliance(evaluator, setup_node, extracted.product_safety)
    await build_ncdor_vendor_requirements(evaluator, setup_node, extracted.ncdor_vendor)
    await build_cost_estimates(evaluator, setup_node, extracted.costs)

    # 4) Return summary
    return evaluator.get_summary()