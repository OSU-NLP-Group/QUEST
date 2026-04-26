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
TASK_ID = "tx_esthetician_mocra_compliance"
TASK_DESCRIPTION = """
You are planning to establish an esthetician practice in Texas that will provide professional skincare treatments. As part of your business model, you will manufacture and retail three cosmetic products: (1) a daily facial moisturizer for general skincare, (2) an eyelash growth serum that contacts the eye area, and (3) a 48-hour wear lip tint.

Provide a comprehensive regulatory compliance analysis that identifies:
- The minimum training hours required for a Texas esthetician license
- Whether FDA facility registration is required for your manufacturing operation and, if so, the required form number, prerequisite federal identifier, and registration renewal period
- Whether FDA product listing is required and, if so, the required form number, mandatory information elements that must be included, and the frequency of required updates
- Which of your three products qualify for FDA small business exemptions under MoCRA, considering the specific exemption exclusion criteria for eye-contact products and long-duration wear products
- The mandatory business licenses required to operate a retail cosmetics business
- Which certification is recognized as the gold standard for cruelty-free cosmetics verification

Your analysis must be grounded in current FDA MoCRA regulations (effective December 2023) and Texas state licensing requirements.
"""

ROOT_DESC = "Regulatory compliance analysis for Texas esthetician practice + MoCRA obligations + retail licensing + cruelty-free certification, grounded in MoCRA (effective Dec 2023) and Texas requirements"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RegulatoryBasisExtraction(BaseModel):
    mentions_mocra_dec_2023: Optional[bool] = None
    mentions_texas_licensing: Optional[bool] = None
    cited_urls: List[str] = Field(default_factory=list)


class TexasLicensingExtraction(BaseModel):
    stated_hours: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class RegistrationDetailsExtraction(BaseModel):
    registration_required: Optional[bool] = None
    form_number: Optional[str] = None
    fei_prerequisite: Optional[bool] = None
    renewal_period: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ListingDetailsExtraction(BaseModel):
    listing_required: Optional[bool] = None
    form_number: Optional[str] = None
    update_frequency: Optional[str] = None
    ingredients_order_predominance: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


class ProductExemptionInfo(BaseModel):
    qualifies: Optional[bool] = None
    justification: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class SmallBusinessExemptionsExtraction(BaseModel):
    moisturizer: Optional[ProductExemptionInfo] = None
    eyelash_serum: Optional[ProductExemptionInfo] = None
    lip_tint_48hr: Optional[ProductExemptionInfo] = None


class RetailLicensingExtraction(BaseModel):
    licenses: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class CrueltyFreeExtraction(BaseModel):
    certification: Optional[str] = None
    core_requirement: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_regulatory_basis() -> str:
    return """
    Determine whether the answer explicitly grounds its analysis in BOTH:
    1) MoCRA regulations effective December 2023 (look for 'MoCRA' and a reference to 'December 2023' or equivalent date language), and
    2) Texas state licensing requirements (explicit mention of Texas licensing or TDLR).
    
    Return:
    - mentions_mocra_dec_2023: true/false depending on explicit mention of MoCRA and the Dec 2023 effective date
    - mentions_texas_licensing: true/false depending on explicit mention of Texas state licensing requirements
    - cited_urls: array of any URLs cited for regulatory basis (FDA, TDLR, Texas Comptroller, etc.). Only include URLs explicitly present in the answer.
    """


def prompt_extract_texas_hours() -> str:
    return """
    Extract the minimum training hours stated in the answer for a Texas esthetician license.
    Return:
    - stated_hours: the hours value exactly as stated (e.g., '750', '750 hours', etc.). If no hours are stated, return null.
    - source_urls: array of any URLs cited for this requirement (e.g., TDLR pages). If none, return an empty array.
    """


def prompt_extract_registration_details() -> str:
    return """
    Extract the answer's stated position and details regarding FDA facility registration under MoCRA:
    - registration_required: true/false depending on whether the answer says facility registration is required
    - form_number: the form number if given (e.g., 'Form FDA 5066'); otherwise null
    - fei_prerequisite: true/false depending on whether the answer states an FEI (Facility Establishment Identifier) is required
    - renewal_period: the renewal period as stated (e.g., 'every 2 years', 'biennial'); otherwise null
    - source_urls: array of URLs cited for these facility registration details (prefer FDA pages). If none, return an empty array.
    """


def prompt_extract_listing_details() -> str:
    return """
    Extract the answer's stated position and details regarding FDA cosmetic product listing under MoCRA:
    - listing_required: true/false depending on whether the answer says product listing is required
    - form_number: the form number if given (e.g., 'Form FDA 5067'); otherwise null
    - update_frequency: the frequency of required updates as stated (e.g., 'annual'); otherwise null
    - ingredients_order_predominance: true/false depending on whether the answer states ingredients must be listed in order of predominance (as a mandatory information element)
    - source_urls: array of URLs cited for these product listing details (prefer FDA pages). If none, return an empty array.
    """


def prompt_extract_small_business_exemptions() -> str:
    return """
    For MoCRA small business exemptions, extract the answer's determination and justification for each product:
    - moisturizer: { qualifies: true/false, justification: text, source_urls: [urls...] }
    - eyelash_serum: { qualifies: true/false, justification: text, source_urls: [urls...] }
    - lip_tint_48hr: { qualifies: true/false, justification: text, source_urls: [urls...] }
    
    The justification should reference the relevant exclusion criteria (e.g., products that contact the eye mucous membrane; products that alter appearance for >24 hours). If the answer does not provide a clear determination, set qualifies to null and justification to null. Extract any URLs cited (prefer FDA pages).
    """


def prompt_extract_retail_licensing() -> str:
    return """
    Extract the mandatory business licenses the answer lists for operating a retail cosmetics business.
    Return:
    - licenses: an array of license names exactly as stated (e.g., 'business license', 'seller's permit', 'sales tax permit'). Include synonyms related to seller's permit such as 'sales and use tax permit'.
    - source_urls: array of URLs cited for these licensing requirements (e.g., Texas Comptroller for sales tax permit). If none, return an empty array.
    """


def prompt_extract_cruelty_free() -> str:
    return """
    Extract the cruelty-free certification identified as the gold-standard and the core requirement stated.
    Return:
    - certification: the certification name (e.g., 'Leaping Bunny') if stated; otherwise null
    - core_requirement: the stated core requirement (e.g., 'supply chain management system free of animal testing at all stages'); otherwise null
    - source_urls: array of URLs cited for the certification and requirement (prefer the official Leaping Bunny site or recognized NGOs). If none, return an empty array.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _contains_str(s: Optional[str], needle: str) -> bool:
    return bool(s and needle.lower() in s.lower())


def _has_number(s: Optional[str], number: str) -> bool:
    return bool(s and number in s)


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_regulatory_basis(
    evaluator: Evaluator,
    parent_node,
    basis: RegulatoryBasisExtraction,
) -> None:
    """
    Verify that the analysis is explicitly grounded in MoCRA (Dec 2023) and Texas licensing requirements.
    """
    node = evaluator.add_leaf(
        id="Regulatory_Basis_Currency",
        desc="Analysis is explicitly grounded in MoCRA regulations effective Dec 2023 and Texas state licensing requirements (as stated in the prompt)",
        parent=parent_node,
        critical=True,
    )

    # If the answer does not explicitly mention both, fail fast
    mentions_both = (basis.mentions_mocra_dec_2023 is True) and (basis.mentions_texas_licensing is True)
    if not mentions_both:
        # Mark as failed without invoking LLM
        node.score = 0.0
        node.status = "failed"
        return

    claim = (
        "The answer explicitly states that its regulatory basis includes MoCRA regulations effective December 2023 "
        "and Texas state licensing requirements."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Check the answer text for explicit reference to 'MoCRA' with a December 2023 effective date and explicit mention of Texas state licensing requirements (e.g., TDLR)."
    )


async def verify_texas_licensing_hours(
    evaluator: Evaluator,
    parent_node,
    tx_hours: TexasLicensingExtraction,
) -> None:
    """
    Verify the minimum training hours for a Texas esthetician license (750 hours).
    """
    group = evaluator.add_parallel(
        id="Texas_Esthetician_Licensing",
        desc="Texas esthetician licensing minimum training hours identified",
        parent=parent_node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="Training_Hours_Minimum",
        desc="States the minimum training hours required for a Texas esthetician license (750 hours)",
        parent=group,
        critical=True,
    )

    # Require that the answer explicitly states 750 hours; otherwise fail
    if not _has_number(tx_hours.stated_hours, "750"):
        leaf.score = 0.0
        leaf.status = "failed"
        return

    claim = "The minimum training hours required for a Texas esthetician license is 750 hours."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=tx_hours.source_urls,
        additional_instruction="Confirm using Texas TDLR or official state sources that esthetician license training requires 750 hours. If the source disagrees or is missing, mark as not supported."
    )


async def verify_mocra_registration_and_listing(
    evaluator: Evaluator,
    parent_node,
    reg: RegistrationDetailsExtraction,
    listing: ListingDetailsExtraction,
) -> None:
    """
    Verify MoCRA facility registration and product listing requirements and key details.
    """
    group = evaluator.add_parallel(
        id="MoCRA_Registration_and_Listing",
        desc="MoCRA facility registration + product listing requirements addressed",
        parent=parent_node,
        critical=True,
    )

    # Facility registration leaf
    reg_leaf = evaluator.add_leaf(
        id="Facility_Registration_Requirement_and_Details",
        desc="States whether FDA facility registration is required for the manufacturing operation; if required, includes Form FDA 5066, FEI prerequisite, and renewal period (every 2 years)",
        parent=group,
        critical=True,
    )

    # Gate: the answer should clearly state requirement and include all details if required
    if reg.registration_required is None:
        reg_leaf.score = 0.0
        reg_leaf.status = "failed"
    else:
        if reg.registration_required:
            has_form = _contains_str(reg.form_number, "5066")
            has_fei = reg.fei_prerequisite is True
            has_renewal = reg.renewal_period is not None and (
                _contains_str(reg.renewal_period, "2") or _contains_str(reg.renewal_period, "biennial")
            )
            if not (has_form and has_fei and has_renewal):
                reg_leaf.score = 0.0
                reg_leaf.status = "failed"
            else:
                claim_reg = (
                    "Under MoCRA, cosmetic manufacturing facilities must register with the FDA; "
                    "the required form is Form FDA 5066; a Facility Establishment Identifier (FEI) is a prerequisite; "
                    "and registration must be renewed every 2 years (biennially)."
                )
                await evaluator.verify(
                    claim=claim_reg,
                    node=reg_leaf,
                    sources=reg.source_urls,
                    additional_instruction="Verify on official FDA MoCRA pages that facility registration is required, uses Form FDA 5066, requires an FEI, and must be renewed biennially."
                )
        else:
            claim_reg = "Under MoCRA, cosmetic manufacturing facility registration is not required for this operation."
            await evaluator.verify(
                claim=claim_reg,
                node=reg_leaf,
                sources=reg.source_urls,
                additional_instruction="Verify whether the claim that facility registration is not required is supported by official FDA MoCRA sources. If not supported, mark as failed."
            )

    # Product listing leaf
    list_leaf = evaluator.add_leaf(
        id="Product_Listing_Requirement_and_Details",
        desc="States whether FDA product listing is required; if required, includes Form FDA 5067, required update frequency (annual), and mandatory info element that ingredients must be listed in order of predominance",
        parent=group,
        critical=True,
    )

    # Gate: the answer should clearly state requirement and include all details if required
    if listing.listing_required is None:
        list_leaf.score = 0.0
        list_leaf.status = "failed"
    else:
        if listing.listing_required:
            has_form = _contains_str(listing.form_number, "5067")
            has_update = listing.update_frequency is not None and _contains_str(listing.update_frequency, "annual")
            has_ingredients = listing.ingredients_order_predominance is True
            if not (has_form and has_update and has_ingredients):
                list_leaf.score = 0.0
                list_leaf.status = "failed"
            else:
                claim_list = (
                    "Under MoCRA, cosmetic product listing is required; the required form is Form FDA 5067; "
                    "updates must be submitted annually; and the mandatory information includes ingredients listed in order of predominance."
                )
                await evaluator.verify(
                    claim=claim_list,
                    node=list_leaf,
                    sources=listing.source_urls,
                    additional_instruction="Verify on official FDA MoCRA pages that product listing uses Form FDA 5067, requires annual updates, and ingredients must be listed in order of predominance."
                )
        else:
            claim_list = "Under MoCRA, cosmetic product listing is not required for these products."
            await evaluator.verify(
                claim=claim_list,
                node=list_leaf,
                sources=listing.source_urls,
                additional_instruction="Verify whether the claim that product listing is not required is supported by official FDA MoCRA sources. If not supported, mark as failed."
            )


async def verify_small_business_exemptions(
    evaluator: Evaluator,
    parent_node,
    ex: SmallBusinessExemptionsExtraction,
) -> None:
    """
    Verify small business exemption eligibility for each product using MoCRA exclusion criteria.
    """
    group = evaluator.add_parallel(
        id="MoCRA_Small_Business_Exemption_By_Product",
        desc="Small business exemption eligibility determined for each of the three products using the specified exclusion criteria (eye mucous membrane contact; >24-hour appearance alteration)",
        parent=parent_node,
        critical=True,
    )

    # Product 1: Daily facial moisturizer
    p1_leaf = evaluator.add_leaf(
        id="Product_1_Daily_Facial_Moisturizer",
        desc="States whether the daily facial moisturizer qualifies for the small business exemption, justified using the eye-contact and >24-hour-wear exclusion criteria as applicable",
        parent=group,
        critical=True,
    )
    if not ex.moisturizer or ex.moisturizer.qualifies is None or not ex.moisturizer.justification:
        p1_leaf.score = 0.0
        p1_leaf.status = "failed"
    else:
        claim_p1 = (
            "Under MoCRA small business exemptions, the daily facial moisturizer qualifies for the exemption "
            "because it does not contact the eye mucous membrane and does not alter appearance for longer than 24 hours."
            if ex.moisturizer.qualifies
            else "Under MoCRA small business exemptions, the daily facial moisturizer does not qualify for the exemption."
        )
        await evaluator.verify(
            claim=claim_p1,
            node=p1_leaf,
            sources=ex.moisturizer.source_urls,
            additional_instruction="Use MoCRA small business exemption rules. Exemptions generally do not apply to products contacting the eye mucous membrane or those that alter appearance for >24 hours. Verify the stated determination with official FDA sources."
        )

    # Product 2: Eyelash growth serum
    p2_leaf = evaluator.add_leaf(
        id="Product_2_Eyelash_Growth_Serum",
        desc="States whether the eyelash growth serum qualifies for the small business exemption, justified using the eye-contact and >24-hour-wear exclusion criteria as applicable",
        parent=group,
        critical=True,
    )
    if not ex.eyelash_serum or ex.eyelash_serum.qualifies is None or not ex.eyelash_serum.justification:
        p2_leaf.score = 0.0
        p2_leaf.status = "failed"
    else:
        claim_p2 = (
            "Under MoCRA small business exemptions, the eyelash growth serum that contacts the eye area does not qualify for the exemption due to the eye mucous membrane contact exclusion."
            if ex.eyelash_serum.qualifies is False
            else "Under MoCRA small business exemptions, the eyelash growth serum qualifies for the exemption."
        )
        await evaluator.verify(
            claim=claim_p2,
            node=p2_leaf,
            sources=ex.eyelash_serum.source_urls,
            additional_instruction="Verify using MoCRA rules that products contacting the eye mucous membrane are excluded from small business exemptions. If the claim contradicts this rule, mark as not supported."
        )

    # Product 3: 48-hour wear lip tint
    p3_leaf = evaluator.add_leaf(
        id="Product_3_48_Hour_Wear_Lip_Tint",
        desc="States whether the 48-hour wear lip tint qualifies for the small business exemption, justified using the eye-contact and >24-hour-wear exclusion criteria as applicable",
        parent=group,
        critical=True,
    )
    if not ex.lip_tint_48hr or ex.lip_tint_48hr.qualifies is None or not ex.lip_tint_48hr.justification:
        p3_leaf.score = 0.0
        p3_leaf.status = "failed"
    else:
        claim_p3 = (
            "Under MoCRA small business exemptions, the 48-hour wear lip tint does not qualify for the exemption due to the >24-hour appearance alteration exclusion."
            if ex.lip_tint_48hr.qualifies is False
            else "Under MoCRA small business exemptions, the 48-hour wear lip tint qualifies for the exemption."
        )
        await evaluator.verify(
            claim=claim_p3,
            node=p3_leaf,
            sources=ex.lip_tint_48hr.source_urls,
            additional_instruction="Verify using MoCRA rules that products intended to alter appearance for longer than 24 hours are excluded from small business exemptions. If the claim contradicts this rule, mark as not supported."
        )


async def verify_retail_and_cruelty_free(
    evaluator: Evaluator,
    parent_node,
    rl: RetailLicensingExtraction,
    cf: CrueltyFreeExtraction,
) -> None:
    """
    Verify retail licensing requirements and cruelty-free gold-standard certification.
    """
    group = evaluator.add_parallel(
        id="Retail_Licensing_and_Cruelty_Free",
        desc="Retail licensing requirements and cruelty-free gold-standard certification identified",
        parent=parent_node,
        critical=True,
    )

    # Retail licenses leaf
    retail_leaf = evaluator.add_leaf(
        id="Retail_Licenses",
        desc="Identifies the mandatory business licenses required to operate the retail cosmetics business (business license and seller's permit)",
        parent=group,
        critical=True,
    )

    # Gate: the answer should list both business license AND seller's permit (accept synonyms)
    licenses_lower = [l.lower() for l in rl.licenses]
    has_business_license = any("business license" in l for l in licenses_lower)
    has_seller_permit = any(
        ("seller" in l and "permit" in l) or ("sales tax" in l) or ("sales and use tax" in l)
        for l in licenses_lower
    )
    if not (has_business_license and has_seller_permit):
        retail_leaf.score = 0.0
        retail_leaf.status = "failed"
    else:
        claim_retail = (
            "To operate a retail cosmetics business, a general business license and a seller's permit (sales and use tax permit) are mandatory."
        )
        await evaluator.verify(
            claim=claim_retail,
            node=retail_leaf,
            sources=rl.source_urls,
            additional_instruction="Verify with credible sources (e.g., Texas Comptroller for the sales/use tax permit and local government guidance for business licenses) that these permits are required."
        )

    # Cruelty-free leaf
    cf_leaf = evaluator.add_leaf(
        id="Cruelty_Free_Gold_Standard",
        desc="Identifies Leaping Bunny as the gold-standard cruelty-free certification; includes the stated core requirement that it requires a supply chain management system free of animal testing at all stages",
        parent=group,
        critical=True,
    )

    # Gate: the answer should name Leaping Bunny and describe the supply chain requirement
    has_leaping_bunny = _contains_str(cf.certification, "Leaping Bunny")
    has_supply_chain_req = cf.core_requirement is not None and (
        _contains_str(cf.core_requirement, "supply chain") and
        (_contains_str(cf.core_requirement, "all stages") or _contains_str(cf.core_requirement, "no animal testing"))
    )
    if not (has_leaping_bunny and has_supply_chain_req):
        cf_leaf.score = 0.0
        cf_leaf.status = "failed"
    else:
        claim_cf = (
            "Leaping Bunny is recognized as the gold-standard cruelty-free certification, and it requires a supply chain management system that is free of animal testing at all stages."
        )
        await evaluator.verify(
            claim=claim_cf,
            node=cf_leaf,
            sources=cf.source_urls,
            additional_instruction="Verify with Leaping Bunny or recognized NGOs that the program requires a comprehensive supply chain monitoring system and prohibition of animal testing at all stages."
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
    Evaluate an answer for the Texas esthetician + MoCRA compliance analysis task.
    """
    # Initialize evaluator (root: parallel aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=ROOT_DESC,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info from the answer (parallelize where possible)
    (
        basis,
        tx_hours,
        reg,
        listing,
        exemptions,
        retail,
        cf,
    ) = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_regulatory_basis(),
            template_class=RegulatoryBasisExtraction,
            extraction_name="regulatory_basis",
        ),
        evaluator.extract(
            prompt=prompt_extract_texas_hours(),
            template_class=TexasLicensingExtraction,
            extraction_name="texas_hours",
        ),
        evaluator.extract(
            prompt=prompt_extract_registration_details(),
            template_class=RegistrationDetailsExtraction,
            extraction_name="facility_registration",
        ),
        evaluator.extract(
            prompt=prompt_extract_listing_details(),
            template_class=ListingDetailsExtraction,
            extraction_name="product_listing",
        ),
        evaluator.extract(
            prompt=prompt_extract_small_business_exemptions(),
            template_class=SmallBusinessExemptionsExtraction,
            extraction_name="small_business_exemptions",
        ),
        evaluator.extract(
            prompt=prompt_extract_retail_licensing(),
            template_class=RetailLicensingExtraction,
            extraction_name="retail_licensing",
        ),
        evaluator.extract(
            prompt=prompt_extract_cruelty_free(),
            template_class=CrueltyFreeExtraction,
            extraction_name="cruelty_free_certification",
        ),
    )

    # Optional ground truth info for reference in summary
    evaluator.add_ground_truth({
        "expected_min_training_hours_tx_esthetician": "750",
        "mocra_facility_registration": {
            "form": "Form FDA 5066",
            "fei_required": True,
            "renewal": "every 2 years (biennial)"
        },
        "mocra_product_listing": {
            "form": "Form FDA 5067",
            "update_frequency": "annual",
            "ingredients_order": "list ingredients in order of predominance"
        },
        "small_business_exemption_exclusions": [
            "Products that contact the eye mucous membrane",
            "Products intended to alter appearance for longer than 24 hours"
        ],
        "retail_mandatory_licenses": [
            "Business license",
            "Seller's permit (sales/use tax permit)"
        ],
        "cruelty_free_gold_standard": {
            "certification": "Leaping Bunny",
            "core_requirement": "supply chain free of animal testing at all stages"
        }
    }, gt_type="reference_expectations")

    # Build tree and verify each rubric item
    # 1) Regulatory basis currency
    await verify_regulatory_basis(evaluator, root, basis)

    # 2) Texas esthetician licensing
    await verify_texas_licensing_hours(evaluator, root, tx_hours)

    # 3) MoCRA facility registration & product listing
    await verify_mocra_registration_and_listing(evaluator, root, reg, listing)

    # 4) Small business exemptions by product
    await verify_small_business_exemptions(evaluator, root, exemptions)

    # 5) Retail licensing + cruelty-free gold standard
    await verify_retail_and_cruelty_free(evaluator, root, retail, cf)

    # Return structured evaluation summary
    return evaluator.get_summary()