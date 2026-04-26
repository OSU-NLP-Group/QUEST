import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "ma_christmas_2025_shopping"
TASK_DESCRIPTION = """
For a family traveling through Massachusetts on Christmas Day 2025 (December 25, 2025) who needs to purchase essential items including prescription medications, food, and fuel, identify 4 different types of retail businesses that will be confirmed open and legally operating on that date in Massachusetts. Your answer must include:

1. One pharmacy chain
2. One convenience store chain
3. One gas station chain with retail services
4. One option for purchasing food/groceries

For each of the 4 businesses identified, provide the following information:

- The business/chain name
- Official confirmation or evidence of their Christmas Day 2025 operating policy, including a reference URL
- Explanation of how this business type is permitted to operate on Christmas Day in Massachusetts under the state's Blue Laws, including a reference URL to the Blue Law information
- Typical Christmas Day operating hours (if standardized across locations)
- What specific services or products are available on Christmas Day (such as pharmacy services, fuel, retail items, food products)

Note: Massachusetts has Blue Laws that restrict most grocery stores and many retailers from opening on Christmas Day. Your answer must demonstrate understanding of these legal restrictions and identify only businesses that are legally permitted to operate on December 25, 2025 in Massachusetts.
"""

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PolicyInfo(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ComplianceInfo(BaseModel):
    exception: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PharmacyServices(BaseModel):
    pharmacy_services: Optional[str] = None
    retail_services: Optional[str] = None


class GasServices(BaseModel):
    fuel_service: Optional[str] = None
    retail_store: Optional[str] = None


class ConvenienceServices(BaseModel):
    services: Optional[str] = None


class GroceryProducts(BaseModel):
    products: Optional[str] = None


class PharmacyChainInfo(BaseModel):
    name: Optional[str] = None
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    hours: Optional[str] = None
    services: PharmacyServices = Field(default_factory=PharmacyServices)


class ConvenienceChainInfo(BaseModel):
    name: Optional[str] = None
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    hours: Optional[str] = None
    services: ConvenienceServices = Field(default_factory=ConvenienceServices)


class GasStationChainInfo(BaseModel):
    name: Optional[str] = None
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    hours: Optional[str] = None
    services: GasServices = Field(default_factory=GasServices)


class GroceryOptionInfo(BaseModel):
    name: Optional[str] = None  # Can be a chain or a type (e.g., small grocery/market)
    policy: PolicyInfo = Field(default_factory=PolicyInfo)
    compliance: ComplianceInfo = Field(default_factory=ComplianceInfo)
    hours: Optional[str] = None
    products: GroceryProducts = Field(default_factory=GroceryProducts)


class ChristmasShoppingExtraction(BaseModel):
    pharmacy: PharmacyChainInfo = Field(default_factory=PharmacyChainInfo)
    convenience: ConvenienceChainInfo = Field(default_factory=ConvenienceChainInfo)
    gas: GasStationChainInfo = Field(default_factory=GasStationChainInfo)
    grocery: GroceryOptionInfo = Field(default_factory=GroceryOptionInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_businesses() -> str:
    return """
Extract the four business options presented in the answer (pharmacy chain, convenience store chain, gas station chain with retail, and a small grocery/food option) with the following fields for each category:

For each of: pharmacy, convenience, gas, grocery, extract:
- name: Chain or option name (for grocery, this may be a type like "small markets" if a chain isn't given)
- policy.statement: The quoted/explicit policy statement or explanation from the answer about Christmas Day 2025 operations
- policy.urls: All URLs cited for the operating policy (list)
- compliance.exception: The explanation of the Massachusetts Blue Law exception that allows this business type to operate on Christmas Day
- compliance.urls: All URLs cited for Blue Law information (list)
- hours: The typical Christmas Day operating hours as stated in the answer (if given; otherwise null)
Additionally:
- pharmacy.services.pharmacy_services: What the answer says about pharmacy prescription counter/service on Christmas Day
- pharmacy.services.retail_services: What the answer says about the front-store/retail section on Christmas Day
- convenience.services.services: What the answer says about services/products available at the convenience chain on Christmas Day
- gas.services.fuel_service: What the answer says about fuel pump availability on Christmas Day
- gas.services.retail_store: What the answer says about the gas station convenience store on Christmas Day
- grocery.products.products: What types of food/grocery products are available for the grocery option on Christmas Day

Rules:
- Extract exactly as written in the answer; do not infer new content.
- For any missing information, return null (for strings) or an empty list (for URLs).
- For URLs, collect all URLs the answer associates with the given field. Preserve them in a list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len([u for u in urls if _has_nonempty_text(u)]) > 0)


def _canonical_status(text: Optional[str]) -> str:
    """Roughly categorize service status from free-form text."""
    if not _has_nonempty_text(text):
        return "unspecified"
    t = text.lower()
    if any(k in t for k in ["closed", "not available", "unavailable", "no service"]):
        return "closed"
    if any(k in t for k in ["open", "available", "24/7", "24x7", "twenty four", "twenty-four"]):
        return "available"
    if any(k in t for k in ["vary", "varies", "limited", "reduced", "check store", "check location"]):
        return "varies"
    return "unspecified"


# --------------------------------------------------------------------------- #
# Verification logic per category                                             #
# --------------------------------------------------------------------------- #
async def verify_pharmacy(evaluator: Evaluator, parent, data: PharmacyChainInfo) -> None:
    node = evaluator.add_parallel(
        id="pharmacy_chain",
        desc="Identify one pharmacy chain confirmed open on Christmas Day 2025 in Massachusetts",
        parent=parent,
        critical=False
    )

    # Name (critical existence)
    evaluator.add_custom_node(
        result=_has_nonempty_text(data.name),
        id="pharmacy_name",
        desc="Provide the name of a national pharmacy chain",
        parent=node,
        critical=True
    )

    # Policy (sequential, critical)
    policy_seq = evaluator.add_sequential(
        id="pharmacy_policy",
        desc="Provide evidence of the pharmacy chain's Christmas Day 2025 operating policy",
        parent=node,
        critical=True
    )

    policy_url_exists = evaluator.add_custom_node(
        result=_has_urls(data.policy.urls),
        id="pharmacy_policy_url_exists",
        desc="Provide URL reference for the policy information",
        parent=policy_seq,
        critical=True
    )

    policy_statement_leaf = evaluator.add_leaf(
        id="pharmacy_policy_statement_supported",
        desc="Include official statement or confirmation about Christmas Day operations",
        parent=policy_seq,
        critical=True
    )
    policy_claim = (
        f"The cited policy page(s) support that {data.name or 'the chain'} stores are open on Christmas Day 2025 "
        f"(December 25, 2025) in Massachusetts or generally (e.g., open 24/7 or with holiday hours). "
        f"Statement provided in the answer: {data.policy.statement or 'None provided'}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_statement_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm the page states being open on Christmas Day or indicates 24/7/holiday hours. If it states 'hours vary by location' but still open, consider it supported."
    )

    # Massachusetts compliance (sequential, critical)
    compliance_seq = evaluator.add_sequential(
        id="pharmacy_compliance",
        desc="Explain how the pharmacy complies with Massachusetts Blue Laws on Christmas",
        parent=node,
        critical=True
    )

    bluelaw_url_exists = evaluator.add_custom_node(
        result=_has_urls(data.compliance.urls),
        id="pharmacy_bluelaw_url_exists",
        desc="Provide URL reference for Blue Law information",
        parent=compliance_seq,
        critical=True
    )

    bluelaw_exception_leaf = evaluator.add_leaf(
        id="pharmacy_bluelaw_exception_supported",
        desc="Identify which Blue Law exception allows this business type to operate",
        parent=compliance_seq,
        critical=True
    )
    bluelaw_claim = (
        f"The cited Massachusetts Blue Law source(s) support that pharmacies/drug stores are permitted to operate "
        f"on Christmas Day (December 25) in Massachusetts. Exception stated: {data.compliance.exception or 'None provided'}."
    )
    await evaluator.verify(
        claim=bluelaw_claim,
        node=bluelaw_exception_leaf,
        sources=data.compliance.urls,
        additional_instruction="Verify the MA blue law page explicitly allows drug stores/pharmacies (or similar essential services) to operate on Christmas Day."
    )

    # Christmas hours (non-critical)
    hours_leaf = evaluator.add_leaf(
        id="pharmacy_hours",
        desc="Provide typical Christmas Day operating hours for this chain",
        parent=node,
        critical=False
    )
    hours_claim = (
        f"The chain's cited policy page(s) support the following Christmas Day hours information: {data.hours or 'None provided'}."
    )
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=data.policy.urls,
        additional_instruction="Check if the holiday/Christmas hours information matches or is reasonably consistent with the provided description. Allow 'hours vary by location' wordings."
    )

    # Services available (non-critical, parallel)
    services_par = evaluator.add_parallel(
        id="pharmacy_services",
        desc="Specify what services are available on Christmas Day",
        parent=node,
        critical=False
    )

    # Pharmacy prescription services
    pharm_status_leaf = evaluator.add_leaf(
        id="pharmacy_service_status",
        desc="Indicate whether pharmacy prescription services are available",
        parent=services_par,
        critical=False
    )
    pharm_status = _canonical_status(data.services.pharmacy_services)
    pharm_claim = (
        f"The cited policy/official page(s) indicate the following about pharmacy prescription services on Christmas Day 2025: {data.services.pharmacy_services or 'None provided'} "
        f"(interpreted as {pharm_status})."
    )
    await evaluator.verify(
        claim=pharm_claim,
        node=pharm_status_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm whether the pharmacy counter is open, closed, or limited/varies on Christmas Day. Allow general statements like 'pharmacy closed but front store open'."
    )

    # Retail/front store services
    retail_status_leaf = evaluator.add_leaf(
        id="pharmacy_retail_status",
        desc="Indicate whether retail/front store is open",
        parent=services_par,
        critical=False
    )
    retail_status = _canonical_status(data.services.retail_services)
    retail_claim = (
        f"The cited policy/official page(s) indicate the following about front-store/retail availability on Christmas Day 2025: {data.services.retail_services or 'None provided'} "
        f"(interpreted as {retail_status})."
    )
    await evaluator.verify(
        claim=retail_claim,
        node=retail_status_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm that non-pharmacy retail area is open/available (or closed) on Christmas Day. Allow 'varies by location' style language."
    )


async def verify_convenience(evaluator: Evaluator, parent, data: ConvenienceChainInfo) -> None:
    node = evaluator.add_parallel(
        id="convenience_chain",
        desc="Identify one convenience store chain confirmed open on Christmas Day 2025 in Massachusetts",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_text(data.name),
        id="convenience_name",
        desc="Provide the name of a convenience store chain",
        parent=node,
        critical=True
    )

    policy_seq = evaluator.add_sequential(
        id="convenience_policy",
        desc="Provide evidence of the convenience store's Christmas Day 2025 operating policy",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.policy.urls),
        id="convenience_policy_url_exists",
        desc="Provide URL reference for the policy information",
        parent=policy_seq,
        critical=True
    )

    policy_statement_leaf = evaluator.add_leaf(
        id="convenience_policy_statement_supported",
        desc="Include official statement or confirmation about Christmas Day operations",
        parent=policy_seq,
        critical=True
    )
    policy_claim = (
        f"The cited policy page(s) support that {data.name or 'the chain'} convenience stores are open on Christmas Day 2025 "
        f"(December 25, 2025) in Massachusetts or generally (e.g., 24/7 or holiday hours). "
        f"Statement provided: {data.policy.statement or 'None provided'}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_statement_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm page(s) indicate Christmas Day operations or 24/7 status. 'Hours vary' but open is acceptable."
    )

    compliance_seq = evaluator.add_sequential(
        id="convenience_compliance",
        desc="Explain how the convenience store complies with Massachusetts Blue Laws on Christmas",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.compliance.urls),
        id="convenience_bluelaw_url_exists",
        desc="Provide URL reference for Blue Law information",
        parent=compliance_seq,
        critical=True
    )

    bluelaw_exception_leaf = evaluator.add_leaf(
        id="convenience_bluelaw_exception_supported",
        desc="Identify which Blue Law exception allows this business type to operate",
        parent=compliance_seq,
        critical=True
    )
    bluelaw_claim = (
        f"The cited Massachusetts Blue Law source(s) support that convenience stores (or similar small food/variety stores) are permitted to operate on Christmas Day in Massachusetts. "
        f"Exception stated: {data.compliance.exception or 'None provided'}."
    )
    await evaluator.verify(
        claim=bluelaw_claim,
        node=bluelaw_exception_leaf,
        sources=data.compliance.urls,
        additional_instruction="Verify that MA Blue Laws explicitly allow convenience or similar stores to open on Christmas Day (e.g., stores selling food, drugstores, etc.)."
    )

    hours_leaf = evaluator.add_leaf(
        id="convenience_hours",
        desc="Provide typical Christmas Day operating hours for this chain",
        parent=node,
        critical=False
    )
    hours_claim = f"The cited policy page(s) support the following Christmas Day hours: {data.hours or 'None provided'}."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=data.policy.urls,
        additional_instruction="Check that the hours information (including 'varies by location') is supported."
    )

    services_leaf = evaluator.add_leaf(
        id="convenience_services",
        desc="Specify what services or products are available on Christmas Day",
        parent=node,
        critical=False
    )
    services_claim = (
        f"The cited policy/official page(s) support that the chain offers the following on Christmas Day 2025: {data.services.services or 'None provided'}."
    )
    await evaluator.verify(
        claim=services_claim,
        node=services_leaf,
        sources=data.policy.urls,
        additional_instruction="Look for confirmation of general convenience items/retail availability on Christmas Day."
    )


async def verify_gas_station(evaluator: Evaluator, parent, data: GasStationChainInfo) -> None:
    node = evaluator.add_parallel(
        id="gas_station_chain",
        desc="Identify one gas station chain with retail store confirmed open on Christmas Day 2025 in Massachusetts",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_text(data.name),
        id="gas_station_name",
        desc="Provide the name of a gas station chain with convenience retail",
        parent=node,
        critical=True
    )

    policy_seq = evaluator.add_sequential(
        id="gas_station_policy",
        desc="Provide evidence of the gas station's Christmas Day 2025 operating policy",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.policy.urls),
        id="gas_station_policy_url_exists",
        desc="Provide URL reference for the policy information",
        parent=policy_seq,
        critical=True
    )

    policy_statement_leaf = evaluator.add_leaf(
        id="gas_station_policy_statement_supported",
        desc="Include official statement or confirmation about Christmas Day operations",
        parent=policy_seq,
        critical=True
    )
    policy_claim = (
        f"The cited policy page(s) support that {data.name or 'the chain'} fuel/convenience locations are open on Christmas Day 2025 "
        f"(December 25, 2025). Statement provided: {data.policy.statement or 'None provided'}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_statement_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm that Christmas Day operations (fuel and/or store) are indicated. Allow 'open 24/7' language."
    )

    compliance_seq = evaluator.add_sequential(
        id="gas_station_compliance",
        desc="Explain how the gas station complies with Massachusetts Blue Laws on Christmas",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.compliance.urls),
        id="gas_station_bluelaw_url_exists",
        desc="Provide URL reference for Blue Law information",
        parent=compliance_seq,
        critical=True
    )

    bluelaw_exception_leaf = evaluator.add_leaf(
        id="gas_station_bluelaw_exception_supported",
        desc="Identify which Blue Law exception allows this business type to operate",
        parent=compliance_seq,
        critical=True
    )
    bluelaw_claim = (
        f"The cited Massachusetts Blue Law source(s) support that gasoline stations (fuel) and associated convenience retail are permitted to operate on Christmas Day. "
        f"Exception stated: {data.compliance.exception or 'None provided'}."
    )
    await evaluator.verify(
        claim=bluelaw_claim,
        node=bluelaw_exception_leaf,
        sources=data.compliance.urls,
        additional_instruction="Verify that MA Blue Laws explicitly allow gasoline filling stations and possibly their attached convenience stores to operate on Christmas Day."
    )

    hours_leaf = evaluator.add_leaf(
        id="gas_station_hours",
        desc="Provide typical Christmas Day operating hours for this chain",
        parent=node,
        critical=False
    )
    hours_claim = f"The cited policy page(s) support the following Christmas Day hours: {data.hours or 'None provided'}."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=data.policy.urls,
        additional_instruction="Check for 'holiday hours' or '24/7' statements; allow 'varies by location'."
    )

    services_par = evaluator.add_parallel(
        id="gas_station_services",
        desc="Specify what services are available on Christmas Day",
        parent=node,
        critical=False
    )

    fuel_leaf = evaluator.add_leaf(
        id="fuel_service_status",
        desc="Indicate whether fuel/gas pumps are available",
        parent=services_par,
        critical=False
    )
    fuel_status = _canonical_status(data.services.fuel_service)
    fuel_claim = (
        f"The cited page(s) indicate fuel/pumps availability on Christmas Day 2025: {data.services.fuel_service or 'None provided'} "
        f"(interpreted as {fuel_status})."
    )
    await evaluator.verify(
        claim=fuel_claim,
        node=fuel_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm that fuel is available; accept 24/7 or standard fuel availability language."
    )

    store_leaf = evaluator.add_leaf(
        id="retail_store_status",
        desc="Indicate whether convenience retail store is open",
        parent=services_par,
        critical=False
    )
    retail_status = _canonical_status(data.services.retail_store)
    store_claim = (
        f"The cited page(s) indicate the gas station convenience store availability on Christmas Day 2025: {data.services.retail_store or 'None provided'} "
        f"(interpreted as {retail_status})."
    )
    await evaluator.verify(
        claim=store_claim,
        node=store_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm that the attached convenience store is open or has limited/varied hours on Christmas Day."
    )


async def verify_grocery_option(evaluator: Evaluator, parent, data: GroceryOptionInfo) -> None:
    node = evaluator.add_parallel(
        id="grocery_option",
        desc="Identify one small grocery store or food retailer option that can legally operate on Christmas Day 2025 in Massachusetts",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=_has_nonempty_text(data.name),
        id="grocery_option_name",
        desc="Provide the type or name of small grocery option available",
        parent=node,
        critical=True
    )

    policy_seq = evaluator.add_sequential(
        id="grocery_policy",
        desc="Provide evidence or explanation of how this option can operate on Christmas Day",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.policy.urls),
        id="grocery_policy_url_exists",
        desc="Provide URL reference for the policy information",
        parent=policy_seq,
        critical=True
    )

    policy_expl_leaf = evaluator.add_leaf(
        id="grocery_policy_explanation_supported",
        desc="Explain the operational status or availability on Christmas Day",
        parent=policy_seq,
        critical=True
    )
    policy_claim = (
        f"The cited page(s) support that the grocery option '{data.name or 'the option'}' can operate on Christmas Day 2025 "
        f"in Massachusetts or generally (e.g., small food retailers permitted/open with holiday hours). "
        f"Explanation provided: {data.policy.statement or 'None provided'}."
    )
    await evaluator.verify(
        claim=policy_claim,
        node=policy_expl_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm that the provided source explains this grocery/food option is available/open on Christmas Day (allow 'varies' or limited hours)."
    )

    compliance_seq = evaluator.add_sequential(
        id="grocery_compliance",
        desc="Explain how this grocery option complies with Massachusetts Blue Laws on Christmas",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(data.compliance.urls),
        id="grocery_bluelaw_url_exists",
        desc="Provide URL reference for Blue Law information",
        parent=compliance_seq,
        critical=True
    )

    bluelaw_exception_leaf = evaluator.add_leaf(
        id="grocery_bluelaw_exception_supported",
        desc="Identify which Blue Law exception allows this business type to operate",
        parent=compliance_seq,
        critical=True
    )
    bluelaw_claim = (
        f"The cited Massachusetts Blue Law source(s) support that this small grocery/food retailer type is permitted to operate on Christmas Day. "
        f"Exception stated: {data.compliance.exception or 'None provided'}."
    )
    await evaluator.verify(
        claim=bluelaw_claim,
        node=bluelaw_exception_leaf,
        sources=data.compliance.urls,
        additional_instruction="Verify that MA Blue Laws list this type of small food retailer (or relevant category) as allowed on Christmas Day."
    )

    hours_leaf = evaluator.add_leaf(
        id="grocery_hours",
        desc="Provide typical Christmas Day operating hours or availability",
        parent=node,
        critical=False
    )
    hours_claim = f"The cited page(s) support the following Christmas Day availability/hours: {data.hours or 'None provided'}."
    await evaluator.verify(
        claim=hours_claim,
        node=hours_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm hours/availability are supported (including 'varies by location')."
    )

    products_leaf = evaluator.add_leaf(
        id="grocery_products_available",
        desc="Specify what types of food or grocery products are available",
        parent=node,
        critical=False
    )
    products_claim = (
        f"The cited page(s) support that the following food/grocery products are available on Christmas Day: {data.products.products or 'None provided'}."
    )
    await evaluator.verify(
        claim=products_claim,
        node=products_leaf,
        sources=data.policy.urls,
        additional_instruction="Confirm the availability of general grocery/food items for this option on Christmas Day."
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
    Evaluate an answer for the Massachusetts Christmas Day 2025 shopping task.
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
        default_model=model
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_businesses(),
        template_class=ChristmasShoppingExtraction,
        extraction_name="extracted_businesses"
    )

    # Top-level root description node (parallel aggregation of four categories)
    top = evaluator.add_parallel(
        id="christmas_day_shopping_root",
        desc="Successfully identify 4 different types of retail businesses open on Christmas Day 2025 in Massachusetts with complete verification details",
        parent=root,
        critical=False
    )

    # Verify each category
    await verify_pharmacy(evaluator, top, extracted.pharmacy)
    await verify_convenience(evaluator, top, extracted.convenience)
    await verify_gas_station(evaluator, top, extracted.gas)
    await verify_grocery_option(evaluator, top, extracted.grocery)

    return evaluator.get_summary()