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
TASK_ID = "va_commercial_props_three_cities"
TASK_DESCRIPTION = (
    "I'm evaluating commercial real estate investment opportunities in Virginia and need to identify suitable rental "
    "properties in three different cities. For each of the following Virginia cities—Richmond, Virginia Beach, and "
    "Arlington—please identify one commercial rental property that is currently available or recently listed.\n\n"
    "For each property, please provide:\n"
    "1. The property address and confirmation of its location in the specified city\n"
    "2. The property type and its suitability for commercial rental purposes\n"
    "3. Confirmation that property management of rental properties in Virginia requires a real estate broker or "
    "salesperson license under Virginia law § 54.1-2135\n"
    "4. Information indicating that current commercial mortgage rates (as of February 2026) for this property type "
    "   fall within the range of 4.73% to 12.75%\n"
    "5. Confirmation that commercial property taxes for investment properties like this are tax-deductible under IRS "
    "   guidelines\n"
    "6. A reference URL supporting the property information"
)

MORTGAGE_RATE_RANGE = (4.73, 12.75)
AS_OF_DATE = "February 2026"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CityProperty(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    property_type: Optional[str] = None
    property_urls: List[str] = Field(default_factory=list)


class PropertiesExtraction(BaseModel):
    # One property per city
    richmond: Optional[CityProperty] = None
    virginia_beach: Optional[CityProperty] = None
    arlington: Optional[CityProperty] = None

    # Global regulatory and finance sources
    management_license_urls: List[str] = Field(default_factory=list)
    mortgage_rate_urls: List[str] = Field(default_factory=list)
    tax_deductibility_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    You must extract structured information from the answer about three Virginia cities: Richmond, Virginia Beach, and Arlington.
    For each city, extract exactly one commercial rental property (choose the first one mentioned for that city if multiple are listed).
    
    For each of the three cities, extract the following fields:
    - address: The street address of the property as written in the answer (include city/state if present).
    - city: The city name associated with the property as written in the answer (e.g., "Richmond", "Virginia Beach", or "Arlington").
    - property_type: The type/category of the property as written (e.g., office, retail, industrial, warehouse, flex, mixed-use, etc.).
    - property_urls: An array of URL(s) explicitly cited in the answer that directly reference the property listing or official page supporting its details (address, type, availability, etc.).
    
    Additionally, extract the following global source URLs mentioned anywhere in the answer (they may be reused for multiple cities):
    - management_license_urls: URL(s) that support the statement that in Virginia, property management of rental properties requires a real estate broker or salesperson license, preferably citing or referencing Virginia law § 54.1-2135 or official DPOR/Virginia Code pages.
    - mortgage_rate_urls: URL(s) that provide current commercial mortgage rates (as of February 2026) for commercial properties (general or property-type-specific).
    - tax_deductibility_urls: URL(s) supporting that commercial property taxes on investment properties are tax-deductible under IRS guidelines (e.g., IRS or other authoritative tax sources).
    
    IMPORTANT URL RULES:
    - Extract only URLs explicitly present in the answer (including plain URLs or markdown links).
    - Include full URLs. If a URL lacks protocol, prepend http://
    - Do not invent or infer URLs.
    
    If any field is missing, set it to null (for strings) or an empty array (for URL arrays).
    Use keys exactly as defined in the schema.
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def has_valid_url(urls: Optional[List[str]]) -> bool:
    if not urls:
        return False
    for u in urls:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            return True
    return False


def safe_text(value: Optional[str], fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification logic per city                                                 #
# --------------------------------------------------------------------------- #
async def verify_city_property(
    evaluator: Evaluator,
    parent_node,
    city_id_prefix: str,     # e.g., "richmond", "vb", "arlington"
    city_node_id: str,       # e.g., "richmond_property", "virginia_beach_property", "arlington_property"
    city_node_desc: str,     # description for city node
    required_city_name: str, # "Richmond", "Virginia Beach", "Arlington"
    property_item: Optional[CityProperty],
    management_license_urls: List[str],
    mortgage_rate_urls: List[str],
    tax_deductibility_urls: List[str],
):
    """
    Build the sub-tree and perform verifications for one city's property.
    All criterion leaves under a city are critical (as per rubric). The city node itself is non-critical
    relative to the root to allow partial credit across cities.
    """
    # City node (parallel aggregation to evaluate criteria independently)
    city_node = evaluator.add_parallel(
        id=city_node_id,
        desc=city_node_desc,
        parent=parent_node,
        critical=False
    )

    # Gather basic fields
    address_text = safe_text(property_item.address if property_item else None, "the referenced property")
    prop_type_text = safe_text(property_item.property_type if property_item else None, "commercial property")
    prop_urls = property_item.property_urls if property_item and property_item.property_urls else []

    # Reference URL presence (Critical) – added first to act as a gate for property-specific checks
    ref_url_exists = has_valid_url(prop_urls)
    evaluator.add_custom_node(
        result=ref_url_exists,
        id=f"{city_id_prefix}_reference_url",
        desc="A valid reference URL is provided that supports the property information.",
        parent=city_node,
        critical=True
    )

    # Location verification (Critical)
    location_leaf = evaluator.add_leaf(
        id=f"{city_id_prefix}_location",
        desc=f"The property is located in {required_city_name}, Virginia.",
        parent=city_node,
        critical=True
    )
    loc_claim = f"The property at '{address_text}' is located in {required_city_name}, Virginia."
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=prop_urls,
        additional_instruction=(
            f"Verify on the referenced property page that the address/location explicitly indicates {required_city_name}, VA. "
            "Minor variations in formatting are acceptable. If the page does not clearly show the city in Virginia, mark as not supported."
        ),
    )

    # Property type & rental suitability (Critical)
    type_leaf = evaluator.add_leaf(
        id=f"{city_id_prefix}_property_type",
        desc="The property is a commercial property suitable for rental purposes.",
        parent=city_node,
        critical=True
    )
    type_claim = (
        f"The property at '{address_text}' is a commercial property suitable for lease/rental, "
        f"with a type described as '{prop_type_text}'."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=prop_urls,
        additional_instruction=(
            "Confirm the page shows it's a commercial property and is for lease/rent (e.g., lease terms, 'for lease', "
            "'for rent', 'available'). Accept common commercial categories like office, retail, industrial, flex, "
            "warehouse, or mixed-use."
        ),
    )

    # Management licensing requirement (Critical)
    if has_valid_url(management_license_urls):
        ml_leaf = evaluator.add_leaf(
            id=f"{city_id_prefix}_management_license",
            desc="Confirmation that property management of this rental property in Virginia requires a real estate broker or salesperson license per Virginia law § 54.1-2135.",
            parent=city_node,
            critical=True
        )
        ml_claim = (
            "In Virginia, managing rental properties requires a real estate broker or salesperson license under Virginia law § 54.1-2135."
        )
        await evaluator.verify(
            claim=ml_claim,
            node=ml_leaf,
            sources=management_license_urls,
            additional_instruction=(
                "Look for authoritative Virginia sources (e.g., Virginia Code, DPOR) explicitly stating that "
                "property management requires a broker or salesperson license, ideally referencing § 54.1-2135."
            ),
        )
    else:
        # Fail the criterion when no source is provided (enforce source-grounding for factual legal claim)
        evaluator.add_custom_node(
            result=False,
            id=f"{city_id_prefix}_management_license",
            desc="Confirmation that property management of this rental property in Virginia requires a real estate broker or salesperson license per Virginia law § 54.1-2135.",
            parent=city_node,
            critical=True
        )

    # Mortgage rate range verification (Critical)
    if has_valid_url(mortgage_rate_urls):
        rate_leaf = evaluator.add_leaf(
            id=f"{city_id_prefix}_mortgage_rate",
            desc=f"Current commercial mortgage rates available for this property type fall within the {AS_OF_DATE} range of {MORTGAGE_RATE_RANGE[0]}% to {MORTGAGE_RATE_RANGE[1]}%.",
            parent=city_node,
            critical=True
        )
        rate_claim = (
            f"As of {AS_OF_DATE}, current commercial mortgage rates applicable to {prop_type_text} properties fall "
            f"within the range of {MORTGAGE_RATE_RANGE[0]}% to {MORTGAGE_RATE_RANGE[1]}%."
        )
        await evaluator.verify(
            claim=rate_claim,
            node=rate_leaf,
            sources=mortgage_rate_urls,
            additional_instruction=(
                f"Check the page for current commercial mortgage rate ranges around {AS_OF_DATE}. "
                "Accept reasonable rounding (e.g., 4.7% for 4.73%). If the page shows typical/average ranges, "
                "ensure they fit within 4.73%–12.75%."
            ),
        )
    else:
        # Fail if no rate source is provided
        evaluator.add_custom_node(
            result=False,
            id=f"{city_id_prefix}_mortgage_rate",
            desc=f"Current commercial mortgage rates available for this property type fall within the {AS_OF_DATE} range of {MORTGAGE_RATE_RANGE[0]}% to {MORTGAGE_RATE_RANGE[1]}%.",
            parent=city_node,
            critical=True
        )

    # Tax deductibility verification (Critical)
    if has_valid_url(tax_deductibility_urls):
        tax_leaf = evaluator.add_leaf(
            id=f"{city_id_prefix}_tax_deductible",
            desc="Commercial property taxes for this investment property are tax-deductible according to IRS guidelines.",
            parent=city_node,
            critical=True
        )
        tax_claim = "Property taxes on commercial investment properties are tax-deductible under IRS business expense rules."
        await evaluator.verify(
            claim=tax_claim,
            node=tax_leaf,
            sources=tax_deductibility_urls,
            additional_instruction=(
                "Look for IRS or authoritative tax sources stating that business property (commercial real estate) taxes "
                "are deductible as an ordinary and necessary business expense."
            ),
        )
    else:
        # Fail if no tax-deductibility source is provided
        evaluator.add_custom_node(
            result=False,
            id=f"{city_id_prefix}_tax_deductible",
            desc="Commercial property taxes for this investment property are tax-deductible according to IRS guidelines.",
            parent=city_node,
            critical=True
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
    Evaluate an answer for the Virginia commercial properties (three-city) task and return a structured result dict.
    """
    # Initialize evaluator (root node as parallel, non-critical to allow partial credit across cities)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify three commercial rental properties, one in each of three specified Virginia cities (Richmond, Virginia Beach, and Arlington), that meet specific investment and operational criteria.",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured information
    extracted: PropertiesExtraction = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=PropertiesExtraction,
        extraction_name="properties_extraction"
    )

    # Record custom expected info
    evaluator.add_ground_truth({
        "target_cities": ["Richmond", "Virginia Beach", "Arlington"],
        "mortgage_rate_range_percent": {"min": MORTGAGE_RATE_RANGE[0], "max": MORTGAGE_RATE_RANGE[1], "as_of": AS_OF_DATE}
    }, gt_type="expected_parameters")

    # Build verification trees per city
    await verify_city_property(
        evaluator=evaluator,
        parent_node=root,
        city_id_prefix="richmond",
        city_node_id="richmond_property",
        city_node_desc="Identify one commercial rental property in Richmond, Virginia that meets all specified criteria.",
        required_city_name="Richmond",
        property_item=extracted.richmond,
        management_license_urls=extracted.management_license_urls,
        mortgage_rate_urls=extracted.mortgage_rate_urls,
        tax_deductibility_urls=extracted.tax_deductibility_urls,
    )

    await verify_city_property(
        evaluator=evaluator,
        parent_node=root,
        city_id_prefix="vb",
        city_node_id="virginia_beach_property",
        city_node_desc="Identify one commercial rental property in Virginia Beach, Virginia that meets all specified criteria.",
        required_city_name="Virginia Beach",
        property_item=extracted.virginia_beach,
        management_license_urls=extracted.management_license_urls,
        mortgage_rate_urls=extracted.mortgage_rate_urls,
        tax_deductibility_urls=extracted.tax_deductibility_urls,
    )

    await verify_city_property(
        evaluator=evaluator,
        parent_node=root,
        city_id_prefix="arlington",
        city_node_id="arlington_property",
        city_node_desc="Identify one commercial rental property in Arlington, Virginia that meets all specified criteria.",
        required_city_name="Arlington",
        property_item=extracted.arlington,
        management_license_urls=extracted.management_license_urls,
        mortgage_rate_urls=extracted.mortgage_rate_urls,
        tax_deductibility_urls=extracted.tax_deductibility_urls,
    )

    # Return the evaluation summary
    return evaluator.get_summary()