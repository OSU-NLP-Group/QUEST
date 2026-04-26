import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ca_amazon_fresh_closure_portfolio_20260127"
TASK_DESCRIPTION = """Amazon announced on January 27, 2026, that it would close all Amazon Fresh stores nationwide, including 22 locations in California. A real estate investment firm is evaluating opportunities to acquire and repurpose these properties.

Identify a portfolio of exactly 4 Amazon Fresh store properties in California that were announced for closure on January 27, 2026, meeting the following real estate investment criteria:

1. All 4 properties must be confirmed Amazon Fresh stores in California from the January 27, 2026 closure announcement
2. Each property must have a complete, verifiable street address (street number, street name, city, CA, ZIP code)
3. The properties must be distributed across at least 3 different California cities
4. The properties must span at least 2 different California counties
5. No single city may contain more than 2 of the 4 properties
6. At least 2 properties must be located in Southern California (defined as counties south of and including San Luis Obispo, Kern, and San Bernardino counties)
7. At least 1 property must be located in Northern California (counties north of the Southern California boundary)
8. At least 2 properties must be in cities within one of the five major California metropolitan statistical areas: Los Angeles-Long Beach-Anaheim, San Francisco-Oakland-Berkeley, San Diego-Chula Vista-Carlsbad, Riverside-San Bernardino-Ontario, or Sacramento-Roseville-Folsom

For each of the 4 properties, provide:
- Complete street address (number, street name, city, CA, ZIP code)
- County name
- Regional classification (Southern or Northern California)
- Reference URL(s) confirming the property details and closure announcement
"""

# --------------------------------------------------------------------------- #
# Geographic helpers                                                          #
# --------------------------------------------------------------------------- #
# Southern California counties (south of and including SLO, Kern, San Bernardino)
SOUTHERN_CA_COUNTIES = {
    "los angeles", "orange", "san diego", "riverside", "san bernardino",
    "ventura", "santa barbara", "san luis obispo", "kern", "imperial"
}

# Major MSAs by county membership
MSA_COUNTIES = {
    "los_angeles_long_beach_anaheim": {"los angeles", "orange"},
    "san_francisco_oakland_berkeley": {"san francisco", "alameda", "contra costa", "san mateo", "marin"},
    "san_diego_chula_vista_carlsbad": {"san diego"},
    "riverside_san_bernardino_ontario": {"riverside", "san bernardino"},
    "sacramento_roseville_folsom": {"sacramento", "placer", "el dorado", "yolo"},
}
ALL_MSA_COUNTIES = set().union(*MSA_COUNTIES.values())

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertySources(BaseModel):
    brand: List[str] = Field(default_factory=list)   # confirms Amazon Fresh brand designation
    closure: List[str] = Field(default_factory=list) # confirms Jan 27, 2026 closure announcement
    street: List[str] = Field(default_factory=list)  # confirms street address (number + name)
    city: List[str] = Field(default_factory=list)    # confirms city
    zip: List[str] = Field(default_factory=list)     # confirms ZIP code
    state: List[str] = Field(default_factory=list)   # confirms California (CA)
    county: List[str] = Field(default_factory=list)  # confirms county
    region: List[str] = Field(default_factory=list)  # supports Southern/Northern classification
    size: List[str] = Field(default_factory=list)    # confirms store size


class PropertyItem(BaseModel):
    store_label: Optional[str] = None  # optional identifier (e.g., city or center name)
    street: Optional[str] = None       # street number + street name (no city/state/zip)
    city: Optional[str] = None
    state: Optional[str] = None        # should be "CA"
    zip: Optional[str] = None          # 5-digit zip
    county: Optional[str] = None       # county name (e.g., "Los Angeles")
    region: Optional[str] = None       # "Southern" or "Northern" if provided
    size: Optional[str] = None         # e.g., "30,000 sq ft"
    sources: PropertySources = Field(default_factory=PropertySources)


class PortfolioExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_portfolio() -> str:
    return """
You must extract exactly 4 Amazon Fresh store properties in California that the answer claims were announced for closure on January 27, 2026.

For each of the 4 properties, extract the following fields exactly as stated in the answer:
- store_label: Optional identifier for the property (e.g., a store name, shopping center name, or city label)
- street: Street number and street name ONLY (e.g., "123 Main St"). Do not include city, state, or ZIP here.
- city: The city name (e.g., "Los Angeles")
- state: The state abbreviation (must be "CA" if present)
- zip: 5-digit ZIP code (e.g., "90210")
- county: California county name (e.g., "Los Angeles")
- region: The regional classification if provided in the answer ("Southern" or "Northern"). If the answer does not explicitly state it, set to null.
- size: The store size as written (e.g., "30,000 sq ft" or "approx. 28,000 square feet"). If not provided, set to null.

Also extract URL sources as separate arrays for each verification category (include only URLs explicitly present in the answer):
- sources.brand: URLs that confirm the store is an "Amazon Fresh" branded store
- sources.closure: URLs that confirm the store (or the list including it) was announced for closure on January 27, 2026
- sources.street: URLs that confirm the street number and street name
- sources.city: URLs that confirm the city name
- sources.zip: URLs that confirm the ZIP code
- sources.state: URLs that confirm the state is California (CA)
- sources.county: URLs that confirm the county
- sources.region: URLs that support the Southern/Northern classification (if provided)
- sources.size: URLs that confirm the size is within the described range

Rules:
- Extract ONLY what is actually present in the answer text and the URLs it lists. Do not invent, infer, or add missing data.
- If more than 4 properties are provided in the answer, extract the FIRST 4 only.
- If fewer than 4 are provided, extract whatever is available (the evaluator may pad placeholders).
- For any missing field, return null; for any missing URL category, return an empty list.
"""


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _normalize_county_name(county: Optional[str]) -> Optional[str]:
    if not county:
        return None
    c = county.strip().lower()
    c = re.sub(r"\s*county\s*$", "", c)  # remove trailing 'county'
    return c


def infer_region_from_county(county: Optional[str]) -> Optional[str]:
    nc = _normalize_county_name(county)
    if not nc:
        return None
    if nc in SOUTHERN_CA_COUNTIES:
        return "Southern"
    return "Northern"


def in_major_msa_by_county(county: Optional[str]) -> bool:
    nc = _normalize_county_name(county)
    if not nc:
        return False
    return nc in ALL_MSA_COUNTIES


def is_valid_zip(zip_code: Optional[str]) -> bool:
    if not zip_code:
        return False
    return bool(re.fullmatch(r"\d{5}", zip_code.strip()))


def dedup_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if uu and uu not in seen:
            seen.add(uu)
            out.append(uu)
    return out


def first_nonempty(*vals: Optional[str]) -> Optional[str]:
    for v in vals:
        if v and v.strip():
            return v.strip()
    return None


# --------------------------------------------------------------------------- #
# Leaf verification helper                                                    #
# --------------------------------------------------------------------------- #
async def add_source_verified_leaf(
    evaluator: Evaluator,
    *,
    id: str,
    desc: str,
    parent,
    claim: str,
    sources: Optional[List[str]],
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    """
    Add a leaf node and verify the claim with provided sources.
    If no sources are provided, fail the leaf immediately to enforce source-grounding.
    """
    sources = dedup_urls(sources)
    leaf = evaluator.add_leaf(
        id=id,
        desc=desc,
        parent=parent,
        critical=critical,
        status="initialized",
        score=0.0
    )
    if not sources:
        # Fail due to missing evidence
        leaf.score = 0.0
        leaf.status = "failed"
        return
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Property verification                                                       #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    parent_node,
    prop: PropertyItem,
    index: int,
) -> None:
    """
    Build the verification subtree for a single property.
    """
    pnum = index + 1
    prop_node = evaluator.add_parallel(
        id=f"property_{index+1}_evaluation",
        desc=f"Complete evaluation of the {'first' if pnum==1 else 'second' if pnum==2 else 'third' if pnum==3 else 'fourth'} Amazon Fresh property",
        parent=parent_node,
        critical=True
    )

    # Identity Verification (critical)
    identity_node = evaluator.add_parallel(
        id=f"property_{index+1}_identity",
        desc=f"Verification that Property {pnum} is a valid California Amazon Fresh store from the closure list",
        parent=prop_node,
        critical=True
    )

    # Store Type -> Source
    store_type_node = evaluator.add_parallel(
        id=f"property_{index+1}_store_type",
        desc=f"Confirmation that Property {pnum} is an Amazon Fresh branded store",
        parent=identity_node,
        critical=True
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_store_type_source",
        desc="URL reference confirming the Amazon Fresh brand designation",
        parent=store_type_node,
        claim=(
            f"This page confirms that the store"
            + (f" at {prop.street}, {prop.city}, CA {prop.zip}" if prop.street or prop.city or prop.zip else "")
            + " is branded as an Amazon Fresh grocery store."
        ),
        sources=prop.sources.brand,
        additional_instruction="Allow reasonable formatting differences. The page should explicitly indicate Amazon Fresh branding for this location or list it as an Amazon Fresh store."
    )

    # Closure Announcement -> Source (Jan 27, 2026)
    closure_node = evaluator.add_parallel(
        id=f"property_{index+1}_closure",
        desc=f"Verification that Property {pnum} was announced for closure on January 27, 2026",
        parent=identity_node,
        critical=True
    )
    closure_location = first_nonempty(
        f"{prop.street}, {prop.city}, CA {prop.zip}" if prop.street and prop.city and prop.zip else None,
        f"{prop.city}, CA" if prop.city else None,
        prop.store_label
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_closure_date_source",
        desc="URL reference confirming the January 27, 2026 closure announcement",
        parent=closure_node,
        claim=(
            f"This page states that the Amazon Fresh store"
            + (f" in {closure_location}" if closure_location else "")
            + " was announced for closure on January 27, 2026."
        ),
        sources=prop.sources.closure,
        additional_instruction="The page may be a corporate announcement or reliable news report listing closures with the date Jan 27, 2026. Allow if the page explicitly ties the store (by address, city, or unique identifier) to the Jan 27, 2026 closure list."
    )

    # California Location -> Source
    state_node = evaluator.add_parallel(
        id=f"property_{index+1}_state",
        desc=f"Confirmation that Property {pnum} is located in California",
        parent=identity_node,
        critical=True
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_state_source",
        desc="URL reference confirming California as the property location state",
        parent=state_node,
        claim="This page confirms the property is located in California (CA).",
        sources=prop.sources.state if prop.sources.state else (prop.sources.city or prop.sources.street),
        additional_instruction="The page should clearly show California (CA) for this store; the address line including CA is acceptable."
    )

    # Size Verification (critical)
    size_node = evaluator.add_parallel(
        id=f"property_{index+1}_size",
        desc=f"Verification that Property {pnum} has a typical Amazon Fresh footprint between 25,000 and 51,000 square feet",
        parent=prop_node,
        critical=True
    )
    size_claim = "This page indicates that the store size is between 25,000 and 51,000 square feet."
    if prop.size and isinstance(prop.size, str):
        size_claim = (
            f"This page indicates that the store size ({prop.size}) is between 25,000 and 51,000 square feet "
            f"(allowing minor rounding or approximations)."
        )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_size_source",
        desc="URL reference or documentation confirming the property size falls within the 25,000-51,000 sq ft range",
        parent=size_node,
        claim=size_claim,
        sources=prop.sources.size,
        additional_instruction="Accept minor rounding and common representations (e.g., 30k, 30,000 sq ft, ~30,000 sf)."
    )

    # Address Information (critical)
    addr_node = evaluator.add_parallel(
        id=f"property_{index+1}_address",
        desc=f"Complete address details for Property {pnum}",
        parent=prop_node,
        critical=True
    )

    # Street Address -> Source
    street_node = evaluator.add_parallel(
        id=f"property_{index+1}_street",
        desc=f"The street number and street name for Property {pnum}",
        parent=addr_node,
        critical=True
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_street_source",
        desc="URL reference providing the street address",
        parent=street_node,
        claim=f"The street number and street name are '{prop.street}'." if prop.street else "The page provides the exact street number and street name for this property.",
        sources=prop.sources.street,
        additional_instruction="The page should show the street number and street name. Minor punctuation or abbreviation differences are acceptable."
    )

    # City Name -> Source
    city_node = evaluator.add_parallel(
        id=f"property_{index+1}_city",
        desc=f"The city where Property {pnum} is located",
        parent=addr_node,
        critical=True
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_city_source",
        desc="URL reference confirming the city name",
        parent=city_node,
        claim=f"The city is '{prop.city}', California." if prop.city else "The page confirms the city in California for this property.",
        sources=prop.sources.city or prop.sources.street,
        additional_instruction="The source should clearly show the city. An address line including the city is acceptable."
    )

    # ZIP Code -> Source
    zip_node = evaluator.add_parallel(
        id=f"property_{index+1}_zip",
        desc=f"The 5-digit ZIP code for Property {pnum}",
        parent=addr_node,
        critical=True
    )
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_zip_source",
        desc="URL reference providing the ZIP code",
        parent=zip_node,
        claim=f"The ZIP code is '{prop.zip}'." if prop.zip else "The page provides the 5-digit ZIP code for this property.",
        sources=prop.sources.zip or prop.sources.street,
        additional_instruction="The page should list a 5-digit ZIP code for this address."
    )

    # Geographic Classification (critical)
    geo_node = evaluator.add_parallel(
        id=f"property_{index+1}_geo",
        desc=f"Geographic classification and administrative boundaries for Property {pnum}",
        parent=prop_node,
        critical=True
    )

    # County -> Source
    county_node = evaluator.add_parallel(
        id=f"property_{index+1}_county",
        desc=f"The California county where Property {pnum} is located",
        parent=geo_node,
        critical=True
    )
    county_claim = f"The store is in {prop.county} County, California." if prop.county else "The page indicates the California county for this store."
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_county_source",
        desc="URL reference confirming the county designation",
        parent=county_node,
        claim=county_claim,
        sources=prop.sources.county or prop.sources.city or prop.sources.street,
        additional_instruction="The county may be shown explicitly or derivable from an official or authoritative address page listing the county."
    )

    # Regional Classification -> Source
    region_node = evaluator.add_parallel(
        id=f"property_{index+1}_region",
        desc=f"Classification of Property {pnum} as Southern California or Northern California",
        parent=geo_node,
        critical=True
    )
    # Prefer extracted region; if missing, infer from county for claim text
    region_value = prop.region or infer_region_from_county(prop.county)
    region_text = f"{region_value} California" if region_value else "the correct California region (Southern or Northern)"
    await add_source_verified_leaf(
        evaluator,
        id=f"property_{index+1}_region_source",
        desc="URL reference or basis for regional classification",
        parent=region_node,
        claim=f"The property's county places it in {region_text}.",
        sources=prop.sources.region or prop.sources.county,
        additional_instruction="The source should reasonably support that the county is considered part of Southern or Northern California as defined (Southern includes counties south of and including SLO, Kern, San Bernardino)."
    )


# --------------------------------------------------------------------------- #
# Portfolio-level checks                                                      #
# --------------------------------------------------------------------------- #
def compute_portfolio_stats(props: List[PropertyItem]) -> Dict[str, Any]:
    # Normalize cities and counties
    city_vals = [p.city.strip().lower() for p in props if p.city and p.city.strip()]
    county_vals = [_normalize_county_name(p.county) for p in props if p.county and p.county.strip()]

    unique_cities = set(city_vals)
    unique_counties = set([c for c in county_vals if c])

    # City concentration
    city_counts: Dict[str, int] = {}
    for c in city_vals:
        city_counts[c] = city_counts.get(c, 0) + 1
    max_per_city = max(city_counts.values()) if city_counts else 0

    # Regional counts
    southern_count = 0
    northern_count = 0
    for p in props:
        region = p.region or infer_region_from_county(p.county)
        if region == "Southern":
            southern_count += 1
        elif region == "Northern":
            northern_count += 1

    # Major MSA counts (by county)
    major_msa_count = sum(1 for p in props if in_major_msa_by_county(p.county))

    # Distinct address keys (street + city + zip)
    def addr_key(pi: PropertyItem) -> Optional[str]:
        if not (pi.street and pi.city and pi.zip):
            return None
        return f"{pi.street.strip().lower()}|{pi.city.strip().lower()}|{pi.zip.strip()}"

    keys = [addr_key(p) for p in props]
    keys_nonnull = [k for k in keys if k]
    unique_keys = set(keys_nonnull)

    stats = {
        "unique_city_count": len(unique_cities),
        "city_counts": city_counts,
        "max_per_city": max_per_city,
        "unique_county_count": len(unique_counties),
        "southern_count": southern_count,
        "northern_count": northern_count,
        "major_msa_count": major_msa_count,
        "address_keys_nonnull": keys_nonnull,
        "unique_address_keys_count": len(unique_keys),
        "total_nonnull_address_keys": len(keys_nonnull),
    }
    return stats


def pad_or_slice_properties(extracted: PortfolioExtraction, target_n: int = 4) -> List[PropertyItem]:
    props = list(extracted.properties or [])
    if len(props) >= target_n:
        return props[:target_n]
    # pad with empty items
    padded = props[:]
    while len(padded) < target_n:
        padded.append(PropertyItem())
    return padded


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the California Amazon Fresh closure portfolio task.
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

    # ------------------------ Extraction ---------------------------------- #
    extracted = await evaluator.extract(
        prompt=prompt_extract_portfolio(),
        template_class=PortfolioExtraction,
        extraction_name="portfolio_extraction"
    )

    # Keep exactly 4 properties for evaluation (pad if needed)
    properties = pad_or_slice_properties(extracted, 4)

    # Record some helpful derived info for transparency
    derived_info = []
    for i, p in enumerate(properties):
        derived_info.append({
            "index": i + 1,
            "street": p.street,
            "city": p.city,
            "state": p.state,
            "zip": p.zip,
            "county": p.county,
            "region_extracted": p.region,
            "region_inferred_from_county": infer_region_from_county(p.county),
            "size": p.size,
            "sources": p.sources.dict() if hasattr(p.sources, "dict") else {}
        })
    evaluator.add_custom_info(
        {"properties": derived_info},
        info_type="derived_debug_info",
        info_name="extracted_properties_debug"
    )

    # -------------------- Build Verification Tree ------------------------- #
    # Portfolio node (critical root for all checks)
    portfolio_node = evaluator.add_parallel(
        id="portfolio_evaluation",
        desc="Comprehensive evaluation of a portfolio of 4 California Amazon Fresh store properties announced for closure, meeting specific real estate investment criteria",
        parent=root,
        critical=True
    )

    # Property evaluations (all critical under portfolio)
    for i, prop in enumerate(properties):
        await verify_property(evaluator, portfolio_node, prop, i)

    # Geographic Diversity (critical)
    geo_div_node = evaluator.add_parallel(
        id="portfolio_geographic_diversity",
        desc="Assessment of geographic diversification across the 4-property portfolio",
        parent=portfolio_node,
        critical=True
    )

    # City distribution analysis (critical)
    city_dist_node = evaluator.add_parallel(
        id="city_distribution_analysis",
        desc="Analysis of city-level distribution across the portfolio",
        parent=geo_div_node,
        critical=True
    )

    stats = compute_portfolio_stats(properties)

    # Minimum city count >= 3
    evaluator.add_custom_node(
        result=stats["unique_city_count"] >= 3,
        id="minimum_city_count",
        desc="The portfolio includes properties from at least 3 different California cities",
        parent=city_dist_node,
        critical=True
    )

    # No city > 2 properties
    evaluator.add_custom_node(
        result=stats["max_per_city"] <= 2,
        id="city_concentration_limit",
        desc="No single city contains more than 2 of the 4 properties",
        parent=city_dist_node,
        critical=True
    )

    # County distribution analysis (critical)
    county_dist_node = evaluator.add_parallel(
        id="county_distribution_analysis",
        desc="Analysis of county-level distribution across the portfolio",
        parent=geo_div_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=stats["unique_county_count"] >= 2,
        id="minimum_county_count",
        desc="The portfolio spans at least 2 different California counties",
        parent=county_dist_node,
        critical=True
    )

    # Regional balance (critical)
    regional_balance_node = evaluator.add_parallel(
        id="regional_balance",
        desc="Evaluation of regional balance between Southern and Northern California",
        parent=geo_div_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=stats["southern_count"] >= 2,
        id="southern_california_count",
        desc="At least 2 properties are located in Southern California (counties south of and including San Luis Obispo, Kern, and San Bernardino counties)",
        parent=regional_balance_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=stats["northern_count"] >= 1,
        id="northern_california_count",
        desc="At least 1 property is located in Northern California (counties north of the Southern California boundary)",
        parent=regional_balance_node,
        critical=True
    )

    # Metropolitan presence (critical)
    metro_presence_node = evaluator.add_parallel(
        id="metropolitan_presence",
        desc="Analysis of presence in major California metropolitan statistical areas",
        parent=geo_div_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=stats["major_msa_count"] >= 2,
        id="major_metro_count",
        desc="At least 2 properties are located in cities within the five major California MSAs (Los Angeles-Long Beach-Anaheim, San Francisco-Oakland-Berkeley, San Diego-Chula Vista-Carlsbad, Riverside-San Bernardino-Ontario, or Sacramento-Roseville-Folsom)",
        parent=metro_presence_node,
        critical=True
    )

    # Portfolio integrity (critical)
    integrity_node = evaluator.add_parallel(
        id="portfolio_integrity",
        desc="Verification of portfolio completeness and uniqueness requirements",
        parent=portfolio_node,
        critical=True
    )

    # Exactly 4 distinct Amazon Fresh properties (by non-null full address key)
    evaluator.add_custom_node(
        result=(len(properties) == 4),
        id="property_count_requirement",
        desc="The portfolio contains exactly 4 distinct Amazon Fresh properties (no more, no fewer)",
        parent=integrity_node,
        critical=True
    )

    # Address uniqueness
    evaluator.add_custom_node(
        result=(stats["unique_address_keys_count"] == stats["total_nonnull_address_keys"] and stats["total_nonnull_address_keys"] == 4),
        id="address_uniqueness",
        desc="All 4 properties have unique street addresses with no duplicates",
        parent=integrity_node,
        critical=True
    )

    # All properties distinct (same as uniqueness but kept as separate requirement)
    evaluator.add_custom_node(
        result=(stats["unique_address_keys_count"] == 4),
        id="all_properties_distinct",
        desc="Each of the 4 properties represents a different physical location",
        parent=integrity_node,
        critical=True
    )

    # Add debug info for portfolio stats
    evaluator.add_custom_info(
        {
            "unique_city_count": stats["unique_city_count"],
            "city_counts": stats["city_counts"],
            "unique_county_count": stats["unique_county_count"],
            "southern_count": stats["southern_count"],
            "northern_count": stats["northern_count"],
            "major_msa_count": stats["major_msa_count"],
            "address_keys_nonnull": stats["address_keys_nonnull"],
        },
        info_type="portfolio_stats",
        info_name="portfolio_statistics"
    )

    return evaluator.get_summary()