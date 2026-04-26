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
TASK_ID = "celebrity_fragrance_pfw_fall_2026"
TASK_DESCRIPTION = """
During Paris Fashion Week Fall 2026 (March 2-10, 2026), identify a celebrity who founded a fragrance brand that meets ALL of the following criteria:

1. The celebrity must be the founder or co-founder of the fragrance brand (not merely a brand ambassador for an existing company)

2. The celebrity attended at least one fashion show during Paris Fashion Week Fall 2026 (provide specific show name(s) and date(s))

3. The fragrance brand has a retail partnership with at least one major beauty retailer (Ulta Beauty, Sephora, Selfridges, or equivalent prestige retailer)

4. The brand has physical retail store presence (not exclusively direct-to-consumer online)

5. The brand has retail distribution in both North America and Europe as of March 2026

6. The brand has expanded to at least 3 different countries/markets by March 2026

7. The brand has confirmed Middle East market expansion that is either completed or actively in progress during Q1 2026 (January-March 2026)

8. The standard size fragrance products (50ml/1.7oz or equivalent) are priced under $100

9. The fragrances feature alcohol-free formulation or clean beauty certification

For your answer, provide:
- The celebrity's name
- The fragrance brand name
- Specific Paris Fashion Week show(s) attended with dates
- Major retail partner(s)
- List of at least 3 markets/countries where the brand is available
- Middle East expansion details and timeline
- Standard product price point
- Formulation type (alcohol-free/clean beauty details)
- URL references supporting each claim
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PFWShow(BaseModel):
    show_name: Optional[str] = None
    date: Optional[str] = None  # Prefer ISO (YYYY-MM-DD) if available; otherwise as stated
    url: Optional[str] = None


class PFWAttendance(BaseModel):
    shows: List[PFWShow] = Field(default_factory=list)


class RetailPartner(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None  # Prefer a direct product/category page at the retailer


class PhysicalRetailEvidence(BaseModel):
    description: Optional[str] = None  # e.g., “Available in-store at Sephora” / “Find in store”
    urls: List[str] = Field(default_factory=list)  # store locator, in-store pickup, press release, etc.


class DistributionPresence(BaseModel):
    north_america_markets: List[str] = Field(default_factory=list)  # e.g., USA, Canada, Mexico
    europe_markets: List[str] = Field(default_factory=list)  # e.g., UK, France, Germany
    urls: List[str] = Field(default_factory=list)  # proof pages for NA/EU availability


class MarketsExpansion(BaseModel):
    markets_list: List[str] = Field(default_factory=list)  # at least 3 markets
    urls: List[str] = Field(default_factory=list)


class MiddleEastExpansion(BaseModel):
    countries: List[str] = Field(default_factory=list)  # e.g., UAE, KSA, Qatar
    timeline: Optional[str] = None  # e.g., “launching March 2026”, “Q1 2026”
    status: Optional[str] = None  # e.g., “launched”, “launching”, “in progress”
    urls: List[str] = Field(default_factory=list)


class PricingInfo(BaseModel):
    size_label: Optional[str] = None  # e.g., “50ml”, “1.7oz”
    price: Optional[str] = None  # as in the answer, e.g., “$95”
    currency: Optional[str] = None  # e.g., “USD”, “GBP”, “EUR”
    urls: List[str] = Field(default_factory=list)  # retailer or brand PDP page for price


class FormulationInfo(BaseModel):
    details: Optional[str] = None  # e.g., “alcohol-free”, “Clean at Sephora”, “IFRA compliant”, etc.
    type: Optional[str] = None  # “alcohol-free” or “clean beauty certification”
    urls: List[str] = Field(default_factory=list)  # proof pages (brand, retailer, certification)


class BrandExtraction(BaseModel):
    celebrity: Optional[str] = None
    brand: Optional[str] = None
    founder_url: Optional[str] = None  # proof that celebrity is founder/co-founder
    pfw_attendance: PFWAttendance = Field(default_factory=PFWAttendance)
    major_retailers: List[RetailPartner] = Field(default_factory=list)
    physical_retail: PhysicalRetailEvidence = Field(default_factory=PhysicalRetailEvidence)
    distribution: DistributionPresence = Field(default_factory=DistributionPresence)
    markets_expansion: MarketsExpansion = Field(default_factory=MarketsExpansion)
    middle_east: MiddleEastExpansion = Field(default_factory=MiddleEastExpansion)
    pricing: PricingInfo = Field(default_factory=PricingInfo)
    formulation: FormulationInfo = Field(default_factory=FormulationInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_candidate() -> str:
    return """
    Extract a single candidate (celebrity + fragrance brand) from the answer that allegedly satisfies ALL the task criteria.
    Return the following fields (extract strictly from the answer; do not invent anything):

    1) celebrity: The celebrity's full name.
    2) brand: The fragrance brand name.
    3) founder_url: A URL that explicitly states the celebrity is a founder or co-founder of this fragrance brand (not merely a brand ambassador).

    4) pfw_attendance:
       - shows: an array of show attendance objects. For each:
         • show_name: the runway show or presentation name (e.g., “Dior”, “Saint Laurent”).
         • date: the attendance date as stated in the answer (use ISO date like 2026-03-05 if provided; otherwise the verbatim text).
         • url: a URL that explicitly documents the celebrity’s attendance at that show.

    5) major_retailers: an array of retailer objects. For each:
       • name: e.g., Ulta Beauty, Sephora, Selfridges, or another prestige beauty retailer.
       • url: a retailer product/category page where the brand’s products are sold.

    6) physical_retail:
       • description: a concise description of the in-store physical presence (e.g., “Available in-store at Sephora; find-in-store shown”).
       • urls: an array of URLs that indicate in-store availability or physical retail presence (store locator, “find in store,” press release).

    7) distribution:
       • north_america_markets: list of NA countries where the brand is available.
       • europe_markets: list of European countries where the brand is available.
       • urls: URLs confirming distribution in NA and/or Europe.

    8) markets_expansion:
       • markets_list: list of distinct countries/markets where the brand is available (include at least three if present in the answer).
       • urls: URLs that confirm multi-market presence.

    9) middle_east:
       • countries: Middle East market(s) mentioned (e.g., UAE, Saudi Arabia).
       • timeline: the timeline statement (e.g., “Q1 2026”, “launching March 2026”).
       • status: “launched”, “launching”, or “in progress”, as stated.
       • urls: URLs confirming Middle East expansion timeline/status.

    10) pricing:
       • size_label: the standard fragrance size (e.g., “50ml”, “1.7oz”).
       • price: the price string for that size (e.g., “$95”).
       • currency: the currency code if stated (e.g., USD, GBP, EUR).
       • urls: URLs confirming the price for the standard size.

    11) formulation:
       • details: statement about alcohol-free or clean beauty certification (e.g., “alcohol-free”; “Clean at Sephora”).
       • type: “alcohol-free” or “clean beauty certification” (choose whichever matches the answer).
       • urls: URLs confirming the formulation/clean certification.

    IMPORTANT:
    - Only extract URLs that are explicitly present in the answer (supporting references).
    - If some field is not present, set it to null (for string) or [] (for list).
    - Do not add any extra items besides what appears in the answer.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _non_empty_list(values: List[Optional[str]]) -> List[str]:
    return [v.strip() for v in values if isinstance(v, str) and v.strip()]


def _collect_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for ul in url_lists:
        for u in ul:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    merged.append(uu)
    return merged


def _collect_partner_urls(partners: List[RetailPartner]) -> List[str]:
    return _collect_urls([p.url for p in partners if p and p.url])


def _pfw_urls(pfw: PFWAttendance) -> List[str]:
    return _collect_urls([s.url for s in pfw.shows if s and s.url])


def _pfw_shows_str(pfw: PFWAttendance) -> str:
    parts = []
    for s in pfw.shows:
        show = (s.show_name or "").strip()
        date = (s.date or "").strip()
        if show or date:
            if show and date:
                parts.append(f"{show} on {date}")
            elif show:
                parts.append(show)
            else:
                parts.append(date)
    return "; ".join(parts) if parts else ""


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: BrandExtraction) -> None:
    """
    Build the verification tree according to the rubric JSON, mapping each leaf to a concrete check.
    All nodes under the top-level task node are marked critical as per rubric.
    """
    # Top-level critical node (acts as rubric root under evaluator.root)
    top = evaluator.add_parallel(
        id="Celebrity_Fragrance_Brand_Identification",
        desc="Identify a celebrity who founded a fragrance brand that meets all specified retail distribution, geographic expansion, and Paris Fashion Week attendance criteria during March 2026",
        parent=evaluator.root,
        critical=True
    )

    celebrity = extracted.celebrity or ""
    brand = extracted.brand or ""

    # 1) Celebrity_Brand_Founder_Status (leaf)
    founder_leaf = evaluator.add_leaf(
        id="Celebrity_Brand_Founder_Status",
        desc="The celebrity is a founder or co-founder of the fragrance brand (not merely an ambassador)",
        parent=top,
        critical=True
    )
    founder_sources = _non_empty_list([extracted.founder_url] if extracted.founder_url else [])
    founder_claim = f"{celebrity} is a founder or co-founder of the fragrance brand {brand}, not merely a brand ambassador."
    await evaluator.verify(
        claim=founder_claim,
        node=founder_leaf,
        sources=founder_sources,
        additional_instruction="Verify explicit founder/co-founder language on the provided page(s). Do not accept mere 'ambassador' or 'face of' roles."
    )

    # 2) Paris_Fashion_Week_Attendance (parallel)
    pfw_node = evaluator.add_parallel(
        id="Paris_Fashion_Week_Attendance",
        desc="The celebrity attended at least one fashion show during Paris Fashion Week Fall 2026 (March 2-10, 2026)",
        parent=top,
        critical=True
    )

    # 2.1) PFW_Attendance_Verification (leaf)
    pfw_verify_leaf = evaluator.add_leaf(
        id="PFW_Attendance_Verification",
        desc="Provide specific show name(s) and date(s) of attendance during March 2-10, 2026",
        parent=pfw_node,
        critical=True
    )
    pfw_urls = _pfw_urls(extracted.pfw_attendance)
    shows_str = _pfw_shows_str(extracted.pfw_attendance)
    pfw_claim = (
        f"{celebrity} attended at least one show at Paris Fashion Week Fall 2026 (March 2–10, 2026): {shows_str}."
    )
    await evaluator.verify(
        claim=pfw_claim,
        node=pfw_verify_leaf,
        sources=pfw_urls,
        additional_instruction="Confirm the event(s) occurred in Paris during March 2–10, 2026 and explicitly show attendance by the named celebrity."
    )

    # 2.2) PFW_Attendance_URL (leaf) - presence of at least one URL
    pfw_url_presence = evaluator.add_custom_node(
        result=len(pfw_urls) > 0,
        id="PFW_Attendance_URL",
        desc="Provide URL reference confirming the celebrity's Paris Fashion Week Fall 2026 attendance",
        parent=pfw_node,
        critical=True
    )

    # 3) Retail_Distribution_Criteria (parallel)
    retail_node = evaluator.add_parallel(
        id="Retail_Distribution_Criteria",
        desc="The fragrance brand meets all retail distribution requirements as of March 2026",
        parent=top,
        critical=True
    )

    # 3.1) Major_Retail_Partnership (parallel)
    major_retail_node = evaluator.add_parallel(
        id="Major_Retail_Partnership",
        desc="The brand has a retail partnership with at least one major beauty retailer (Ulta Beauty, Sephora, Selfridges, or equivalent prestige retailer)",
        parent=retail_node,
        critical=True
    )
    partner_names = [p.name for p in extracted.major_retailers if p and p.name]
    partners_str = ", ".join(_non_empty_list(partner_names))
    partner_urls = _collect_partner_urls(extracted.major_retailers)

    # 3.1.a) Major_Retailer_Name (leaf)
    major_retailer_name_leaf = evaluator.add_leaf(
        id="Major_Retailer_Name",
        desc="Identify the major beauty retailer(s) carrying the brand",
        parent=major_retail_node,
        critical=True
    )
    major_retail_claim = f"The brand {brand} is sold at the following major beauty retailer(s): {partners_str}."
    await evaluator.verify(
        claim=major_retail_claim,
        node=major_retailer_name_leaf,
        sources=partner_urls,
        additional_instruction="Confirm that at least one listed retailer is a major prestige retailer (e.g., Ulta Beauty, Sephora, Selfridges or equivalent), and that the page shows the brand's products for sale."
    )

    # 3.1.b) Major_Retailer_URL (leaf) - at least one partner URL present
    major_retailer_url_presence = evaluator.add_custom_node(
        result=len(partner_urls) > 0,
        id="Major_Retailer_URL",
        desc="Provide URL reference confirming the major retail partnership",
        parent=major_retail_node,
        critical=True
    )

    # 3.2) Physical_Retail_Presence (parallel)
    physical_retail_node = evaluator.add_parallel(
        id="Physical_Retail_Presence",
        desc="The brand has physical retail store presence, not exclusively direct-to-consumer online",
        parent=retail_node,
        critical=True
    )
    physical_urls = _collect_urls(extracted.physical_retail.urls, partner_urls)
    physical_desc = extracted.physical_retail.description or ""

    # 3.2.a) Physical_Store_Evidence (leaf)
    physical_store_leaf = evaluator.add_leaf(
        id="Physical_Store_Evidence",
        desc="Provide evidence of physical retail locations or in-store availability",
        parent=physical_retail_node,
        critical=True
    )
    physical_claim = f"The brand {brand} has in-store availability or physical retail presence. Evidence: {physical_desc}"
    await evaluator.verify(
        claim=physical_claim,
        node=physical_store_leaf,
        sources=physical_urls,
        additional_instruction="Accept evidence like 'Find in store', 'In-store pickup', store locator with brand stocked, or press releases confirming in-store availability."
    )

    # 3.2.b) Physical_Retail_URL (leaf) - presence of at least one URL
    physical_url_presence = evaluator.add_custom_node(
        result=len(physical_urls) > 0,
        id="Physical_Retail_URL",
        desc="Provide URL reference confirming physical retail presence",
        parent=physical_retail_node,
        critical=True
    )

    # 3.3) Multi_Continent_Distribution (parallel)
    multi_cont_node = evaluator.add_parallel(
        id="Multi_Continent_Distribution",
        desc="The brand has retail presence in both North America and Europe as of March 2026",
        parent=retail_node,
        critical=True
    )
    dist_urls = _collect_urls(extracted.distribution.urls, partner_urls)

    # 3.3.a) North_America_Presence (leaf)
    na_markets = _non_empty_list(extracted.distribution.north_america_markets)
    na_markets_str = ", ".join(na_markets)
    na_leaf = evaluator.add_leaf(
        id="North_America_Presence",
        desc="Identify specific North American market(s) where the brand is available",
        parent=multi_cont_node,
        critical=True
    )
    na_claim = f"The brand {brand} has retail distribution in North America in the following market(s): {na_markets_str}."
    await evaluator.verify(
        claim=na_claim,
        node=na_leaf,
        sources=dist_urls,
        additional_instruction="Confirm retail distribution availability in at least one North American country (e.g., US, Canada, Mexico) as of March 2026."
    )

    # 3.3.b) Europe_Presence (leaf)
    eu_markets = _non_empty_list(extracted.distribution.europe_markets)
    eu_markets_str = ", ".join(eu_markets)
    eu_leaf = evaluator.add_leaf(
        id="Europe_Presence",
        desc="Identify specific European market(s) where the brand is available",
        parent=multi_cont_node,
        critical=True
    )
    eu_claim = f"The brand {brand} has retail distribution in Europe in the following market(s): {eu_markets_str}."
    await evaluator.verify(
        claim=eu_claim,
        node=eu_leaf,
        sources=dist_urls,
        additional_instruction="Confirm retail distribution availability in at least one European country as of March 2026."
    )

    # 3.3.c) Multi_Continent_URL (leaf) - at least one URL provided for distribution proof
    multi_cont_url_presence = evaluator.add_custom_node(
        result=len(dist_urls) > 0,
        id="Multi_Continent_URL",
        desc="Provide URL reference(s) confirming North American and European distribution",
        parent=multi_cont_node,
        critical=True
    )

    # 4) Geographic_Expansion_Criteria (parallel)
    geo_node = evaluator.add_parallel(
        id="Geographic_Expansion_Criteria",
        desc="The brand meets geographic expansion requirements showing multi-market presence",
        parent=top,
        critical=True
    )

    # 4.1) Minimum_Three_Markets (parallel)
    min3_node = evaluator.add_parallel(
        id="Minimum_Three_Markets",
        desc="The brand has expanded to at least 3 different countries/markets by March 2026",
        parent=geo_node,
        critical=True
    )
    markets = _non_empty_list(extracted.markets_expansion.markets_list)
    markets_str = ", ".join(markets)
    markets_urls = _collect_urls(extracted.markets_expansion.urls, dist_urls, partner_urls)

    # 4.1.a) Three_Markets_List (leaf)
    three_markets_leaf = evaluator.add_leaf(
        id="Three_Markets_List",
        desc="List at least 3 distinct countries/markets where the brand is available",
        parent=min3_node,
        critical=True
    )
    three_markets_claim = f"The brand {brand} is available in at least three markets: {markets_str}."
    await evaluator.verify(
        claim=three_markets_claim,
        node=three_markets_leaf,
        sources=markets_urls,
        additional_instruction="Confirm that at least three distinct countries/markets are represented by the provided sources."
    )

    # 4.1.b) Three_Markets_URL (leaf) - at least one URL provided for multi-market proof
    three_markets_url_presence = evaluator.add_custom_node(
        result=len(markets_urls) > 0,
        id="Three_Markets_URL",
        desc="Provide URL reference confirming the multi-market expansion",
        parent=min3_node,
        critical=True
    )

    # 4.2) Middle_East_Expansion_Q1_2026 (parallel)
    me_node = evaluator.add_parallel(
        id="Middle_East_Expansion_Q1_2026",
        desc="The brand has confirmed Middle East market expansion that is either completed or actively in progress during Q1 2026 (January-March 2026)",
        parent=geo_node,
        critical=True
    )
    me_countries = _non_empty_list(extracted.middle_east.countries)
    me_countries_str = ", ".join(me_countries)
    me_timeline = extracted.middle_east.timeline or ""
    me_status = extracted.middle_east.status or ""
    me_urls = _collect_urls(extracted.middle_east.urls)

    # 4.2.a) Middle_East_Status (leaf)
    me_status_leaf = evaluator.add_leaf(
        id="Middle_East_Status",
        desc="Provide specific information about Middle East market entry, including country/countries and timeline",
        parent=me_node,
        critical=True
    )
    me_claim = (
        f"The brand {brand} has Middle East expansion in Q1 2026 (January–March 2026), "
        f"with countries: {me_countries_str}; timeline/status: {me_timeline} / {me_status}."
    )
    await evaluator.verify(
        claim=me_claim,
        node=me_status_leaf,
        sources=me_urls,
        additional_instruction="Confirm that expansion is completed or actively in progress during Q1 2026, and that at least one Middle Eastern country (e.g., UAE, KSA, Qatar) is identified."
    )

    # 4.2.b) Middle_East_URL (leaf) - at least one URL for ME expansion
    me_url_presence = evaluator.add_custom_node(
        result=len(me_urls) > 0,
        id="Middle_East_URL",
        desc="Provide URL reference confirming Middle East expansion during Q1 2026",
        parent=me_node,
        critical=True
    )

    # 5) Product_Specifications (parallel)
    prod_node = evaluator.add_parallel(
        id="Product_Specifications",
        desc="The fragrance products meet specified formulation and pricing criteria",
        parent=top,
        critical=True
    )

    # 5.1) Pricing_Under_100 (parallel)
    pricing_node = evaluator.add_parallel(
        id="Pricing_Under_100",
        desc="Standard size fragrance products (50ml/1.7oz or equivalent) are priced under $100",
        parent=prod_node,
        critical=True
    )
    size_label = (extracted.pricing.size_label or "").strip()
    price_str = (extracted.pricing.price or "").strip()
    price_currency = (extracted.pricing.currency or "").strip()
    price_urls = _collect_urls(extracted.pricing.urls, partner_urls)

    # 5.1.a) Price_Point (leaf)
    price_leaf = evaluator.add_leaf(
        id="Price_Point",
        desc="Provide specific price for standard size product",
        parent=pricing_node,
        critical=True
    )
    price_claim = (
        f"The standard size fragrance ({size_label}) of {brand} is priced at {price_str} "
        f"({price_currency if price_currency else 'currency as shown'}), and this price is under $100 USD."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=price_urls,
        additional_instruction="Confirm the price for the standard size (50ml/1.7oz). If the currency is not USD, allow reasonable conversion to verify it's under $100 USD (rounding acceptable)."
    )

    # 5.1.b) Pricing_URL (leaf) - at least one URL for price
    pricing_url_presence = evaluator.add_custom_node(
        result=len(price_urls) > 0,
        id="Pricing_URL",
        desc="Provide URL reference confirming the product pricing",
        parent=pricing_node,
        critical=True
    )

    # 5.2) Clean_Formulation (parallel)
    clean_node = evaluator.add_parallel(
        id="Clean_Formulation",
        desc="The fragrances feature alcohol-free formulation or clean beauty certification",
        parent=prod_node,
        critical=True
    )
    formulation_details = (extracted.formulation.details or "").strip()
    formulation_type = (extracted.formulation.type or "").strip()
    formulation_urls = _collect_urls(extracted.formulation.urls, partner_urls)

    # 5.2.a) Formulation_Details (leaf)
    formulation_leaf = evaluator.add_leaf(
        id="Formulation_Details",
        desc="Describe the clean/alcohol-free formulation characteristics",
        parent=clean_node,
        critical=True
    )
    formulation_claim = (
        f"The brand {brand} fragrances feature {formulation_type or 'clean/alcohol-free'} formulation: {formulation_details}."
    )
    await evaluator.verify(
        claim=formulation_claim,
        node=formulation_leaf,
        sources=formulation_urls,
        additional_instruction="Confirm explicit statements such as 'alcohol-free' or recognized clean-beauty certifications/labels (e.g., 'Clean at Sephora')."
    )

    # 5.2.b) Formulation_URL (leaf) - at least one URL for formulation
    formulation_url_presence = evaluator.add_custom_node(
        result=len(formulation_urls) > 0,
        id="Formulation_URL",
        desc="Provide URL reference confirming the formulation type",
        parent=clean_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the celebrity-founded fragrance brand task (PFW Fall 2026).
    Returns a structured summary including the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregator (we add a critical child node under root)
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
    extracted: BrandExtraction = await evaluator.extract(
        prompt=prompt_extract_brand_candidate(),
        template_class=BrandExtraction,
        extraction_name="celebrity_fragrance_candidate"
    )

    # Record a compact custom info snapshot (optional)
    evaluator.add_custom_info(
        {
            "celebrity": extracted.celebrity,
            "brand": extracted.brand,
            "pfw_shows_count": len(extracted.pfw_attendance.shows) if extracted.pfw_attendance else 0,
            "major_retailers": [p.name for p in (extracted.major_retailers or []) if p and p.name],
            "na_markets": extracted.distribution.north_america_markets,
            "eu_markets": extracted.distribution.europe_markets,
            "me_countries": extracted.middle_east.countries,
            "price": extracted.pricing.price,
            "size_label": extracted.pricing.size_label,
            "formulation_type": extracted.formulation.type,
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return summary
    return evaluator.get_summary()