import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mi_ferry_nps_2026"
TASK_DESCRIPTION = (
    "A 22-year-old U.S. resident from Wisconsin is planning a summer 2026 trip from Milwaukee to visit Michigan's "
    "federal recreation sites by crossing Lake Michigan via ferry. They are traveling during the June 1-August 31, 2026 period and want to make cost-effective choices.\n\n"
    "Provide the following information:\n\n"
    "1. Ferry Transportation: Identify the Lake Michigan ferry that operates between Milwaukee, Wisconsin and Muskegon, Michigan. "
    "Include the ferry name, crossing duration, the special promotion available during June 1-August 31, 2026, the one-way adult fare (including all fees), "
    "and the official website URL.\n\n"
    "2. Federal Recreation Sites: Identify two different National Park Service sites in Michigan that are covered by the America the Beautiful Pass. "
    "For each site, provide the official name and location, confirmation that it's a National Park Service property covered by the America the Beautiful Pass, "
    "and the direct link to the official NPS.gov page for the site.\n\n"
    "3. America the Beautiful Pass: State the 2026 Resident Annual Pass price for U.S. residents and confirm it covers entrance fees at both identified sites.\n\n"
    "4. Cost Calculation: Calculate the estimated one-way trip cost including: the Lake Express ferry one-way fare for one adult "
    "(base fare + fuel surcharge + port/security fee), the 2026 America the Beautiful Resident Annual Pass, and the total cost (ferry + pass)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FerryInfo(BaseModel):
    name: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    duration: Optional[str] = None  # e.g., "2.5 hours"
    promotion_name: Optional[str] = None  # e.g., "Kids Ride Free"
    promotion_dates: Optional[str] = None  # e.g., "June 1–August 31, 2026"
    promo_url: Optional[str] = None
    website_url: Optional[str] = None

    adult_one_way_base_fare: Optional[str] = None  # e.g., "$114.50"
    fuel_surcharge: Optional[str] = None  # e.g., "$8.00"
    port_security_fee: Optional[str] = None  # e.g., "$9.00"
    adult_one_way_total: Optional[str] = None  # e.g., "$131.50"
    fares_url: Optional[str] = None  # if a dedicated fares page was cited


class SiteInfo(BaseModel):
    name: Optional[str] = None  # Official site name
    location: Optional[str] = None  # Should indicate Michigan
    nps_url: Optional[str] = None  # Direct NPS.gov page URL


class SitesExtraction(BaseModel):
    sites: List[SiteInfo] = Field(default_factory=list)


class PassInfo(BaseModel):
    pass_name: Optional[str] = None  # e.g., "America the Beautiful Annual Pass"
    year: Optional[str] = None  # e.g., "2026"
    price: Optional[str] = None  # e.g., "$80"
    pass_urls: List[str] = Field(default_factory=list)  # URLs cited for the pass information
    coverage_statement: Optional[str] = None  # any statement text from the answer about coverage


class CostInfo(BaseModel):
    ferry_base: Optional[str] = None
    ferry_fuel: Optional[str] = None
    ferry_port: Optional[str] = None
    ferry_total: Optional[str] = None
    pass_price: Optional[str] = None
    trip_total: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ferry() -> str:
    return """
    Extract ferry transportation details as presented in the answer. Return a JSON object with:
    - name: the ferry service name
    - origin: the departure city/state for the crossing (e.g., "Milwaukee, WI")
    - destination: the arrival city/state for the crossing (e.g., "Muskegon, MI")
    - duration: the stated crossing time (e.g., "2.5 hours")
    - promotion_name: the name of any special promotion mentioned for summer 2026 (e.g., "Kids Ride Free")
    - promotion_dates: the date range of that promotion if stated (e.g., "June 1–August 31, 2026")
    - promo_url: a URL cited that describes the promotion (if any)
    - website_url: the official ferry website URL cited
    - adult_one_way_base_fare: the adult one-way base fare as stated
    - fuel_surcharge: the fuel surcharge per one-way ticket as stated
    - port_security_fee: the port/security fee per one-way ticket as stated
    - adult_one_way_total: the total adult one-way cost including all fees as stated
    - fares_url: a direct fares or pricing page URL if cited; otherwise null
    If any fields are not explicitly present in the answer, set them to null.
    """


def prompt_extract_sites() -> str:
    return """
    Extract up to the first two National Park Service sites in Michigan that the answer claims are covered by the America the Beautiful pass.
    Return a JSON object with:
    - sites: an array of up to two items; for each item include:
      - name: official site name as stated
      - location: the Michigan location as stated (it should indicate that the site is in Michigan)
      - nps_url: a direct link to the official NPS.gov page for this site
    If fewer than two sites are mentioned, return as many as provided. Do not invent URLs.
    """


def prompt_extract_pass() -> str:
    return """
    Extract America the Beautiful pass details as presented in the answer. Return a JSON object with:
    - pass_name: the pass name (e.g., "America the Beautiful Annual Pass")
    - year: the year of the price cited (e.g., "2026") if specified; else null
    - price: the 2026 Resident Annual Pass price for U.S. residents as stated (e.g., "$80")
    - pass_urls: an array of URLs cited for this pass information (USGS/NPS official sources preferred)
    - coverage_statement: any statement in the answer claiming that the pass covers entrance fees at the identified sites
    If any fields are not explicitly present, set them to null or an empty array as appropriate.
    """


def prompt_extract_costs() -> str:
    return """
    Extract any explicit cost calculation values provided in the answer. Return a JSON object with:
    - ferry_base: base adult one-way ferry fare as stated (e.g., "$114.50")
    - ferry_fuel: fuel surcharge per one-way ticket (e.g., "$8.00")
    - ferry_port: port/security fee per one-way ticket (e.g., "$9.00")
    - ferry_total: total adult one-way ferry cost including all fees as stated (e.g., "$131.50")
    - pass_price: the 2026 Resident Annual Pass cost used for calculation (e.g., "$80")
    - trip_total: the total one-way trip cost (ferry + pass) as stated (e.g., "$211.50")
    If any fields are missing in the answer, set them to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_money(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    # keep digits, decimal point, and minus sign
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", ".", "-"):
        return None
    try:
        return round(float(s), 2)
    except Exception:
        return None


def norm(s: Optional[str]) -> str:
    return (s or "").strip()


def is_domain(url: Optional[str], domain: str) -> bool:
    u = (url or "").lower()
    return domain.lower() in u


def collect_sources(*urls: Optional[str]) -> List[str]:
    out = []
    for u in urls:
        if u and isinstance(u, str) and len(u.strip()) > 0:
            out.append(u.strip())
    # deduplicate preserving order
    seen = set()
    deduped = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_ferry_section(evaluator: Evaluator, parent, ferry: FerryInfo):
    """
    Build and verify the Ferry_Information subtree.
    """
    ferry_node = evaluator.add_parallel(
        id="Ferry_Information",
        desc="Identify Lake Express ferry and provide complete details",
        parent=parent,
        critical=False
    )

    # 1) Ferry_URL validity (critical)
    url_valid = is_domain(ferry.website_url, "lake-express.com")
    ferry_url_node = evaluator.add_custom_node(
        result=url_valid,
        id="Ferry_URL",
        desc="Direct link to lake-express.com official website",
        parent=ferry_node,
        critical=True
    )

    ferry_sources = collect_sources(ferry.fares_url, ferry.website_url)

    # 2) Ferry_Name (critical)
    name_leaf = evaluator.add_leaf(
        id="Ferry_Name",
        desc="Ferry name must be Lake Express",
        parent=ferry_node,
        critical=True
    )
    claim_name = f"The ferry service is named '{norm(ferry.name)}'."
    await evaluator.verify(
        claim=claim_name,
        node=name_leaf,
        sources=ferry_sources,
        additional_instruction="Verify that the official ferry brand/service name is Lake Express (minor phrasing variants acceptable)."
    )

    # 3) Ferry_Route (critical)
    route_leaf = evaluator.add_leaf(
        id="Ferry_Route",
        desc="Route must be Milwaukee, WI to Muskegon, MI",
        parent=ferry_node,
        critical=True
    )
    route_text = f"{norm(ferry.origin)} to {norm(ferry.destination)}"
    claim_route = (
        "This ferry operates a route between Milwaukee, Wisconsin and Muskegon, Michigan "
        "(any direction: Milwaukee↔Muskegon)."
        if not norm(ferry.origin) and not norm(ferry.destination)
        else f"The ferry route operates between {norm(ferry.origin)} and {norm(ferry.destination)}."
    )
    await evaluator.verify(
        claim=claim_route,
        node=route_leaf,
        sources=ferry_sources,
        additional_instruction="Confirm that Lake Express connects Milwaukee, WI and Muskegon, MI (either direction acceptable)."
    )

    # 4) Ferry_Duration (non-critical)
    duration_leaf = evaluator.add_leaf(
        id="Ferry_Duration",
        desc="Crossing duration must be stated as 2.5 hours",
        parent=ferry_node,
        critical=False
    )
    dur = norm(ferry.duration)
    if dur:
        claim_dur = f"The Lake Michigan crossing takes about {dur}."
    else:
        # still form a claim to be checked; will likely fail if unsupported
        claim_dur = "The Lake Michigan crossing takes about 2.5 hours."
    await evaluator.verify(
        claim=claim_dur,
        node=duration_leaf,
        sources=ferry_sources,
        additional_instruction="Check the official site for the advertised crossing duration; 2.5 hours is commonly cited."
    )

    # 5) Summer_Promotion (non-critical)
    promo_leaf = evaluator.add_leaf(
        id="Summer_Promotion",
        desc="Kids Ride Free promotion June 1-August 31, 2026 must be mentioned",
        parent=ferry_node,
        critical=False
    )
    promo_name = norm(ferry.promotion_name) or "Kids Ride Free"
    promo_dates = norm(ferry.promotion_dates) or "June 1–August 31, 2026"
    promo_sources = collect_sources(ferry.promo_url, ferry.website_url)
    claim_promo = f"The ferry offers a '{promo_name}' promotion during {promo_dates}."
    await evaluator.verify(
        claim=claim_promo,
        node=promo_leaf,
        sources=promo_sources,
        additional_instruction="Verify that the summer 2026 promotion exists and matches the stated date window (June 1–August 31, 2026)."
    )


async def build_sites_section(evaluator: Evaluator, parent, sites: List[SiteInfo], pass_info: PassInfo):
    """
    Build and verify the Federal_Recreation_Sites subtree with Site_1 and Site_2.
    """
    sites_node = evaluator.add_parallel(
        id="Federal_Recreation_Sites",
        desc="Identify two National Park Service sites in Michigan covered by the America the Beautiful Pass",
        parent=parent,
        critical=False
    )

    # ensure two entries (pad with empty if fewer)
    s1 = sites[0] if len(sites) > 0 else SiteInfo()
    s2 = sites[1] if len(sites) > 1 else SiteInfo()

    async def verify_site(site: SiteInfo, idx: int, other_site: Optional[SiteInfo] = None):
        site_id = f"Site_{idx}"
        site_node = evaluator.add_parallel(
            id=site_id,
            desc=f"{'First' if idx == 1 else 'Second'} National Park Service site meeting criteria",
            parent=sites_node,
            critical=False
        )

        # URL validity (critical leaf as per rubric "Site_X_URL")
        url_ok = is_domain(site.nps_url, "nps.gov")
        url_leaf = evaluator.add_custom_node(
            result=url_ok,
            id=f"{site_id}_URL",
            desc=f"Direct link to official NPS.gov page for this site",
            parent=site_node,
            critical=True
        )

        # Name + Location (critical)
        nl_leaf = evaluator.add_leaf(
            id=f"{site_id}_Name_Location",
            desc="Provide official name and Michigan location",
            parent=site_node,
            critical=True
        )
        nm = norm(site.name)
        loc = norm(site.location)
        claim_nl = f"The official site name is '{nm}' and it is located in Michigan."
        await evaluator.verify(
            claim=claim_nl,
            node=nl_leaf,
            sources=site.nps_url,
            additional_instruction="Confirm that the page shows the site's official name and that it is located in Michigan (MI)."
        )

        # NPS property (critical)
        nps_leaf = evaluator.add_leaf(
            id=f"{site_id}_NPS_Property",
            desc="Site must be National Park Service property (National Lakeshore, National Park, etc.)",
            parent=site_node,
            critical=True
        )
        claim_nps = "This site is administered by the National Park Service (an official NPS unit/property)."
        await evaluator.verify(
            claim=claim_nps,
            node=nps_leaf,
            sources=site.nps_url,
            additional_instruction="Look for evidence on the page that it is an official National Park Service site (e.g., labeled National Park, National Lakeshore, etc.)."
        )

        # Pass coverage (critical)
        passcov_leaf = evaluator.add_leaf(
            id=f"{site_id}_Pass_Coverage",
            desc="Confirm site is covered by America the Beautiful Pass for entrance fees",
            parent=site_node,
            critical=True
        )
        cov_claim = "The America the Beautiful (Interagency) Annual Pass is accepted for entrance fees at this site (if entrance fees are charged)."
        pass_sources = collect_sources(site.nps_url, *(pass_info.pass_urls or []))
        await evaluator.verify(
            claim=cov_claim,
            node=passcov_leaf,
            sources=pass_sources,
            additional_instruction="Verify that the site accepts the Interagency Annual Pass; if the site does not charge entrance fees, it may still indicate pass acceptance where applicable."
        )

        # Site_2_Different (only for second site)
        if idx == 2 and other_site:
            diff_leaf = evaluator.add_custom_node(
                result=(norm(site.name).lower() != norm(other_site.name).lower() and norm(site.nps_url) != norm(other_site.nps_url)),
                id=f"{site_id}_Different",
                desc="Second site must be different from first site",
                parent=site_node,
                critical=True
            )

    await verify_site(s1, 1, None)
    await verify_site(s2, 2, s1)


async def build_pass_section(evaluator: Evaluator, parent, pass_info: PassInfo, sites: List[SiteInfo]):
    """
    Build and verify the Pass_Information subtree.
    """
    pass_node = evaluator.add_parallel(
        id="Pass_Information",
        desc="America the Beautiful Pass details and coverage confirmation",
        parent=parent,
        critical=False
    )

    # Pass_Price (critical)
    price_leaf = evaluator.add_leaf(
        id="Pass_Price",
        desc="2026 Resident Annual Pass price must be stated as $80",
        parent=pass_node,
        critical=True
    )
    claim_price = f"The 2026 America the Beautiful Annual Pass (Resident Annual Pass) price is {norm(pass_info.price)}."
    await evaluator.verify(
        claim=claim_price,
        node=price_leaf,
        sources=pass_info.pass_urls,
        additional_instruction="Verify that the Interagency America the Beautiful Annual Pass price for 2026 is $80 (allowing for standard naming variations)."
    )

    # Pass_Coverage_Confirmation (critical)
    cov_leaf = evaluator.add_leaf(
        id="Pass_Coverage_Confirmation",
        desc="Confirmation that the pass covers entrance fees at both identified sites",
        parent=pass_node,
        critical=True
    )
    s_names = [norm(s.name) for s in sites[:2]]
    s_urls = [norm(s.nps_url) for s in sites[:2]]
    claim_cov = (
        f"The America the Beautiful (Interagency) Annual Pass covers entrance fees at both {', '.join([n for n in s_names if n])}."
        if any(s_names) else
        "The America the Beautiful (Interagency) Annual Pass covers entrance fees at both identified sites."
    )
    sources = collect_sources(*(pass_info.pass_urls or []), *s_urls)
    await evaluator.verify(
        claim=claim_cov,
        node=cov_leaf,
        sources=sources,
        additional_instruction="Confirm that the Interagency Annual Pass is valid for entrance fees at both sites (NPS units)."
    )


async def build_cost_section(
    evaluator: Evaluator,
    parent,
    ferry: FerryInfo,
    pass_info: PassInfo,
    cost: CostInfo
):
    """
    Build and verify the Cost_Calculation subtree.
    """
    cost_node = evaluator.add_sequential(
        id="Cost_Calculation",
        desc="Calculate total trip costs for ferry and pass",
        parent=parent,
        critical=False
    )

    # Determine numeric values using either dedicated cost extraction or ferry/pass extraction
    base = parse_money(cost.ferry_base) or parse_money(ferry.adult_one_way_base_fare)
    fuel = parse_money(cost.ferry_fuel) or parse_money(ferry.fuel_surcharge)
    port = parse_money(cost.ferry_port) or parse_money(ferry.port_security_fee)
    ferry_total_extracted = parse_money(cost.ferry_total) or parse_money(ferry.adult_one_way_total)

    pass_price_num = parse_money(cost.pass_price) or parse_money(pass_info.price)
    trip_total_extracted = parse_money(cost.trip_total)

    fares_sources = collect_sources(ferry.fares_url, ferry.website_url)

    # Ferry_Cost_Breakdown (sequential)
    fbreak_node = evaluator.add_sequential(
        id="Ferry_Cost_Breakdown",
        desc="Calculate total one-way ferry cost including all fees",
        parent=cost_node,
        critical=False
    )

    # Base_Fare (critical)
    base_leaf = evaluator.add_leaf(
        id="Base_Fare",
        desc="Classic Adult one-way base fare of $114.50",
        parent=fbreak_node,
        critical=True
    )
    claim_base_val = f"The Classic adult one-way base fare is {('$' + f'{base:.2f}') if base is not None else 'unknown'}."
    await evaluator.verify(
        claim=claim_base_val,
        node=base_leaf,
        sources=fares_sources,
        additional_instruction="Confirm the adult one-way base fare on the official ferry fares/pricing page. Accept minor formatting differences."
    )

    # Fuel_Surcharge (critical)
    fuel_leaf = evaluator.add_leaf(
        id="Fuel_Surcharge",
        desc="Fuel surcharge of $8.00 per one-way ticket",
        parent=fbreak_node,
        critical=True
    )
    claim_fuel_val = f"The fuel surcharge is {('$' + f'{fuel:.2f}') if fuel is not None else 'unknown'} per one-way ticket."
    await evaluator.verify(
        claim=claim_fuel_val,
        node=fuel_leaf,
        sources=fares_sources,
        additional_instruction="Verify the stated fuel surcharge per one-way ticket on the ferry website."
    )

    # Port_Security_Fee (critical)
    port_leaf = evaluator.add_leaf(
        id="Port_Security_Fee",
        desc="Port and security fee of $9.00 per one-way ticket",
        parent=fbreak_node,
        critical=True
    )
    claim_port_val = f"The port and security fee is {('$' + f'{port:.2f}') if port is not None else 'unknown'} per one-way ticket."
    await evaluator.verify(
        claim=claim_port_val,
        node=port_leaf,
        sources=fares_sources,
        additional_instruction="Verify the stated port and security fee per one-way ticket on the ferry website."
    )

    # Total_Ferry_Cost (critical): validate arithmetic consistency and target $131.50 if data allows
    expected_ferry_total = None
    if base is not None and fuel is not None and port is not None:
        expected_ferry_total = round(base + fuel + port, 2)

    # Create a custom node to check total calculation integrity
    total_correct = False
    if expected_ferry_total is not None:
        # If the answer provided a total, check against their total; otherwise check against 131.50 expectation.
        if ferry_total_extracted is not None:
            total_correct = abs(ferry_total_extracted - expected_ferry_total) <= 0.01
        else:
            total_correct = abs(expected_ferry_total - 131.50) <= 0.01

    ferry_total_desc = "Total one-way ferry cost: $114.50 + $8.00 + $9.00 = $131.50"
    evaluator.add_custom_node(
        result=bool(total_correct),
        id="Total_Ferry_Cost",
        desc=ferry_total_desc,
        parent=fbreak_node,
        critical=True
    )

    # Pass_Cost (non-critical) - verify via pass URLs
    pass_cost_leaf = evaluator.add_leaf(
        id="Pass_Cost",
        desc="America the Beautiful 2026 Resident Annual Pass cost of $80",
        parent=cost_node,
        critical=False
    )
    claim_pass_cost = f"The America the Beautiful Annual Pass used in the calculation costs {('$' + f'{pass_price_num:.2f}') if pass_price_num is not None else 'unknown'}."
    await evaluator.verify(
        claim=claim_pass_cost,
        node=pass_cost_leaf,
        sources=pass_info.pass_urls,
        additional_instruction="Verify the Annual Pass price used for the cost calculation. It should be $80 for 2026."
    )

    # Trip_Total (critical): ferry total + pass price = $211.50 expected
    trip_ok = False
    if expected_ferry_total is not None and pass_price_num is not None:
        computed_total = round(expected_ferry_total + pass_price_num, 2)
        if trip_total_extracted is not None:
            trip_ok = abs(trip_total_extracted - computed_total) <= 0.01
        else:
            trip_ok = abs(computed_total - 211.50) <= 0.01

    evaluator.add_custom_node(
        result=bool(trip_ok),
        id="Trip_Total",
        desc="Total cost calculation: Ferry ($131.50) + Pass ($80) = $211.50",
        parent=cost_node,
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
    Evaluate an answer for the Lake Express + Michigan NPS + Pass + Costs task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract all structured data (in parallel)
    ferry_extraction_task = evaluator.extract(
        prompt=prompt_extract_ferry(),
        template_class=FerryInfo,
        extraction_name="ferry_info"
    )
    sites_extraction_task = evaluator.extract(
        prompt=prompt_extract_sites(),
        template_class=SitesExtraction,
        extraction_name="sites_info"
    )
    pass_extraction_task = evaluator.extract(
        prompt=prompt_extract_pass(),
        template_class=PassInfo,
        extraction_name="pass_info"
    )
    cost_extraction_task = evaluator.extract(
        prompt=prompt_extract_costs(),
        template_class=CostInfo,
        extraction_name="cost_info"
    )

    ferry_info, sites_info, pass_info, cost_info = await asyncio.gather(
        ferry_extraction_task,
        sites_extraction_task,
        pass_extraction_task,
        cost_extraction_task
    )

    # Build and verify subtrees
    await build_ferry_section(evaluator, root, ferry_info)

    # Ensure we use exactly two sites (pad if fewer)
    sites_list = sites_info.sites[:2]
    while len(sites_list) < 2:
        sites_list.append(SiteInfo())
    await build_sites_section(evaluator, root, sites_list, pass_info)

    await build_pass_section(evaluator, root, pass_info, sites_list)

    await build_cost_section(evaluator, root, ferry_info, pass_info, cost_info)

    # Return evaluation summary
    return evaluator.get_summary()