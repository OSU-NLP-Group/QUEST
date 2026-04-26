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
TASK_ID = "jbf_best_chef_2025"
TASK_DESCRIPTION = """
Identify three restaurants in the United States that won James Beard Best Chef awards in 2025, where each restaurant must be from a different James Beard regional category. For each of the three restaurants, provide the following information:

1. The full name of the chef who won the 2025 James Beard Best Chef award
2. The name of the restaurant where the chef works
3. The specific James Beard regional category for the award (e.g., Best Chef: Midwest, Best Chef: California, etc.)
4. The restaurant's complete physical address (street address, city, and state)
5. The restaurant's official website URL
6. A reference URL from jamesbeard.org that confirms the award

Note: The James Beard Foundation divides Best Chef awards into 12 regional categories: California, Great Lakes (IL, IN, MI, OH), Mid-Atlantic (DC, DE, MD, NJ, PA, VA), Midwest (IA, KS, MN, MO, NE, ND, SD, WI), Mountain (CO, ID, MT, UT, WY), New York State, Northeast (CT, MA, ME, NH, RI, VT), Northwest and Pacific (AK, HI, OR, WA), South (AL, AR, FL, LA, MS, PR), Southeast (GA, KY, NC, SC, TN, WV), Southwest (AZ, NM, NV, OK), and Texas. Each of your three selected restaurants must be from a different regional category.
"""

# Canonical James Beard regions and state membership (for logical checks)
JBF_REGION_STATES: Dict[str, List[str]] = {
    "California": ["CA"],
    "Great Lakes": ["IL", "IN", "MI", "OH"],
    "Mid-Atlantic": ["DC", "DE", "MD", "NJ", "PA", "VA"],
    "Midwest": ["IA", "KS", "MN", "MO", "NE", "ND", "SD", "WI"],
    "Mountain": ["CO", "ID", "MT", "UT", "WY"],
    "New York State": ["NY"],
    "Northeast": ["CT", "MA", "ME", "NH", "RI", "VT"],
    "Northwest and Pacific": ["AK", "HI", "OR", "WA"],
    "South": ["AL", "AR", "FL", "LA", "MS", "PR"],
    "Southeast": ["GA", "KY", "NC", "SC", "TN", "WV"],
    "Southwest": ["AZ", "NM", "NV", "OK"],
    "Texas": ["TX"],
}

REGION_SYNONYMS: Dict[str, List[str]] = {
    "northwest and pacific": ["northwest & pacific", "northwest & pacific region", "northwest-and-pacific"],
    "mid-atlantic": ["midatlantic", "mid atlantic"],
    "new york state": ["new york", "ny state"],
    "great lakes": ["greatlakes", "great lakes region"],
    "southwest": ["south west"],
    "southeast": ["south east"],
}

ALLOWED_REGION_NAMES = list(JBF_REGION_STATES.keys())


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AwardItem(BaseModel):
    chef_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    region_category: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer 2-letter abbreviation if available
    website_url: Optional[str] = None
    award_reference_url: Optional[str] = None  # Must be from jamesbeard.org


class AwardsExtraction(BaseModel):
    restaurants: List[AwardItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_awards() -> str:
    region_list = "; ".join(ALLOWED_REGION_NAMES)
    return f"""
    Extract up to three entries describing restaurants associated with 2025 James Beard Best Chef awards, as presented in the answer text.

    For each entry, extract these fields exactly as stated in the answer:
    - chef_name: Full name of the chef who won the 2025 James Beard Best Chef award.
    - restaurant_name: Name of the restaurant where the chef works.
    - region_category: The explicit James Beard regional category (e.g., "Best Chef: Midwest", "Best Chef: California", etc.). If the answer only provides the region label (e.g., "Midwest"/"California") without "Best Chef:", still extract the region string that is associated with the Best Chef award.
    - street_address: The street address of the restaurant (include suite numbers if present).
    - city: City of the restaurant.
    - state: State of the restaurant (prefer the 2-letter abbreviation like CA, NY, TX; if not given, extract as-is).
    - website_url: The official website URL of the restaurant (must be an actual URL present in the answer).
    - award_reference_url: A URL from jamesbeard.org that confirms the award for the chef/restaurant.

    IMPORTANT RULES:
    1) Extract only information explicitly stated in the answer. Do not invent or infer missing values.
    2) If any required information for an entry is missing, set that field to null.
    3) Only extract valid URLs explicitly present in the answer text. If a URL lacks protocol, prepend http://.
    4) The 'award_reference_url' must be a URL on the jamesbeard.org domain.
    5) The 'region_category' should correspond to one of the Best Chef regional categories. Common category names include: {region_list}. Minor formatting differences (e.g., "Northwest & Pacific" vs "Northwest and Pacific") are acceptable to extract as provided.

    Return a JSON object with one field:
    - restaurants: an array of up to 3 objects following the schema above, in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_region_name(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    base = s.strip().lower()
    base = base.replace("best chef:", "").strip()
    base = base.replace("best chef -", "").strip()
    base = base.replace("best chef", "").strip(": ").strip()
    # normalize common punctuation
    base = base.replace("&", "and")
    base = base.replace("  ", " ").strip()

    # direct match against allowed names
    for canonical in ALLOWED_REGION_NAMES:
        if base == canonical.lower():
            return canonical

    # synonyms mapping
    for canonical, syns in REGION_SYNONYMS.items():
        if base == canonical or base in syns:
            # find actual canonical title case
            for k in ALLOWED_REGION_NAMES:
                if k.lower() == canonical:
                    return k

    # try fuzzy simple startswith check
    for canonical in ALLOWED_REGION_NAMES:
        if base.startswith(canonical.lower()):
            return canonical

    return s  # fallback to original if cannot normalize


def _is_state_in_region(state: Optional[str], region: Optional[str]) -> bool:
    if not state or not region:
        return False
    normalized_region = _normalize_region_name(region)
    if not normalized_region:
        return False
    # normalize state (accept variations like "DC", "D.C.", "District of Columbia")
    st = state.strip().upper().replace(".", "")
    if st in ("DISTRICT OF COLUMBIA", "WASHINGTON DC", "WASHINGTON D C", "WASHINGTON, DC"):
        st = "DC"
    # Some answers might include full state names; quick mapping
    name_to_abbrev = {
        "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
        "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
        "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
        "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
        "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
        "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH",
        "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC",
        "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA",
        "PUERTO RICO": "PR", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
        "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
        "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
        "DISTRICT OF COLUMBIA": "DC",
    }
    if st in name_to_abbrev:
        st = name_to_abbrev[st]
    # final check
    allowed = JBF_REGION_STATES.get(normalized_region, [])
    return st in allowed


def _non_empty_urls(*urls: Optional[str]) -> List[str]:
    return [u for u in urls if u and u.strip() != ""]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_restaurant(
    evaluator: Evaluator,
    parent_node,
    item: AwardItem,
    index: int,
    prior_regions: List[str],
) -> None:
    """
    Build the verification subtree for one restaurant and perform all checks.
    """
    idx1 = index + 1
    rest_node = evaluator.add_parallel(
        id=f"restaurant_{idx1}",
        desc=f"Restaurant #{idx1} meeting all specified criteria",
        parent=parent_node,
        critical=True,  # Child of critical Task_Compliance must be critical
    )

    # ----------------- Chef Information ----------------- #
    chef_info = evaluator.add_parallel(
        id=f"chef_info_{idx1}",
        desc="Complete information about the award-winning chef",
        parent=rest_node,
        critical=True,
    )

    # Chef Full Name (verify against jamesbeard.org reference URL)
    chef_name_leaf = evaluator.add_leaf(
        id=f"chef_full_name_{idx1}",
        desc="Chef full name is correctly stated and supported by jamesbeard.org",
        parent=chef_info,
        critical=True,
    )
    chef_claim = f"The chef who won the 2025 James Beard Best Chef award is named '{item.chef_name}'."
    await evaluator.verify(
        claim=chef_claim,
        node=chef_name_leaf,
        sources=item.award_reference_url,
        additional_instruction="Use the jamesbeard.org reference page to confirm the chef's name. Allow minor variations like middle initials or accents; treat them as the same person if obviously referring to the same chef.",
    )

    # Restaurant Affiliation (verify chef-restaurant association on jamesbeard.org)
    chef_rest_leaf = evaluator.add_leaf(
        id=f"restaurant_affiliation_{idx1}",
        desc="Chef's associated restaurant is correctly stated and supported by jamesbeard.org",
        parent=chef_info,
        critical=True,
    )
    aff_claim = f"The award page indicates that {item.chef_name} is affiliated with the restaurant '{item.restaurant_name}'."
    await evaluator.verify(
        claim=aff_claim,
        node=chef_rest_leaf,
        sources=item.award_reference_url,
        additional_instruction="Confirm that the jamesbeard.org page associates the chef with the specified restaurant. Minor naming variations (e.g., punctuation or LLC) are acceptable.",
    )

    # ----------------- Award Verification ----------------- #
    award_ver = evaluator.add_parallel(
        id=f"award_verification_{idx1}",
        desc="Verification of the James Beard award details",
        parent=rest_node,
        critical=True,
    )

    # Award Reference (must confirm the award on jamesbeard.org)
    award_ref_leaf = evaluator.add_leaf(
        id=f"award_reference_{idx1}",
        desc="URL reference from jamesbeard.org confirms the award",
        parent=award_ver,
        critical=True,
    )
    ref_claim = f"This page on jamesbeard.org confirms that {item.chef_name} won a 2025 Best Chef regional award connected to {item.restaurant_name}."
    await evaluator.verify(
        claim=ref_claim,
        node=award_ref_leaf,
        sources=item.award_reference_url,
        additional_instruction="First verify the domain is jamesbeard.org (see the provided URL). Then check the page text or screenshot clearly indicates the chef is a 2025 'Best Chef' winner and the associated restaurant.",
    )

    # Award Category (Best Chef)
    award_cat_leaf = evaluator.add_leaf(
        id=f"award_category_{idx1}",
        desc="Award is a 'Best Chef' award from a regional category",
        parent=award_ver,
        critical=True,
    )
    cat_claim = "This award is a 'Best Chef' award in a James Beard regional category."
    await evaluator.verify(
        claim=cat_claim,
        node=award_cat_leaf,
        sources=item.award_reference_url,
        additional_instruction="Look for phrasing such as 'Best Chef:' followed by a region (e.g., 'Best Chef: Texas').",
    )

    # Award Year (2025)
    award_year_leaf = evaluator.add_leaf(
        id=f"award_year_{idx1}",
        desc="Award year is 2025",
        parent=award_ver,
        critical=True,
    )
    year_claim = "This award was given in 2025."
    await evaluator.verify(
        claim=year_claim,
        node=award_year_leaf,
        sources=item.award_reference_url,
        additional_instruction="Verify the year 2025 explicitly on the page (winner lists or press release).",
    )

    # Award Region (specific region supported by jamesbeard.org)
    award_region_leaf = evaluator.add_leaf(
        id=f"award_region_{idx1}",
        desc="Specific James Beard regional category for the award is correctly identified",
        parent=award_ver,
        critical=True,
    )
    region_claim = f"The award region category for this Best Chef award is '{item.region_category}'."
    allowed_regions_text = "; ".join(ALLOWED_REGION_NAMES)
    await evaluator.verify(
        claim=region_claim,
        node=award_region_leaf,
        sources=item.award_reference_url,
        additional_instruction=f"Confirm the exact region label (allow minor formatting variations like '&' vs 'and'). Valid regions include: {allowed_regions_text}.",
    )

    # Region-State consistency (logic check, non-web factual)
    region_state_consistent = _is_state_in_region(item.state, item.region_category)
    evaluator.add_custom_node(
        result=region_state_consistent,
        id=f"region_state_consistency_{idx1}",
        desc=f"State '{item.state}' belongs to the James Beard region '{_normalize_region_name(item.region_category) or item.region_category}'",
        parent=award_ver,
        critical=True,
    )

    # Region uniqueness across restaurants (for restaurant #2 and #3)
    if index >= 1:
        prev_norm = [_normalize_region_name(r) for r in prior_regions if r]
        current_norm = _normalize_region_name(item.region_category)
        unique_ok = current_norm is not None and (current_norm not in prev_norm)
        evaluator.add_custom_node(
            result=unique_ok,
            id=f"region_uniqueness_{idx1}",
            desc=f"Region for restaurant #{idx1} is different from earlier selections",
            parent=award_ver,
            critical=True,
        )

    # ----------------- Restaurant Location ----------------- #
    location_node = evaluator.add_parallel(
        id=f"restaurant_location_{idx1}",
        desc="Complete physical location information for the restaurant",
        parent=rest_node,
        critical=True,
    )
    loc_sources = _non_empty_urls(item.website_url, item.award_reference_url)

    # Street Address
    street_leaf = evaluator.add_leaf(
        id=f"street_address_{idx1}",
        desc="Restaurant street address is correctly stated",
        parent=location_node,
        critical=True,
    )
    street_claim = f"The restaurant's street address is '{item.street_address}'."
    await evaluator.verify(
        claim=street_claim,
        node=street_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction="Prefer confirming via the official restaurant website. If not shown, cross-check on the jamesbeard.org page.",
    )

    # City
    city_leaf = evaluator.add_leaf(
        id=f"city_{idx1}",
        desc="Restaurant city is correctly stated",
        parent=location_node,
        critical=True,
    )
    city_claim = f"The restaurant is located in the city '{item.city}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction="Confirm the city from the official restaurant website; use the jamesbeard.org page as backup.",
    )

    # State
    state_leaf = evaluator.add_leaf(
        id=f"state_{idx1}",
        desc="Restaurant state is correctly stated",
        parent=location_node,
        critical=True,
    )
    state_claim = f"The restaurant is located in the state '{item.state}'."
    await evaluator.verify(
        claim=state_claim,
        node=state_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction="Confirm the state (prefer 2-letter abbreviation) from the official restaurant website; use the jamesbeard.org page as backup.",
    )

    # ----------------- Restaurant Website ----------------- #
    website_node = evaluator.add_sequential(
        id=f"restaurant_website_{idx1}",
        desc="Official website URL for the restaurant",
        parent=rest_node,
        critical=True,
    )

    # Website URL existence/format
    has_working_url = item.website_url is not None and item.website_url.strip() != "" and (
        item.website_url.strip().lower().startswith("http://") or item.website_url.strip().lower().startswith("https://")
    )
    evaluator.add_custom_node(
        result=has_working_url,
        id=f"website_url_{idx1}",
        desc="A working URL for the restaurant's official website is provided",
        parent=website_node,
        critical=True,
    )

    # Website accessibility and correctness (leads to the correct restaurant)
    site_access_leaf = evaluator.add_leaf(
        id=f"website_accessibility_{idx1}",
        desc="The website URL is accessible and leads to the correct restaurant",
        parent=website_node,
        critical=True,
    )
    site_claim = f"This website belongs to the restaurant named '{item.restaurant_name}'."
    await evaluator.verify(
        claim=site_claim,
        node=site_access_leaf,
        sources=item.website_url,
        additional_instruction="Check page title, header, or 'About/Contact' sections to confirm the website is for the named restaurant. Allow minor naming variations.",
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the James Beard 2025 Best Chef restaurants task.
    """
    # Initialize evaluator with a neutral root, then add a critical Task_Compliance node
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

    task_node = evaluator.add_parallel(
        id="Task_Compliance",
        desc="Overall compliance with task requirements to identify three US restaurants that won James Beard Best Chef awards in 2025 from different regions, providing comprehensive details for each",
        parent=root,
        critical=True,
    )

    # Extract structured items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_awards(),
        template_class=AwardsExtraction,
        extraction_name="awards_extraction",
    )

    # Keep only the first 3 restaurants; pad if fewer
    items: List[AwardItem] = list(extracted.restaurants[:3])
    while len(items) < 3:
        items.append(AwardItem())

    # Record region mapping info for transparency
    evaluator.add_custom_info(
        info={"regions": JBF_REGION_STATES, "synonyms": REGION_SYNONYMS},
        info_type="region_mapping",
        info_name="jbf_region_mapping"
    )

    # Verify each restaurant subtree
    prior_regions: List[str] = []
    for idx, item in enumerate(items):
        await verify_restaurant(evaluator, task_node, item, idx, prior_regions)
        if item.region_category:
            prior_regions.append(item.region_category)

    # Return evaluation summary
    return evaluator.get_summary()