import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yellowstone_first_park"
TASK_DESCRIPTION = (
    "Identify the first national park established in the United States, which was "
    "designated on March 1, 1872, by President Ulysses S. Grant. For this park, provide: "
    "(1) the UNESCO World Heritage Site designation year, "
    "(2) the number of geysers (the park has the world's largest concentration), "
    "(3) the names of the two major waterfalls in its Grand Canyon area, and "
    "(4) the height in feet of the Lower Falls, which is the tallest waterfall in the park. "
    "Include authoritative reference URLs (e.g., UNESCO WHC, NPS) for each piece of information."
)

EXPECTED_PARK_NAME = "Yellowstone National Park"
EXPECTED_ESTABLISHMENT_DATE = "March 1, 1872"
EXPECTED_ESTABLISHED_BY = "Ulysses S. Grant"
EXPECTED_UNESCO_YEAR = "1978"
EXPECTED_CRITERIA_SET = {"vii", "viii", "ix", "x"}  # For reference in instructions
EXPECTED_LOWER_FALLS_HEIGHT_FT = "308"
EXPECTED_LOWER_FALLS_HEIGHT_M = "94"

# --------------------------------------------------------------------------- #
# Data extraction model                                                       #
# --------------------------------------------------------------------------- #
class YellowstoneExtraction(BaseModel):
    # Park identity
    park_name: Optional[str] = None
    establishment_date: Optional[str] = None
    established_by: Optional[str] = None
    first_national_park_claim: Optional[str] = None
    identity_urls: List[str] = Field(default_factory=list)

    # UNESCO designation
    unesco_year: Optional[str] = None
    unesco_year_urls: List[str] = Field(default_factory=list)

    unesco_criteria: List[str] = Field(default_factory=list)  # e.g., ["(vii)", "(viii)", "(ix)", "(x)"]
    unesco_criteria_urls: List[str] = Field(default_factory=list)

    # Geysers
    geyser_more_than_300_claim: Optional[str] = None  # e.g., "more than 300", "over 500"
    geyser_two_thirds_claim: Optional[str] = None     # e.g., "about two-thirds", "≈2/3"
    geyser_urls: List[str] = Field(default_factory=list)

    # Waterfalls (Grand Canyon of the Yellowstone)
    waterfalls_upper_name: Optional[str] = None
    waterfalls_lower_name: Optional[str] = None
    waterfalls_urls: List[str] = Field(default_factory=list)

    # Lower Falls details
    lower_falls_height_ft: Optional[str] = None       # e.g., "308", "308 ft", "308 feet"
    lower_falls_height_m: Optional[str] = None        # e.g., "94", "94 m", "94 meters"
    lower_falls_height_urls: List[str] = Field(default_factory=list)

    lower_falls_tallest_claim: Optional[str] = None   # e.g., "tallest waterfall in the park"
    lower_falls_tallest_urls: List[str] = Field(default_factory=list)

    lower_falls_twice_niagara_claim: Optional[str] = None  # e.g., "about twice the height of Niagara Falls"
    lower_falls_twice_niagara_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_yellowstone() -> str:
    return """
Extract the following fields exactly as they appear from the provided answer text. Do NOT invent or infer any information. If an item is not explicitly present in the answer, set it to null or an empty list accordingly.

1) Park identity (the first U.S. national park established on March 1, 1872 by President Ulysses S. Grant):
   - park_name: The park name mentioned (e.g., "Yellowstone National Park").
   - establishment_date: The establishment date as written in the answer (e.g., "March 1, 1872").
   - established_by: The person who established/signed the act (e.g., "Ulysses S. Grant").
   - first_national_park_claim: The exact phrasing or short phrase that the answer uses to claim it's the first U.S. national park (or null if not explicitly claimed).
   - identity_urls: A list of URLs in the answer that directly support the park identification/establishment facts.

2) UNESCO designation:
   - unesco_year: The year the site was designated/inscribed as a UNESCO World Heritage Site (as written in the answer).
   - unesco_year_urls: URLs that the answer cites to support the UNESCO year.
   - unesco_criteria: The list of UNESCO natural criteria the answer lists for this site (e.g., ["(vii)", "(viii)", "(ix)", "(x)"]). Keep the original formatting from the answer (including parentheses/casing).
   - unesco_criteria_urls: URLs that support the listed UNESCO criteria.

3) Geysers (largest concentration in the world):
   - geyser_more_than_300_claim: The answer’s phrasing about having more than 300 geysers (or a logically stronger statement like “over 500”), or null if absent.
   - geyser_two_thirds_claim: The answer’s phrasing about approximately two-thirds of the world’s geysers being in the park, or null if absent.
   - geyser_urls: URLs the answer cites to support the geyser claims.

4) Grand Canyon of the Yellowstone waterfalls:
   - waterfalls_upper_name: The answer’s name for the upper waterfall (e.g., "Upper Falls", "Upper Yellowstone Falls").
   - waterfalls_lower_name: The answer’s name for the lower waterfall (e.g., "Lower Falls", "Lower Yellowstone Falls").
   - waterfalls_urls: URLs the answer cites to support the waterfall names.

5) Lower Falls details:
   - lower_falls_height_ft: The height in feet as written in the answer (e.g., "308", "308 ft", "308 feet").
   - lower_falls_height_m: The height in meters as written in the answer (e.g., "94", "94 m", "94 meters"), if provided.
   - lower_falls_height_urls: URLs that support the height value.
   - lower_falls_tallest_claim: The answer’s phrasing claiming the Lower Falls is the tallest waterfall in the park, or null.
   - lower_falls_tallest_urls: URLs that support that tallest-in-park claim.
   - lower_falls_twice_niagara_claim: The answer’s phrasing that Lower Falls is approximately twice the height of Niagara Falls, or null.
   - lower_falls_twice_niagara_urls: URLs that support the Niagara comparison claim.

SPECIAL RULES FOR URL FIELDS:
- Extract only URLs explicitly present in the answer text (plain, markdown, or similar).
- Include full URLs with protocol (prepend http:// if missing).
- Do not add or infer URLs that are not present in the answer.

Return a single JSON object matching the specified fields.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*url_lists: Optional[List[str]]) -> List[str]:
    """Merge and deduplicate URL lists while preserving order."""
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def authoritative_instruction(allowed_domains: List[str], extra_requirements: Optional[str] = None) -> str:
    """Build an instruction requiring authoritative domains for support."""
    domains_list = ", ".join(allowed_domains)
    extra = f"\nAdditional requirement: {extra_requirements}" if extra_requirements else ""
    return (
        "Only mark the claim as supported if at least one provided source URL is from an authoritative site. "
        f"Acceptable authoritative domains include: {domains_list}. "
        "If none of the provided URLs are from these domains, return 'Incorrect' even if the content seems to agree. "
        "Allow minor phrasing variations (e.g., 'inscribed' vs 'designated', 'signed into law' vs 'established') "
        "and treat factual equivalences (e.g., 'over 500' logically implies 'more than 300')."
        f"{extra}"
    )


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_identify_park_subtree(evaluator: Evaluator, parent, data: YellowstoneExtraction) -> None:
    """
    Identify the correct park and support with authoritative reference URLs.
    """
    node = evaluator.add_parallel(
        id="identify_park",
        desc="Identify the correct park and support the identification with an authoritative reference URL.",
        parent=parent,
        critical=True,
    )

    # 1) Value presence in the answer (no URL; check answer content)
    leaf_value = evaluator.add_leaf(
        id="park_identity_and_establishment",
        desc="States the park name and that it was established on March 1, 1872 by President Ulysses S. Grant as the first U.S. national park.",
        parent=node,
        critical=True,
    )
    claim_value = (
        f"The answer identifies the park as {EXPECTED_PARK_NAME} and states it was established "
        f"on {EXPECTED_ESTABLISHMENT_DATE} by President {EXPECTED_ESTABLISHED_BY} as the first U.S. national park."
    )
    await evaluator.verify(
        claim=claim_value,
        node=leaf_value,
        sources=None,
        additional_instruction=(
            "Judge only based on whether the answer text explicitly states these facts. "
            "Minor wording variants are acceptable (e.g., 'signed into law' by the President). "
            "If any of the key facts (name, date, president, first national park) are missing or incorrect, return Incorrect."
        ),
    )

    # 2) Authoritative reference URL support (URL-based)
    leaf_ref = evaluator.add_leaf(
        id="park_identity_reference_url",
        desc="Provides an authoritative reference URL supporting the park’s establishment/first-national-park claim.",
        parent=node,
        critical=True,
    )
    identity_sources = data.identity_urls or []
    claim_ref = (
        f"{EXPECTED_PARK_NAME} was established on {EXPECTED_ESTABLISHMENT_DATE} by U.S. President "
        f"{EXPECTED_ESTABLISHED_BY} and is the first U.S. national park."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=leaf_ref,
        sources=identity_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov", "whc.unesco.org", "archives.gov", "doi.gov", "loc.gov"]
        ),
    )


async def build_required_attributes_subtree(evaluator: Evaluator, parent, data: YellowstoneExtraction) -> None:
    """
    Provide all required UNESCO, geyser, and waterfall attributes with authoritative URLs.
    """
    req = evaluator.add_parallel(
        id="required_attributes",
        desc="Provide all required UNESCO, geyser, and waterfall attributes for the identified park, each supported by authoritative reference URLs.",
        parent=parent,
        critical=True,
    )

    # --- UNESCO Year ---
    unesco_year = evaluator.add_parallel(
        id="unesco_designation_year",
        desc="Gives the UNESCO World Heritage Site designation year (1978) with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (answer states 1978)
    leaf_uy_val = evaluator.add_leaf(
        id="unesco_year_value",
        desc="States the UNESCO designation year as 1978.",
        parent=unesco_year,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The answer states that {EXPECTED_PARK_NAME} was designated/inscribed as a UNESCO World Heritage Site in {EXPECTED_UNESCO_YEAR}.",
        node=leaf_uy_val,
        sources=None,
        additional_instruction="Check only the answer content. Accept synonyms like 'inscribed in 1978'. If the year isn't 1978, return Incorrect.",
    )

    # Reference URL (confirm 1978 on authoritative page)
    leaf_uy_ref = evaluator.add_leaf(
        id="unesco_year_reference_url",
        desc="Provides an authoritative reference URL confirming the 1978 designation year.",
        parent=unesco_year,
        critical=True,
    )
    unesco_year_sources = combine_sources(data.unesco_year_urls, data.unesco_criteria_urls, data.identity_urls)
    await evaluator.verify(
        claim=f"{EXPECTED_PARK_NAME} was inscribed as a UNESCO World Heritage Site in {EXPECTED_UNESCO_YEAR}.",
        node=leaf_uy_ref,
        sources=unesco_year_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["whc.unesco.org", "nps.gov"]
        ),
    )

    # --- UNESCO Criteria ---
    unesco_criteria = evaluator.add_parallel(
        id="unesco_natural_criteria",
        desc="States the site meets UNESCO natural criteria (vii), (viii), (ix), and (x) with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (answer lists criteria)
    leaf_uc_val = evaluator.add_leaf(
        id="unesco_criteria_values",
        desc="Lists UNESCO natural criteria (vii), (viii), (ix), and (x).",
        parent=unesco_criteria,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer lists the UNESCO natural criteria for Yellowstone as (vii), (viii), (ix), and (x), "
            "or an equivalent representation (allowing minor casing/parentheses variations)."
        ),
        node=leaf_uc_val,
        sources=None,
        additional_instruction=(
            "Judge only the answer content. Consider '(VII)' and 'vii' equivalent; allow missing parentheses or commas. "
            "All four criteria must be present to pass."
        ),
    )

    # Reference URL (confirm criteria on authoritative page)
    leaf_uc_ref = evaluator.add_leaf(
        id="unesco_criteria_reference_url",
        desc="Provides an authoritative reference URL supporting the listed criteria.",
        parent=unesco_criteria,
        critical=True,
    )
    uc_sources = combine_sources(data.unesco_criteria_urls, data.unesco_year_urls, data.identity_urls)
    await evaluator.verify(
        claim=f"The UNESCO listing for {EXPECTED_PARK_NAME} shows it meets natural criteria (vii), (viii), (ix), and (x).",
        node=leaf_uc_ref,
        sources=uc_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["whc.unesco.org", "nps.gov"]
        ),
    )

    # --- Geyser Concentration ---
    geysers = evaluator.add_parallel(
        id="geyser_concentration",
        desc="States the park has the world’s largest concentration of geysers, including the quantitative constraints, with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (more than 300)
    leaf_g300_val = evaluator.add_leaf(
        id="geyser_more_than_300_value",
        desc="States the park has more than 300 geysers.",
        parent=geysers,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the park has more than 300 geysers. Treat stronger statements such as "
            "'over 500 geysers' or 'more than five hundred' as satisfying 'more than 300'."
        ),
        node=leaf_g300_val,
        sources=None,
        additional_instruction="Judge only the answer content. Logical implication is acceptable (e.g., >500 implies >300).",
    )

    # Value (approximately two-thirds)
    leaf_g23_val = evaluator.add_leaf(
        id="geyser_approx_two_thirds_value",
        desc="States the park has approximately two-thirds of all geysers on Earth.",
        parent=geysers,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the park contains approximately two-thirds of the world's geysers. "
            "Accept variants like 'about two-thirds', 'around 2/3', or '~66%'."
        ),
        node=leaf_g23_val,
        sources=None,
        additional_instruction="Judge only the answer content. Minor phrasing variations are acceptable.",
    )

    # Reference URL (confirm quantitative facts on authoritative page)
    leaf_g_ref = evaluator.add_leaf(
        id="geyser_reference_url",
        desc="Provides an authoritative reference URL supporting the geyser concentration/quantity claim(s).",
        parent=geysers,
        critical=True,
    )
    geyser_sources = combine_sources(data.geyser_urls, data.identity_urls)
    await evaluator.verify(
        claim=(
            f"{EXPECTED_PARK_NAME} has more than 300 geysers and contains approximately two-thirds of all geysers on Earth."
        ),
        node=leaf_g_ref,
        sources=geyser_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov", "whc.unesco.org"],
            extra_requirements="If only one of the two quantitative facts is supported by the URL(s), return Incorrect.",
        ),
    )

    # --- Grand Canyon Waterfalls Names ---
    wnames = evaluator.add_parallel(
        id="grand_canyon_waterfalls_names",
        desc="Names the two major waterfalls in the park’s Grand Canyon area (Upper Falls and Lower Falls) with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (Upper & Lower Falls)
    leaf_wn_val = evaluator.add_leaf(
        id="waterfalls_names_value",
        desc="States the two major waterfalls are the Upper Falls and the Lower Falls.",
        parent=wnames,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the two major waterfalls in the Grand Canyon of the Yellowstone are the Upper Falls "
            "and the Lower Falls (allowing variants like 'Upper Yellowstone Falls'/'Lower Yellowstone Falls')."
        ),
        node=leaf_wn_val,
        sources=None,
        additional_instruction="Judge only the answer content. Minor name variants are acceptable.",
    )

    # Reference URL (confirm names on authoritative page)
    leaf_wn_ref = evaluator.add_leaf(
        id="waterfalls_names_reference_url",
        desc="Provides an authoritative reference URL supporting the waterfall names.",
        parent=wnames,
        critical=True,
    )
    waterfalls_sources = combine_sources(data.waterfalls_urls, data.identity_urls)
    await evaluator.verify(
        claim="The two major waterfalls in the Grand Canyon of the Yellowstone are the Upper Falls and the Lower Falls.",
        node=leaf_wn_ref,
        sources=waterfalls_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov"]
        ),
    )

    # --- Lower Falls Height ---
    lf_height = evaluator.add_parallel(
        id="lower_falls_height",
        desc="Provides the height of the Lower Falls (308 feet / 94 meters) with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (308 feet / 94 meters)
    leaf_lfh_val = evaluator.add_leaf(
        id="lower_falls_height_value",
        desc="States the Lower Falls height is 308 feet (94 meters).",
        parent=lf_height,
        critical=True,
    )
    await evaluator.verify(
        claim=(
            "The answer states that the Lower Falls is 308 feet (94 meters) tall. "
            "Accept minor wording variants like '308 ft', '308-foot', '94 m'."
        ),
        node=leaf_lfh_val,
        sources=None,
        additional_instruction="Judge only the answer content.",
    )

    # Reference URL (confirm height on authoritative page)
    leaf_lfh_ref = evaluator.add_leaf(
        id="lower_falls_height_reference_url",
        desc="Provides an authoritative reference URL confirming the Lower Falls height.",
        parent=lf_height,
        critical=True,
    )
    lf_height_sources = combine_sources(data.lower_falls_height_urls, data.waterfalls_urls, data.identity_urls)
    await evaluator.verify(
        claim=f"The Lower Falls is {EXPECTED_LOWER_FALLS_HEIGHT_FT} feet ({EXPECTED_LOWER_FALLS_HEIGHT_M} meters) tall.",
        node=leaf_lfh_ref,
        sources=lf_height_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov"]
        ),
    )

    # --- Lower Falls Tallest in Park ---
    lf_tallest = evaluator.add_parallel(
        id="lower_falls_tallest_in_park",
        desc="States the Lower Falls is the tallest waterfall in the park with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (tallest in park)
    leaf_lft_val = evaluator.add_leaf(
        id="lower_falls_tallest_value",
        desc="States the Lower Falls is the tallest waterfall in the park.",
        parent=lf_tallest,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Lower Falls is the tallest waterfall in Yellowstone National Park.",
        node=leaf_lft_val,
        sources=None,
        additional_instruction="Judge only the answer content. Accept 'highest' as equivalent to 'tallest'.",
    )

    # Reference URL (confirm tallest on authoritative page)
    leaf_lft_ref = evaluator.add_leaf(
        id="lower_falls_tallest_reference_url",
        desc="Provides an authoritative reference URL supporting that the Lower Falls is the tallest in the park.",
        parent=lf_tallest,
        critical=True,
    )
    lf_tallest_sources = combine_sources(data.lower_falls_tallest_urls, data.waterfalls_urls, data.identity_urls)
    await evaluator.verify(
        claim="The Lower Falls is the tallest waterfall in Yellowstone National Park.",
        node=leaf_lft_ref,
        sources=lf_tallest_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov"]
        ),
    )

    # --- Lower Falls ≈ Twice Niagara ---
    lf_niagara = evaluator.add_parallel(
        id="lower_falls_twice_niagara",
        desc="States the Lower Falls is approximately twice the height of Niagara Falls with an authoritative reference URL.",
        parent=req,
        critical=True,
    )

    # Value (≈2x Niagara Falls)
    leaf_lfn_val = evaluator.add_leaf(
        id="lower_falls_twice_niagara_value",
        desc="States the Lower Falls is approximately twice the height of Niagara Falls.",
        parent=lf_niagara,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer states that the Lower Falls is approximately twice the height of Niagara Falls.",
        node=leaf_lfn_val,
        sources=None,
        additional_instruction="Judge only the answer content. Accept phrasing like 'about 2x' or 'twice as high'.",
    )

    # Reference URL (confirm Niagara comparison on authoritative page)
    leaf_lfn_ref = evaluator.add_leaf(
        id="lower_falls_twice_niagara_reference_url",
        desc="Provides an authoritative reference URL supporting the Niagara comparison.",
        parent=lf_niagara,
        critical=True,
    )
    lf_niagara_sources = combine_sources(data.lower_falls_twice_niagara_urls, data.waterfalls_urls, data.identity_urls)
    await evaluator.verify(
        claim="The Lower Falls is approximately twice the height of Niagara Falls.",
        node=leaf_lfn_ref,
        sources=lf_niagara_sources,
        additional_instruction=authoritative_instruction(
            allowed_domains=["nps.gov", "whc.unesco.org", "archives.gov", "doi.gov", "loc.gov"]
        ),
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
    Evaluate an answer for the Yellowstone first national park task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Identify first, then verify attributes
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

    # Extraction
    extracted: YellowstoneExtraction = await evaluator.extract(
        prompt=prompt_extract_yellowstone(),
        template_class=YellowstoneExtraction,
        extraction_name="yellowstone_structured_extraction",
    )

    # Build verification tree
    await build_identify_park_subtree(evaluator, root, extracted)
    await build_required_attributes_subtree(evaluator, root, extracted)

    # Optional: add ground truth reference info for transparency
    evaluator.add_ground_truth({
        "expected_park_name": EXPECTED_PARK_NAME,
        "expected_establishment_date": EXPECTED_ESTABLISHMENT_DATE,
        "expected_established_by": EXPECTED_ESTABLISHED_BY,
        "expected_unesco_year": EXPECTED_UNESCO_YEAR,
        "expected_unesco_criteria": sorted(list(EXPECTED_CRITERIA_SET)),
        "expected_lower_falls_height_ft": EXPECTED_LOWER_FALLS_HEIGHT_FT,
        "expected_lower_falls_height_m": EXPECTED_LOWER_FALLS_HEIGHT_M,
        "notes": "Authoritative sources should primarily be from nps.gov and whc.unesco.org for this task."
    }, gt_type="expected_facts")

    return evaluator.get_summary()