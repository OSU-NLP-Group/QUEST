import asyncio
import logging
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from mind2web2.eval_toolkit import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.evaluator import Evaluator, AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "conference_sponsor_search"
TASK_DESCRIPTION = """
Can you identify 15 sponsors that have supported at least four of the past five CVPR conferences? Please provide the sponsor names, the years they sponsored in the past 5 years, and their sponsorship tier for each year.
"""

# Ground truth data for CVPR sponsors by year and tier
GROUND_TRUTH = {
    "2025": {
        "platinum": [
            "Adobe", "Amazon", "Apple", "ByteDance", "Captions", "GMI Cloud",
            "Google", "Intel", "Meta", "Qualcomm", "Sony", "Tencent"
        ],
        "gold": [
            "Baidu", "Johns Hopkins University, Whiting School of Engineering",
            "Lambda", "Nvidia", "Toloka AI", "Voxel51", "Wayve"
        ],
        "silver": [
            "Ant Research", "Atmanity", "Aurora", "CEB MetaSystems",
            "Commonwealth Bank of Australia", "Encord", "fal.ai", "firstsource",
            "helm.ai", "Kitware", "lakeFS", "Meituan", "Motional", "nexdata",
            "Oad Ridge National Laboratory", "oppo", "Pinterest", "sapien",
            "Scale", "Toyota Research Institute", "Waymo", "Zoox"
        ]
    },
    "2024": {
        "platinum": [
            "Amazon Science", "AMD", "Apple", "ByteDance", "Captions", "Google",
            "Hyperbolic Labs", "Intel", "Lambda", "Meituan", "Meta", "Microsoft",
            "QUALCOMM", "Sony Group Corporation", "The AI Institute"
        ],
        "gold": [
            "Adobe", "Alibaba Cloud", "Ant Research", "Baidu", "Comet", "Gatik",
            "Hyundai Motor Company", "Jane Street", "Latitude AI", "Scale AI",
            "Tianqiao and Chrissy Chen Institute", "Toyota Research Institute",
            "Voxel51", "Wayve", "Weights & Biases", "WYZE", "Zoox"
        ],
        "silver": [
            "Akool Inc", "Anduril Industries", "ASML", "B GARAGE", "CEB Metasystems, Inc.",
            "DeepAuto.ai", "Disney Research", "GE Aerospace Research", "Helm.ai",
            "HPE AI Software Solutions Open Source Community Team", "iMerit", "Kitware",
            "Kuaishou", "LG", "Lightning AI", "NEC Laboratories America, Inc.",
            "Nexdata Technology Inc.", "OPPO", "Pinterest", "Prophesee", "Sapien",
            "Simular", "Snap Inc.", "Synthesia", "Tencent", "Tenyks Ltd.", "Waymo"
        ]
    },
    "2023": {
        "platinum": [
            "Ant Research", "Amazon Science", "Apple", "Cruise LLC", "Facebook AI",
            "Google", "Lambda", "Qualcomm", "Toyota Research Institute"
        ],
        "gold": [
            "Adobe", "Alibaba Cloud", "Baidu", "Beijing Haitian Ruisheng Science technology Ltd. (SpeechOcean)",
            "Datagen Technologies, Inc.", "FuriosaAI", "Iterative", "Latitude AI",
            "Neural Magic", "Novarc Technologies", "Synthesis AI", "TELUS International AI Data Solutions",
            "Tesla, Inc.", "TikTok", "Voxel51", "Weights & Biases", "Zoox"
        ],
        "silver": [
            "adeia", "Dataminr", "Datatang Technology Inc.", "Digital Divide Data",
            "Hyundai Motor Company", "Kitware", "LG", "Lightning AI", "Manot",
            "Meitu MTlab", "National Security Agency", "Prophesee", "Rivian",
            "Scale AI", "Snap Inc.", "Tianqiao & Chrissy Chen Institute",
            "Visual Layer", "Waymo"
        ]
    },
    "2022": {
        "platinum": [
            "Amazon Science", "Anduril", "Apple", "Argo AI", "Cruise", "Datagen",
            "Google", "iMerit", "Inspur", "Intel", "Meta", "Microsoft", "Motional",
            "Qualcomm", "Roku", "Saudi Federation for Cybersecurity, Programming and Drones",
            "Sea AI Lab", "Tencent Applied Research Center", "TikTok", "Toyota Research Institute", "Waymo"
        ],
        "gold": [
            "Activeloop", "Adobe", "Aibee", "Alegion", "Alibaba Group", "Appen",
            "Baidu", "Baobab", "Bosch", "FuriosaAI", "Graphcore", "Labelbox",
            "NAVER Corp", "Neural Magic", "Snap Inc.", "Sony AI", "SuperAnnotate",
            "Superb AI", "TELUS International AI Data Solutions", "Tesla", "V7",
            "Voxel51", "Weights & Biases", "Woven Planet", "Zoox"
        ],
        "silver": [
            "Algolux", "AMAX", "Cogito Tech", "Datatang", "Digital Divide Data",
            "Disney Research", "Hitachi", "Kitware", "KLA", "LG AI Research",
            "Matterport", "Meitu MT Lab", "Micron", "NVIDIA", "OPPO", "Panasonic",
            "Pinterest", "Prophesee", "Samsung AI Center", "Scale AI", "Synthetaic",
            "Tencent Youtu", "Unity", "Zillow"
        ]
    },
    "2021": {
        "champion": [
            "Alibaba Group", "Amazon Science", "Apple", "ByteDance", "Facebook",
            "Google Research", "Kuaishou", "Microsoft", "National Security Agency",
            "Qualcomm", "Sea Limited", "Tencent", "Tencent Youtu", "Toyota Research Institute"
        ],
        "spotlight": [
            "Activeloop", "Adobe", "Baidu", "Futurewei", "iMerit", "Intel",
            "LG AI Research", "Motorola Solutions", "Naver Line", "SuperAnnotate",
            "Superb AI", "Weights & Biases"
        ],
        "premium": [
            "Aibee", "Alegion", "Algolux", "Argo AI", "Dataloop", "Deepen AI",
            "Exxact", "Guangzhou Shiyuan Electronic Technology Co., Ltd.", "Hitachi",
            "Intel RealSense", "Labelbox", "Pinterest", "Snap Inc.", "Waymo"
        ],
        "standard": [
            "AMAX", "Appen", "Beijing Surfing Technology", "Bosch AI", "Caliber Data Labs",
            "Digital Divide Data", "dlabel", "Kitware", "MVTec Software GmbH",
            "NVIDIA", "OPPO", "Sama", "Samsung AI Center Toronto", "Science Robotics",
            "Vzense Technology Inc."
        ]
    }
}

AVAILABLE_YEARS = sorted(int(year) for year in GROUND_TRUTH.keys())
RECENT_YEARS_INT = AVAILABLE_YEARS[-5:] if len(AVAILABLE_YEARS) >= 5 else AVAILABLE_YEARS
RECENT_YEARS = [str(year) for year in RECENT_YEARS_INT]
if RECENT_YEARS:
    RECENT_YEAR_RANGE = f"{RECENT_YEARS[0]}-{RECENT_YEARS[-1]}" if len(RECENT_YEARS) > 1 else RECENT_YEARS[0]
else:
    RECENT_YEAR_RANGE = ""

def get_sponsors_list_for_year(year: str) -> str:
    """Get formatted sponsor list string for a specific year."""
    if year not in GROUND_TRUTH:
        return f"No ground truth data available for {year}"

    year_data = GROUND_TRUTH[year]
    sponsor_list_parts = []

    for tier, sponsors in year_data.items():
        sponsors_str = ", ".join(sponsors)
        sponsor_list_parts.append(f"{tier.upper()} sponsors: {sponsors_str}")

    return f"CVPR {year} sponsors: " + "; ".join(sponsor_list_parts)


# --------------------------------------------------------------------------- #
# Data models for extracted info                                              #
# --------------------------------------------------------------------------- #
class SponsorNamesList(BaseModel):
    """List of sponsor names only."""
    sponsor_names: List[str] = Field(default_factory=list, description="List of sponsor company names")


class SponsorYear(BaseModel):
    """Sponsorship information for a specific year."""
    year: Optional[str] = Field(default=None, description="Year of sponsorship")
    tier: Optional[str] = Field(default=None, description="Sponsorship tier")


class DetailedSponsorInfo(BaseModel):
    """Detailed sponsorship information for a single sponsor."""
    years: List[SponsorYear] = Field(default_factory=list, description="Years and tiers of sponsorship")
    urls: List[str] = Field(default_factory=list, description="Supporting URLs")


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_sponsor_names() -> str:
    """Extract just the sponsor names from the answer."""
    return """
    Extract the list of sponsor company names that are mentioned in the answer as sponsors who have supported at least four of the past five CVPR conferences.

    IMPORTANT:
    - Extract ONLY the company/sponsor names, nothing else
    - Include ALL sponsor names mentioned, even if there are more than 15
    - Extract names exactly as they appear in the answer
    - Do NOT include years, tiers, or any other information - just the names
    - If the answer provides a numbered or bulleted list, extract all names from that list
    """


def prompt_extract_sponsor_details(sponsor_name: str) -> str:
    """Extract detailed sponsorship information for a specific sponsor."""
    return f"""
    For the sponsor "{sponsor_name}", extract the following information from the answer:

    1. All years they sponsored CVPR (within the window {RECENT_YEAR_RANGE}) and their sponsorship tier for each year
    2. Any URLs provided as sources/citations for this sponsor's information

    IMPORTANT:
    - Extract ONLY information related to "{sponsor_name}"
    - Include all years mentioned for this sponsor that fall within {RECENT_YEAR_RANGE}
    - Include the sponsorship tier for each year (e.g., "platinum", "gold", "silver", etc.)
    - Extract ALL URLs that are provided as sources for this sponsor
    - If no information is found for a field, leave it empty
    """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_basic_requirements(
        evaluator: Evaluator,
        parent_node,
        sponsor_names: List[str]
) -> None:
    """Verify basic requirements: at least one sponsor provided."""

    # Check if we have any sponsors at all
    evaluator.add_custom_node(
        result=len(sponsor_names) > 0,
        id="sponsors_provided",
        desc="At least one sponsor is provided in the answer",
        parent=parent_node,
        critical=True
    )


async def verify_sponsor_details(
        evaluator: Evaluator,
        parent_node,
        sponsor_name: str,
        sponsor_details: DetailedSponsorInfo,
        sponsor_index: int
) -> None:
    """Verify individual sponsor details against ground truth."""

    # Create sponsor verification node
    sponsor_node = evaluator.add_parallel(
        id=f"sponsor_{sponsor_index}",
        desc=f"Verification for sponsor {sponsor_index + 1}: {sponsor_name}",
        parent=parent_node,
        critical=False  # Non-critical for partial scoring across sponsors
    )

    # Check if sponsor has at least 4 years of data within the recent window
    valid_years_data: List[SponsorYear] = []
    for year_entry in sponsor_details.years:
        if not year_entry.year:
            continue
        year_str = year_entry.year.strip()
        if year_str in RECENT_YEARS:
            if not year_str.isdigit():
                continue
            year_entry.year = year_str
            valid_years_data.append(year_entry)

    # Sort years in descending order so we check the most recent ones first
    valid_years_data.sort(key=lambda y: int(y.year), reverse=True)

    has_enough_years = len(valid_years_data) >= 4
    evaluator.add_custom_node(
        result=has_enough_years,
        id=f"sponsor_{sponsor_index}_enough_years",
        desc=f"Sponsor {sponsor_index + 1} ({sponsor_name}) provides at least 4 years of sponsorship data within {RECENT_YEAR_RANGE} (provided {len(valid_years_data)})",
        parent=sponsor_node,
        critical=True  # Critical: must have at least 4 years to qualify
    )

    # Verify first 4 years only
    years_to_verify = valid_years_data[:4]

    years_node = evaluator.add_parallel(
        id=f"sponsor_{sponsor_index}_years",
        desc=f"Year-by-year verification for {sponsor_name} (first 4 years)",
        parent=sponsor_node,
        critical=True
    )

    for year_idx, year_info in enumerate(years_to_verify):
        year = year_info.year.strip()
        claimed_tier = year_info.tier or "unknown"

        # Create year verification container node
        year_container_node = evaluator.add_sequential(
            id=f"sponsor_{sponsor_index}_year_{year_idx}",
            desc=f"{sponsor_name} sponsored CVPR {year} as {claimed_tier}",
            parent=years_node,
            critical=True
        )

        # Step 1: Use LLM to verify sponsor presence in ground truth for that year
        gt_verification_node = evaluator.add_leaf(
            id=f"sponsor_{sponsor_index}_year_{year_idx}_gt_match",
            desc=f"{sponsor_name} appears in CVPR {year} ground truth sponsor list",
            parent=year_container_node,
            critical=True  # Critical: must be in GT to proceed
        )

        # Build sponsor list for the year
        year_sponsors_list = get_sponsors_list_for_year(year)

        # Create claim for LLM verification
        gt_claim = f"The company name '{sponsor_name}' appears in the CVPR {year} sponsor list at tier {claimed_tier}. For your reference, here is the complete sponsor list with tiers for CVPR {year}: {year_sponsors_list}"

        await evaluator.verify(
            claim=gt_claim,
            node=gt_verification_node,
            sources=None,
            additional_instruction="Check with the provided sponsor list. Check both 1) whether it appears in the correct year 2) the tier information. Allow for reasonable variations in company names and tier names (e.g., 'Amazon' vs 'Amazon Science', 'Meta' vs 'Facebook', company name changes over time, abbreviations, etc.). The sponsor must be in the list to pass."
        )

        # # Step 2: Check tier accuracy using LLM
        # tier_verification_node = evaluator.add_leaf(
        #     id_=f"sponsor_{sponsor_index}_year_{year_idx}_tier_match",
        #     desc=f"Tier matches: claimed '{claimed_tier}' vs ground truth",
        #     parent=year_container_node,
        #     critical=False  # Non-critical: tier mismatch shouldn't fail the year entirely
        # )
        #
        # tier_claim = f"The sponsor '{sponsor_name}' is listed as a '{claimed_tier}' tier sponsor for CVPR {year}. Here is the complete sponsor list for CVPR {year} organized by tiers: {year_sponsors_list}"
        #
        # await evaluator.verify(
        #     claim=tier_claim,
        #     node=tier_verification_node,
        #     sources=None,
        #     additional_instruction="Verify if the claimed tier matches the actual tier in the sponsor list. The tier names should match (e.g., 'platinum' matches 'platinum', 'gold' matches 'gold'). Case variations are acceptable. If the sponsor is not found in the list at all, this should fail.",
        #     extra_prerequisites=[gt_verification_node]  # Only check tier if sponsor is in GT
        # )

        # Step 3: URL attribution (REQUIRED)
        url_attribution_node = evaluator.add_custom_node(
            result=bool(sponsor_details.urls and len(sponsor_details.urls) > 0),
            id=f"sponsor_{sponsor_index}_year_{year_idx}_urls_provided",
            desc=f"URLs provided for attribution (found {len(sponsor_details.urls) if sponsor_details.urls else 0} URLs)",
            parent=year_container_node,
            critical=True  # Critical: URLs are required
        )

        # Step 4: Verify URL content (only if URLs provided)
        # if sponsor_details.urls:
        url_content_node = evaluator.add_leaf(
            id=f"sponsor_{sponsor_index}_year_{year_idx}_url_verification",
            desc=f"URL content supports {sponsor_name} sponsorship of CVPR {year}",
            parent=year_container_node,
            critical=True
        )

        # Create verification claim
        claim = f"The sponsor '{sponsor_name}' (or company variants/historical names) sponsored CVPR {year}. Or, this is a full list of sponsors for CVPR {year} (if so, no need to further check the sponsors and companies)."

        await evaluator.verify(
            claim=claim,
            node=url_content_node,
            sources=sponsor_details.urls,
            additional_instruction=f"Verify that the provided webpage contain information about CVPR {year} sponsor lists (e.g., official sponsor lists), or explicitly mention that '{sponsor_name}' sponsored CVPR {year} (e.g., news or announcements about sponsorship). Allow for reasonable company name variations and historical name changes."
        )

    # Create placeholder nodes for missing years (if less than 4 provided)
    for missing_idx in range(len(years_to_verify), 4):
        placeholder_node = evaluator.add_leaf(
            id=f"sponsor_{sponsor_index}_year_placeholder_{missing_idx}",
            desc=f"{sponsor_name} year {missing_idx + 1} (not provided)",
            parent=years_node,
            critical=True,
            status="skipped"
        )


async def create_placeholder_sponsor(
        evaluator: Evaluator,
        parent_node,
        sponsor_index: int
) -> None:
    """Create placeholder nodes for missing sponsors up to 15."""

    sponsor_node = evaluator.add_parallel(
        id=f"sponsor_placeholder_{sponsor_index}",
        desc=f"Placeholder for sponsor {sponsor_index + 1} (not provided)",
        parent=parent_node,
        critical=False
    )

    # Add placeholder subnodes - simplified since we don't have a name
    evaluator.add_leaf(
        id=f"sponsor_placeholder_{sponsor_index}_years_check",
        desc=f"Sponsor {sponsor_index + 1} has 4+ years (not provided)",
        parent=sponsor_node,
        critical=True,
        status="skipped"
    )

    years_node = evaluator.add_parallel(
        id=f"sponsor_placeholder_{sponsor_index}_years",
        desc=f"Sponsor {sponsor_index + 1} years verification (not provided)",
        parent=sponsor_node,
        critical=False
    )

    for year_idx in range(4):
        evaluator.add_leaf(
            id=f"sponsor_placeholder_{sponsor_index}_year_{year_idx}",
            desc=f"Sponsor {sponsor_index + 1} year {year_idx + 1} (not provided)",
            parent=years_node,
            critical=False,
            status="skipped"
        )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
    """
    Evaluate an answer to the CVPR sponsors identification task.
    """

    # -------- 1. Initialize evaluator ----------------------------- #
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

    # -------- 2. Extract sponsor names first ---------------------- #
    sponsor_names_result = await evaluator.extract(
        prompt=prompt_extract_sponsor_names(),
        template_class=SponsorNamesList,
        extraction_name="sponsor_names_extraction",
        source=None,
    )

    sponsor_names = sponsor_names_result.sponsor_names
    logger.info(f"Extracted {len(sponsor_names)} sponsor names from the answer")

    # Add ground truth information
    evaluator.add_ground_truth(GROUND_TRUTH, "cvpr_sponsors_ground_truth")

    # -------- 3. Extract detailed info for each sponsor ----------- #
    sponsors_details = []
    for idx, sponsor_name in enumerate(sponsor_names[:15]):  # Only process first 15
        logger.info(f"Extracting details for sponsor {idx + 1}: {sponsor_name}")

        details = await evaluator.extract(
            prompt=prompt_extract_sponsor_details(sponsor_name),
            template_class=DetailedSponsorInfo,
            extraction_name=f"sponsor_{idx}_details",
            source=None,
        )
        sponsors_details.append((sponsor_name, details))

    # Add extraction statistics
    evaluator.add_custom_info({
        "total_sponsors_extracted": len(sponsor_names),
        "sponsors_processed": len(sponsors_details),
        "sponsors_with_years": len([d for _, d in sponsors_details if d.years]),
        "sponsors_with_urls": len([d for _, d in sponsors_details if d.urls]),
    }, "extraction_statistics")

    # -------- 4. Build verification tree -------------------------- #
    #
    # # Basic requirement verification
    # await verify_basic_requirements(evaluator, root, sponsor_names)

    # Verify each sponsor's details
    for i, (sponsor_name, details) in enumerate(sponsors_details):
        await verify_sponsor_details(evaluator, root, sponsor_name, details, i)

    # Create placeholders for missing sponsors (if less than 15 provided)
    for i in range(len(sponsors_details), 15):
        await create_placeholder_sponsor(evaluator, root, i)

    # Add final statistics
    qualifying_sponsors = 0
    for sponsor_name, details in sponsors_details:
        valid_years = len([y for y in details.years if y.year and y.year.strip()])
        if valid_years >= 4:
            qualifying_sponsors += 1

    evaluator.add_custom_info({
        "sponsors_provided": len(sponsors_details),
        "potential_qualifying_sponsors": qualifying_sponsors,
        "target_sponsors": 15,
        "coverage_rate": len(sponsors_details) / 15,
    }, "final_results")

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()
