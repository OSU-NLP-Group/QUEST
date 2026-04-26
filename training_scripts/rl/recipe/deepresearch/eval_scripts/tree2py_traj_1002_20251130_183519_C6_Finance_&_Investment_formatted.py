import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "sp500_dec2023_ca"
TASK_DESCRIPTION = (
    "In December 2023, three companies were added to the S&P 500 index. Identify which one of these three companies is "
    "headquartered in California, and provide the following information:\n\n"
    "1. The company's full name\n"
    "2. The complete headquarters location, including:\n"
    "   - City\n"
    "   - State\n"
    "   - Street address\n"
    "3. The specific date when this company was added to the S&P 500\n"
    "4. California's ranking among U.S. states in terms of having the most Fortune 500 companies, and the number of "
    "Fortune 500 companies headquartered in California as of 2024\n"
    "5. The names of the three companies that were removed from the S&P 500 on the same date\n\n"
    "Additionally, confirm that the identified company met the S&P 500 eligibility requirements regarding positive "
    "earnings and U.S. domicile.\n\n"
    "Provide reference URLs for all factual claims."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class CompanySources(BaseModel):
    company_identification_urls: List[str] = Field(default_factory=list)
    hq_location_urls: List[str] = Field(default_factory=list)
    sp500_addition_date_urls: List[str] = Field(default_factory=list)
    ca_f500_context_urls: List[str] = Field(default_factory=list)
    removed_companies_urls: List[str] = Field(default_factory=list)
    positive_earnings_urls: List[str] = Field(default_factory=list)
    us_domicile_urls: List[str] = Field(default_factory=list)


class ExtractedCompanyInfo(BaseModel):
    company_full_name: Optional[str] = None

    hq_city: Optional[str] = None
    hq_state: Optional[str] = None
    hq_street_address: Optional[str] = None

    sp500_addition_date: Optional[str] = None

    california_ranking: Optional[str] = None  # e.g., "1st", "2", "second"
    california_f500_count_2024: Optional[str] = None  # keep as string for robustness

    removed_companies: List[str] = Field(default_factory=list)

    positive_earnings_confirmation: Optional[str] = None  # text confirming requirement met, if provided
    us_domicile_confirmation: Optional[str] = None  # text confirming U.S. domicile, if provided

    sources: CompanySources = Field(default_factory=CompanySources)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_company_info() -> str:
    return """
    From the answer text, extract the structured information requested by the task about the S&P 500 addition in December 2023 that is headquartered in California.

    REQUIRED FIELDS:
    1) company_full_name: The full, official name of the identified company.
    2) hq_city: The city of the company's headquarters.
    3) hq_state: The state of the company's headquarters (e.g., "California" or "CA").
    4) hq_street_address: The street address line (number + street name) for the company's headquarters (e.g., "123 Main St").
    5) sp500_addition_date: The specific date when this company was added to the S&P 500 (e.g., "December 18, 2023").
    6) california_ranking: California’s ranking among U.S. states by number of Fortune 500 company headquarters as of 2024 (e.g., "1st", "2nd", "second", "2").
    7) california_f500_count_2024: The number of Fortune 500 companies headquartered in California as of 2024 (e.g., "57").
    8) removed_companies: An array of the names of the three companies removed from the S&P 500 on the same date as the identified company’s addition. If more than three are mentioned, include the first three. If fewer than three are present, include all mentioned.
    9) positive_earnings_confirmation: If the answer explicitly states that the identified company met the positive-earnings S&P 500 eligibility requirement at the time of index addition (positive most recent quarter and positive sum of the trailing four quarters), return that statement or a concise summary. Otherwise return null.
    10) us_domicile_confirmation: If the answer explicitly states the company is U.S.-domiciled, return the statement or a concise summary. Otherwise return null.

    SOURCE URLS:
    Extract the actual URLs the answer cites to support each factual claim. Only include URLs explicitly present in the answer (plain links or markdown links).
    - sources.company_identification_urls: URLs supporting that the identified company is indeed one of the three added in December 2023.
    - sources.hq_location_urls: URLs supporting the full HQ address (city/state/street) and that the company is headquartered in California.
    - sources.sp500_addition_date_urls: URLs supporting the specific S&P 500 addition date.
    - sources.ca_f500_context_urls: URLs supporting California’s Fortune 500 ranking and the 2024 count.
    - sources.removed_companies_urls: URLs listing the three removed companies on the same date.
    - sources.positive_earnings_urls: URLs supporting that the company met the positive-earnings eligibility requirement at time of addition.
    - sources.us_domicile_urls: URLs supporting that the company is U.S.-domiciled.

    IMPORTANT RULES:
    - Do not invent or infer any URL. Only include URLs that appear in the answer text.
    - If a field is not present, return null (or [] for arrays).
    - Keep all fields as strings (except arrays) even if they look numeric (e.g., counts, dates).
    - For removed_companies, provide a clean list of organization names as written in the answer (strip extra punctuation).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _combine_sources(*lists_or_strs: Any) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in lists_or_strs:
        if not item:
            continue
        if isinstance(item, list):
            for u in item:
                if u and isinstance(u, str) and u not in seen:
                    seen.add(u)
                    out.append(u)
        elif isinstance(item, str):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ExtractedCompanyInfo) -> None:
    """
    Build and execute the verification tree according to the rubric.
    """
    # Create the main critical sequential node (ResearchTaskCompletion)
    main = evaluator.add_sequential(
        id="ResearchTaskCompletion",
        desc="Complete the task: identify the California-headquartered company among the December 2023 S&P 500 additions and provide all requested details with supporting references.",
        parent=evaluator.root,
        critical=True
    )

    # 1) IdentifyCaliforniaHeadquarteredAddition (critical, parallel)
    identify_node = evaluator.add_parallel(
        id="IdentifyCaliforniaHeadquarteredAddition",
        desc="Correctly identify which of the three December 2023 S&P 500 additions is headquartered in California.",
        parent=main,
        critical=True
    )

    # 1.a) ProvidesCompanyFullName (critical leaf) - existence check
    evaluator.add_custom_node(
        result=_non_empty_str(extracted.company_full_name),
        id="ProvidesCompanyFullName",
        desc="Provides the company’s full name.",
        parent=identify_node,
        critical=True
    )

    # 1.b) IsOneOfThreeDecember2023Additions (critical leaf) - verify via sources
    is_dec_add_leaf = evaluator.add_leaf(
        id="IsOneOfThreeDecember2023Additions",
        desc="The identified company is one of the three companies added to the S&P 500 in December 2023 (per the prompt/constraints).",
        parent=identify_node,
        critical=True
    )
    claim_dec_add = (
        f"{extracted.company_full_name or 'UNKNOWN'} was one of the three companies added to the S&P 500 in December 2023."
    )
    id_sources = _combine_sources(
        extracted.sources.company_identification_urls,
        extracted.sources.sp500_addition_date_urls
    )
    await evaluator.verify(
        claim=claim_dec_add,
        node=is_dec_add_leaf,
        sources=id_sources,
        additional_instruction=(
            "Check that the referenced page(s) state the company was added to the S&P 500 in December 2023 and that there "
            "were three additions. Accept S&P Dow Jones Indices press releases or reputable financial news coverage."
        )
    )

    # 1.c) HeadquarteredInCalifornia (critical leaf) - verify via HQ sources
    hq_in_ca_leaf = evaluator.add_leaf(
        id="HeadquarteredInCalifornia",
        desc="Correctly indicates the identified company’s headquarters is in California.",
        parent=identify_node,
        critical=True
    )
    claim_hq_ca = (
        f"The headquarters of {extracted.company_full_name or 'UNKNOWN'} is in California."
    )
    await evaluator.verify(
        claim=claim_hq_ca,
        node=hq_in_ca_leaf,
        sources=extracted.sources.hq_location_urls,
        additional_instruction=(
            "Verify that the official or authoritative source indicates the HQ is located in a California city. "
            "Company websites, SEC filings, or reliable encyclopedic sources are acceptable."
        )
    )

    # 2) ProvideRequestedInformation (critical, parallel)
    provide_info = evaluator.add_parallel(
        id="ProvideRequestedInformation",
        desc="Provide all requested information for the identified company and related context.",
        parent=main,
        critical=True
    )

    # 2.a) HeadquartersLocation (critical, parallel)
    hq_node = evaluator.add_parallel(
        id="HeadquartersLocation",
        desc="Provide the complete headquarters location (city, state, street address).",
        parent=provide_info,
        critical=True
    )

    # 2.a.i) HeadquartersCityProvided
    hq_city_leaf = evaluator.add_leaf(
        id="HeadquartersCityProvided",
        desc="Headquarters city is provided.",
        parent=hq_node,
        critical=True
    )
    claim_hq_city = (
        f"The headquarters city of {extracted.company_full_name or 'UNKNOWN'} is {extracted.hq_city or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_hq_city,
        node=hq_city_leaf,
        sources=extracted.sources.hq_location_urls,
        additional_instruction=(
            "Confirm the HQ city exactly or near-exactly matches the provided value. Minor formatting differences are acceptable."
        )
    )

    # 2.a.ii) HeadquartersStateProvided
    hq_state_leaf = evaluator.add_leaf(
        id="HeadquartersStateProvided",
        desc="Headquarters state is provided.",
        parent=hq_node,
        critical=True
    )
    claim_hq_state = (
        f"The headquarters state of {extracted.company_full_name or 'UNKNOWN'} is {extracted.hq_state or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_hq_state,
        node=hq_state_leaf,
        sources=extracted.sources.hq_location_urls,
        additional_instruction=(
            "Confirm the HQ state matches the provided value (e.g., 'California' or 'CA'). Minor variations and abbreviations are acceptable."
        )
    )

    # 2.a.iii) HeadquartersStreetAddressProvided
    hq_street_leaf = evaluator.add_leaf(
        id="HeadquartersStreetAddressProvided",
        desc="Headquarters street address (street number + street name) is provided.",
        parent=hq_node,
        critical=True
    )
    claim_hq_street = (
        f"The headquarters street address of {extracted.company_full_name or 'UNKNOWN'} is {extracted.hq_street_address or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_hq_street,
        node=hq_street_leaf,
        sources=extracted.sources.hq_location_urls,
        additional_instruction=(
            "Verify that the street address (number + street name) matches. Allow minor formatting differences (e.g., 'St' vs 'Street')."
        )
    )

    # 2.b) SP500AdditionDateProvided (critical leaf)
    add_date_leaf = evaluator.add_leaf(
        id="SP500AdditionDateProvided",
        desc="Provides the specific date the identified company was added to the S&P 500.",
        parent=provide_info,
        critical=True
    )
    claim_add_date = (
        f"{extracted.company_full_name or 'UNKNOWN'} was added to the S&P 500 on {extracted.sp500_addition_date or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_add_date,
        node=add_date_leaf,
        sources=extracted.sources.sp500_addition_date_urls,
        additional_instruction=(
            "Confirm the effective date of index addition (e.g., 'effective prior to the open on December 18, 2023'). "
            "Match the date string, allowing minor phrasing differences."
        )
    )

    # 2.c) CaliforniaFortune500Context (critical, parallel)
    ca_f500_node = evaluator.add_parallel(
        id="CaliforniaFortune500Context",
        desc="Provide California’s Fortune 500 context as of 2024 (ranking among states and the number of Fortune 500 HQs).",
        parent=provide_info,
        critical=True
    )

    ca_rank_leaf = evaluator.add_leaf(
        id="CaliforniaRankingProvided",
        desc="States California’s ranking among U.S. states for having the most Fortune 500 companies.",
        parent=ca_f500_node,
        critical=True
    )
    claim_ca_rank = (
        f"As of 2024, California's ranking among U.S. states by number of Fortune 500 company headquarters is {extracted.california_ranking or 'UNKNOWN'}."
    )
    await evaluator.verify(
        claim=claim_ca_rank,
        node=ca_rank_leaf,
        sources=extracted.sources.ca_f500_context_urls,
        additional_instruction=(
            "Check a 2024 Fortune 500 state breakdown or equivalent credible source. Accept ordinal words or numerals (e.g., '2nd', 'second', '2')."
        )
    )

    ca_count_leaf = evaluator.add_leaf(
        id="CaliforniaCompanyCountProvided",
        desc="Provides the number of Fortune 500 companies headquartered in California as of 2024.",
        parent=ca_f500_node,
        critical=True
    )
    claim_ca_count = (
        f"As of 2024, California has {extracted.california_f500_count_2024 or 'UNKNOWN'} Fortune 500 companies headquartered in the state."
    )
    await evaluator.verify(
        claim=claim_ca_count,
        node=ca_count_leaf,
        sources=extracted.sources.ca_f500_context_urls,
        additional_instruction=(
            "Verify the count using a 2024 Fortune 500 listing or state-by-state analysis from Fortune or equivalent credible sources."
        )
    )

    # 2.d) RemovedCompaniesOnSameDate (critical leaf)
    removed_leaf = evaluator.add_leaf(
        id="RemovedCompaniesOnSameDate",
        desc="Provides the names of the three companies removed from the S&P 500 on the same date as the identified company’s addition.",
        parent=provide_info,
        critical=True
    )
    removed_list_for_claim = ", ".join(extracted.removed_companies) if extracted.removed_companies else "UNKNOWN"
    claim_removed = (
        f"The three companies removed from the S&P 500 on the same date were: {removed_list_for_claim}."
    )
    await evaluator.verify(
        claim=claim_removed,
        node=removed_leaf,
        sources=extracted.sources.removed_companies_urls,
        additional_instruction=(
            "Verify that the cited source lists those three removed companies for the same effective date. "
            "Name variants (Inc., Corp., Co.) and ordering differences are acceptable."
        )
    )

    # 2.e) EligibilityConfirmation (critical, parallel)
    eligibility_node = evaluator.add_parallel(
        id="EligibilityConfirmation",
        desc="Confirm the identified company met S&P 500 eligibility requirements regarding positive earnings and U.S. domicile.",
        parent=provide_info,
        critical=True
    )

    pos_earn_leaf = evaluator.add_leaf(
        id="PositiveEarningsRequirementConfirmed",
        desc="Confirms the positive-earnings requirement (most recent quarter and trailing four consecutive quarters) is met.",
        parent=eligibility_node,
        critical=True
    )
    claim_pos_earn = (
        f"At the time of index addition, {extracted.company_full_name or 'UNKNOWN'} satisfied the S&P 500 positive-earnings requirement "
        f"(positive earnings in the most recent quarter and positive sum of the previous four quarters)."
    )
    await evaluator.verify(
        claim=claim_pos_earn,
        node=pos_earn_leaf,
        sources=extracted.sources.positive_earnings_urls,
        additional_instruction=(
            "Use company financial statements, SEC filings, or reputable analyses indicating positive GAAP earnings in the most recent quarter "
            "and positive cumulative earnings over the trailing four quarters as of the addition date."
        )
    )

    us_dom_leaf = evaluator.add_leaf(
        id="USDomicileRequirementConfirmed",
        desc="Confirms the company is U.S.-domiciled.",
        parent=eligibility_node,
        critical=True
    )
    claim_us_dom = (
        f"{extracted.company_full_name or 'UNKNOWN'} is U.S.-domiciled."
    )
    await evaluator.verify(
        claim=claim_us_dom,
        node=us_dom_leaf,
        sources=extracted.sources.us_domicile_urls,
        additional_instruction=(
            "Confirm the company is organized/registered as a U.S. company (U.S. domicile). "
            "SEC filings, company legal info, or authoritative profiles are acceptable."
        )
    )

    # 3) ReferencesProvided (critical leaf) - Check presence of URLs for each required claim
    # We require at least one URL for each of the key factual groups
    refs_ok = all([
        len(id_sources) > 0,  # identification (added in Dec 2023 among three)
        len(extracted.sources.hq_location_urls) > 0,  # HQ including CA & address
        len(extracted.sources.sp500_addition_date_urls) > 0,  # addition date
        len(extracted.sources.ca_f500_context_urls) > 0,  # CA Fortune 500 ranking & count
        len(extracted.sources.removed_companies_urls) > 0,  # removed companies
        len(extracted.sources.positive_earnings_urls) > 0,  # positive earnings
        len(extracted.sources.us_domicile_urls) > 0,  # U.S. domicile
    ])
    evaluator.add_custom_node(
        result=refs_ok,
        id="ReferencesProvided",
        desc="Provides reference URL(s) supporting each required factual claim (company identification, HQ location, addition date, CA Fortune 500 ranking/count, removed companies, and eligibility confirmations).",
        parent=main,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the December 2023 S&P 500 California-headquartered addition task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Root aggregation strategy; main logic in a child critical node
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

    # Extract structured info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_company_info(),
        template_class=ExtractedCompanyInfo,
        extraction_name="extracted_company_info",
    )

    # Build and run verification tree
    await build_verification_tree(evaluator, extracted_info)

    # Return standardized summary
    return evaluator.get_summary()