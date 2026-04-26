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
TASK_ID = "eu_extradition_travel_requirements"
TASK_DESCRIPTION = """
Identify a European country that meets ALL of the following criteria:

1. Has a bilateral extradition treaty with the United States that was signed after January 1, 2000
2. The extradition treaty entered into force before December 31, 2015
3. Is a party to the Schengen Agreement
4. Allows US citizens to enter for tourism without a visa for stays under 90 days
5. Requires passports to be valid for at least 3 months beyond the planned departure date from the Schengen area
6. Has a yellow fever vaccination requirement only for travelers arriving from countries with yellow fever transmission risk (not a blanket requirement for all travelers)
7. The extradition treaty must be bilateral (between the specific country and the United States), not just a multilateral agreement

Please provide:
- The name of the country
- The signing date of the extradition treaty
- The date the treaty entered into force
- Official government sources (URLs) that verify the treaty details
- Official government sources (URLs) that verify the travel and entry requirements for US citizens
"""


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class SolutionExtraction(BaseModel):
    """
    Extract the essential fields from the agent's answer.
    All URLs must be explicitly present in the answer text.
    """
    country_name: Optional[str] = None
    treaty_signing_date: Optional[str] = None
    treaty_entry_into_force_date: Optional[str] = None
    treaty_sources: List[str] = Field(default_factory=list)
    travel_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_solution() -> str:
    return """
    Extract the following fields from the answer:

    1) country_name: The specific country proposed by the answer.
    2) treaty_signing_date: The signing date of the bilateral extradition treaty with the United States, as stated in the answer. Keep it as a string exactly as written (e.g., "June 25, 2003" or "2003-06-25").
    3) treaty_entry_into_force_date: The date the treaty entered into force, as stated in the answer. Keep it as a string exactly as written.
    4) treaty_sources: A list of all official government URLs in the answer that support the treaty details (e.g., state.gov, travel.state.gov, justice.gov, or the partner country's official government websites).
    5) travel_sources: A list of all official government URLs in the answer that support the travel and entry requirements for U.S. citizens (e.g., travel.state.gov or the partner country's official government websites).

    IMPORTANT:
    - Only include URLs explicitly present in the answer. Do not invent or infer URLs.
    - If a URL is missing the protocol, prepend "http://".
    - If any field is missing in the answer, set it to null (or an empty list for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order."""
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def _all_urls_accessible(evaluator: Evaluator, urls: List[str]) -> bool:
    """
    Check if all provided URLs are publicly accessible by attempting to retrieve each.
    Returns False if the list is empty or any URL fails to load content.
    """
    urls = _dedup_urls(urls)
    if not urls:
        return False

    async def _fetch(u: str) -> bool:
        try:
            screenshot_b64, web_text = await evaluator.verifier.get_page_info(u)
            if screenshot_b64 is None or web_text is None:
                return False
            if isinstance(screenshot_b64, list):
                has_image = any(bool(x) for x in screenshot_b64)
            else:
                has_image = bool(screenshot_b64)
            has_text = bool(web_text.strip()) if isinstance(web_text, str) else False
            return has_image or has_text
        except Exception:
            return False

    results = await asyncio.gather(*[asyncio.create_task(_fetch(u)) for u in urls], return_exceptions=True)
    return all((r is True) for r in results)


# --------------------------------------------------------------------------- #
# Verification procedure                                                      #
# --------------------------------------------------------------------------- #
async def _build_and_verify_tree(evaluator: Evaluator, extraction: SolutionExtraction) -> None:
    """
    Construct the verification tree according to the rubric and execute all verifications.
    """
    # Create the top-level critical aggregation node
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify a European country with a bilateral US extradition treaty signed after 2000, in force before 2016, meeting specific travel requirements",
        parent=evaluator.root,
        critical=True,
    )

    # Prepare data
    country = extraction.country_name or ""
    treaty_sources = _dedup_urls(extraction.treaty_sources)
    travel_sources = _dedup_urls(extraction.travel_sources)
    all_sources = _dedup_urls(treaty_sources + travel_sources)

    # --------------------------- Source_Documentation --------------------------- #
    sources_node = evaluator.add_parallel(
        id="Source_Documentation",
        desc="Provide verifiable official sources for all claims",
        parent=task_node,
        critical=True,
    )

    # Treaty official source (must be official gov; fail if none provided)
    treaty_official_leaf = evaluator.add_leaf(
        id="Treaty_Official_Source",
        desc="Official government source (US State Department or partner country) documenting the treaty details including signing and entry into force dates is provided",
        parent=sources_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided URLs is an official government source (US State Department or the government of {country}) that documents the US–{country} extradition treaty and includes the signing and entry-into-force dates.",
        node=treaty_official_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "If no URLs are provided, judge as Incorrect. Consider a URL 'official' if it is clearly a government or embassy/consulate website "
            "(e.g., state.gov, travel.state.gov, justice.gov, usembassy.gov subdomains, or the partner country's official government domains). "
            "Also verify that the page mentions the extradition treaty with the United States and explicitly lists the signing and entry-into-force dates."
        ),
    )

    # Travel official source (must be official gov; fail if none provided)
    travel_official_leaf = evaluator.add_leaf(
        id="Travel_Requirements_Official_Source",
        desc="Official government source (US State Department or partner country) documenting travel and entry requirements is provided",
        parent=sources_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided URLs is an official government source (US State Department or the government of {country}) that documents travel/entry requirements for U.S. citizens.",
        node=travel_official_leaf,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "If no URLs are provided, judge as Incorrect. Treat travel.state.gov and official partner-country government websites (including embassy/consulate sites) as official. "
            "The page should cover U.S. citizen entry policies for short stays, passport validity, and health (e.g., yellow fever) if applicable."
        ),
    )

    # Sources accessibility check (all provided sources should be accessible)
    sources_accessible_result = await _all_urls_accessible(evaluator, all_sources)
    evaluator.add_custom_node(
        result=sources_accessible_result,
        id="Sources_Are_Accessible",
        desc="All provided sources are publicly accessible and verifiable",
        parent=sources_node,
        critical=True,
    )

    # --------------------------- Country_Identification ------------------------ #
    country_node = evaluator.add_parallel(
        id="Country_Identification",
        desc="Identify the country and verify basic eligibility",
        parent=task_node,
        critical=True,
    )

    # Country name provided
    evaluator.add_custom_node(
        result=bool(extraction.country_name and extraction.country_name.strip()),
        id="Country_Name_Provided",
        desc="A specific country name is provided",
        parent=country_node,
        critical=True,
    )

    # Geographic location Europe/Mediterranean
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Location_Europe",
        desc="The country is located in Europe or the Mediterranean region",
        parent=country_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{country} is located in Europe or the Mediterranean region.",
        node=geo_leaf,
        additional_instruction="Use widely accepted geographic classifications. If the country is transcontinental, it still qualifies if commonly regarded as European or Mediterranean.",
    )

    # Country existence and recognition
    exists_leaf = evaluator.add_leaf(
        id="Country_Exists_And_Recognized",
        desc="The country is a recognized sovereign nation",
        parent=country_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{country} is a recognized sovereign nation (widely recognized internationally, e.g., UN member state).",
        node=exists_leaf,
        additional_instruction="Use general knowledge. Do not require a URL; this is a basic recognition check.",
    )

    # ---------------------- Extradition_Treaty_Verification -------------------- #
    treaty_node = evaluator.add_parallel(
        id="Extradition_Treaty_Verification",
        desc="Verify all extradition treaty requirements are met",
        parent=task_node,
        critical=True,
    )

    # Provided dates existence checks
    evaluator.add_custom_node(
        result=bool(extraction.treaty_signing_date and extraction.treaty_signing_date.strip()),
        id="Treaty_Signing_Date_Provided",
        desc="The specific signing date of the treaty is provided in the solution",
        parent=treaty_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(extraction.treaty_entry_into_force_date and extraction.treaty_entry_into_force_date.strip()),
        id="Treaty_Entry_Into_Force_Date_Provided",
        desc="The specific entry into force date is provided in the solution",
        parent=treaty_node,
        critical=True,
    )

    # Bilateral treaty with US (not multilateral or EU-only)
    bilateral_leaf = evaluator.add_leaf(
        id="Bilateral_Treaty_With_US",
        desc="The country has a bilateral extradition treaty specifically with the United States",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"There is a bilateral extradition treaty specifically between the United States and {country}. Multilateral or EU-wide agreements alone do not satisfy this.",
        node=bilateral_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Accept only a direct bilateral treaty between the United States and the named country. "
            "Do not accept references solely to multilateral conventions (e.g., Council of Europe) or EU frameworks. "
            "Prefer official treaty pages (e.g., state.gov Treaties in Force or partner's official treaty portal). "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # Signing date criterion (after 2000-01-01)
    signing_criterion_leaf = evaluator.add_leaf(
        id="Treaty_Signing_Date_Criteria",
        desc="The treaty was signed after January 1, 2000",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extradition treaty between the United States and {country} was signed after January 1, 2000.",
        node=signing_criterion_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Look for the signing date on the official page and confirm it is strictly after January 1, 2000 (2000-01-01). "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # Entry-into-force date criterion (before 2015-12-31)
    eif_criterion_leaf = evaluator.add_leaf(
        id="Treaty_Entry_Into_Force_Date_Criteria",
        desc="The treaty entered into force before December 31, 2015",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extradition treaty between the United States and {country} entered into force before December 31, 2015.",
        node=eif_criterion_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Find the entry-into-force date on the official page and confirm it is strictly before December 31, 2015 (2015-12-31). "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # US ratification
    us_rat_leaf = evaluator.add_leaf(
        id="US_Ratification",
        desc="The United States ratified the treaty",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The United States ratified (e.g., Senate advice and consent given and/or instrument of ratification deposited) the extradition treaty with {country}.",
        node=us_rat_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Accept synonymous phrasing such as 'Senate advice and consent', 'ratified', 'instrument of ratification deposited', or equivalent official statements. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # Partner country ratification
    partner_rat_leaf = evaluator.add_leaf(
        id="Partner_Country_Ratification",
        desc="The partner country ratified the treaty",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{country} ratified the extradition treaty with the United States (e.g., parliamentary approval and/or instrument of ratification deposited).",
        node=partner_rat_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Accept synonymous phrasing such as 'ratified', 'approved by parliament', or equivalent official statements indicating ratification/deposit. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # Treaty currently in force
    in_force_leaf = evaluator.add_leaf(
        id="Treaty_Currently_In_Force",
        desc="The treaty is currently in force and has not been terminated",
        parent=treaty_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The extradition treaty between the United States and {country} is currently in force and has not been terminated.",
        node=in_force_leaf,
        sources=treaty_sources if treaty_sources else None,
        additional_instruction=(
            "Look for explicit 'in force' status or equivalent on official sources (e.g., State Department 'Treaties in Force'). "
            "If the source indicates the treaty is in force (and not terminated), mark Correct. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[treaty_official_leaf],
    )

    # -------------------- Travel_And_Entry_Requirements ------------------------ #
    travel_node = evaluator.add_parallel(
        id="Travel_And_Entry_Requirements",
        desc="Verify all travel and entry requirements for US citizens",
        parent=task_node,
        critical=True,
    )

    # Schengen party
    schengen_leaf = evaluator.add_leaf(
        id="Schengen_Agreement_Party",
        desc="The country is a party to the Schengen Agreement",
        parent=travel_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{country} is a Schengen Area member (party to the Schengen Agreement).",
        node=schengen_leaf,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "Accept phrasing like 'Schengen Area member', 'Schengen state', or equivalent. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[travel_official_leaf],
    )

    # Visa exemption under 90 days
    visa_leaf = evaluator.add_leaf(
        id="Visa_Exemption_Under_90_Days",
        desc="US citizens can enter for tourism without a visa for stays under 90 days",
        parent=travel_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"U.S. citizens can enter {country} for tourism without a visa for short stays up to 90 days (within any 180-day period).",
        node=visa_leaf,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "Look for statements like 'visa-free for 90 days' or 'short-stay up to 90 days in any 180-day period' for U.S. citizens. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[travel_official_leaf],
    )

    # Passport validity 3 months beyond departure
    passport_leaf = evaluator.add_leaf(
        id="Passport_Validity_3_Months",
        desc="The country requires passports to be valid for at least 3 months beyond planned departure date from the Schengen area",
        parent=travel_node,
        critical=True,
    )
    await evaluator.verify(
        claim="U.S. passports must be valid for at least three months beyond the planned date of departure from the Schengen Area.",
        node=passport_leaf,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "Accept equivalent phrasing indicating 'three months beyond departure' requirement in the Schengen context. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[travel_official_leaf],
    )

    # Yellow fever conditional requirement
    yellow_leaf = evaluator.add_leaf(
        id="Yellow_Fever_Conditional_Requirement",
        desc="The country requires yellow fever vaccination only for travelers from countries with transmission risk, not as a blanket requirement for all travelers",
        parent=travel_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Yellow fever vaccination is only required if arriving from (or transiting through) countries with risk of yellow fever transmission, not required for all travelers.",
        node=yellow_leaf,
        sources=travel_sources if travel_sources else None,
        additional_instruction=(
            "Confirm the policy is conditional (e.g., only if arriving from or transiting through a risk country) and not a blanket requirement for all travelers. "
            "If no URLs are provided, judge as Incorrect."
        ),
        extra_prerequisites=[travel_official_leaf],
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
    Evaluate an answer for the European country extradition/travel criteria task.
    """
    # Initialize evaluator with a parallel root (we attach a critical Task_Completion node under it)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_solution(),
        template_class=SolutionExtraction,
        extraction_name="solution_extraction",
    )

    # Record constraints as "ground truth" context for transparency (not used for scoring)
    evaluator.add_ground_truth({
        "criteria": {
            "treaty_signed_after": "2000-01-01",
            "treaty_in_force_before": "2015-12-31",
            "schengen_required": True,
            "visa_free_us_90_days": True,
            "passport_validity_3m_beyond_departure": True,
            "yellow_fever_conditional_only": True,
            "treaty_bilateral": True
        }
    })

    # Build tree and run verifications
    await _build_and_verify_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()