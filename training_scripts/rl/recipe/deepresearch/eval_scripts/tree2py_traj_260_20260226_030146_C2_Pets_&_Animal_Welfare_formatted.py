import asyncio
import logging
from typing import Optional, List, Dict, Any

from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aza_tx_service_animals_ada_2026"
TASK_DESCRIPTION = (
    "Identify an AZA-accredited zoo located in the state of Texas that has a publicly stated service animal policy "
    "complying with the Americans with Disabilities Act (ADA). Your answer must include: (1) The name of the zoo, "
    "(2) Verification that it is currently AZA-accredited (as of February 2026), (3) Verification that it is located "
    "in Texas, and (4) Evidence of its ADA-compliant service animal policy. Provide reference URLs from official "
    "sources (the zoo's official website and the AZA website) to support each requirement."
)

AS_OF_LABEL = "February 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ZooSelectionExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for a single Texas zoo.
    """
    zoo_name: Optional[str] = None
    # Any URLs that are clearly the zoo's own official website (homepage or other official pages)
    official_site_urls: List[str] = Field(default_factory=list)
    # URLs used to support the Texas location claim (ideally from the zoo's own website)
    texas_location_urls: List[str] = Field(default_factory=list)
    # URLs used to support AZA accreditation (must be from AZA official website)
    aza_accreditation_urls: List[str] = Field(default_factory=list)
    # URLs for the zoo's service animal policy (ideally from the zoo website; ADA.gov also acceptable as a supporting reference)
    policy_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_zoo_selection() -> str:
    return """
    Extract the information for exactly one zoo mentioned in the answer that the user selected to meet the requirements.
    Return the following fields:

    - zoo_name: The exact name of the zoo selected.
    - official_site_urls: An array of URLs to the zoo's own official website (homepage or other pages on the same official domain).
    - texas_location_urls: An array of URLs cited in the answer that support the zoo's physical location in the state of Texas (prefer official zoo pages such as Contact/Visit/About with a postal address).
    - aza_accreditation_urls: An array of URL(s) from the official AZA website (aza.org or related AZA subdomains) that confirm the zoo's current accreditation status.
    - policy_urls: An array of URL(s) to the zoo’s publicly stated service animal policy (prefer the zoo's official website policy page or guest guidelines; if the answer cites official ADA guidance, include that ADA.gov URL as well).

    Rules:
    1) Only include URLs that actually appear in the answer text.
    2) For aza_accreditation_urls, include only official AZA website URLs (domains ending with 'aza.org').
    3) For official_site_urls, include the zoo's own domain URLs (not third-party sites).
    4) For texas_location_urls and policy_urls, prefer the zoo's official website; if the answer provides multiple URLs, include them all.
    5) If a field is not present in the answer, return an empty array for URLs or null for the zoo name.

    Output strictly as JSON matching the schema.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _domain(url: str) -> Optional[str]:
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or None
    except Exception:
        return None


def _is_sub_or_equal(domain: Optional[str], base: Optional[str]) -> bool:
    if not domain or not base:
        return False
    return domain == base or domain.endswith("." + base)


def _has_domain(urls: List[str], base_domain: str) -> bool:
    for u in urls:
        d = _domain(u)
        if _is_sub_or_equal(d, base_domain):
            return True
    return False


def _filter_urls_by_domain(urls: List[str], base_domain: Optional[str]) -> List[str]:
    if not base_domain:
        return []
    return [u for u in urls if _is_sub_or_equal(_domain(u), base_domain)]


def _is_aza_url(url: str) -> bool:
    d = _domain(url)
    return bool(d and d.endswith("aza.org"))


def _is_ada_official_url(url: str) -> bool:
    d = _domain(url)
    return bool(d and d.endswith("ada.gov"))


def _infer_official_domains(official_site_urls: List[str]) -> List[str]:
    domains = []
    for u in official_site_urls:
        d = _domain(u)
        if d and d not in domains:
            domains.append(d)
    return domains


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    root,
    extracted: ZooSelectionExtraction,
    logger: logging.Logger
) -> None:
    # Record some helpful debugging info
    official_domains = _infer_official_domains(extracted.official_site_urls)
    aza_domains_present = list({(_domain(u) or "") for u in extracted.aza_accreditation_urls if _domain(u)})
    policy_domains_present = list({(_domain(u) or "") for u in extracted.policy_urls if _domain(u)})
    loc_domains_present = list({(_domain(u) or "") for u in extracted.texas_location_urls if _domain(u)})

    evaluator.add_custom_info(
        info={
            "zoo_name": extracted.zoo_name,
            "official_domains": official_domains,
            "location_url_domains": loc_domains_present,
            "policy_url_domains": policy_domains_present,
            "aza_url_domains": aza_domains_present,
        },
        info_type="debug_domains",
    )

    # Facility Identification (critical, parallel)
    facility_node = evaluator.add_parallel(
        id="Facility_Identification",
        desc="Identify a single AZA-accredited zoo located in Texas that complies with federal ADA service animal requirements",
        parent=root,
        critical=True
    )

    # 0) Zoo name must be provided (critical)
    name_present = bool(extracted.zoo_name and extracted.zoo_name.strip())
    evaluator.add_custom_node(
        result=name_present,
        id="Zoo_Name_Provided",
        desc="The answer explicitly provides the zoo's name",
        parent=facility_node,
        critical=True
    )

    # 1) Geographic Location (critical group)
    geo_node = evaluator.add_parallel(
        id="Geographic_Location",
        desc="The facility must be physically located in the state of Texas",
        parent=facility_node,
        critical=True
    )

    # 1a) Reference_URL_Geographic (critical leaf via custom check)
    # Require: at least one location URL AND at least one comes from an official zoo domain
    has_loc_urls = len(extracted.texas_location_urls) > 0
    has_official_domain = False
    for od in official_domains:
        if _has_domain(extracted.texas_location_urls, od):
            has_official_domain = True
            break

    evaluator.add_custom_node(
        result=(has_loc_urls and has_official_domain),
        id="Reference_URL_Geographic",
        desc="Provide a reference URL confirming the zoo's Texas location (must include at least one official zoo website URL)",
        parent=geo_node,
        critical=True
    )

    # 1b) Texas_Location_Verification (critical, verify against URLs)
    # Prefer to use only official location URLs for verification
    official_loc_urls: List[str] = []
    for od in official_domains:
        official_loc_urls.extend(_filter_urls_by_domain(extracted.texas_location_urls, od))
    if not official_loc_urls:
        # Fallback: use all provided location URLs (verification will still run; presence is gated by the reference URL node)
        official_loc_urls = extracted.texas_location_urls

    texas_loc_leaf = evaluator.add_leaf(
        id="Texas_Location_Verification",
        desc="Verify the zoo's physical address shows it is located in Texas",
        parent=geo_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The official page(s) confirm that the zoo named '{extracted.zoo_name or 'the selected zoo'}' is physically located in the state of Texas (TX), United States.",
        node=texas_loc_leaf,
        sources=official_loc_urls,
        additional_instruction="Confirm the page belongs to the zoo and clearly shows a Texas address or mentions Texas explicitly (e.g., a postal address with 'TX' or 'Texas')."
    )

    # 2) AZA Accreditation Status (critical group)
    aza_node = evaluator.add_parallel(
        id="AZA_Accreditation_Status",
        desc=f"The facility must hold current AZA accreditation as of {AS_OF_LABEL}",
        parent=facility_node,
        critical=True
    )

    # 2a) Reference_URL_Accreditation (critical custom)
    has_aza_urls = len(extracted.aza_accreditation_urls) > 0
    has_official_aza = any(_is_aza_url(u) for u in extracted.aza_accreditation_urls)
    evaluator.add_custom_node(
        result=(has_aza_urls and has_official_aza),
        id="Reference_URL_Accreditation",
        desc="Provide the AZA official website URL confirming the zoo's accreditation status",
        parent=aza_node,
        critical=True
    )

    # 2b) Current_Accreditation (critical, verify with AZA URL(s))
    accreditation_leaf = evaluator.add_leaf(
        id="Current_Accreditation",
        desc=f"Verify the zoo appears on the official AZA list of accredited institutions and is current as of {AS_OF_LABEL}",
        parent=aza_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The AZA official website confirms that '{extracted.zoo_name or 'the selected zoo'}' is currently AZA-accredited as of {AS_OF_LABEL} (i.e., accreditation has not expired).",
        node=accreditation_leaf,
        sources=extracted.aza_accreditation_urls,
        additional_instruction=(
            "Use only the AZA official website. If an accreditation expiration date is present, ensure it is not before the as-of month/year. "
            "If the AZA directory/entry lists the zoo as 'Accredited' or appears in the official 'Find a Zoo' accredited directory, consider it valid."
        )
    )

    # 3) ADA Service Animal Policy (critical group)
    ada_node = evaluator.add_parallel(
        id="ADA_Service_Animal_Policy",
        desc="The facility must have a publicly stated service animal policy that complies with ADA requirements",
        parent=facility_node,
        critical=True
    )

    # 3a) Reference_URL_Policy (critical custom)
    # Accept either an official zoo policy URL or an ADA.gov supplement, but the primary requirement is that the zoo has a public policy.
    # Therefore: require at least one policy URL on an official zoo domain; ADA.gov is allowed as supplemental, but not sufficient alone.
    has_policy_urls = len(extracted.policy_urls) > 0
    official_policy_found = False
    for od in official_domains:
        if _has_domain(extracted.policy_urls, od):
            official_policy_found = True
            break

    # If no official domain policy page, allow ADA.gov only if also the answer clearly ties it to the zoo policy page,
    # but our extraction cannot infer that; thus we keep it strict: require official zoo policy URL.
    evaluator.add_custom_node(
        result=(has_policy_urls and official_policy_found),
        id="Reference_URL_Policy",
        desc="Provide a reference URL from the zoo's official website confirming their service animal policy (ADA compliant)",
        parent=ada_node,
        critical=True
    )

    # 3b) Policy_Compliance (critical, verify)
    # Use only official zoo policy URLs for verification (to confirm the zoo's own policy, not generic ADA guidance)
    official_policy_urls: List[str] = []
    for od in official_domains:
        official_policy_urls.extend(_filter_urls_by_domain(extracted.policy_urls, od))
    if not official_policy_urls:
        # Still pass whatever the answer provided; but presence is gated by the reference URL node above
        official_policy_urls = extracted.policy_urls

    policy_leaf = evaluator.add_leaf(
        id="Policy_Compliance",
        desc="Verify the zoo acknowledges service animals (dogs) are permitted in accordance with the ADA",
        parent=ada_node,
        critical=True
    )

    await evaluator.verify(
        claim=f"The official website of '{extracted.zoo_name or 'the selected zoo'}' states that service animals (dogs) are permitted in accordance with the Americans with Disabilities Act (ADA).",
        node=policy_leaf,
        sources=official_policy_urls,
        additional_instruction=(
            "Look for explicit language such as 'service animals as defined by the ADA', 'service dogs are permitted', or similar. "
            "It is acceptable if the policy excludes emotional support animals. "
            "The page should be an official zoo policy/guest guidelines page."
        )
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
) -> Dict[str, Any]:
    """
    Entry point for evaluating an answer for the Texas AZA-accredited zoo with ADA service animal policy task.
    """
    # Initialize evaluator with a parallel root
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_zoo_selection(),
        template_class=ZooSelectionExtraction,
        extraction_name="zoo_selection_extraction"
    )

    # Add ground truth/context info (as-of date requirement)
    evaluator.add_ground_truth({
        "as_of": AS_OF_LABEL,
        "requirements": [
            "Zoo name provided",
            "Texas location verified with official zoo URL",
            "AZA accreditation verified on AZA official website",
            "Service animal policy (ADA-compliant) published on zoo's official website"
        ]
    })

    # Build verification tree and run checks
    await build_and_verify(evaluator, root, extracted, logger)

    # Return evaluation summary
    return evaluator.get_summary()