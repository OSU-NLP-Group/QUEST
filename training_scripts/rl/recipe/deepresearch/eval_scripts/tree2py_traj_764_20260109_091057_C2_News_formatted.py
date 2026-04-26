import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "news_org_awards_2024"
TASK_DESCRIPTION = (
    "Identify one news organization that won both a 2024 Pulitzer Prize in a journalism category and a 2024 George Polk Award. "
    "Provide the following information: (1) The full physical headquarters address of the organization (including street address, city, state/region, and postal code), "
    "(2) A direct URL link to the official Pulitzer Prize website (pulitzer.org) page that confirms the organization's 2024 win, and "
    "(3) A direct URL link to the official George Polk Awards website (liu.edu/polk-awards) page that confirms the organization's 2024 win."
)


# ----------------------------- Data Models --------------------------------- #
class AddressInfo(BaseModel):
    full_address: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state_region: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class OrganizationExtraction(BaseModel):
    organization_name: Optional[str] = None
    additional_organizations: List[str] = Field(default_factory=list)
    pulitzer_url: Optional[str] = None
    polk_url: Optional[str] = None
    address: Optional[AddressInfo] = None


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_organization_info() -> str:
    return """
    Extract the single news organization the answer identifies as meeting the criteria (won both a 2024 Pulitzer Prize in a journalism category and a 2024 George Polk Award), along with the required documentation.

    Return a JSON with:
    - organization_name: The primary organization the answer clearly identifies as the one meeting the criteria.
    - additional_organizations: A list of other organizations explicitly named in the answer (exclude the primary organization). Include any other outlets/winners mentioned, even if referenced for context.
    - pulitzer_url: A direct URL to pulitzer.org that the answer provides to confirm the organization's 2024 Pulitzer win. If none or not from pulitzer.org, return null.
    - polk_url: A direct URL to liu.edu/polk-awards that the answer provides to confirm the organization's 2024 George Polk Award win. If none or not from liu.edu/polk-awards, return null.
    - address:
        - full_address: The complete headquarters address string as presented in the answer.
        - street_address: Street address component if provided.
        - city: City component if provided.
        - state_region: State/region component if provided.
        - postal_code: Postal/ZIP code component if provided.
        - country: Country if provided.
        - source_urls: Any URL(s) the answer provides to verify the headquarters address (prefer official organization site). If none provided, return an empty list.

    Rules:
    - Extract strictly what appears in the answer text. Do not invent or alter URLs or addresses.
    - For pulitzer_url, only accept URLs whose domain is pulitzer.org; for polk_url, only accept URLs on liu.edu that include 'polk-awards' in the path.
    - If the address components are not separated but a single full string exists, set full_address and leave missing components as null.
    - If the answer mentions multiple organizations, set organization_name to the one the answer presents as the qualifying organization and list the rest in additional_organizations.
    """


# ------------------------------ Helpers ------------------------------------ #
def _is_valid_pulitzer_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        return host.endswith("pulitzer.org")
    except Exception:
        return False


def _is_valid_polk_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        return host.endswith("liu.edu") and ("polk-awards" in path)
    except Exception:
        return False


def _has_full_address(addr: Optional[AddressInfo]) -> bool:
    if not addr:
        return False
    required = [
        addr.full_address,
        addr.street_address,
        addr.city,
        addr.state_region,
        addr.postal_code,
    ]
    return all(isinstance(x, str) and x.strip() for x in required)


# --------------------------- Verification Logic ---------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: OrganizationExtraction) -> None:
    # Root: sequential to enforce ordering (identify -> qualification -> documentation)
    root = evaluator.root

    # 1) IdentifyOrganization (Critical)
    ident_result = (
        isinstance(extracted.organization_name, str)
        and extracted.organization_name.strip() != ""
        and (len(extracted.additional_organizations) == 0)
    )
    evaluator.add_custom_node(
        result=ident_result,
        id="IdentifyOrganization",
        desc="Exactly one news organization is clearly identified (named) in the response",
        parent=root,
        critical=True,
    )

    # 2) OrganizationQualification (Critical, parallel)
    qual_node = evaluator.add_parallel(
        id="OrganizationQualification",
        desc="The identified news organization satisfies the 2024 award constraints",
        parent=root,
        critical=True,
    )

    org_name = extracted.organization_name or ""

    # 2.1 PulitzerPrize2024 (Critical)
    pulitzer_leaf = evaluator.add_leaf(
        id="PulitzerPrize2024",
        desc="The organization won at least one 2024 Pulitzer Prize in a journalism category",
        parent=qual_node,
        critical=True,
    )
    pulitzer_claim = (
        f"This page confirms that the organization '{org_name}' won a 2024 Pulitzer Prize in a journalism category."
    )
    pulitzer_instruction = (
        "Verify on the provided page that it explicitly confirms a 2024 Pulitzer win for the named organization in any journalism category. "
        "Accept newsroom/team/staff credits under the organization's name as an organization win. "
        "If the URL is missing or not on pulitzer.org, or the page relates to non-journalism categories (e.g., Letters, Drama, Music), return 'not supported'."
    )
    await evaluator.verify(
        claim=pulitzer_claim,
        node=pulitzer_leaf,
        sources=extracted.pulitzer_url if _is_valid_pulitzer_url(extracted.pulitzer_url) else None,
        additional_instruction=pulitzer_instruction,
    )

    # 2.2 GeorgePolkAward2024 (Critical)
    polk_leaf = evaluator.add_leaf(
        id="GeorgePolkAward2024",
        desc="The organization won at least one 2024 George Polk Award",
        parent=qual_node,
        critical=True,
    )
    polk_claim = (
        f"This page confirms that the organization '{org_name}' won a George Polk Award in 2024."
    )
    polk_instruction = (
        "Verify on the provided page that it explicitly confirms a 2024 George Polk Award for the named organization. "
        "If the URL is missing or not on liu.edu/polk-awards, return 'not supported'."
    )
    await evaluator.verify(
        claim=polk_claim,
        node=polk_leaf,
        sources=extracted.polk_url if _is_valid_polk_url(extracted.polk_url) else None,
        additional_instruction=polk_instruction,
    )

    # 3) RequiredDocumentation (Critical, parallel)
    docs_node = evaluator.add_parallel(
        id="RequiredDocumentation",
        desc="All required fields are provided and are verifiable from official sources",
        parent=root,
        critical=True,
    )

    # 3.1 HeadquartersAddressProvided (Critical)
    addr_provided = _has_full_address(extracted.address)
    evaluator.add_custom_node(
        result=addr_provided,
        id="HeadquartersAddressProvided",
        desc="The full physical headquarters address is provided, including street address, city, state/region, and postal code where applicable",
        parent=docs_node,
        critical=True,
    )

    # 3.2 HeadquartersAddressOfficiallyVerifiable (Critical)
    addr_verify_leaf = evaluator.add_leaf(
        id="HeadquartersAddressOfficiallyVerifiable",
        desc="The provided headquarters address is accurate and verifiable from an official source for the organization",
        parent=docs_node,
        critical=True,
    )
    addr = extracted.address or AddressInfo()
    addr_claim = (
        f"The official headquarters address of '{org_name}' is '{addr.full_address}' "
        f"(street: '{addr.street_address}', city: '{addr.city}', state/region: '{addr.state_region}', postal code: '{addr.postal_code}')."
    )
    addr_instruction = (
        "Use ONLY the provided source URLs to check the address. Prefer the official organization website (e.g., About/Contact page). "
        "If no source URLs are provided, or the provided URLs are not official or do not explicitly confirm the address, return 'not supported'."
    )
    await evaluator.verify(
        claim=addr_claim,
        node=addr_verify_leaf,
        sources=addr.source_urls if addr.source_urls else None,
        additional_instruction=addr_instruction,
    )

    # 3.3 PulitzerURLVerification (Critical)
    pulitzer_url_ok = _is_valid_pulitzer_url(extracted.pulitzer_url)
    evaluator.add_custom_node(
        result=(isinstance(extracted.pulitzer_url, str) and pulitzer_url_ok),
        id="PulitzerURLVerification",
        desc="A direct URL link to the official Pulitzer Prize website (pulitzer.org) page confirming the organization's 2024 win is provided",
        parent=docs_node,
        critical=True,
    )

    # 3.4 PolkURLVerification (Critical)
    polk_url_ok = _is_valid_polk_url(extracted.polk_url)
    evaluator.add_custom_node(
        result=(isinstance(extracted.polk_url, str) and polk_url_ok),
        id="PolkURLVerification",
        desc="A direct URL link to the official George Polk Awards website (liu.edu/polk-awards) page confirming the organization's 2024 win is provided",
        parent=docs_node,
        critical=True,
    )


# ------------------------ Main Evaluation Function ------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    evaluator = Evaluator()
    evaluator.initialize(
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
        default_model=model,
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_organization_info(),
        template_class=OrganizationExtraction,
        extraction_name="organization_info",
    )

    # Optionally record a summary of extracted fields for transparency
    evaluator.add_custom_info(
        info={
            "organization_name": extracted.organization_name,
            "additional_organizations": extracted.additional_organizations,
            "pulitzer_url": extracted.pulitzer_url,
            "polk_url": extracted.polk_url,
            "address": extracted.address.dict() if extracted.address else None,
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
    )

    await build_verification_tree(evaluator, extracted)

    return evaluator.get_summary()