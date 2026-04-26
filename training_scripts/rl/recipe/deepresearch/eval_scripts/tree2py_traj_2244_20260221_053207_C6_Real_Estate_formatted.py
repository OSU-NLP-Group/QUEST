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
TASK_ID = "tokenized_real_estate_2026"
TASK_DESCRIPTION = """
Identify four distinct tokenized real estate properties currently available for investment in 2026 that meet ALL of the following criteria:

1. Blockchain Infrastructure: The property tokens must be issued on one of the following blockchain networks: Ethereum, Algorand, XRP Ledger, or Hedera.

2. Investment Accessibility: The minimum investment amount must be $100 or less per token or fractional share.

3. Regulatory Compliance: The platform must explicitly operate under an SEC regulatory framework, either through SEC registration or a specific SEC exemption (such as Regulation Crowdfunding, Regulation D, or Regulation A+).

4. Investor Eligibility: The platform must accept non-accredited U.S. investors.

5. Property Location: At least three of the four properties must be located in different U.S. states. Properties may include both residential and commercial real estate.

6. Income Distribution: The platform must use an automated smart contract-based system for distributing rental income or dividends to token holders, with a specified distribution frequency (daily, weekly, or monthly).

7. Secondary Market Access: The platform must provide a secondary market mechanism for trading property tokens.

For each of the four properties, provide:
- The complete property address (street address, city, and state)
- The tokenization platform name
- The blockchain network used
- The minimum investment amount (price per token)
- The regulatory framework under which the platform operates
- The income distribution frequency and mechanism (e.g., daily via smart contracts)
- The secondary market trading mechanism
- A direct URL reference to the property listing page
"""

ALLOWED_NETWORKS = ["Ethereum", "Algorand", "XRP Ledger", "XRPL", "Hedera", "Hedera Hashgraph"]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PropertyItem(BaseModel):
    platform_name: Optional[str] = None
    property_listing_url: Optional[str] = None

    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Two-letter or full state name if present

    blockchain_network: Optional[str] = None
    network_evidence_urls: List[str] = Field(default_factory=list)

    min_investment: Optional[str] = None
    min_investment_evidence_urls: List[str] = Field(default_factory=list)

    regulatory_framework: Optional[str] = None
    regulatory_evidence_urls: List[str] = Field(default_factory=list)

    accepts_non_accredited_statement: Optional[str] = None
    eligibility_evidence_urls: List[str] = Field(default_factory=list)

    income_distribution_mechanism: Optional[str] = None
    automation_evidence_urls: List[str] = Field(default_factory=list)

    income_distribution_frequency: Optional[str] = None  # daily / weekly / monthly (or equivalent phrasing)
    frequency_evidence_urls: List[str] = Field(default_factory=list)

    secondary_market_mechanism: Optional[str] = None
    secondary_market_evidence_urls: List[str] = Field(default_factory=list)

    address_evidence_urls: List[str] = Field(default_factory=list)


class RealEstateExtraction(BaseModel):
    properties: List[PropertyItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_properties() -> str:
    return """
    Extract up to four distinct tokenized real estate properties mentioned in the answer. For each property, return an object with the following fields, using the exact wording from the answer text wherever possible. If any field is missing, set it to null (or an empty array for URL lists).

    Required fields per property:
    1) platform_name: Name of the tokenization platform.
    2) property_listing_url: Direct URL to the property's listing page on the platform.
    3) address: The complete property address (street address, city, and state), as written in the answer.
    4) city: City component of the address (if explicitly present).
    5) state: State component (two-letter code like 'CA' or full name like 'California', as explicitly present).
    6) blockchain_network: Blockchain network used (e.g., Ethereum, Algorand, XRP Ledger/XRPL, Hedera).
    7) network_evidence_urls: URLs cited that show or confirm the blockchain network for this property.
    8) min_investment: Minimum investment amount or price per token (as text, e.g., '$50', 'USD 75', '0.05 ETH').
    9) min_investment_evidence_urls: URLs cited that show the minimum investment for this property.
    10) regulatory_framework: The SEC regulatory framework under which the platform operates (e.g., SEC Registration, Regulation Crowdfunding (Reg CF), Regulation D, Regulation A+).
    11) regulatory_evidence_urls: URLs cited that disclose the regulatory framework or exemption.
    12) accepts_non_accredited_statement: A statement indicating acceptance of non-accredited U.S. investors (as text).
    13) eligibility_evidence_urls: URLs cited that show investor eligibility, specifically acceptance of non-accredited U.S. investors.
    14) income_distribution_mechanism: Mechanism for distributing rental income/dividends (e.g., 'automated via smart contracts').
    15) automation_evidence_urls: URLs cited that show smart contract automation for distribution.
    16) income_distribution_frequency: Distribution frequency (daily, weekly, or monthly), as explicitly stated.
    17) frequency_evidence_urls: URLs cited that show the distribution frequency.
    18) secondary_market_mechanism: The secondary market trading mechanism (e.g., 'ATS marketplace', 'exchange', 'platform secondary market').
    19) secondary_market_evidence_urls: URLs cited that show the existence of the secondary market for trading tokens.
    20) address_evidence_urls: URLs cited that show the property address.

    Rules:
    - Only extract URLs explicitly present in the answer text (including markdown links).
    - Do not invent or infer values; return null or [] if not present in the answer.
    - Keep text fields as provided (free-form strings are acceptable).
    - If multiple properties are mentioned, extract them in the order they appear; limit to at most four.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def compile_sources(main_url: Optional[str], extra_urls: List[str]) -> List[str]:
    """Combine main listing URL and extra evidence URLs, deduplicated, preserving order."""
    sources: List[str] = []
    if main_url and main_url.strip():
        sources.append(main_url.strip())
    for u in extra_urls:
        if u and u.strip():
            if u.strip() not in sources:
                sources.append(u.strip())
    return sources


US_STATE_NAME_TO_ABBR: Dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
    "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC"
}


def normalize_state(state_str: Optional[str]) -> Optional[str]:
    """Normalize full state name to 2-letter abbreviation; uppercase if already abbreviation."""
    if not state_str:
        return None
    s = state_str.strip()
    if len(s) == 2:
        return s.upper()
    # Title-case name
    name = s.strip()
    # Try direct mapping ignoring case
    for k, v in US_STATE_NAME_TO_ABBR.items():
        if k.lower() == name.lower():
            return v
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_property(
    evaluator: Evaluator,
    property_parent_node,
    prop: PropertyItem,
    ordinal_label: str,
    idx: int,
) -> None:
    """
    Build verification sub-tree for a single property.
    ordinal_label: "First", "Second", "Third", "Fourth"
    """
    # ---- Platform Identification (critical) ----
    platform_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Platform_Identification",
        desc="Identify the tokenization platform and provide reference URL",
        parent=property_parent_node,
        critical=True
    )

    # Platform Name leaf
    platform_name_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Platform_Name",
        desc="Provide the name of the tokenization platform",
        parent=platform_node,
        critical=True
    )
    platform_name = prop.platform_name or ""
    platform_sources = compile_sources(prop.property_listing_url, [])
    await evaluator.verify(
        claim=f"The property listing page shows the tokenization platform as '{platform_name}'.",
        node=platform_name_leaf,
        sources=platform_sources if platform_sources else None,
        additional_instruction="Confirm the platform brand/name indicated on the listing or site header/footer."
    )

    # Platform Reference URL leaf
    platform_url_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Platform_Reference_URL",
        desc="Provide a valid URL to the platform or property listing page",
        parent=platform_node,
        critical=True
    )
    url_for_verify = prop.property_listing_url or ""
    await evaluator.verify(
        claim="This URL is a property listing page on the tokenization platform for a tokenized real estate property.",
        node=platform_url_leaf,
        sources=url_for_verify if url_for_verify else None,
        additional_instruction="Check if the page content is a property listing or platform page with property details (address, investment terms)."
    )

    # ---- Property Address (critical) ----
    address_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Property_Address",
        desc="Provide complete property address and evidence",
        parent=property_parent_node,
        critical=True
    )

    # Complete Address leaf
    address_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Complete_Address",
        desc="Provide the complete property address including street address, city, and state",
        parent=address_node,
        critical=True
    )
    address_claim = f"The property address is '{prop.address or ''}'" + (
        f", located in {prop.city or ''}, {prop.state or ''}." if prop.city or prop.state else "."
    )
    address_sources = compile_sources(prop.property_listing_url, prop.address_evidence_urls)
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=address_sources if address_sources else None,
        additional_instruction="Verify the full address text on the listing page; allow minor formatting variations (punctuation, abbreviations)."
    )

    # Address Evidence URL leaf
    address_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Address_Evidence_URL",
        desc="Provide URL evidence showing the property address",
        parent=address_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided evidence page displays the same property address: '{prop.address or ''}'.",
        node=address_evidence_leaf,
        sources=prop.address_evidence_urls if prop.address_evidence_urls else address_sources if address_sources else None,
        additional_instruction="Check for the address string anywhere on the evidence page (including screenshot text)."
    )

    # ---- Blockchain Network (critical) ----
    network_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Blockchain_Network",
        desc="Verify blockchain network and provide evidence",
        parent=property_parent_node,
        critical=True
    )

    # Network Requirement leaf
    network_req_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Network_Requirement",
        desc="The property tokens must be issued on Ethereum, Algorand, XRP Ledger, or Hedera",
        parent=network_node,
        critical=True
    )
    network_req_sources = compile_sources(prop.property_listing_url, prop.network_evidence_urls)
    await evaluator.verify(
        claim="The property tokens are issued on one of the allowed networks: Ethereum, Algorand, XRP Ledger (XRPL), or Hedera.",
        node=network_req_leaf,
        sources=network_req_sources if network_req_sources else None,
        additional_instruction="Look for explicit mentions of Ethereum, Algorand, XRPL/XRP Ledger, or Hedera/Hedera Hashgraph on the listing or platform pages."
    )

    # Network Evidence URL leaf
    network_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Network_Evidence_URL",
        desc="Provide URL evidence of the blockchain network used",
        parent=network_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page confirms the blockchain network used is '{prop.blockchain_network or ''}'.",
        node=network_evidence_leaf,
        sources=prop.network_evidence_urls if prop.network_evidence_urls else network_req_sources if network_req_sources else None,
        additional_instruction="Accept common synonyms: 'XRPL' for 'XRP Ledger'; 'Hedera Hashgraph' for 'Hedera'."
    )

    # ---- Minimum Investment (critical) ----
    invest_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Minimum_Investment",
        desc="Verify minimum investment amount and provide evidence",
        parent=property_parent_node,
        critical=True
    )

    # Investment Amount leaf
    invest_amt_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Investment_Amount",
        desc="The minimum investment amount must be $100 or less per token",
        parent=invest_node,
        critical=True
    )
    invest_sources = compile_sources(prop.property_listing_url, prop.min_investment_evidence_urls)
    await evaluator.verify(
        claim="The listing states a minimum investment per token or fractional share that is $100 or less.",
        node=invest_amt_leaf,
        sources=invest_sources if invest_sources else None,
        additional_instruction="Find the minimum investment figure; if denominated in USD, verify it is ≤ $100. If shown per token/share unit, interpret accordingly."
    )

    # Investment Evidence URL leaf
    invest_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Investment_Evidence_URL",
        desc="Provide URL evidence of the minimum investment amount",
        parent=invest_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page explicitly indicates the minimum investment amount as '{prop.min_investment or ''}'.",
        node=invest_evidence_leaf,
        sources=prop.min_investment_evidence_urls if prop.min_investment_evidence_urls else invest_sources if invest_sources else None,
        additional_instruction="Verify the text snippet for minimum investment/price per token; allow currency formatting variations."
    )

    # ---- Regulatory Framework (critical) ----
    reg_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Regulatory_Framework",
        desc="Verify SEC regulatory framework and provide evidence",
        parent=property_parent_node,
        critical=True
    )

    reg_ident_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Framework_Identification",
        desc="Identify the specific SEC framework (registration or exemption such as Reg CF, Reg D, or Reg A+)",
        parent=reg_node,
        critical=True
    )
    reg_sources = compile_sources(prop.property_listing_url, prop.regulatory_evidence_urls)
    await evaluator.verify(
        claim="The platform explicitly operates under an SEC framework: SEC Registration or an exemption (Reg CF, Reg D, or Reg A+).",
        node=reg_ident_leaf,
        sources=reg_sources if reg_sources else None,
        additional_instruction="Look for legal disclosures, offering memoranda, or footers stating Reg CF, Reg D (e.g., Rule 506), Reg A+, or direct SEC registration."
    )

    reg_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Framework_Evidence_URL",
        desc="Provide URL evidence of the regulatory framework disclosure",
        parent=reg_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page discloses the regulatory framework as '{prop.regulatory_framework or ''}'.",
        node=reg_evidence_leaf,
        sources=prop.regulatory_evidence_urls if prop.regulatory_evidence_urls else reg_sources if reg_sources else None,
        additional_instruction="Accept variants of the exemption names and shorthand (e.g., 'Regulation Crowdfunding' equals 'Reg CF')."
    )

    # ---- Investor Eligibility (critical) ----
    elig_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Investor_Eligibility",
        desc="Verify platform accepts non-accredited U.S. investors and provide evidence",
        parent=property_parent_node,
        critical=True
    )

    elig_accept_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Non_Accredited_Acceptance",
        desc="Platform must explicitly accept non-accredited U.S. investors",
        parent=elig_node,
        critical=True
    )
    elig_sources = compile_sources(prop.property_listing_url, prop.eligibility_evidence_urls)
    await evaluator.verify(
        claim="The platform explicitly accepts non-accredited U.S. investors for this property/offering.",
        node=elig_accept_leaf,
        sources=elig_sources if elig_sources else None,
        additional_instruction="Look for statements like 'open to non-accredited investors', 'available to all investors', or Reg CF context allowing non-accredited."
    )

    elig_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Eligibility_Evidence_URL",
        desc="Provide URL evidence of investor eligibility requirements",
        parent=elig_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The evidence page states acceptance of non-accredited U.S. investors ({prop.accepts_non_accredited_statement or ''}).",
        node=elig_evidence_leaf,
        sources=prop.eligibility_evidence_urls if prop.eligibility_evidence_urls else elig_sources if elig_sources else None,
        additional_instruction="Confirm eligibility wording; ignore unrelated risk or KYC statements unless they specifically preclude non-accredited investors."
    )

    # ---- Income Distribution (critical) ----
    income_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Income_Distribution",
        desc="Verify automated income distribution system",
        parent=property_parent_node,
        critical=True
    )

    # Smart Contract Automation (with evidence child)
    automation_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Smart_Contract_Automation",
        desc="Confirm use of smart contract-based automation for income distribution",
        parent=income_node,
        critical=True
    )
    automation_claim_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Automation_Claim",
        desc="Confirm smart contract-based automation for income distribution",
        parent=automation_node,
        critical=True
    )
    automation_sources = compile_sources(prop.property_listing_url, prop.automation_evidence_urls)
    await evaluator.verify(
        claim="The platform uses automated smart contract-based distribution of rental income or dividends to token holders.",
        node=automation_claim_leaf,
        sources=automation_sources if automation_sources else None,
        additional_instruction="Find mentions like 'smart contracts automate payouts', 'on-chain distribution', or similar technical descriptions."
    )
    automation_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Automation_Evidence_URL",
        desc="Provide URL evidence of smart contract automation",
        parent=automation_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The evidence page explicitly mentions smart contract-based automation ({prop.income_distribution_mechanism or ''}).",
        node=automation_evidence_leaf,
        sources=prop.automation_evidence_urls if prop.automation_evidence_urls else automation_sources if automation_sources else None,
        additional_instruction="Accept explicit references to smart contracts or on-chain payout mechanisms, even if technical."
    )

    # Distribution Frequency (with evidence child)
    freq_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Distribution_Frequency",
        desc="Specify the distribution frequency (daily, weekly, or monthly)",
        parent=income_node,
        critical=True
    )
    frequency_claim_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Frequency_Claim",
        desc="Distribution frequency is daily, weekly, or monthly",
        parent=freq_node,
        critical=True
    )
    freq_sources = compile_sources(prop.property_listing_url, prop.frequency_evidence_urls)
    await evaluator.verify(
        claim="The platform specifies income distribution frequency as daily, weekly, or monthly.",
        node=frequency_claim_leaf,
        sources=freq_sources if freq_sources else None,
        additional_instruction="Look for explicit cadence words: 'daily', 'weekly', 'monthly'; accept phrases like 'paid every month'."
    )
    frequency_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Frequency_Evidence_URL",
        desc="Provide URL evidence of distribution frequency",
        parent=freq_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The evidence page states the distribution frequency as '{prop.income_distribution_frequency or ''}'.",
        node=frequency_evidence_leaf,
        sources=prop.frequency_evidence_urls if prop.frequency_evidence_urls else freq_sources if freq_sources else None,
        additional_instruction="Confirm exact phrasing or equivalent periodic descriptions."
    )

    # ---- Secondary Market (critical) ----
    secondary_node = evaluator.add_parallel(
        id=f"{ordinal_label}_Secondary_Market",
        desc="Verify secondary market mechanism and provide evidence",
        parent=property_parent_node,
        critical=True
    )

    secondary_mech_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Trading_Mechanism",
        desc="Describe the secondary market mechanism for trading property tokens",
        parent=secondary_node,
        critical=True
    )
    secondary_sources = compile_sources(prop.property_listing_url, prop.secondary_market_evidence_urls)
    await evaluator.verify(
        claim="The platform provides a secondary market mechanism for trading property tokens (e.g., ATS marketplace, exchange, or platform secondary market).",
        node=secondary_mech_leaf,
        sources=secondary_sources if secondary_sources else None,
        additional_instruction="Look for 'secondary market', 'ATS', 'trading', 'marketplace', or partner exchange facilitating token trading."
    )

    secondary_evidence_leaf = evaluator.add_leaf(
        id=f"{ordinal_label}_Trading_Evidence_URL",
        desc="Provide URL evidence of secondary market capabilities",
        parent=secondary_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The evidence page describes the secondary market mechanism as '{prop.secondary_market_mechanism or ''}'.",
        node=secondary_evidence_leaf,
        sources=prop.secondary_market_evidence_urls if prop.secondary_market_evidence_urls else secondary_sources if secondary_sources else None,
        additional_instruction="Confirm specific mechanism or partner platform enabling token trading; accept ATS disclosures."
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
    Evaluate an answer for tokenized real estate properties meeting 2026 criteria.
    """
    # Initialize evaluator (root as non-critical parallel to allow partial credit across properties)
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

    # Record allowed networks info
    evaluator.add_ground_truth({"allowed_networks": ALLOWED_NETWORKS})

    # Extract properties from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_properties(),
        template_class=RealEstateExtraction,
        extraction_name="properties_extraction",
    )

    # Prepare first four properties (pad if fewer)
    props: List[PropertyItem] = extraction.properties[:4]
    while len(props) < 4:
        props.append(PropertyItem())

    ordinal_labels = ["First_Property", "Second_Property", "Third_Property", "Fourth_Property"]

    # Build property subtrees
    for i, prop in enumerate(props):
        prop_node = evaluator.add_parallel(
            id=ordinal_labels[i],
            desc=f"{['First','Second','Third','Fourth'][i]} tokenized real estate property",
            parent=root,
            critical=False  # Each property contributes partial credit; internal checks are critical
        )
        await verify_property(evaluator, prop_node, prop, ordinal_labels[i], i)

    # Geographic diversity check (critical under root)
    # Collect normalized states from extracted properties
    states: List[str] = []
    for p in props:
        normalized = normalize_state(p.state)
        if normalized:
            states.append(normalized)

    unique_states = sorted(list(set(states)))
    evaluator.add_custom_info(
        info={"extracted_states": states, "unique_states": unique_states},
        info_type="geo_states",
        info_name="geographic_states_info"
    )

    geo_diversity_pass = len(set(states)) >= 3
    evaluator.add_custom_node(
        result=geo_diversity_pass,
        id="Geographic_Diversity",
        desc="At least three of the four properties must be located in different U.S. states",
        parent=root,
        critical=True
    )

    # Return final summary
    return evaluator.get_summary()