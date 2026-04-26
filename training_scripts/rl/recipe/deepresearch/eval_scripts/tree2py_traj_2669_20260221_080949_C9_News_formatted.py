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
TASK_ID = "eu_defense_2025_countries"
TASK_DESCRIPTION = """
Based on European defense policy developments in 2025, identify four European countries that satisfy the following specific criteria:

Country A: A European country that meets ALL of the following conditions:
- Participates in the European Sky Shield Initiative (ESSI)
- Maintains permanent military neutrality (is not a NATO member)
- Is a member of the European Union
- Has publicly committed to increasing its defense spending to 2% of GDP by the year 2032

Country B: A NATO member country that meets ALL of the following conditions:
- Participates in the European Sky Shield Initiative (ESSI)
- Received an official exemption from the 5% GDP defense spending target agreed upon at the June 2025 NATO Summit in The Hague

Countries C and D: Two additional European countries (different from Countries A and B) that each meet ALL of the following conditions:
- Is a member of NATO
- Participates in the European Sky Shield Initiative (ESSI)
- Is a member of the European Union

For each country identified, provide:
1. The country's name
2. Supporting evidence with reference URLs confirming each required criterion
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CountryAExtraction(BaseModel):
    name: Optional[str] = None
    essi_urls: List[str] = Field(default_factory=list)
    neutrality_urls: List[str] = Field(default_factory=list)
    non_nato_urls: List[str] = Field(default_factory=list)
    eu_urls: List[str] = Field(default_factory=list)
    spending_target_2032_urls: List[str] = Field(default_factory=list)


class CountryBExtraction(BaseModel):
    name: Optional[str] = None
    essi_urls: List[str] = Field(default_factory=list)
    nato_urls: List[str] = Field(default_factory=list)
    hague_exemption_urls: List[str] = Field(default_factory=list)


class CountryCDExtraction(BaseModel):
    name: Optional[str] = None
    essi_urls: List[str] = Field(default_factory=list)
    eu_urls: List[str] = Field(default_factory=list)
    nato_urls: List[str] = Field(default_factory=list)


class DefenseCountriesExtraction(BaseModel):
    country_a: Optional[CountryAExtraction] = None
    country_b: Optional[CountryBExtraction] = None
    country_c: Optional[CountryCDExtraction] = None
    country_d: Optional[CountryCDExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_countries_defense() -> str:
    return """
    Extract structured information for four countries labeled in the answer as Country A, Country B, Country C, and Country D, following the task requirements.

    For each country, return the following fields (extract only URLs explicitly present in the answer; do not invent any):
    - name: The country's name as written in the answer (string or null if missing).
    - essi_urls: Array of URLs cited that explicitly support or confirm participation in the European Sky Shield Initiative (ESSI).
    - eu_urls: Array of URLs cited that explicitly confirm membership in the European Union. (For Country B, C, and D this applies; for Country A it applies too.)
    - nato_urls: Array of URLs cited that explicitly confirm NATO membership. (Applicable for B, C, D. For A, set it to an empty array or omit; do NOT mark A as NATO member.)
    - neutrality_urls: Array of URLs cited that explicitly confirm permanent military neutrality. (Applicable for Country A; empty array for others.)
    - non_nato_urls: Array of URLs cited that explicitly confirm the country is NOT a NATO member. (Applicable for Country A; empty array for others.)
    - spending_target_2032_urls: Array of URLs cited that explicitly confirm a public commitment to reach 2% of GDP in defense spending by 2032. (Applicable for Country A; empty array for others.)
    - hague_exemption_urls: Array of URLs cited that explicitly confirm that the country received an official exemption from the 5% GDP defense spending target at the June 2025 NATO Summit in The Hague. (Applicable for Country B; empty array for others.)

    Return a single JSON object with fields: country_a, country_b, country_c, country_d.
    Each country field must itself be a JSON object with the fields above. If any field is not mentioned in the answer, set it to null (for name) or an empty array (for URLs).
    Ensure all URLs are complete and valid; include full protocol (http:// or https://). Do not include non-URL citations or generic references without an explicit URL.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_urls(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) else []


def _name_or_placeholder(name: Optional[str], placeholder: str) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else placeholder


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_country_a(evaluator: Evaluator, parent_node, info: CountryAExtraction) -> None:
    country_name = _name_or_placeholder(info.name, "Country A")

    # Country A Identification (non-critical aggregator)
    a_node = evaluator.add_parallel(
        id="Country_A_Identification",
        desc="Evaluate whether Country A is correctly identified and satisfies all required criteria: ESSI participation, military neutrality (non-NATO), EU membership, and 2% GDP by 2032 defense spending commitment",
        parent=parent_node,
        critical=False
    )

    # ESSI Participation (critical aggregator)
    essi_part_node = evaluator.add_parallel(
        id="Country_A_ESSI_Participation",
        desc="Verify that the identified Country A participates in the European Sky Shield Initiative (ESSI)",
        parent=a_node,
        critical=True
    )
    essi_verif_node = evaluator.add_parallel(
        id="Country_A_ESSI_Verification",
        desc="Verification of Country A's ESSI participation through evidence and references",
        parent=essi_part_node,
        critical=True
    )
    # Evidence leaf (verify with URLs)
    essi_evidence_leaf = evaluator.add_leaf(
        id="Country_A_ESSI_Evidence",
        desc="Evidence is provided confirming Country A's participation in ESSI",
        parent=essi_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} participates in the European Sky Shield Initiative (ESSI).",
        node=essi_evidence_leaf,
        sources=_safe_urls(info.essi_urls),
        additional_instruction="Confirm explicit participation/membership in ESSI. Accept reasonable wording variants such as 'joins ESSI', 'part of ESSI', or 'participates in ESSI'. Use the cited URLs to ground your decision."
    )
    # URL existence check (critical custom)
    evaluator.add_custom_node(
        result=len(_safe_urls(info.essi_urls)) > 0,
        id="Country_A_ESSI_URL",
        desc="A reference URL is provided that verifies Country A's ESSI participation",
        parent=essi_verif_node,
        critical=True
    )

    # Military Neutrality (critical aggregator)
    neutrality_part_node = evaluator.add_parallel(
        id="Country_A_Military_Neutrality",
        desc="Verify that the identified Country A maintains permanent military neutrality and is not a NATO member",
        parent=a_node,
        critical=True
    )
    neutrality_verif_node = evaluator.add_parallel(
        id="Country_A_Neutrality_Verification",
        desc="Verification of Country A's neutrality status through evidence and references",
        parent=neutrality_part_node,
        critical=True
    )
    # Neutrality status leaf
    neutrality_status_leaf = evaluator.add_leaf(
        id="Country_A_Neutrality_Status",
        desc="Evidence confirms that Country A maintains permanent military neutrality",
        parent=neutrality_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} maintains permanent military neutrality.",
        node=neutrality_status_leaf,
        sources=_safe_urls(info.neutrality_urls),
        additional_instruction="Verify that the country has a permanent policy of military neutrality. Look for official policy descriptions, constitutional provisions, or government statements affirming neutrality."
    )
    # Non-NATO status leaf
    non_nato_status_leaf = evaluator.add_leaf(
        id="Country_A_Non_NATO_Status",
        desc="Evidence confirms that Country A is not a member of NATO",
        parent=neutrality_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is not a member of NATO.",
        node=non_nato_status_leaf,
        sources=_safe_urls(info.non_nato_urls),
        additional_instruction="Confirm that the country is not a NATO member. Use official NATO membership lists or credible sources that explicitly state non-membership."
    )
    # Neutrality URL existence check
    evaluator.add_custom_node(
        result=len(_safe_urls(info.neutrality_urls)) > 0,
        id="Country_A_Neutrality_URL",
        desc="A reference URL is provided that verifies Country A's military neutrality status",
        parent=neutrality_verif_node,
        critical=True
    )

    # EU Membership (critical aggregator)
    eu_part_node = evaluator.add_parallel(
        id="Country_A_EU_Membership",
        desc="Verify that the identified Country A is a member of the European Union",
        parent=a_node,
        critical=True
    )
    eu_verif_node = evaluator.add_parallel(
        id="Country_A_EU_Verification",
        desc="Verification of Country A's EU membership through evidence and references",
        parent=eu_part_node,
        critical=True
    )
    eu_status_leaf = evaluator.add_leaf(
        id="Country_A_EU_Status",
        desc="Evidence confirms that Country A is an EU member",
        parent=eu_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of the European Union.",
        node=eu_status_leaf,
        sources=_safe_urls(info.eu_urls),
        additional_instruction="Confirm EU membership using official EU sources or credible references. Allow typical naming variants and abbreviations."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.eu_urls)) > 0,
        id="Country_A_EU_URL",
        desc="A reference URL is provided that verifies Country A's EU membership",
        parent=eu_verif_node,
        critical=True
    )

    # Defense Spending Commitment (critical aggregator)
    spend_part_node = evaluator.add_parallel(
        id="Country_A_Defense_Spending_Commitment",
        desc="Verify that the identified Country A has publicly committed to increasing its defense spending to 2% of GDP by the year 2032",
        parent=a_node,
        critical=True
    )
    spend_verif_node = evaluator.add_parallel(
        id="Country_A_Spending_Verification",
        desc="Verification of Country A's 2% GDP by 2032 commitment through evidence and references",
        parent=spend_part_node,
        critical=True
    )
    target_leaf = evaluator.add_leaf(
        id="Country_A_2032_Target",
        desc="Evidence confirms that Country A has committed to reach 2% GDP defense spending specifically by 2032",
        parent=spend_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} has publicly committed to reach defense spending of 2% of GDP by 2032.",
        node=target_leaf,
        sources=_safe_urls(info.spending_target_2032_urls),
        additional_instruction="Check for explicit language committing to 2% of GDP by the year 2032. If the commitment is for a different year or lacks timeframe, do not pass."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.spending_target_2032_urls)) > 0,
        id="Country_A_Spending_URL",
        desc="A reference URL is provided that verifies Country A's 2% GDP by 2032 commitment",
        parent=spend_verif_node,
        critical=True
    )


async def verify_country_b(evaluator: Evaluator, parent_node, info: CountryBExtraction) -> None:
    country_name = _name_or_placeholder(info.name, "Country B")

    b_node = evaluator.add_parallel(
        id="Country_B_Identification",
        desc="Evaluate whether Country B is correctly identified and satisfies all required criteria: NATO membership, ESSI participation, and Hague Summit exemption from 5% GDP target",
        parent=parent_node,
        critical=False
    )

    # NATO membership
    nato_part_node = evaluator.add_parallel(
        id="Country_B_NATO_Membership",
        desc="Verify that the identified Country B is a member of NATO",
        parent=b_node,
        critical=True
    )
    nato_verif_node = evaluator.add_parallel(
        id="Country_B_NATO_Verification",
        desc="Verification of Country B's NATO membership through evidence and references",
        parent=nato_part_node,
        critical=True
    )
    nato_status_leaf = evaluator.add_leaf(
        id="Country_B_NATO_Status",
        desc="Evidence confirms that Country B is a NATO member",
        parent=nato_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of NATO.",
        node=nato_status_leaf,
        sources=_safe_urls(info.nato_urls),
        additional_instruction="Verify NATO membership using official NATO membership lists or credible sources."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.nato_urls)) > 0,
        id="Country_B_NATO_URL",
        desc="A reference URL is provided that verifies Country B's NATO membership",
        parent=nato_verif_node,
        critical=True
    )

    # ESSI participation
    essi_part_node = evaluator.add_parallel(
        id="Country_B_ESSI_Participation",
        desc="Verify that the identified Country B participates in the European Sky Shield Initiative (ESSI)",
        parent=b_node,
        critical=True
    )
    essi_verif_node = evaluator.add_parallel(
        id="Country_B_ESSI_Verification",
        desc="Verification of Country B's ESSI participation through evidence and references",
        parent=essi_part_node,
        critical=True
    )
    essi_evidence_leaf = evaluator.add_leaf(
        id="Country_B_ESSI_Evidence",
        desc="Evidence is provided confirming Country B's participation in ESSI",
        parent=essi_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} participates in the European Sky Shield Initiative (ESSI).",
        node=essi_evidence_leaf,
        sources=_safe_urls(info.essi_urls),
        additional_instruction="Confirm explicit participation/membership in ESSI using the cited URLs."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.essi_urls)) > 0,
        id="Country_B_ESSI_URL",
        desc="A reference URL is provided that verifies Country B's ESSI participation",
        parent=essi_verif_node,
        critical=True
    )

    # Hague exemption
    exempt_part_node = evaluator.add_parallel(
        id="Country_B_Hague_Exemption",
        desc="Verify that the identified Country B received an official exemption from the 5% GDP defense spending target agreed upon at the June 2025 NATO Hague Summit",
        parent=b_node,
        critical=True
    )
    exempt_verif_node = evaluator.add_parallel(
        id="Country_B_Exemption_Verification",
        desc="Verification of Country B's Hague Summit exemption through evidence and references",
        parent=exempt_part_node,
        critical=True
    )
    exempt_status_leaf = evaluator.add_leaf(
        id="Country_B_Exemption_Status",
        desc="Evidence confirms that Country B received an exemption from the 5% GDP defense spending target at the June 2025 Hague Summit",
        parent=exempt_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} received an official exemption from the 5% GDP defense spending target at the June 2025 NATO Summit in The Hague.",
        node=exempt_status_leaf,
        sources=_safe_urls(info.hague_exemption_urls),
        additional_instruction="Verify that the source explicitly mentions an official exemption from the 5% GDP target at the June 2025 NATO Summit in The Hague. General commentary without explicit exemption should not pass."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.hague_exemption_urls)) > 0,
        id="Country_B_Exemption_URL",
        desc="A reference URL is provided that verifies Country B's exemption from the Hague Summit 5% target",
        parent=exempt_verif_node,
        critical=True
    )


async def verify_country_c(evaluator: Evaluator, parent_node, info: CountryCDExtraction, a_info: CountryAExtraction, b_info: CountryBExtraction) -> None:
    country_name = _name_or_placeholder(info.name, "Country C")
    a_name = _name_or_placeholder(a_info.name, "Country A")
    b_name = _name_or_placeholder(b_info.name, "Country B")

    c_node = evaluator.add_parallel(
        id="Country_C_Identification",
        desc="Evaluate whether Country C is correctly identified and satisfies all required criteria: NATO membership, ESSI participation, and EU membership",
        parent=parent_node,
        critical=False
    )

    # NATO membership
    nato_part_node = evaluator.add_parallel(
        id="Country_C_NATO_Membership",
        desc="Verify that the identified Country C is a member of NATO",
        parent=c_node,
        critical=True
    )
    nato_verif_node = evaluator.add_parallel(
        id="Country_C_NATO_Verification",
        desc="Verification of Country C's NATO membership through evidence and references",
        parent=nato_part_node,
        critical=True
    )
    nato_status_leaf = evaluator.add_leaf(
        id="Country_C_NATO_Status",
        desc="Evidence confirms that Country C is a NATO member",
        parent=nato_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of NATO.",
        node=nato_status_leaf,
        sources=_safe_urls(info.nato_urls),
        additional_instruction="Confirm NATO membership using official or credible sources."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.nato_urls)) > 0,
        id="Country_C_NATO_URL",
        desc="A reference URL is provided that verifies Country C's NATO membership",
        parent=nato_verif_node,
        critical=True
    )

    # ESSI participation
    essi_part_node = evaluator.add_parallel(
        id="Country_C_ESSI_Participation",
        desc="Verify that the identified Country C participates in the European Sky Shield Initiative (ESSI)",
        parent=c_node,
        critical=True
    )
    essi_verif_node = evaluator.add_parallel(
        id="Country_C_ESSI_Verification",
        desc="Verification of Country C's ESSI participation through evidence and references",
        parent=essi_part_node,
        critical=True
    )
    essi_evidence_leaf = evaluator.add_leaf(
        id="Country_C_ESSI_Evidence",
        desc="Evidence is provided confirming Country C's participation in ESSI",
        parent=essi_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} participates in the European Sky Shield Initiative (ESSI).",
        node=essi_evidence_leaf,
        sources=_safe_urls(info.essi_urls),
        additional_instruction="Confirm explicit participation/membership in ESSI using the cited URLs."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.essi_urls)) > 0,
        id="Country_C_ESSI_URL",
        desc="A reference URL is provided that verifies Country C's ESSI participation",
        parent=essi_verif_node,
        critical=True
    )

    # EU membership
    eu_part_node = evaluator.add_parallel(
        id="Country_C_EU_Membership",
        desc="Verify that the identified Country C is a member of the European Union",
        parent=c_node,
        critical=True
    )
    eu_verif_node = evaluator.add_parallel(
        id="Country_C_EU_Verification",
        desc="Verification of Country C's EU membership through evidence and references",
        parent=eu_part_node,
        critical=True
    )
    eu_status_leaf = evaluator.add_leaf(
        id="Country_C_EU_Status",
        desc="Evidence confirms that Country C is an EU member",
        parent=eu_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of the European Union.",
        node=eu_status_leaf,
        sources=_safe_urls(info.eu_urls),
        additional_instruction="Verify EU membership using official EU sources or credible references."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.eu_urls)) > 0,
        id="Country_C_EU_URL",
        desc="A reference URL is provided that verifies Country C's EU membership",
        parent=eu_verif_node,
        critical=True
    )

    # Distinctness from A and B
    distinct_leaf = evaluator.add_leaf(
        id="Country_C_Distinctness",
        desc="Verify that Country C is different from Countries A and B",
        parent=c_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is different from {a_name} and {b_name}.",
        node=distinct_leaf,
        additional_instruction="Compare names case-insensitively and allow minor naming variants (accents, official vs. common names). The countries must be different entities."
    )


async def verify_country_d(evaluator: Evaluator, parent_node, info: CountryCDExtraction, a_info: CountryAExtraction, b_info: CountryBExtraction, c_info: CountryCDExtraction) -> None:
    country_name = _name_or_placeholder(info.name, "Country D")
    a_name = _name_or_placeholder(a_info.name, "Country A")
    b_name = _name_or_placeholder(b_info.name, "Country B")
    c_name = _name_or_placeholder(c_info.name, "Country C")

    d_node = evaluator.add_parallel(
        id="Country_D_Identification",
        desc="Evaluate whether Country D is correctly identified and satisfies all required criteria: NATO membership, ESSI participation, and EU membership",
        parent=parent_node,
        critical=False
    )

    # NATO membership
    nato_part_node = evaluator.add_parallel(
        id="Country_D_NATO_Membership",
        desc="Verify that the identified Country D is a member of NATO",
        parent=d_node,
        critical=True
    )
    nato_verif_node = evaluator.add_parallel(
        id="Country_D_NATO_Verification",
        desc="Verification of Country D's NATO membership through evidence and references",
        parent=nato_part_node,
        critical=True
    )
    nato_status_leaf = evaluator.add_leaf(
        id="Country_D_NATO_Status",
        desc="Evidence confirms that Country D is a NATO member",
        parent=nato_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of NATO.",
        node=nato_status_leaf,
        sources=_safe_urls(info.nato_urls),
        additional_instruction="Verify NATO membership using official or credible sources."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.nato_urls)) > 0,
        id="Country_D_NATO_URL",
        desc="A reference URL is provided that verifies Country D's NATO membership",
        parent=nato_verif_node,
        critical=True
    )

    # ESSI participation
    essi_part_node = evaluator.add_parallel(
        id="Country_D_ESSI_Participation",
        desc="Verify that the identified Country D participates in the European Sky Shield Initiative (ESSI)",
        parent=d_node,
        critical=True
    )
    essi_verif_node = evaluator.add_parallel(
        id="Country_D_ESSI_Verification",
        desc="Verification of Country D's ESSI participation through evidence and references",
        parent=essi_part_node,
        critical=True
    )
    essi_evidence_leaf = evaluator.add_leaf(
        id="Country_D_ESSI_Evidence",
        desc="Evidence is provided confirming Country D's participation in ESSI",
        parent=essi_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} participates in the European Sky Shield Initiative (ESSI).",
        node=essi_evidence_leaf,
        sources=_safe_urls(info.essi_urls),
        additional_instruction="Confirm explicit participation/membership in ESSI using the cited URLs."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.essi_urls)) > 0,
        id="Country_D_ESSI_URL",
        desc="A reference URL is provided that verifies Country D's ESSI participation",
        parent=essi_verif_node,
        critical=True
    )

    # EU membership
    eu_part_node = evaluator.add_parallel(
        id="Country_D_EU_Membership",
        desc="Verify that the identified Country D is a member of the European Union",
        parent=d_node,
        critical=True
    )
    eu_verif_node = evaluator.add_parallel(
        id="Country_D_EU_Verification",
        desc="Verification of Country D's EU membership through evidence and references",
        parent=eu_part_node,
        critical=True
    )
    eu_status_leaf = evaluator.add_leaf(
        id="Country_D_EU_Status",
        desc="Evidence confirms that Country D is an EU member",
        parent=eu_verif_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is a member of the European Union.",
        node=eu_status_leaf,
        sources=_safe_urls(info.eu_urls),
        additional_instruction="Verify EU membership using official EU sources or credible references."
    )
    evaluator.add_custom_node(
        result=len(_safe_urls(info.eu_urls)) > 0,
        id="Country_D_EU_URL",
        desc="A reference URL is provided that verifies Country D's EU membership",
        parent=eu_verif_node,
        critical=True
    )

    # Distinctness from A, B, C
    distinct_leaf = evaluator.add_leaf(
        id="Country_D_Distinctness",
        desc="Verify that Country D is different from Countries A, B, and C",
        parent=d_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{country_name} is different from {a_name}, {b_name}, and {c_name}.",
        node=distinct_leaf,
        additional_instruction="Compare names case-insensitively and allow minor naming variants (accents, official vs. common names). The countries must be different entities."
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
    Evaluate whether the response correctly identifies four European countries (A, B, C, and D) that each satisfy their respective specified criteria based on 2025 European defense policy developments.
    """
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_countries_defense(),
        template_class=DefenseCountriesExtraction,
        extraction_name="countries_defense_2025",
    )

    # Record short custom info for debugging
    evaluator.add_custom_info(
        info={
            "country_a_name": extraction.country_a.name if extraction.country_a else None,
            "country_b_name": extraction.country_b.name if extraction.country_b else None,
            "country_c_name": extraction.country_c.name if extraction.country_c else None,
            "country_d_name": extraction.country_d.name if extraction.country_d else None,
            "policy_context_year": 2025
        },
        info_type="extraction_summary",
        info_name="extraction_summary"
    )

    # Build verification tree: Root parallel children are the country identifications
    # Country A
    await verify_country_a(
        evaluator=evaluator,
        parent_node=root,
        info=extraction.country_a or CountryAExtraction()
    )

    # Country B
    await verify_country_b(
        evaluator=evaluator,
        parent_node=root,
        info=extraction.country_b or CountryBExtraction()
    )

    # Country C
    await verify_country_c(
        evaluator=evaluator,
        parent_node=root,
        info=extraction.country_c or CountryCDExtraction(),
        a_info=extraction.country_a or CountryAExtraction(),
        b_info=extraction.country_b or CountryBExtraction()
    )

    # Country D
    await verify_country_d(
        evaluator=evaluator,
        parent_node=root,
        info=extraction.country_d or CountryCDExtraction(),
        a_info=extraction.country_a or CountryAExtraction(),
        b_info=extraction.country_b or CountryBExtraction(),
        c_info=extraction.country_c or CountryCDExtraction()
    )

    return evaluator.get_summary()