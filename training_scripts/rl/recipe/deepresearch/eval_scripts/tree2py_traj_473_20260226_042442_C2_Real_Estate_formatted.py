import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ohfa_target_area_eligibility"
TASK_DESCRIPTION = """A first-time homebuyer household consisting of 3 persons with a combined annual income of $140,000 wants to purchase a single-family home in Franklin County, Ohio using the Ohio Housing Finance Agency (OHFA) financing programs. They are specifically interested in properties located in OHFA-designated target areas. Research OHFA's current program requirements and determine: (1) whether this household is income-eligible for OHFA first-time homebuyer programs in target areas of Franklin County, (2) the maximum purchase price limit for a 1-family home in a target area of Franklin County under OHFA programs, and (3) one mandatory program requirement (beyond income and purchase price qualifications) that all OHFA first-time homebuyers must complete. Provide official OHFA sources to support your answers.
"""

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def parse_usd_to_number(text: Optional[str]) -> Optional[float]:
    """Parse a USD-like string (e.g., '$145,000', '145000', '145,000.00') into a float."""
    if not text:
        return None
    # Find all number-like chunks
    nums = re.findall(r'[\d,]+(?:\.\d+)?', text)
    if not nums:
        return None
    # Heuristic: choose the largest numeric token in case multiple appear
    vals = []
    for n in nums:
        try:
            vals.append(float(n.replace(',', '')))
        except Exception:
            continue
    if not vals:
        return None
    return max(vals)


def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    """Return a cleaned list of plausible URLs."""
    if not urls:
        return []
    cleaned = []
    for u in urls:
        if not u:
            continue
        us = u.strip()
        if not us:
            continue
        if not (us.startswith("http://") or us.startswith("https://")):
            # Basic normalization: prepend http:// if missing
            us = "http://" + us
        cleaned.append(us)
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in cleaned:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def interpret_eligibility_text(text: Optional[str]) -> Optional[bool]:
    """Interpret 'eligible' vs 'ineligible' from free text."""
    if text is None:
        return None
    t = text.strip().lower()
    # Look for explicit negatives first
    if "not eligible" in t or "ineligible" in t or "does not qualify" in t:
        return False
    if "eligible" in t or "qualifies" in t:
        return True
    return None


# --------------------------------------------------------------------------- #
# Data Models for Extraction                                                  #
# --------------------------------------------------------------------------- #
class IncomeSection(BaseModel):
    # Example: "$145,000"
    target_area_income_limit_3plus: Optional[str] = None
    # URLs the answer cites for income limits (ideally official OHFA)
    income_source_urls: List[str] = Field(default_factory=list)
    # Whether the answer explicitly states a determination (True/False), nullable
    eligibility_stated: Optional[bool] = None
    # If stated, True means "eligible", False means "not eligible"
    eligibility_value: Optional[bool] = None
    # The literal phrase used in the answer (e.g., "eligible", "not eligible")
    eligibility_text: Optional[str] = None


class PriceSection(BaseModel):
    # Example: "$588,000" (maximum purchase price in target area for 1-family)
    max_price_target_area_1family: Optional[str] = None
    # Example: "Franklin County" or "Columbus area"
    geographic_area_label: Optional[str] = None
    # Example: "1-family", "single-family", "one-unit"
    property_type_label: Optional[str] = None
    # Example: "target area"
    area_classification_label: Optional[str] = None
    # URLs the answer cites for purchase price limits
    price_source_urls: List[str] = Field(default_factory=list)


class EducationSection(BaseModel):
    # The requirement text identified by the answer (e.g., "complete a homebuyer education course")
    requirement_text: Optional[str] = None
    # URLs the answer cites for the requirement (ideally official OHFA)
    education_source_urls: List[str] = Field(default_factory=list)


class OHFAExtraction(BaseModel):
    income: Optional[IncomeSection] = None
    price: Optional[PriceSection] = None
    education: Optional[EducationSection] = None


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_ohfa() -> str:
    return """
Extract the following structured information as it appears in the answer. Do not infer or fabricate values.

SECTION 1: INCOME ELIGIBILITY (Target Area – Franklin County / Columbus area)
- income.target_area_income_limit_3plus: The target area income limit explicitly stated in the answer for a household size category that applies to 3-person households (for OHFA this is often "3 or more persons"). Keep the text exactly (e.g., "$145,000"). If missing, return null.
- income.income_source_urls: All URLs the answer cites as sources for the income limits (ideally official OHFA pages). Return only URLs explicitly present in the answer.
- income.eligibility_text: The answer’s explicit determination about whether the household with $140,000 income is "eligible" or "not eligible" (or equivalent wording) under target area limits. If not explicitly stated, return null.
- income.eligibility_stated: true if the answer explicitly states a determination (eligible / not eligible). Otherwise false or null.
- income.eligibility_value: If the answer explicitly states a determination, set to true for "eligible/qualifies" or false for "ineligible/does not qualify". If no explicit determination, return null.

SECTION 2: MAXIMUM PURCHASE PRICE (Target Area – Franklin County / Columbus area)
- price.max_price_target_area_1family: The maximum purchase price limit as stated for a 1-family (one-unit/single-family) home located in a target area of Franklin County (or Columbus area if OHFA organizes by region). Keep the text exactly (e.g., "$588,000"). If missing, return null.
- price.geographic_area_label: The geographic label used in the answer that ties the price limit to Franklin County (or the Columbus area). If missing, return null.
- price.property_type_label: The property type label used in the answer (e.g., "1-family", "single-family", "one-unit"). If missing, return null.
- price.area_classification_label: The area classification label used in the answer (e.g., "target area"). If missing, return null.
- price.price_source_urls: All URLs the answer cites as sources for purchase price limits. Return only URLs explicitly present in the answer.

SECTION 3: MANDATORY PROGRAM REQUIREMENT
- education.requirement_text: One mandatory program requirement (beyond income and purchase price) that the answer claims all OHFA first-time homebuyers must complete. Typically this might be "complete a homebuyer education course"; extract exactly what the answer states. If missing, return null.
- education.education_source_urls: All URLs the answer cites as sources for the requirement. Return only URLs explicitly present in the answer.

RULES:
- Do not invent URLs; return only URLs explicitly present in the answer text.
- Preserve price/income numeric strings exactly as shown (including $ and commas).
- If any requested field is not mentioned, return null (or an empty list for URLs).
"""


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_income_section(evaluator: Evaluator, parent_node, income: Optional[IncomeSection]) -> None:
    # Build Income Eligibility node (critical)
    inc_node = evaluator.add_parallel(
        id="Income_Eligibility",
        desc="Determination of household income eligibility for OHFA target area programs",
        parent=parent_node,
        critical=True
    )

    # Prepare data
    income_limit_text = income.target_area_income_limit_3plus if income else None
    income_urls = sanitize_urls(income.income_source_urls if income else [])
    elig_text = income.eligibility_text if income else None
    elig_stated = income.eligibility_stated if income else None
    elig_value = income.eligibility_value if income else None

    # 1) Income_Source_Reference (leaf)
    # Verify that at least one cited source is an OHFA page that presents income limits for target areas (Franklin County / Columbus area)
    inc_src_leaf = evaluator.add_leaf(
        id="Income_Source_Reference",
        desc="Reference to official OHFA document containing target area income limits for Franklin County",
        parent=inc_node,
        critical=True
    )
    claim_inc_src = ("At least one of the cited sources is an official Ohio Housing Finance Agency (OHFA) webpage "
                     "that presents OHFA household income limits for target areas relevant to Franklin County "
                     "(often organized as the 'Columbus area').")
    await evaluator.verify(
        claim=claim_inc_src,
        node=inc_src_leaf,
        sources=income_urls,
        additional_instruction="Look for 'Income Limits', 'Target Area', and either 'Franklin County' or 'Columbus area' on the OHFA page."
    )

    # 2) Income_Eligibility_Verification (critical parallel group)
    inc_verify_node = evaluator.add_parallel(
        id="Income_Eligibility_Verification",
        desc="Accurate verification that household qualifies based on correctly identified income limits",
        parent=inc_node,
        critical=True
    )

    # 2.a) Target_Area_Income_Limit_Identified (leaf)
    limit_leaf = evaluator.add_leaf(
        id="Target_Area_Income_Limit_Identified",
        desc="Correct target area income limit for 3 or more persons in Franklin County identified from official OHFA sources",
        parent=inc_verify_node,
        critical=True
    )
    if income_limit_text:
        claim_limit = (f"The target area income limit for a household size category applicable to 3-person households "
                       f"in Franklin County (often listed under the 'Columbus area') is {income_limit_text}.")
    else:
        claim_limit = ("The answer identifies the correct OHFA target area income limit for a 3-person household "
                       "in Franklin County (often listed under the 'Columbus area').")
    await evaluator.verify(
        claim=claim_limit,
        node=limit_leaf,
        sources=income_urls,
        additional_instruction=("Check the OHFA source for a table or listing of income limits. "
                                "Use the 'Target Area' column if present and the row for '3 or more persons' or '3-person household'. "
                                "Minor formatting differences (e.g., commas, currency symbols) are acceptable.")
    )

    # 2.b) Eligibility_Determination (leaf, simple verification)
    elig_leaf = evaluator.add_leaf(
        id="Eligibility_Determination",
        desc="Accurate determination of whether the household's $140,000 income qualifies under the identified limit",
        parent=inc_verify_node,
        critical=True
    )

    # Construct a robust claim using answer context and math
    # Prefer to reference what the answer stated if available
    limit_for_math = parse_usd_to_number(income_limit_text) if income_limit_text else None
    if elig_stated is True and elig_value is not None:
        human_label = "eligible" if elig_value else "not eligible"
        claim_elig = (f"The answer explicitly states that the household is {human_label} under OHFA target area income limits for Franklin County. "
                      f"Given a household income of $140,000 and the identified target area income limit '{income_limit_text}', "
                      f"this determination is mathematically correct (treat income as eligible if it is less than or equal to the limit).")
    else:
        # Require the answer to clearly state the determination
        claim_elig = (f"The answer clearly states whether the household is income-eligible under OHFA target area income limits "
                      f"for Franklin County, and the stated determination is mathematically correct given the identified limit "
                      f"'{income_limit_text}' and the $140,000 income (eligible if income <= limit).")

    await evaluator.verify(
        claim=claim_elig,
        node=elig_leaf,
        additional_instruction=("Evaluate two things: (1) Does the answer clearly state 'eligible' or 'not eligible' for the income test? "
                                "(2) Is that determination correct given the provided target area limit (parse currency/commas) "
                                "and $140,000 income, using the rule 'eligible if income <= limit'? "
                                "If the answer did not explicitly state a determination, mark this as incorrect.")
    )


async def verify_price_section(evaluator: Evaluator, parent_node, price: Optional[PriceSection]) -> None:
    price_node = evaluator.add_parallel(
        id="Maximum_Purchase_Price",
        desc="Identification of maximum purchase price for target area 1-family home in Franklin County",
        parent=parent_node,
        critical=True
    )

    # Prepare data
    price_limit_text = price.max_price_target_area_1family if price else None
    price_urls = sanitize_urls(price.price_source_urls if price else [])
    geo_label = (price.geographic_area_label or "Franklin County or the Columbus area") if price else "Franklin County or the Columbus area"
    prop_label = (price.property_type_label or "1-family (single-family/one-unit)") if price else "1-family (single-family/one-unit)"
    area_class_label = (price.area_classification_label or "Target Area") if price else "Target Area"

    # 1) Price_Source_Reference (leaf)
    price_src_leaf = evaluator.add_leaf(
        id="Price_Source_Reference",
        desc="Reference to official OHFA document containing purchase price limits for Columbus area/Franklin County",
        parent=price_node,
        critical=True
    )
    claim_price_src = ("At least one of the cited sources is an official OHFA webpage that lists purchase price limits "
                       "for Franklin County or the Columbus area under OHFA programs.")
    await evaluator.verify(
        claim=claim_price_src,
        node=price_src_leaf,
        sources=price_urls,
        additional_instruction=("Look for a table or listing of OHFA purchase price limits by area. "
                                "OHFA may group Franklin County under the 'Columbus area'.")
    )

    # 2) Purchase_Price_Verification (critical parallel group)
    ppv_node = evaluator.add_parallel(
        id="Purchase_Price_Verification",
        desc="Correct maximum purchase price limit identified with proper specifications",
        parent=price_node,
        critical=True
    )

    # 2.a) Geographic_Area_Specified
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Area_Specified",
        desc="Correct geographic area (Franklin County or Columbus area) specified",
        parent=ppv_node,
        critical=True
    )
    claim_geo = (f"The OHFA source confirms that the purchase price limit referenced applies to the correct geographic area, "
                 f"specifically {geo_label} (Franklin County is commonly grouped within the 'Columbus area').")
    await evaluator.verify(
        claim=claim_geo,
        node=geo_leaf,
        sources=price_urls,
        additional_instruction=("Confirm that the purchase price limits shown on the OHFA source correspond to Franklin County "
                                "or to the Columbus area region that includes Franklin County.")
    )

    # 2.b) Property_Type_Specified
    prop_leaf = evaluator.add_leaf(
        id="Property_Type_Specified",
        desc="Correct property type (1-family) specified",
        parent=ppv_node,
        critical=True
    )
    claim_prop = (f"The referenced purchase price limit in the OHFA source applies to a 1-family home "
                  f"(consider 'one-unit' or 'single-family' as equivalent to '{prop_label}').")
    await evaluator.verify(
        claim=claim_prop,
        node=prop_leaf,
        sources=price_urls,
        additional_instruction="Accept synonyms such as 'single-family' or 'one-unit' as equivalent to '1-family'."
    )

    # 2.c) Area_Classification_Specified
    area_class_leaf = evaluator.add_leaf(
        id="Area_Classification_Specified",
        desc="Correct area classification (target area) specified",
        parent=ppv_node,
        critical=True
    )
    claim_area_class = ("The purchase price limit referenced is specifically for Target Areas (as distinct from Non-Target Areas).")
    await evaluator.verify(
        claim=claim_area_class,
        node=area_class_leaf,
        sources=price_urls,
        additional_instruction="Look for 'Target Area' vs 'Non-Target Area' columns or labels on the OHFA page."
    )

    # 2.d) Price_Limit_Stated
    price_limit_leaf = evaluator.add_leaf(
        id="Price_Limit_Stated",
        desc="Maximum purchase price limit for target area 1-family home accurately stated from official OHFA sources",
        parent=ppv_node,
        critical=True
    )
    if price_limit_text:
        claim_price_limit = (f"The maximum purchase price limit for a {prop_label} home in a Target Area in {geo_label} is {price_limit_text}.")
    else:
        claim_price_limit = (f"The answer correctly states the maximum purchase price limit for a {prop_label} home "
                             f"in a Target Area in {geo_label}, as shown on the OHFA source.")
    await evaluator.verify(
        claim=claim_price_limit,
        node=price_limit_leaf,
        sources=price_urls,
        additional_instruction=("Verify the numeric amount against the OHFA source. "
                                "Minor formatting differences (currency symbols/commas) are acceptable if the value matches.")
    )


async def verify_education_section(evaluator: Evaluator, parent_node, edu: Optional[EducationSection]) -> None:
    edu_node = evaluator.add_parallel(
        id="Education_Requirement",
        desc="Identification of mandatory homebuyer education requirement",
        parent=parent_node,
        critical=True
    )

    edu_urls = sanitize_urls(edu.education_source_urls if edu else [])
    requirement_text = edu.requirement_text if edu else None

    edu_leaf = evaluator.add_leaf(
        id="Education_Course_Requirement",
        desc="Homebuyer education course requirement correctly identified as mandatory for OHFA first-time homebuyer programs",
        parent=edu_node,
        critical=True
    )

    if requirement_text:
        claim_edu = (f"The answer identifies the mandatory requirement: '{requirement_text}', and this requirement is indeed required "
                     f"for OHFA first-time homebuyer programs.")
    else:
        claim_edu = ("OHFA first-time homebuyer programs mandate completion of a homebuyer education course (or equivalent "
                     "homeownership education) as a requirement beyond income and purchase price limits.")

    await evaluator.verify(
        claim=claim_edu,
        node=edu_leaf,
        sources=edu_urls,
        additional_instruction=("Check the official OHFA page(s) for language indicating that homebuyer education is required/mandatory "
                                "for first-time homebuyers using OHFA programs. Accept reasonable synonyms like 'homebuyer education', "
                                "'homeownership education', or 'homebuyer counseling'.")
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict:
    """
    Evaluate an answer for the OHFA target area eligibility and requirements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Overall, three major aspects can be checked in parallel
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

    # 1) Extract structured information from the answer
    extraction: OHFAExtraction = await evaluator.extract(
        prompt=prompt_extract_ohfa(),
        template_class=OHFAExtraction,
        extraction_name="ohfa_extraction"
    )

    # 2) Build top-level assessment node (critical)
    ohfa_node = evaluator.add_parallel(
        id="OHFA_Program_Assessment",
        desc="Complete assessment of OHFA first-time homebuyer program eligibility and requirements for the specified household",
        parent=root,
        critical=True
    )

    # 3) Verify Income Eligibility section
    await verify_income_section(evaluator, ohfa_node, extraction.income)

    # 4) Verify Maximum Purchase Price section
    await verify_price_section(evaluator, ohfa_node, extraction.price)

    # 5) Verify Education Requirement section
    await verify_education_section(evaluator, ohfa_node, extraction.education)

    # 6) Return evaluation summary
    return evaluator.get_summary()