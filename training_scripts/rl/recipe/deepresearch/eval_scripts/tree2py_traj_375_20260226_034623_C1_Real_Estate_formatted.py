import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tn_retirement_taxes"
TASK_DESCRIPTION = """I am considering retiring to Tennessee and want to understand the tax implications. Please verify the following two aspects of Tennessee's tax policies:

1. Does Tennessee have a state income tax on retirement income (including Social Security benefits, pensions, 401(k), and IRA distributions)?

2. Does Tennessee offer any property tax relief or freeze programs specifically for senior homeowners age 65 and older?

For each aspect, provide a clear yes or no answer with supporting information from official or reputable sources."""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RetirementTypesAddressed(BaseModel):
    mentions_social_security: Optional[bool] = None
    mentions_pensions: Optional[bool] = None
    mentions_401k: Optional[bool] = None
    mentions_ira: Optional[bool] = None


class RetirementIncomeSection(BaseModel):
    conclusion_yes_no: Optional[str] = None  # expected "no" if correctly answered
    conclusion_sentence: Optional[str] = None
    addressed: RetirementTypesAddressed = Field(default_factory=RetirementTypesAddressed)
    sources: List[str] = Field(default_factory=list)


class SeniorPropertyTaxSection(BaseModel):
    conclusion_yes_no: Optional[str] = None  # expected "yes" if correctly answered
    conclusion_sentence: Optional[str] = None
    mentions_age_65_plus: Optional[bool] = None
    age_65_quote: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class TaxPolicyExtraction(BaseModel):
    retirement_income: RetirementIncomeSection = Field(default_factory=RetirementIncomeSection)
    senior_property_tax: SeniorPropertyTaxSection = Field(default_factory=SeniorPropertyTaxSection)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_tax_policy() -> str:
    return """
    Extract structured information from the answer for two separate aspects of Tennessee tax policy.

    Aspect A (Retirement Income State Income Tax):
    - retirement_income.conclusion_yes_no: Extract a single word "yes" or "no" that reflects the answer's explicit conclusion about whether Tennessee has a state income tax on retirement income overall. If unclear or not stated, use null.
    - retirement_income.conclusion_sentence: Extract the exact sentence(s) where the answer gives its yes/no conclusion. If missing, null.
    - retirement_income.addressed.mentions_social_security: true if the answer explicitly mentions Social Security benefits in the context of Tennessee taxation; otherwise false.
    - retirement_income.addressed.mentions_pensions: true if explicitly mentions pensions; otherwise false.
    - retirement_income.addressed.mentions_401k: true if explicitly mentions 401(k) distributions; otherwise false.
    - retirement_income.addressed.mentions_ira: true if explicitly mentions IRA distributions; otherwise false.
    - retirement_income.sources: list of URLs the answer cites to support the retirement-income tax discussion. If the answer provides a single combined source list for the whole response, you may include those URLs here as well. Only include explicit URLs present in the answer.

    Aspect B (Senior Property Tax Relief/Freeze for 65+):
    - senior_property_tax.conclusion_yes_no: Extract a single word "yes" or "no" that reflects the answer's explicit conclusion about whether Tennessee offers any property tax relief or freeze programs (statewide or local) for seniors. If unclear, null.
    - senior_property_tax.conclusion_sentence: Extract the exact sentence(s) where the answer gives its yes/no conclusion. If missing, null.
    - senior_property_tax.mentions_age_65_plus: true if the answer explicitly states age 65 or older as an eligibility for a senior program; otherwise false.
    - senior_property_tax.age_65_quote: The exact phrase/sentence mentioning "age 65" or "65 and older" if present; else null.
    - senior_property_tax.sources: list of URLs the answer cites to support the senior property-tax program(s). If the answer provides a single combined source list for the whole response, you may include those URLs here as well. Only include explicit URLs present in the answer.

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (including markdown links).
    - Do not invent URLs.
    - Ensure URLs are complete and valid; if protocol missing, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _all_true(values: List[Optional[bool]]) -> bool:
    return all(bool(v) for v in values)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_retirement_income_verification(
    evaluator: Evaluator,
    parent_node,
    extraction: TaxPolicyExtraction
) -> None:
    """
    Build and verify the subtree for:
    Verification of whether Tennessee has a state income tax on retirement income (incl. Social Security, pensions, 401(k), IRA).
    """
    retirement_node = evaluator.add_parallel(
        id="Retirement_Income_State_Income_Tax",
        desc="Verification of whether Tennessee has a state income tax on retirement income (incl. Social Security, pensions, 401(k), IRA).",
        parent=parent_node,
        critical=True
    )

    # 1) Clear yes/no conclusion that Tennessee has NO state income tax on retirement income (answer content check)
    concl_leaf = evaluator.add_leaf(
        id="Retirement_Income_Tax_Correct_YesNo_Conclusion",
        desc="Gives a clear yes/no conclusion that Tennessee has NO state income tax on retirement income.",
        parent=retirement_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly concludes that Tennessee has NO state income tax on retirement income.",
        node=concl_leaf,
        additional_instruction="Judge only the answer text: accept synonymous phrasing such as 'no state income tax', 'Tennessee does not tax wages or retirement income', or mention that the Hall income tax was repealed and thus there is no current state income tax. Do not check external sources for this leaf."
    )

    # 2) Explicitly addresses the listed retirement income types (answer content check)
    addr = extraction.retirement_income.addressed or RetirementTypesAddressed()
    addresses_all = _all_true([
        addr.mentions_social_security,
        addr.mentions_pensions,
        addr.mentions_401k,
        addr.mentions_ira
    ])
    evaluator.add_custom_node(
        result=addresses_all,
        id="Retirement_Income_Tax_Addresses_Listed_Retirement_Types",
        desc="Explicitly addresses the listed retirement income types: Social Security benefits, pensions, 401(k) distributions, and IRA distributions.",
        parent=retirement_node,
        critical=True
    )

    # 3) Supporting sources (evidence-grounded)
    sources_leaf = evaluator.add_leaf(
        id="Retirement_Income_Tax_Supporting_Sources",
        desc="Includes supporting information from official or otherwise reputable sources for the retirement-income tax claim.",
        parent=retirement_node,
        critical=True
    )
    ret_sources = extraction.retirement_income.sources or []
    if not ret_sources:
        # No sources provided in the answer -> fail this critical leaf
        sources_leaf.score = 0.0
        sources_leaf.status = "failed"
    else:
        retirement_claim = (
            "Tennessee does not impose a state income tax on individual income, so the state does not tax retirement "
            "income (including Social Security benefits, pensions, 401(k), or IRA distributions)."
        )
        await evaluator.verify(
            claim=retirement_claim,
            node=sources_leaf,
            sources=ret_sources,
            additional_instruction="Verify that at least one provided source supports this claim. Prefer official sources (e.g., tn.gov, Department of Revenue) or widely reputable sources (e.g., major news or recognized tax policy orgs). If any provided URL clearly supports the claim, pass."
        )


async def build_property_tax_verification(
    evaluator: Evaluator,
    parent_node,
    extraction: TaxPolicyExtraction
) -> None:
    """
    Build and verify the subtree for:
    Verification of whether Tennessee offers property tax relief or freeze programs specifically for senior homeowners age 65+.
    """
    property_node = evaluator.add_parallel(
        id="Senior_Property_Tax_Relief_or_Freeze_65plus",
        desc="Verification of whether Tennessee offers property tax relief or freeze programs specifically for senior homeowners age 65+.",
        parent=parent_node,
        critical=True
    )

    # 1) Clear yes/no conclusion that Tennessee DOES offer some property tax relief/freeze program(s) (answer content check)
    concl_leaf = evaluator.add_leaf(
        id="Senior_Property_Tax_Programs_Correct_YesNo_Conclusion",
        desc="Gives a clear yes/no conclusion that Tennessee DOES offer some property tax relief and/or freeze program(s).",
        parent=property_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer clearly concludes that Tennessee DOES offer some property tax relief and/or property tax freeze program(s) for homeowners.",
        node=concl_leaf,
        additional_instruction="Judge only the answer text for a clear 'Yes' conclusion. Accept synonymous phrasing like 'Tennessee offers property tax relief for seniors' or 'there is a property tax freeze program'. Do not check external sources for this leaf."
    )

    # 2) States that the programs are specifically available to senior homeowners age 65+ (answer content check)
    mentions_65 = bool(extraction.senior_property_tax.mentions_age_65_plus)
    evaluator.add_custom_node(
        result=mentions_65,
        id="Senior_Property_Tax_Programs_Are_Specific_to_65plus_Seniors",
        desc="States that the property tax relief/freeze program(s) are specifically available to senior homeowners age 65 and older (not only general programs).",
        parent=property_node,
        critical=True
    )

    # 3) Supporting sources (evidence-grounded)
    sources_leaf = evaluator.add_leaf(
        id="Senior_Property_Tax_Programs_Supporting_Sources",
        desc="Includes supporting information from official or otherwise reputable sources for the senior property-tax relief/freeze program claim.",
        parent=property_node,
        critical=True
    )
    prop_sources = extraction.senior_property_tax.sources or []
    if not prop_sources:
        # No sources provided in the answer -> fail this critical leaf
        sources_leaf.score = 0.0
        sources_leaf.status = "failed"
    else:
        property_claim = (
            "Tennessee offers property tax relief and/or property tax freeze program(s) specifically available "
            "to senior homeowners age 65 or older."
        )
        await evaluator.verify(
            claim=property_claim,
            node=sources_leaf,
            sources=prop_sources,
            additional_instruction="Verify that at least one provided source supports this claim. Prefer official TN government sources (e.g., tn.gov, comptroller.tn.gov) or county assessor/treasurer sites; otherwise accept highly reputable sources. The source should indicate eligibility for seniors (age 65+)."
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
    Evaluate an answer for Tennessee retirement-related tax policies.
    """
    # Initialize evaluator
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_tax_policy(),
        template_class=TaxPolicyExtraction,
        extraction_name="tax_policy_extraction"
    )

    # Rubric root (critical)
    rubric_root = evaluator.add_parallel(
        id="Tennessee_Retirement_Tax_Verification",
        desc="Evaluate whether the response correctly verifies Tennessee retirement-related tax policies and supports each conclusion with reputable/official sources.",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_retirement_income_verification(evaluator, rubric_root, extraction)
    await build_property_tax_verification(evaluator, rubric_root, extraction)

    # Return evaluation summary
    return evaluator.get_summary()