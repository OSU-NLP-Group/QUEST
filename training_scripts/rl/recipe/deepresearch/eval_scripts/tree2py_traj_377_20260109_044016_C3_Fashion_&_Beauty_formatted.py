import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_fashion_brand_multi_cert"
TASK_DESCRIPTION = """Identify a United States-based fashion apparel brand that meets all of the following criteria:

1. The brand must be a Certified B Corporation with a verified B Impact Assessment score of at least 100 points out of 200.

2. The brand must have Fair Trade certification, with at least 80% of its product line manufactured in Fair Trade Certified factories.

3. The brand must use organic cotton that is certified to the Global Organic Textile Standard (GOTS) in its cotton products.

Provide the following information:
- The brand name
- The brand's B Corp score
- The percentage of its products that are Fair Trade Certified
- URL references that document: (a) the brand's B Corp certification and score, (b) the Fair Trade certification percentage, and (c) the GOTS-certified organic cotton usage.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BrandCertificationExtraction(BaseModel):
    # Identity and scope
    brand_name: Optional[str] = None
    us_based_statement: Optional[str] = None
    apparel_focus_statement: Optional[str] = None

    # B Corp
    bcorp_certified_statement: Optional[str] = None
    bcorp_score: Optional[str] = None
    bcorp_urls: List[str] = Field(default_factory=list)

    # Fair Trade
    fair_trade_certified_statement: Optional[str] = None
    fair_trade_percentage: Optional[str] = None
    fair_trade_urls: List[str] = Field(default_factory=list)

    # GOTS (organic cotton)
    gots_statement: Optional[str] = None
    gots_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_brand_multi_cert() -> str:
    return """
Extract the following fields exactly as presented in the answer. Do not invent any values.

Required fields:
- brand_name: The brand's name.
- us_based_statement: The exact phrase or sentence from the answer that indicates the brand is based in the United States (e.g., mentions like “U.S.-based”, “United States”, “USA”, HQ in a U.S. state). If not explicitly stated, set to null.
- apparel_focus_statement: The exact phrase or sentence indicating the brand is primarily a fashion apparel/clothing brand. If not explicitly stated, set to null.

B Corp:
- bcorp_certified_statement: Exact phrase/sentence claiming the brand is a Certified B Corporation. If not explicitly stated, set to null.
- bcorp_score: The numeric B Impact Assessment score as written (e.g., "108.5", "100", "102.3"). Return only the number text if possible; otherwise return the exact text given. If missing, set to null.
- bcorp_urls: All URLs provided that substantiate B Corp certification and score. Return an array. If none are provided, return [].

Fair Trade:
- fair_trade_certified_statement: Exact phrase/sentence claiming the brand has Fair Trade certification or manufactures in Fair Trade Certified factories. If not explicitly stated, set to null.
- fair_trade_percentage: The percentage of the product line manufactured in Fair Trade Certified factories as written (e.g., "85%", "80 percent", "over 90%"). If missing, set to null.
- fair_trade_urls: All URLs provided that substantiate the Fair Trade coverage percentage. Return an array. If none are provided, return [].

GOTS (organic cotton):
- gots_statement: Exact phrase/sentence that the brand uses organic cotton certified to GOTS in its cotton products. If not explicitly stated, set to null.
- gots_urls: All URLs provided that substantiate GOTS-certified organic cotton usage. Return an array. If none are provided, return [].

General rules:
- Extract only what is explicitly present in the answer text.
- For URLs, include full URLs actually present in the answer (including those in markdown links).
- If a field is not explicitly provided, return null (or [] for url lists).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _first_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _first_percentage_value(text: Optional[str]) -> Optional[float]:
    # Accept formats like "85%", "80 percent", "at least 90%", "90+%"
    if not text:
        return None
    text_norm = text.lower().replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(%|percent)", text_norm)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Fallback: look for any number if symbol not present
    val = _first_number(text_norm)
    return val


def _non_empty_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _safe_brand(brand: Optional[str]) -> str:
    return brand.strip() if isinstance(brand, str) and brand.strip() else "the brand"


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, data: BrandCertificationExtraction) -> None:
    brand = _safe_brand(data.brand_name)

    # ---------------- Brand Eligibility ----------------
    brand_node = evaluator.add_parallel(
        id="Brand_Eligibility",
        desc="Brand satisfies basic identity constraints (name, U.S.-based, apparel-focused).",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.brand_name and data.brand_name.strip()),
        id="Brand_Name_Provided",
        desc="Provide the brand name.",
        parent=brand_node,
        critical=True
    )

    us_based_leaf = evaluator.add_leaf(
        id="US_Based_Status",
        desc="Brand is based in the United States.",
        parent=brand_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{brand} is based in the United States (USA/U.S.).",
        node=us_based_leaf,
        additional_instruction="Judge only based on the provided answer text. Accept evidence like HQ or principal office in a U.S. state/city."
    )

    apparel_focus_leaf = evaluator.add_leaf(
        id="Apparel_Focus",
        desc="Brand is primarily a fashion apparel/clothing company.",
        parent=brand_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{brand} is primarily a fashion apparel/clothing brand (focuses on garments/apparel).",
        node=apparel_focus_leaf,
        additional_instruction="Judge only from the answer text; allow synonyms like apparel, clothing, garments."
    )

    # ---------------- B Corp Requirements ----------------
    bcorp_node = evaluator.add_parallel(
        id="B_Corp_Requirements",
        desc="Verify B Corp certification and the required minimum B Impact score, and provide supporting URL(s).",
        parent=root,
        critical=True
    )

    # URL existence (gating for URL-based checks)
    bcorp_urls = _non_empty_urls(data.bcorp_urls)
    bcorp_url_provided = evaluator.add_custom_node(
        result=len(bcorp_urls) > 0,
        id="B_Corp_Certification_And_Score_URL_Provided",
        desc="B Corp certification/score URL(s) provided.",
        parent=bcorp_node,
        critical=True
    )

    # Certified status (verify, ideally with URLs)
    bcorp_cert_leaf = evaluator.add_leaf(
        id="B_Corp_Certified_Status",
        desc="Brand is a Certified B Corporation.",
        parent=bcorp_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{brand} is a Certified B Corporation (B Corp).",
        node=bcorp_cert_leaf,
        sources=bcorp_urls if bcorp_urls else None,
        extra_prerequisites=[bcorp_url_provided],
        additional_instruction="If the provided URL(s) are irrelevant or do not support B Corp certification, judge as not supported."
    )

    # Score provided
    bscore_val = _first_number(data.bcorp_score)
    bscore_provided_leaf = evaluator.add_custom_node(
        result=bscore_val is not None,
        id="B_Corp_Score_Value_Provided",
        desc="Provide the brand's verified B Impact Assessment score (numeric value).",
        parent=bcorp_node,
        critical=True
    )

    # Score threshold
    evaluator.add_custom_node(
        result=(bscore_val is not None and bscore_val >= 100.0),
        id="B_Corp_Score_Minimum_Met",
        desc="B Impact Assessment score is at least 100 out of 200.",
        parent=bcorp_node,
        critical=True
    )

    # URLs support both certification and score
    bcorp_url_support_leaf = evaluator.add_leaf(
        id="B_Corp_Certification_And_Score_URL",
        desc="Provide URL reference(s) documenting the brand's B Corp certification and the score value.",
        parent=bcorp_node,
        critical=True
    )
    # Use the extracted numeric string (with tolerance for rounding)
    score_text = data.bcorp_score if data.bcorp_score else ""
    await evaluator.verify(
        claim=f"The provided webpage(s) show that {brand} is a Certified B Corporation and that its B Impact Assessment score is {score_text}.",
        node=bcorp_url_support_leaf,
        sources=bcorp_urls if bcorp_urls else None,
        extra_prerequisites=[bcorp_url_provided, bscore_provided_leaf],
        additional_instruction="Treat minor rounding differences (e.g., 100 vs 100.1) as acceptable. The page(s) must clearly indicate both the certification and the score."
    )

    # ---------------- Fair Trade Requirements ----------------
    ft_node = evaluator.add_parallel(
        id="Fair_Trade_Requirements",
        desc="Verify Fair Trade certification/coverage, provide the percentage value, and provide supporting URL(s).",
        parent=root,
        critical=True
    )

    # URL existence
    ft_urls = _non_empty_urls(data.fair_trade_urls)
    ft_url_provided = evaluator.add_custom_node(
        result=len(ft_urls) > 0,
        id="Fair_Trade_Percentage_URL_Provided",
        desc="Fair Trade coverage percentage URL(s) provided.",
        parent=ft_node,
        critical=True
    )

    # Certified status (or uses Fair Trade Certified factories)
    ft_cert_leaf = evaluator.add_leaf(
        id="Fair_Trade_Certified_Status",
        desc="Brand has Fair Trade certification (i.e., offers Fair Trade Certified products / uses Fair Trade Certified factories as claimed).",
        parent=ft_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{brand} is Fair Trade certified and/or manufactures in Fair Trade Certified factories.",
        node=ft_cert_leaf,
        sources=ft_urls if ft_urls else None,
        extra_prerequisites=[ft_url_provided],
        additional_instruction="The supporting page(s) should clearly indicate Fair Trade Certified products or manufacturing."
    )

    # Percentage provided and >= 80%
    ft_pct_val = _first_percentage_value(data.fair_trade_percentage)
    ft_pct_provided_leaf = evaluator.add_custom_node(
        result=ft_pct_val is not None,
        id="Fair_Trade_Percentage_Value_Provided",
        desc="Provide the percentage of the brand's product line manufactured in Fair Trade Certified factories.",
        parent=ft_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(ft_pct_val is not None and ft_pct_val >= 80.0),
        id="Fair_Trade_Coverage_Minimum_Met",
        desc="Fair Trade coverage is at least 80% of the product line manufactured in Fair Trade Certified factories.",
        parent=ft_node,
        critical=True
    )

    # URL(s) support the percentage claim
    ft_pct_url_leaf = evaluator.add_leaf(
        id="Fair_Trade_Percentage_URL",
        desc="Provide URL reference(s) documenting the Fair Trade coverage percentage.",
        parent=ft_node,
        critical=True
    )
    pct_text = data.fair_trade_percentage if data.fair_trade_percentage else ""
    await evaluator.verify(
        claim=f"The provided webpage(s) state that at least {pct_text} of {brand}'s product line is manufactured in Fair Trade Certified factories.",
        node=ft_pct_url_leaf,
        sources=ft_urls if ft_urls else None,
        extra_prerequisites=[ft_url_provided, ft_pct_provided_leaf],
        additional_instruction="If the page shows a slightly different phrasing (e.g., 'X% of products are made in Fair Trade Certified factories'), treat it as equivalent."
    )

    # ---------------- GOTS Organic Cotton Requirements ----------------
    gots_node = evaluator.add_parallel(
        id="GOTS_Organic_Cotton_Requirements",
        desc="Verify use of GOTS-certified organic cotton in cotton products and provide supporting URL(s).",
        parent=root,
        critical=True
    )

    gots_urls = _non_empty_urls(data.gots_urls)
    gots_url_provided = evaluator.add_custom_node(
        result=len(gots_urls) > 0,
        id="GOTS_Organic_Cotton_URL_Provided",
        desc="GOTS-certified organic cotton usage URL(s) provided.",
        parent=gots_node,
        critical=True
    )

    # Claim of usage (from answer text)
    gots_used_leaf = evaluator.add_leaf(
        id="GOTS_Certified_Organic_Cotton_Used",
        desc="Brand uses organic cotton that is certified to the Global Organic Textile Standard (GOTS) in its cotton products.",
        parent=gots_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{brand} uses GOTS-certified organic cotton in its cotton products.",
        node=gots_used_leaf,
        additional_instruction="Judge based on the answer text only. Accept equivalent phrases like 'GOTS certified' or 'Global Organic Textile Standard certified'."
    )

    # URLs support the GOTS usage claim
    gots_url_leaf = evaluator.add_leaf(
        id="GOTS_Organic_Cotton_URL",
        desc="Provide URL reference(s) documenting the brand's GOTS-certified organic cotton usage.",
        parent=gots_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided webpage(s) show that {brand} uses GOTS-certified organic cotton in its cotton products.",
        node=gots_url_leaf,
        sources=gots_urls if gots_urls else None,
        extra_prerequisites=[gots_url_provided],
        additional_instruction="Look for explicit mentions of 'GOTS certified' or 'Global Organic Textile Standard' tied to cotton products."
    )

    # Record some parsed values as custom info for transparency
    evaluator.add_custom_info(
        {
            "parsed_bcorp_score": bscore_val,
            "parsed_fair_trade_percentage": ft_pct_val,
            "bcorp_urls_count": len(bcorp_urls),
            "fair_trade_urls_count": len(ft_urls),
            "gots_urls_count": len(gots_urls),
        },
        info_type="parsed_numbers",
        info_name="parsed_numeric_overview"
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_brand_multi_cert(),
        template_class=BrandCertificationExtraction,
        extraction_name="brand_multi_cert_extraction"
    )

    await build_and_verify_tree(evaluator, root, extracted)

    return evaluator.get_summary()