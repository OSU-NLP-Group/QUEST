import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "fashion_awards_2024_designers"
TASK_DESCRIPTION = """
Based on major fashion awards ceremonies held in 2024, identify the winners of four specific designer awards and provide detailed information about each winner according to the following requirements:

Award Category 1: CFDA American Womenswear Designer of the Year 2024
Award Category 2: CFDA American Menswear Designer of the Year 2024
Award Category 3: CFDA American Accessory Designer of the Year 2024
Award Category 4: British Fashion Awards Designer of the Year 2024

For each piece of information provided, include at least one reference URL that supports your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SourcedText(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SourcedList(BaseModel):
    items: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class AwardWomenswear(BaseModel):
    # Winner identity + supporting URLs
    winner: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    # Founder/designer identity for nationality/background context (optional)
    founder_or_designer_name: Optional[str] = None
    founder_background: Optional[SourcedText] = None  # nationality or cultural background

    # Brand context
    brand_name: Optional[str] = None
    brand_founding_year: Optional[SourcedText] = None

    # Additional CFDA awards/recognitions in 2024
    additional_cfda_awards_2024: Optional[SourcedList] = None


class AwardMenswear(BaseModel):
    # Winner identity + supporting URLs
    winner: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    # Previous wins of the same CFDA award (years list)
    previous_menswear_award_years: Optional[SourcedList] = None

    # NYFW September 2024 participation confirmation
    nyfw_sep_2024_participation: Optional[SourcedText] = None  # e.g., "Yes, presented SS25 at NYFW" or "No"

    # Cultural or social themes addressed by designer's work
    cultural_or_social_themes: Optional[SourcedList] = None


class AwardAccessory(BaseModel):
    # Winner identity + supporting URLs
    winner: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    # Signature accessory product
    signature_accessory: Optional[SourcedText] = None

    # Whether won the same award in 2022
    won_in_2022: Optional[SourcedText] = None  # e.g., "Yes (won in 2022)" or "No"

    # New product category launched/announced in late 2024
    new_product_category_late_2024: Optional[SourcedText] = None


class AwardBFA(BaseModel):
    # Winner identity + supporting URLs
    winner: Optional[str] = None
    winner_urls: List[str] = Field(default_factory=list)

    # All brands for which winner serves as creative director
    creative_director_brands: Optional[SourcedList] = None

    # Years won (Designer of the Year)
    years_won_designer_of_year: Optional[SourcedList] = None

    # PFW September/October 2024 shows under this designer's creative direction
    pfw_sep_oct_2024_shows: Optional[SourcedText] = None


class FashionAwardsExtraction(BaseModel):
    womenswear: Optional[AwardWomenswear] = None
    menswear: Optional[AwardMenswear] = None
    accessory: Optional[AwardAccessory] = None
    bfa: Optional[AwardBFA] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_fashion_awards() -> str:
    return """
Extract structured information from the answer about four 2024 fashion designer awards. Populate the JSON exactly according to the following schema and instructions. Only extract information explicitly present in the answer; do not invent. For each attribute that requires citation, extract all URLs (if any) the answer provides for that attribute.

JSON schema (keys must match exactly):
{
  "womenswear": {
    "winner": string or null,
    "winner_urls": [urls...],
    "founder_or_designer_name": string or null,
    "founder_background": {"text": string or null, "urls": [urls...]},
    "brand_name": string or null,
    "brand_founding_year": {"text": string or null, "urls": [urls...]},
    "additional_cfda_awards_2024": {"items": [strings...], "urls": [urls...]}
  },
  "menswear": {
    "winner": string or null,
    "winner_urls": [urls...],
    "previous_menswear_award_years": {"items": [strings...], "urls": [urls...]},
    "nyfw_sep_2024_participation": {"text": string or null, "urls": [urls...]},
    "cultural_or_social_themes": {"items": [strings...], "urls": [urls...]}
  },
  "accessory": {
    "winner": string or null,
    "winner_urls": [urls...],
    "signature_accessory": {"text": string or null, "urls": [urls...]},
    "won_in_2022": {"text": string or null, "urls": [urls...]},
    "new_product_category_late_2024": {"text": string or null, "urls": [urls...]}
  },
  "bfa": {
    "winner": string or null,
    "winner_urls": [urls...],
    "creative_director_brands": {"items": [strings...], "urls": [urls...]},
    "years_won_designer_of_year": {"items": [strings...], "urls": [urls...]},
    "pfw_sep_oct_2024_shows": {"text": string or null, "urls": [urls...]}
  }
}

Specific instructions:
- Award Category 1 (CFDA American Womenswear Designer of the Year 2024, Oct 28, 2024):
  • womenswear.winner: designer/brand winner name.
  • womenswear.winner_urls: all URLs cited in the answer supporting the winner identity (prefer official CFDA or equivalent official announcements).
  • womenswear.founder_or_designer_name: founder/designer name if mentioned.
  • womenswear.founder_background: nationality/cultural background text + all URLs cited for it.
  • womenswear.brand_name: brand name (if winner is a brand, or synonymous with designer).
  • womenswear.brand_founding_year: founding year text + all URLs cited for it.
  • womenswear.additional_cfda_awards_2024: list all CFDA awards/recognitions in 2024 mentioned for this winner + all URLs cited.

- Award Category 2 (CFDA American Menswear Designer of the Year 2024, Oct 28, 2024):
  • menswear.winner and menswear.winner_urls as above.
  • menswear.previous_menswear_award_years: list of years (strings) the winner previously won the same award; if the answer states none, leave items empty but still include any URLs cited that support this.
  • menswear.nyfw_sep_2024_participation: short text stating whether they showed a collection during NYFW Sep 2024 + all cited URLs.
  • menswear.cultural_or_social_themes: list of themes (strings) + all cited URLs.

- Award Category 3 (CFDA American Accessory Designer of the Year 2024, Oct 28, 2024):
  • accessory.winner and accessory.winner_urls as above.
  • accessory.signature_accessory: the most iconic/signature accessory product + URLs cited.
  • accessory.won_in_2022: text explicitly stating whether they won this award in 2022 (e.g., "Yes (won in 2022)" or "No") + URLs cited.
  • accessory.new_product_category_late_2024: text describing any new product category launched/announced in late 2024 + URLs cited.

- Award Category 4 (British Fashion Awards Designer of the Year 2024, Dec 2, 2024, London):
  • bfa.winner and bfa.winner_urls as above (prefer official BFC/Fashion Awards links or equivalent official announcements).
  • bfa.creative_director_brands: list all fashion brands for which the winner serves as creative director + URLs cited.
  • bfa.years_won_designer_of_year: list of years (strings) the winner won Designer of the Year; include 2023 if mentioned to evaluate consecutive wins + URLs cited.
  • bfa.pfw_sep_oct_2024_shows: short text confirming that collections under the winner's creative direction were shown at Paris Fashion Week in Sep/Oct 2024 + URLs cited.

General URL rules:
- Extract only URLs explicitly present in the answer (including markdown links).
- If a field has no cited URLs in the answer, set its urls array to [].
- If a text value is not present, set it to null; if a list is not present, use an empty list.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(lst: Optional[List[str]]) -> List[str]:
    return lst if lst else []


def _safe_text(st: Optional[SourcedText]) -> str:
    return st.text if (st and st.text) else ""


def _safe_urls(st: Optional[SourcedText | SourcedList]) -> List[str]:
    if st is None:
        return []
    if isinstance(st, SourcedText):
        return _safe_list(st.urls)
    if isinstance(st, SourcedList):
        return _safe_list(st.urls)
    return []


def _add_sources_presence_check(
    evaluator: Evaluator,
    parent: VerificationNode,
    base_id: str,
    human_desc: str,
    urls: List[str],
) -> VerificationNode:
    return evaluator.add_custom_node(
        result=bool(urls),
        id=f"{base_id}_sources_present",
        desc=f"At least one reference URL is provided for {human_desc}",
        parent=parent,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Verification builders for each award category                               #
# --------------------------------------------------------------------------- #
async def build_womenswear_checks(evaluator: Evaluator, parent: VerificationNode, ww: Optional[AwardWomenswear]) -> None:
    award_node = evaluator.add_parallel(
        id="award_1_cfda_womenswear_2024",
        desc="CFDA American Womenswear Designer of the Year 2024 (Oct 28, 2024): winner + required attributes with citations.",
        parent=parent,
        critical=False
    )

    # 1) Winner identity with official citation
    winner_urls = _safe_list(ww.winner_urls if ww else [])
    winner_name = ww.winner if ww and ww.winner else ""
    sp1 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="womenswear_winner_identity",
        human_desc="CFDA Womenswear winner identity",
        urls=winner_urls
    )
    node1 = evaluator.add_leaf(
        id="womenswear_winner_identity_with_official_citation",
        desc="Correctly identifies the CFDA American Womenswear Designer of the Year 2024 winner (Oct 28, 2024) with supporting official reference URL(s).",
        parent=award_node,
        critical=True
    )
    claim1 = f"The winner of the CFDA American Womenswear Designer of the Year 2024 (announced October 28, 2024) is '{winner_name}'."
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=winner_urls,
        extra_prerequisites=[sp1],
        additional_instruction="Accept only if at least one provided webpage explicitly states this winner. Prefer CFDA official announcements or well-established fashion publications."
    )

    # 2) Founder nationality/cultural background with citation
    founder_name = ww.founder_or_designer_name if ww and ww.founder_or_designer_name else (winner_name or "the winner/founder")
    bg_text = _safe_text(ww.founder_background if ww else None)
    bg_urls = _safe_urls(ww.founder_background if ww else None)
    sp2 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="womenswear_founder_background",
        human_desc="founder/designer nationality or cultural background",
        urls=bg_urls
    )
    node2 = evaluator.add_leaf(
        id="womenswear_founder_nationality_or_cultural_background_with_citation",
        desc="Provides the nationality or cultural background of the founder/designer with a credible reference URL.",
        parent=award_node,
        critical=True
    )
    claim2 = f"The nationality/cultural background of {founder_name} is: {bg_text}."
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=bg_urls,
        extra_prerequisites=[sp2],
        additional_instruction="Judge supported only if the provided URL(s) explicitly confirm this person's nationality or cultural background."
    )

    # 3) Brand founding year with citation
    brand_name = ww.brand_name if ww and ww.brand_name else (winner_name or "the brand")
    fy_text = _safe_text(ww.brand_founding_year if ww else None)
    fy_urls = _safe_urls(ww.brand_founding_year if ww else None)
    sp3 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="womenswear_brand_founding_year",
        human_desc="brand founding year",
        urls=fy_urls
    )
    node3 = evaluator.add_leaf(
        id="womenswear_brand_founding_year_with_citation",
        desc="Provides the brand founding year with a credible reference URL.",
        parent=award_node,
        critical=True
    )
    claim3 = f"The brand '{brand_name}' was founded/established in {fy_text}."
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=fy_urls,
        extra_prerequisites=[sp3],
        additional_instruction="Support only if the provided URL(s) clearly state the brand's founding year."
    )

    # 4) Additional CFDA awards/recognitions in 2024 with citation
    add_awards_items = (ww.additional_cfda_awards_2024.items if ww and ww.additional_cfda_awards_2024 else [])
    add_awards_urls = _safe_urls(ww.additional_cfda_awards_2024 if ww else None)
    sp4 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="womenswear_additional_cfda_awards_2024",
        human_desc="additional CFDA awards/recognitions in 2024",
        urls=add_awards_urls
    )
    node4 = evaluator.add_leaf(
        id="womenswear_additional_cfda_awards_2024_with_citation",
        desc="Identifies any additional CFDA awards/recognitions in 2024 with citation (or explicitly 'none' with citation).",
        parent=award_node,
        critical=True
    )
    human_list = ", ".join(add_awards_items) if add_awards_items else "none"
    claim4 = f"In 2024, '{winner_name}' also received the following CFDA awards/recognitions: {human_list}."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=add_awards_urls,
        extra_prerequisites=[sp4],
        additional_instruction="Support only if the provided URL(s) explicitly confirm the listed 2024 CFDA awards/recognitions (or confirm none if that's the claim)."
    )


async def build_menswear_checks(evaluator: Evaluator, parent: VerificationNode, mw: Optional[AwardMenswear]) -> None:
    award_node = evaluator.add_parallel(
        id="award_2_cfda_menswear_2024",
        desc="CFDA American Menswear Designer of the Year 2024 (Oct 28, 2024): winner + required attributes with citations.",
        parent=parent,
        critical=False
    )

    # 1) Winner identity with official citation
    winner_urls = _safe_list(mw.winner_urls if mw else [])
    winner_name = mw.winner if mw and mw.winner else ""
    sp1 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="menswear_winner_identity",
        human_desc="CFDA Menswear winner identity",
        urls=winner_urls
    )
    node1 = evaluator.add_leaf(
        id="menswear_winner_identity_with_official_citation",
        desc="Correctly identifies the CFDA American Menswear Designer of the Year 2024 winner with official/credible citation.",
        parent=award_node,
        critical=True
    )
    claim1 = f"The winner of the CFDA American Menswear Designer of the Year 2024 (announced October 28, 2024) is '{winner_name}'."
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=winner_urls,
        extra_prerequisites=[sp1],
        additional_instruction="Accept only if at least one provided webpage explicitly states this winner."
    )

    # 2) Previous wins of the same award with citation
    prev_years_items = (mw.previous_menswear_award_years.items if mw and mw.previous_menswear_award_years else [])
    prev_years_urls = _safe_urls(mw.previous_menswear_award_years if mw else None)
    sp2 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="menswear_previous_win_same_award",
        human_desc="previous wins (years) of CFDA American Menswear Designer of the Year",
        urls=prev_years_urls
    )
    node2 = evaluator.add_leaf(
        id="menswear_previous_win_same_award_with_citation",
        desc="States whether the winner previously won the same CFDA award (years) with citation.",
        parent=award_node,
        critical=True
    )
    human_prev = ", ".join(prev_years_items) if prev_years_items else "none"
    claim2 = f"Prior to 2024, '{winner_name}' has previously won the CFDA American Menswear Designer of the Year in the following years: {human_prev}."
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=prev_years_urls,
        extra_prerequisites=[sp2],
        additional_instruction="Support only if the provided URL(s) explicitly list the same award wins and years for this designer."
    )

    # 3) NYFW September 2024 participation with citation
    nyfw_text = _safe_text(mw.nyfw_sep_2024_participation if mw else None)
    nyfw_urls = _safe_urls(mw.nyfw_sep_2024_participation if mw else None)
    sp3 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="menswear_nyfw_sep_2024_participation",
        human_desc="NYFW Sep 2024 participation",
        urls=nyfw_urls
    )
    node3 = evaluator.add_leaf(
        id="menswear_nyfw_sep_2024_participation_with_citation",
        desc="Confirms whether the winner showed a collection during NYFW Sep 2024 with citation.",
        parent=award_node,
        critical=True
    )
    normalized = (nyfw_text or "").strip().lower()
    if any(tok in normalized for tok in ["no", "did not", "didn't", "absent", "not show"]):
        claim3 = f"'{winner_name}' did not show a collection during New York Fashion Week in September 2024."
    else:
        claim3 = f"'{winner_name}' showed a collection during New York Fashion Week in September 2024."
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=nyfw_urls,
        extra_prerequisites=[sp3],
        additional_instruction="Support only if the provided URL(s) explicitly confirm the presence or absence of a NYFW Sep 2024 show for the winner."
    )

    # 4) Cultural/social themes with citation
    themes_items = (mw.cultural_or_social_themes.items if mw and mw.cultural_or_social_themes else [])
    themes_urls = _safe_urls(mw.cultural_or_social_themes if mw else None)
    sp4 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="menswear_cultural_or_social_themes",
        human_desc="cultural or social themes addressed by the designer's work",
        urls=themes_urls
    )
    node4 = evaluator.add_leaf(
        id="menswear_cultural_or_social_themes_with_citation",
        desc="Describes the cultural or social themes the designer's work addresses with citation.",
        parent=award_node,
        critical=True
    )
    human_themes = ", ".join(themes_items) if themes_items else "none"
    claim4 = f"The designer's work prominently addresses the following cultural/social themes: {human_themes}."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=themes_urls,
        extra_prerequisites=[sp4],
        additional_instruction="Support only if the provided URL(s) explicitly discuss these themes in the designer's work."
    )


async def build_accessory_checks(evaluator: Evaluator, parent: VerificationNode, acc: Optional[AwardAccessory]) -> None:
    award_node = evaluator.add_parallel(
        id="award_3_cfda_accessory_2024",
        desc="CFDA American Accessory Designer of the Year 2024 (Oct 28, 2024): winner + required attributes with citations.",
        parent=parent,
        critical=False
    )

    # 1) Winner identity with official citation
    winner_urls = _safe_list(acc.winner_urls if acc else [])
    winner_name = acc.winner if acc and acc.winner else ""
    sp1 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="accessory_winner_identity",
        human_desc="CFDA Accessory winner identity",
        urls=winner_urls
    )
    node1 = evaluator.add_leaf(
        id="accessory_winner_identity_with_official_citation",
        desc="Correctly identifies the CFDA American Accessory Designer of the Year 2024 winner with citation.",
        parent=award_node,
        critical=True
    )
    claim1 = f"The winner of the CFDA American Accessory Designer of the Year 2024 (announced October 28, 2024) is '{winner_name}'."
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=winner_urls,
        extra_prerequisites=[sp1],
        additional_instruction="Accept only if at least one provided webpage explicitly states this winner."
    )

    # 2) Signature accessory product with citation
    sig_text = _safe_text(acc.signature_accessory if acc else None)
    sig_urls = _safe_urls(acc.signature_accessory if acc else None)
    sp2 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="accessory_signature_product",
        human_desc="signature/most iconic accessory product",
        urls=sig_urls
    )
    node2 = evaluator.add_leaf(
        id="accessory_signature_product_with_citation",
        desc="Identifies the brand/designer's most iconic or signature accessory product with citation.",
        parent=award_node,
        critical=True
    )
    claim2 = f"The brand/designer's most iconic or signature accessory product is '{sig_text}'."
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=sig_urls,
        extra_prerequisites=[sp2],
        additional_instruction="Support only if the provided URL(s) explicitly identify this product as the signature/iconic accessory."
    )

    # 3) Won CFDA Accessory award in 2022 (yes/no) with citation
    won2022_text = _safe_text(acc.won_in_2022 if acc else None)
    won2022_urls = _safe_urls(acc.won_in_2022 if acc else None)
    sp3 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="accessory_won_2022",
        human_desc="confirmation whether the same award was won in 2022",
        urls=won2022_urls
    )
    node3 = evaluator.add_leaf(
        id="accessory_won_cfda_accessory_award_2022_with_citation",
        desc="States whether the same brand/designer won the CFDA American Accessory Designer award in 2022 with citation.",
        parent=award_node,
        critical=True
    )
    normalized = (won2022_text or "").strip().lower()
    if any(tok in normalized for tok in ["yes", "won", "recipient", "winner"]):
        claim3 = f"'{winner_name}' won the CFDA American Accessory Designer of the Year in 2022."
    elif any(tok in normalized for tok in ["no", "did not", "didn't", "not win"]):
        claim3 = f"'{winner_name}' did not win the CFDA American Accessory Designer of the Year in 2022."
    else:
        # Default to positive phrasing unless the text clearly negates; adjust as needed by URLs
        claim3 = f"'{winner_name}' won the CFDA American Accessory Designer of the Year in 2022."
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=won2022_urls,
        extra_prerequisites=[sp3],
        additional_instruction="Support only if the provided URL(s) explicitly confirm (or refute) the 2022 win status."
    )

    # 4) New product category in late 2024 with citation
    npc_text = _safe_text(acc.new_product_category_late_2024 if acc else None)
    npc_urls = _safe_urls(acc.new_product_category_late_2024 if acc else None)
    sp4 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="accessory_new_product_category_late_2024",
        human_desc="new product category launched/announced in late 2024",
        urls=npc_urls
    )
    node4 = evaluator.add_leaf(
        id="accessory_new_product_category_late_2024_with_citation",
        desc="Identifies any new product category the brand launched/announced in late 2024 with citation (or explicitly none with citation).",
        parent=award_node,
        critical=True
    )
    claim4 = f"In late 2024, '{winner_name}' launched/announced a new product category: {npc_text if npc_text else 'none'}."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=npc_urls,
        extra_prerequisites=[sp4],
        additional_instruction="Support only if the provided URL(s) explicitly confirm the new product category (or confirm none, if that's the claim)."
    )


async def build_bfa_checks(evaluator: Evaluator, parent: VerificationNode, bfa: Optional[AwardBFA]) -> None:
    award_node = evaluator.add_parallel(
        id="award_4_british_fashion_awards_designer_of_year_2024",
        desc="British Fashion Awards Designer of the Year 2024 (Dec 2, 2024, London): winner + required attributes with citations.",
        parent=parent,
        critical=False
    )

    # 1) Winner identity with official citation
    winner_urls = _safe_list(bfa.winner_urls if bfa else [])
    winner_name = bfa.winner if bfa and bfa.winner else ""
    sp1 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="bfa_designer_winner_identity",
        human_desc="BFA Designer of the Year 2024 winner identity",
        urls=winner_urls
    )
    node1 = evaluator.add_leaf(
        id="bfa_designer_winner_identity_with_official_citation",
        desc="Correctly identifies the BFA Designer of the Year 2024 winner with official/credible citation.",
        parent=award_node,
        critical=True
    )
    claim1 = f"The winner of the British Fashion Awards Designer of the Year 2024 (December 2, 2024, London) is '{winner_name}'."
    await evaluator.verify(
        claim=claim1,
        node=node1,
        sources=winner_urls,
        extra_prerequisites=[sp1],
        additional_instruction="Accept only if at least one provided webpage (prefer BFC/Fashion Awards) explicitly states this winner."
    )

    # 2) All creative director brands with citation
    cdb_items = (bfa.creative_director_brands.items if bfa and bfa.creative_director_brands else [])
    cdb_urls = _safe_urls(bfa.creative_director_brands if bfa else None)
    sp2 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="bfa_all_creative_director_brands",
        human_desc="list of all brands for which the winner serves as creative director",
        urls=cdb_urls
    )
    node2 = evaluator.add_leaf(
        id="bfa_all_creative_director_brands_with_citation",
        desc="Lists all fashion brands for which the winner serves as creative director with citation.",
        parent=award_node,
        critical=True
    )
    human_brands = ", ".join(cdb_items) if cdb_items else "none"
    claim2 = f"'{winner_name}' serves as creative director for the following fashion brands: {human_brands}."
    await evaluator.verify(
        claim=claim2,
        node=node2,
        sources=cdb_urls,
        extra_prerequisites=[sp2],
        additional_instruction="Support only if the provided URL(s) explicitly confirm all listed creative director roles."
    )

    # 3) Consecutive wins 2023 and 2024 with citation
    years_items = (bfa.years_won_designer_of_year.items if bfa and bfa.years_won_designer_of_year else [])
    years_urls = _safe_urls(bfa.years_won_designer_of_year if bfa else None)
    sp3 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="bfa_consecutive_wins",
        human_desc="consecutive 'Designer of the Year' wins (2023 and 2024)",
        urls=years_urls
    )
    node3 = evaluator.add_leaf(
        id="bfa_consecutive_wins_2023_and_2024_with_citation",
        desc="States whether the winner won 'Designer of the Year' in both 2023 and 2024 with citation.",
        parent=award_node,
        critical=True
    )
    has_2023 = any(y.strip() == "2023" for y in years_items) if years_items else False
    has_2024 = any(y.strip() == "2024" for y in years_items) if years_items else False
    if has_2023 and has_2024:
        claim3 = f"'{winner_name}' won the British Fashion Awards 'Designer of the Year' in consecutive years 2023 and 2024."
    else:
        claim3 = f"'{winner_name}' did not win the British Fashion Awards 'Designer of the Year' in both 2023 and 2024."
    await evaluator.verify(
        claim=claim3,
        node=node3,
        sources=years_urls,
        extra_prerequisites=[sp3],
        additional_instruction="Support only if the provided URL(s) clearly list the relevant winning years for this category."
    )

    # 4) Paris Fashion Week Sep/Oct 2024 shown with citation
    pfw_text = _safe_text(bfa.pfw_sep_oct_2024_shows if bfa else None)
    pfw_urls = _safe_urls(bfa.pfw_sep_oct_2024_shows if bfa else None)
    sp4 = _add_sources_presence_check(
        evaluator, award_node,
        base_id="bfa_paris_fashion_week_sep_oct_2024_shown",
        human_desc="PFW Sep/Oct 2024 showings under the winner's creative direction",
        urls=pfw_urls
    )
    node4 = evaluator.add_leaf(
        id="bfa_paris_fashion_week_sep_oct_2024_shown_with_citation",
        desc="Confirms that collections under the winner's creative direction were shown at PFW Sep/Oct 2024 with citation.",
        parent=award_node,
        critical=True
    )
    normalized_pfw = (pfw_text or "").strip().lower()
    if any(tok in normalized_pfw for tok in ["no", "did not", "didn't", "absent", "not show"]):
        claim4 = f"Collections under '{winner_name}'s creative direction were not shown during Paris Fashion Week in September/October 2024."
    else:
        claim4 = f"Collections under '{winner_name}'s creative direction were shown during Paris Fashion Week in September/October 2024."
    await evaluator.verify(
        claim=claim4,
        node=node4,
        sources=pfw_urls,
        extra_prerequisites=[sp4],
        additional_instruction="Support only if the provided URL(s) explicitly confirm PFW Sep/Oct 2024 showings tied to this designer's creative direction."
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
    Evaluate an answer for 2024 fashion designer awards identifications and attributes.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Awards are independent; allow partial credit across categories
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify the winners for the four specified 2024 fashion designer awards and provide all required attributes for each winner with supporting reference URL(s).",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_fashion_awards(),
        template_class=FashionAwardsExtraction,
        extraction_name="fashion_awards_2024_extraction"
    )

    # Build subtrees for each award
    await build_womenswear_checks(evaluator, root, extraction.womenswear)
    await build_menswear_checks(evaluator, root, extraction.menswear)
    await build_accessory_checks(evaluator, root, extraction.accessory)
    await build_bfa_checks(evaluator, root, extraction.bfa)

    # Return evaluator's summary
    return evaluator.get_summary()