import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "british_model_rose_inc_2026"
TASK_DESCRIPTION = """Identify the British model who meets ALL of the following criteria as of March 2026:

Personal & Basic Career Criteria:
- Born on April 18, 1987
- Height of exactly 5 feet 9 inches (175 cm)
- Started working with Victoria's Secret in early 2006
- Became a Victoria's Secret Angel in February 2010
- Left Victoria's Secret in 2011

Beauty Brand Venture:
- Founded an editorial site/beauty publication called Rose Inc in 2018
- Expanded Rose Inc to sell makeup and skincare products in August 2021
- Rose Inc products were sold at Sephora
- Rose Inc's joint venture partner was Amyris Inc
- Amyris Inc declared bankruptcy in August 2023
- Stepped down from Rose Inc in May 2024

Fashion Campaigns:
- Participated in a Burberry campaign in 2008
- Participated in the Burberry Body fragrance campaign in 2011
- Participated in the Burberry Festive campaign in 2015
- Participated in the Burberry High Summer 2025 campaign
- Participated in Mackage fashion campaigns during 2025-2026

Provide the model's full name and include reference URLs supporting each major aspect of her career (Victoria's Secret years, Rose Inc founding and details, Amyris bankruptcy, Rose Inc exit, and fashion campaigns).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PersonalInfo(BaseModel):
    nationality: Optional[str] = None
    birthdate: Optional[str] = None
    height: Optional[str] = None


class ReferenceURLs(BaseModel):
    # Victoria's Secret timeline
    vs_start: List[str] = Field(default_factory=list)
    vs_angel: List[str] = Field(default_factory=list)
    vs_exit: List[str] = Field(default_factory=list)

    # Rose Inc + Amyris
    rose_founding: List[str] = Field(default_factory=list)
    rose_expansion: List[str] = Field(default_factory=list)
    rose_sephora: List[str] = Field(default_factory=list)
    rose_jv_amyris: List[str] = Field(default_factory=list)
    amyris_bankruptcy: List[str] = Field(default_factory=list)
    rose_exit: List[str] = Field(default_factory=list)
    rose_makeup_sku: List[str] = Field(default_factory=list)
    rose_skincare: List[str] = Field(default_factory=list)

    # Fashion campaigns
    burberry_2008: List[str] = Field(default_factory=list)
    burberry_body_2011: List[str] = Field(default_factory=list)
    burberry_festive_2015: List[str] = Field(default_factory=list)
    burberry_high_summer_2025: List[str] = Field(default_factory=list)
    mackage_2025_2026: List[str] = Field(default_factory=list)

    # Personal info (optional sources)
    personal_nationality: List[str] = Field(default_factory=list)
    personal_birthdate: List[str] = Field(default_factory=list)
    personal_height: List[str] = Field(default_factory=list)


class ModelExtraction(BaseModel):
    full_name: Optional[str] = None
    personal_info: PersonalInfo = PersonalInfo()
    refs: ReferenceURLs = ReferenceURLs()


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_model_info() -> str:
    return """
    You must extract from the answer:
    1) full_name: The single identified model’s full name (as written in the answer).
    2) personal_info:
       - nationality: The nationality string provided (e.g., "British", "English").
       - birthdate: The date string provided for the model’s birth (e.g., "April 18, 1987" or "18 April 1987").
       - height: The height string provided (e.g., "5 ft 9 in (175 cm)").
    3) refs: Group all reference URLs the answer cites into the following fields (include only valid URLs explicitly present in the answer text; accept plain, markdown, or inline URLs):
       - vs_start: URLs supporting that the model started working with Victoria's Secret in early 2006.
       - vs_angel: URLs supporting that she became a Victoria's Secret Angel in February 2010.
       - vs_exit: URLs supporting that she left Victoria's Secret in 2011.
       - rose_founding: URLs supporting that she founded Rose Inc in 2018 as an editorial site/beauty publication.
       - rose_expansion: URLs supporting that Rose Inc expanded to sell makeup and skincare products in August 2021.
       - rose_sephora: URLs supporting that Rose Inc products were sold at Sephora.
       - rose_jv_amyris: URLs supporting that Amyris Inc was the joint venture partner for Rose Inc.
       - amyris_bankruptcy: URLs supporting that Amyris Inc declared bankruptcy in August 2023.
       - rose_exit: URLs supporting that she stepped down from Rose Inc in May 2024.
       - rose_makeup_sku: URLs supporting that Rose Inc offers makeup including foundation, concealer, and blush.
       - rose_skincare: URLs supporting that Rose Inc offers skincare products.
       - burberry_2008: URLs supporting participation in a Burberry campaign in 2008.
       - burberry_body_2011: URLs supporting participation in the 2011 Burberry Body fragrance campaign.
       - burberry_festive_2015: URLs supporting participation in the 2015 Burberry Festive campaign.
       - burberry_high_summer_2025: URLs supporting participation in the Burberry High Summer 2025 campaign.
       - mackage_2025_2026: URLs supporting participation in Mackage campaigns during 2025–2026.
       - personal_nationality: URLs supporting the nationality claim (optional).
       - personal_birthdate: URLs supporting the birthdate claim (optional).
       - personal_height: URLs supporting the height claim (optional).
    IMPORTANT:
    - Do not fabricate URLs. Only include URLs that are explicitly present in the answer.
    - If some references are not provided in the answer for a given field, return an empty array for that field.
    - If any personal_info fields are not mentioned, set them to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _urls(lst: Optional[List[str]]) -> List[str]:
    return [u for u in (lst or []) if _nonempty(u)]


def _claim_subject(name: Optional[str]) -> str:
    return name if _nonempty(name) else "the identified model"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_personal_criteria(
    evaluator: Evaluator,
    parent,
    model_name: Optional[str],
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="PersonalCriteria",
        desc="Personal/basic criteria match.",
        parent=parent,
        critical=True,
    )

    # Nationality
    nationality_leaf = evaluator.add_leaf(
        id="NationalityVerification",
        desc="Model is British.",
        parent=node,
        critical=True,
    )
    nationality_claim = f"{_claim_subject(model_name)} is British."
    await evaluator.verify(
        claim=nationality_claim,
        node=nationality_leaf,
        sources=_urls(refs.personal_nationality) or None,
        additional_instruction="Allow 'British'/'English' and England-origin descriptions (e.g., from Plymouth, Devon, England) to satisfy British nationality."
    )

    # Birthdate
    birth_leaf = evaluator.add_leaf(
        id="BirthDateVerification",
        desc="Model was born on April 18, 1987.",
        parent=node,
        critical=True,
    )
    birth_claim = f"{_claim_subject(model_name)} was born on April 18, 1987."
    await evaluator.verify(
        claim=birth_claim,
        node=birth_leaf,
        sources=_urls(refs.personal_birthdate) or None,
        additional_instruction="Accept date formats like 'April 18, 1987' or '18 April 1987'."
    )

    # Height
    height_leaf = evaluator.add_leaf(
        id="HeightVerification",
        desc="Model's height is exactly 5 feet 9 inches (175 cm / 1.75 m).",
        parent=node,
        critical=True,
    )
    height_claim = f"{_claim_subject(model_name)} has a height of 5 feet 9 inches (175 cm or 1.75 m)."
    await evaluator.verify(
        claim=height_claim,
        node=height_leaf,
        sources=_urls(refs.personal_height) or None,
        additional_instruction="Accept equivalent forms: '5 ft 9 in', '5'9\"', '175 cm', or '1.75 m'. Do not accept 174 cm or 176 cm."
    )


async def build_vs_criteria(
    evaluator: Evaluator,
    parent,
    model_name: Optional[str],
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="VictoriaSecretCriteria",
        desc="Victoria's Secret timeline matches all stated dates.",
        parent=parent,
        critical=True,
    )

    # Start early 2006
    vs_start_leaf = evaluator.add_leaf(
        id="VictoriaSecretStart",
        desc="Started working with Victoria's Secret in early 2006.",
        parent=node,
        critical=True,
    )
    vs_start_claim = f"{_claim_subject(model_name)} started working with Victoria's Secret in early 2006."
    await evaluator.verify(
        claim=vs_start_claim,
        node=vs_start_leaf,
        sources=_urls(refs.vs_start),
        additional_instruction="Accept phrasing indicating first work/contract/shoot with Victoria's Secret in early 2006 (e.g., Jan–Apr 2006). If only 'in 2006' is stated, accept if the context/evidence clearly supports early 2006."
    )

    # Angel Feb 2010
    vs_angel_leaf = evaluator.add_leaf(
        id="VictoriaSecretAngel",
        desc="Became a Victoria's Secret Angel in February 2010.",
        parent=node,
        critical=True,
    )
    vs_angel_claim = f"{_claim_subject(model_name)} became a Victoria's Secret Angel in February 2010."
    await evaluator.verify(
        claim=vs_angel_claim,
        node=vs_angel_leaf,
        sources=_urls(refs.vs_angel),
        additional_instruction="Accept if the source states she was named/announced as a Victoria's Secret Angel in Feb 2010."
    )

    # Exit 2011
    vs_exit_leaf = evaluator.add_leaf(
        id="VictoriaSecretExit",
        desc="Left Victoria's Secret in 2011.",
        parent=node,
        critical=True,
    )
    vs_exit_claim = f"{_claim_subject(model_name)} left Victoria's Secret in 2011."
    await evaluator.verify(
        claim=vs_exit_claim,
        node=vs_exit_leaf,
        sources=_urls(refs.vs_exit),
        additional_instruction="Accept if the source indicates she left/did not continue with Victoria's Secret as of 2011 (e.g., departure news, last involvement year 2011)."
    )


async def build_roseinc_amyris_criteria(
    evaluator: Evaluator,
    parent,
    model_name: Optional[str],
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="RoseIncAndAmyrisCriteria",
        desc="Rose Inc venture details and Amyris details match all stated constraints.",
        parent=parent,
        critical=True,
    )

    # Founded Rose Inc in 2018 (editorial)
    leaf_found = evaluator.add_leaf(
        id="RoseIncFounded2018Editorial",
        desc="Founded Rose Inc in 2018 as an editorial site/beauty publication.",
        parent=node,
        critical=True,
    )
    claim_found = f"{_claim_subject(model_name)} founded Rose Inc in 2018 as an editorial site/beauty publication."
    await evaluator.verify(
        claim=claim_found,
        node=leaf_found,
        sources=_urls(refs.rose_founding),
        additional_instruction="Look for launch/announcement in 2018 that describes Rose Inc as an editorial content site or beauty publication."
    )

    # Expanded to sell products in Aug 2021
    leaf_expand = evaluator.add_leaf(
        id="RoseIncExpandedAug2021ToMakeupAndSkincare",
        desc="Expanded Rose Inc to sell makeup and skincare products in August 2021.",
        parent=node,
        critical=True,
    )
    claim_expand = f"{_claim_subject(model_name)} expanded Rose Inc to sell makeup and skincare products in August 2021."
    await evaluator.verify(
        claim=claim_expand,
        node=leaf_expand,
        sources=_urls(refs.rose_expansion),
        additional_instruction="Accept official launch/press coverage stating the product launch in August 2021 (makeup and skincare)."
    )

    # Sold at Sephora
    leaf_sephora = evaluator.add_leaf(
        id="RoseIncSoldAtSephora",
        desc="Rose Inc products were sold at Sephora.",
        parent=node,
        critical=True,
    )
    claim_sephora = "Rose Inc products were sold at Sephora."
    await evaluator.verify(
        claim=claim_sephora,
        node=leaf_sephora,
        sources=_urls(refs.rose_sephora),
        additional_instruction="Accept Sephora product/brand pages or reputable coverage explicitly showing Rose Inc available at Sephora."
    )

    # JV with Amyris
    leaf_jv = evaluator.add_leaf(
        id="RoseIncJointVentureAmyris",
        desc="Rose Inc's joint venture partner was Amyris Inc.",
        parent=node,
        critical=True,
    )
    claim_jv = "Rose Inc operated as a joint venture with Amyris Inc."
    await evaluator.verify(
        claim=claim_jv,
        node=leaf_jv,
        sources=_urls(refs.rose_jv_amyris),
        additional_instruction="Look for official statements/press that Rose Inc was a JV with Amyris (e.g., 'Rose Inc, a joint venture with Amyris')."
    )

    # Amyris bankruptcy Aug 2023
    leaf_bk = evaluator.add_leaf(
        id="AmyrisBankruptcyAug2023",
        desc="Amyris Inc declared bankruptcy in August 2023.",
        parent=node,
        critical=True,
    )
    claim_bk = "Amyris Inc filed for bankruptcy in August 2023."
    await evaluator.verify(
        claim=claim_bk,
        node=leaf_bk,
        sources=_urls(refs.amyris_bankruptcy),
        additional_instruction="Accept reputable reports on Amyris filing Chapter 11 in August 2023."
    )

    # Rose Inc exit May 2024
    leaf_exit = evaluator.add_leaf(
        id="RoseIncExitMay2024",
        desc="Model stepped down from Rose Inc in May 2024.",
        parent=node,
        critical=True,
    )
    claim_exit = f"{_claim_subject(model_name)} stepped down from Rose Inc in May 2024."
    await evaluator.verify(
        claim=claim_exit,
        node=leaf_exit,
        sources=_urls(refs.rose_exit),
        additional_instruction="Accept official statements or credible news indicating she stepped down/left Rose Inc in May 2024."
    )

    # Makeup SKU includes foundation, concealer, blush
    leaf_sku = evaluator.add_leaf(
        id="RoseIncMakeupIncludesFoundationConcealerBlush",
        desc="Rose Inc offers makeup products including foundation, concealer, and blush.",
        parent=node,
        critical=True,
    )
    claim_sku = "Rose Inc offers makeup products including foundation, concealer, and blush."
    await evaluator.verify(
        claim=claim_sku,
        node=leaf_sku,
        sources=_urls(refs.rose_makeup_sku),
        additional_instruction="Accept official brand/product pages or reputable listings showing Rose Inc makeup SKUs include foundation, concealer, and blush."
    )

    # Offers skincare
    leaf_skin = evaluator.add_leaf(
        id="RoseIncOffersSkincareProducts",
        desc="Rose Inc offers skincare products.",
        parent=node,
        critical=True,
    )
    claim_skin = "Rose Inc offers skincare products."
    await evaluator.verify(
        claim=claim_skin,
        node=leaf_skin,
        sources=_urls(refs.rose_skincare),
        additional_instruction="Accept official brand/product pages or reputable listings indicating Rose Inc offers skincare."
    )


async def build_fashion_campaign_criteria(
    evaluator: Evaluator,
    parent,
    model_name: Optional[str],
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="FashionCampaignCriteria",
        desc="Campaign participation constraints match.",
        parent=parent,
        critical=True,
    )

    # Burberry 2008
    leaf_b08 = evaluator.add_leaf(
        id="BurberryCampaign2008",
        desc="Participated in a Burberry campaign in 2008.",
        parent=node,
        critical=True,
    )
    claim_b08 = f"{_claim_subject(model_name)} participated in a Burberry campaign in 2008."
    await evaluator.verify(
        claim=claim_b08,
        node=leaf_b08,
        sources=_urls(refs.burberry_2008),
        additional_instruction="Accept Burberry official content or reputable fashion press confirming her participation in 2008 campaigns."
    )

    # Burberry Body 2011
    leaf_body = evaluator.add_leaf(
        id="BurberryBody2011",
        desc="Participated in the Burberry Body fragrance campaign in 2011.",
        parent=node,
        critical=True,
    )
    claim_body = f"{_claim_subject(model_name)} participated in the Burberry Body fragrance campaign in 2011."
    await evaluator.verify(
        claim=claim_body,
        node=leaf_body,
        sources=_urls(refs.burberry_body_2011),
        additional_instruction="Look for 'Burberry Body' 2011 fragrance campaign with her as face/participant."
    )

    # Burberry Festive 2015
    leaf_fest = evaluator.add_leaf(
        id="BurberryFestive2015",
        desc="Participated in the Burberry Festive campaign in 2015.",
        parent=node,
        critical=True,
    )
    claim_fest = f"{_claim_subject(model_name)} participated in the Burberry Festive 2015 campaign."
    await evaluator.verify(
        claim=claim_fest,
        node=leaf_fest,
        sources=_urls(refs.burberry_festive_2015),
        additional_instruction="Accept holiday/festive 2015 Burberry campaign credits."
    )

    # Burberry High Summer 2025
    leaf_hs25 = evaluator.add_leaf(
        id="BurberryHighSummer2025",
        desc="Participated in the Burberry High Summer 2025 campaign.",
        parent=node,
        critical=True,
    )
    claim_hs25 = f"{_claim_subject(model_name)} participated in the Burberry High Summer 2025 campaign."
    await evaluator.verify(
        claim=claim_hs25,
        node=leaf_hs25,
        sources=_urls(refs.burberry_high_summer_2025),
        additional_instruction="Accept official Burberry or reputable fashion press referencing 'High Summer 2025' campaign featuring her."
    )

    # Mackage campaigns 2025–2026
    leaf_mackage = evaluator.add_leaf(
        id="MackageCampaigns2025to2026",
        desc="Participated in Mackage fashion campaigns during 2025–2026.",
        parent=node,
        critical=True,
    )
    claim_mackage = f"{_claim_subject(model_name)} participated in Mackage fashion campaigns during 2025–2026."
    await evaluator.verify(
        claim=claim_mackage,
        node=leaf_mackage,
        sources=_urls(refs.mackage_2025_2026),
        additional_instruction="Accept if sources show her in Mackage campaigns in 2025 and/or 2026."
    )


def build_reference_existence_nodes(
    evaluator: Evaluator,
    parent,
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="ProvideReferenceURLsForMajorAspects",
        desc="Answer includes reference URLs supporting each major aspect requested.",
        parent=parent,
        critical=True,
    )

    # VS references
    vs_refs = evaluator.add_parallel(
        id="VSReferences",
        desc="Includes reference URL(s) supporting the Victoria's Secret timeline claims.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.vs_start)) > 0,
        id="VSStartReference",
        desc="Includes ≥1 reference URL supporting the early 2006 Victoria's Secret start claim.",
        parent=vs_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.vs_angel)) > 0,
        id="VSAngelReference",
        desc="Includes ≥1 reference URL supporting the February 2010 Victoria's Secret Angel claim.",
        parent=vs_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.vs_exit)) > 0,
        id="VSExitReference",
        desc="Includes ≥1 reference URL supporting the 2011 Victoria's Secret exit claim.",
        parent=vs_refs,
        critical=True,
    )

    # Rose Inc references
    ri_refs = evaluator.add_parallel(
        id="RoseIncReferences",
        desc="Includes reference URL(s) supporting Rose Inc founding and business details claims.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_founding)) > 0,
        id="RoseIncFoundingReference",
        desc="Includes ≥1 reference URL supporting the 2018 Rose Inc founding/editorial publication claim.",
        parent=ri_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_expansion)) > 0,
        id="RoseIncExpansionReference",
        desc="Includes ≥1 reference URL supporting the August 2021 expansion to selling makeup and skincare products claim.",
        parent=ri_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_sephora)) > 0,
        id="RoseIncSephoraReference",
        desc="Includes ≥1 reference URL supporting the claim that Rose Inc products were sold at Sephora.",
        parent=ri_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_jv_amyris)) > 0,
        id="RoseIncAmyrisJVReference",
        desc="Includes ≥1 reference URL supporting the claim that Amyris Inc was the joint venture partner.",
        parent=ri_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_makeup_sku)) > 0,
        id="RoseIncMakeupSKUReference",
        desc="Includes ≥1 reference URL supporting that Rose Inc offers makeup including foundation, concealer, and blush.",
        parent=ri_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_skincare)) > 0,
        id="RoseIncSkincareReference",
        desc="Includes ≥1 reference URL supporting that Rose Inc offers skincare products.",
        parent=ri_refs,
        critical=True,
    )

    # Amyris bankruptcy reference
    evaluator.add_custom_node(
        result=len(_urls(refs.amyris_bankruptcy)) > 0,
        id="AmyrisBankruptcyReferences",
        desc="Includes ≥1 reference URL supporting Amyris Inc bankruptcy in August 2023.",
        parent=node,
        critical=True,
    )

    # Rose Inc exit references
    evaluator.add_custom_node(
        result=len(_urls(refs.rose_exit)) > 0,
        id="RoseIncExitReferences",
        desc="Includes ≥1 reference URL supporting stepping down from Rose Inc in May 2024.",
        parent=node,
        critical=True,
    )

    # Fashion campaign references
    fc_refs = evaluator.add_parallel(
        id="FashionCampaignReferences",
        desc="Includes reference URL(s) supporting the specified fashion campaign participation claims.",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.burberry_2008)) > 0,
        id="Burberry2008Reference",
        desc="Includes ≥1 reference URL supporting participation in a Burberry campaign in 2008.",
        parent=fc_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.burberry_body_2011)) > 0,
        id="BurberryBody2011Reference",
        desc="Includes ≥1 reference URL supporting participation in the Burberry Body fragrance campaign in 2011.",
        parent=fc_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.burberry_festive_2015)) > 0,
        id="BurberryFestive2015Reference",
        desc="Includes ≥1 reference URL supporting participation in the Burberry Festive campaign in 2015.",
        parent=fc_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.burberry_high_summer_2025)) > 0,
        id="BurberryHighSummer2025Reference",
        desc="Includes ≥1 reference URL supporting participation in the Burberry High Summer 2025 campaign.",
        parent=fc_refs,
        critical=True,
    )
    evaluator.add_custom_node(
        result=len(_urls(refs.mackage_2025_2026)) > 0,
        id="Mackage2025to2026Reference",
        desc="Includes ≥1 reference URL supporting participation in Mackage fashion campaigns during 2025–2026.",
        parent=fc_refs,
        critical=True,
    )


async def build_verify_all_constraints(
    evaluator: Evaluator,
    parent,
    model_name: Optional[str],
    refs: ReferenceURLs,
):
    node = evaluator.add_parallel(
        id="VerifyAllConstraints",
        desc="Identified model satisfies every listed constraint (personal, Victoria's Secret, Rose Inc/Amyris, and campaign participation).",
        parent=parent,
        critical=True,
    )

    # Build each critical parallel group
    await build_personal_criteria(evaluator, node, model_name, refs)
    await build_vs_criteria(evaluator, node, model_name, refs)
    await build_roseinc_amyris_criteria(evaluator, node, model_name, refs)
    await build_fashion_campaign_criteria(evaluator, node, model_name, refs)


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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
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

    # Extract structured info
    extracted: ModelExtraction = await evaluator.extract(
        prompt=prompt_extract_model_info(),
        template_class=ModelExtraction,
        extraction_name="model_extraction",
    )

    # Top-level critical sequential node mirroring rubric
    top = evaluator.add_sequential(
        id="ModelIdentificationAndVerification",
        desc="Identify the British model meeting all constraints and provide supporting URLs for each major aspect requested.",
        parent=root,
        critical=True,
    )

    # ProvideModelFullName (existence/uniqueness proxy)
    evaluator.add_custom_node(
        result=_nonempty(extracted.full_name),
        id="ProvideModelFullName",
        desc="Answer provides the model's full name (single identified person).",
        parent=top,
        critical=True,
    )

    # Verify all constraints with evidence
    await build_verify_all_constraints(
        evaluator=evaluator,
        parent=top,
        model_name=extracted.full_name,
        refs=extracted.refs,
    )

    # Reference URL existence checks for required major aspects
    build_reference_existence_nodes(
        evaluator=evaluator,
        parent=top,
        refs=extracted.refs,
    )

    # Return evaluation summary
    return evaluator.get_summary()