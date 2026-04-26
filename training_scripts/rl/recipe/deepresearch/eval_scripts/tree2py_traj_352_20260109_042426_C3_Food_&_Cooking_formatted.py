import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "jbf_2024_pikliz_pepper"
TASK_DESCRIPTION = (
    "In June 2024, the James Beard Foundation announced its Restaurant and Chef Award winners. "
    "Identify the chef who won the 'Best Chef: Northwest and Pacific' award and operates a Haitian-inspired restaurant in Portland, Oregon. "
    "At this chef's restaurant, there is a traditional Haitian pickled vegetable condiment made with cabbage, carrots, and hot peppers. "
    "Identify the name of this condiment, and then determine which specific variety of Caribbean hot pepper is traditionally essential to this condiment. "
    "For the identified pepper variety, provide: (1) The Scoville Heat Unit (SHU) range, (2) The primary Caribbean country where this pepper variety originated and is traditionally grown, "
    "and (3) A description of its flavor profile. Your answer must include URL references for the chef's 2024 James Beard Award, information about the Haitian condiment and its ingredients, "
    "and the pepper variety's heat rating and origin."
)


# ----------------------------
# Extraction models
# ----------------------------
class ChefExtraction(BaseModel):
    chef_name: Optional[str] = None
    restaurant_name: Optional[str] = None
    restaurant_location_text: Optional[str] = None
    restaurant_cuisine_text: Optional[str] = None
    restaurant_techniques_text: Optional[str] = None
    chef_award_urls: List[str] = Field(default_factory=list)
    restaurant_urls: List[str] = Field(default_factory=list)


class CondimentExtraction(BaseModel):
    condiment_name: Optional[str] = None
    ingredients_text: Optional[str] = None
    ingredient_list: List[str] = Field(default_factory=list)
    condiment_urls: List[str] = Field(default_factory=list)


class PepperExtraction(BaseModel):
    pepper_common_name: Optional[str] = None
    origin_country: Optional[str] = None
    shu_range_text: Optional[str] = None
    flavor_profile_text: Optional[str] = None
    traditional_uses_text: Optional[str] = None
    botanical_species: Optional[str] = None
    climate_text: Optional[str] = None
    pepper_urls: List[str] = Field(default_factory=list)
    pepper_heat_urls: List[str] = Field(default_factory=list)
    pepper_origin_urls: List[str] = Field(default_factory=list)


# ----------------------------
# Extraction prompts
# ----------------------------
def prompt_extract_chef() -> str:
    return """
Extract the chef and restaurant details explicitly stated in the answer:
- chef_name: full name of the chef identified as the 2024 James Beard Award winner for "Best Chef: Northwest and Pacific".
- restaurant_name: the name of the chef's restaurant (if provided).
- restaurant_location_text: verbatim text describing the restaurant location (e.g., "Portland, Oregon", "Portland, OR").
- restaurant_cuisine_text: verbatim text describing the cuisine (e.g., "Haitian-inspired").
- restaurant_techniques_text: verbatim text describing live-fire or related cooking techniques (e.g., "live fire", "live-fire", "wood-fired", "hearth").
- chef_award_urls: list all URLs in the answer that support the chef's 2024 James Beard Award in the 'Best Chef: Northwest and Pacific' category. Extract the actual URLs only.
- restaurant_urls: list all URLs in the answer that describe the restaurant, its location/cuisine/techniques, or profiles of the chef and restaurant.

Rules:
- Only extract information that appears in the answer.
- For URLs, extract only valid, explicit URLs present in the answer (including markdown links).
- If a field is missing, set it to null (or empty list for URLs).
"""


def prompt_extract_condiment() -> str:
    return """
Extract the Haitian condiment details from the answer:
- condiment_name: the name of the condiment referenced at the restaurant.
- ingredients_text: the text describing the ingredients of the condiment as written in the answer.
- ingredient_list: list the individual ingredients explicitly named in the answer (normalize to lowercase single words where possible, e.g., "cabbage", "carrots", "scotch bonnet", "hot peppers").
- condiment_urls: all URLs that describe the condiment and confirm its ingredients.

Rules:
- Extract only what appears in the answer.
- The condiment in question is a traditional Haitian pickled vegetable relish; ensure you capture the name (if present).
- For URLs, extract only explicit URLs mentioned in the answer. If none, return an empty list.
"""


def prompt_extract_pepper() -> str:
    return """
Extract the pepper variety details from the answer:
- pepper_common_name: the common name of the Caribbean hot pepper essential to the condiment.
- origin_country: the primary Caribbean country where this pepper originated and is traditionally grown (as stated in the answer).
- shu_range_text: the Scoville Heat Unit (SHU) range provided in the answer (as text).
- flavor_profile_text: the described flavor profile (as text).
- traditional_uses_text: text describing traditional uses; include mentions of jerk seasoning and/or pikliz if present.
- botanical_species: the botanical classification (e.g., "Capsicum chinense" or "C. chinense").
- climate_text: the described climate in which the pepper thrives (e.g., "hot, humid, tropical").
- pepper_urls: general URLs in the answer about the pepper.
- pepper_heat_urls: URLs specifically used to support the SHU range (if any).
- pepper_origin_urls: URLs specifically used to support the origin country (if any).

Rules:
- Extract exactly what appears in the answer.
- For URLs, extract explicit URLs only. If specific heat/origin URLs are not separated, put them in pepper_urls.
"""


# ----------------------------
# Helpers
# ----------------------------
def merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str):
                s = u.strip()
                if s and s not in seen:
                    seen.add(s)
                    result.append(s)
    return result


def has_any_url(urls: Optional[List[str]]) -> bool:
    return bool([u for u in (urls or []) if isinstance(u, str) and u.strip()])


# ----------------------------
# Verification construction
# ----------------------------
async def build_chef_phase(evaluator: Evaluator, parent, chef: ChefExtraction) -> None:
    chef_phase = evaluator.add_sequential(
        id="chef_phase",
        desc="Identify the relevant 2024 James Beard Award-winning chef and verify required restaurant constraints with an award URL.",
        parent=parent,
        critical=False,
    )

    # chef_name existence (critical)
    evaluator.add_custom_node(
        result=bool(chef.chef_name and chef.chef_name.strip()),
        id="chef_name",
        desc="Provide the full name of the chef.",
        parent=chef_phase,
        critical=True
    )

    # award URL evidence (critical) - require URL(s) present and supporting the claim
    award_node = evaluator.add_leaf(
        id="chef_award_url_evidence",
        desc="Provide a URL reference that supports the claim that this chef won the 2024 James Beard Award in the 'Best Chef: Northwest and Pacific' category.",
        parent=chef_phase,
        critical=True,
    )
    if not has_any_url(chef.chef_award_urls):
        award_node.score = 0.0
        award_node.status = "failed"
    else:
        chef_name_disp = chef.chef_name or "the chef"
        claim = (
            f"{chef_name_disp} won the 2024 James Beard Foundation Restaurant and Chef Awards "
            f"'Best Chef: Northwest and Pacific' honor."
        )
        await evaluator.verify(
            claim=claim,
            node=award_node,
            sources=chef.chef_award_urls,
            additional_instruction=(
                "Verify the 2024 winners page or credible coverage states this chef won the 2024 'Best Chef: Northwest and Pacific' award. "
                "Allow minor variations in category punctuation or hyphenation. Focus on the 2024 award and this exact category."
            ),
        )

    # restaurant constraints (critical group, parallel children)
    rest_group = evaluator.add_parallel(
        id="restaurant_constraints_verification",
        desc="Verify required constraints about the chef's restaurant.",
        parent=chef_phase,
        critical=True
    )
    rest_sources = merge_sources(chef.restaurant_urls, chef.chef_award_urls)

    # location
    rest_loc = evaluator.add_leaf(
        id="restaurant_location",
        desc="Confirm the chef operates a restaurant in Portland, Oregon.",
        parent=rest_group,
        critical=True
    )
    claim_loc = "The chef operates a restaurant located in Portland, Oregon (Portland, OR)."
    await evaluator.verify(
        claim=claim_loc,
        node=rest_loc,
        sources=rest_sources if rest_sources else None,
        additional_instruction="Confirm the restaurant is in Portland, Oregon. Accept 'Portland, OR' or 'Portland, Ore.' as equivalent."
    )

    # cuisine
    rest_cui = evaluator.add_leaf(
        id="restaurant_cuisine",
        desc="Confirm the restaurant specializes in Haitian-inspired cuisine.",
        parent=rest_group,
        critical=True
    )
    claim_cui = "The restaurant specializes in Haitian or Haitian-inspired cuisine."
    await evaluator.verify(
        claim=claim_cui,
        node=rest_cui,
        sources=rest_sources if rest_sources else None,
        additional_instruction="Accept phrasing like 'Haitian', 'Haitian-inspired', or references to Haitian flavors and heritage."
    )

    # live-fire
    rest_fire = evaluator.add_leaf(
        id="restaurant_live_fire",
        desc="Confirm the restaurant uses live-fire cooking techniques.",
        parent=rest_group,
        critical=True
    )
    claim_fire = "The restaurant uses live-fire cooking techniques (e.g., live fire, live-fire, wood-fired cooking, hearth, or open flame)."
    await evaluator.verify(
        claim=claim_fire,
        node=rest_fire,
        sources=rest_sources if rest_sources else None,
        additional_instruction="Look for mentions of 'live fire', 'live-fire', 'wood-fired', 'hearth', or cooking over open fire."
    )


async def build_condiment_phase(evaluator: Evaluator, parent, condiment: CondimentExtraction) -> None:
    cond_phase = evaluator.add_sequential(
        id="condiment_phase",
        desc="Identify the Haitian condiment at the restaurant and verify its required ingredient composition with a URL reference.",
        parent=parent,
        critical=False
    )

    # Condiment must be pikliz (critical)
    is_pikliz = bool(condiment.condiment_name) and condiment.condiment_name.strip().lower() == "pikliz"
    evaluator.add_custom_node(
        result=is_pikliz,
        id="condiment_name",
        desc="Identify the name of the traditional Haitian pickled vegetable condiment (must match the constraint that it is called 'pikliz').",
        parent=cond_phase,
        critical=True
    )

    # Ingredient requirements (critical) using URLs
    cond_ing = evaluator.add_leaf(
        id="condiment_ingredient_requirements",
        desc="Confirm the condiment is a pickled vegetable relish made with cabbage, carrots, and hot peppers.",
        parent=cond_phase,
        critical=True
    )
    claim_ing = "Pikliz is a Haitian pickled vegetable relish made with cabbage, carrots, and hot peppers."
    await evaluator.verify(
        claim=claim_ing,
        node=cond_ing,
        sources=condiment.condiment_urls if has_any_url(condiment.condiment_urls) else None,
        additional_instruction="Confirm that descriptions of pikliz include cabbage, carrots, and hot peppers (often Scotch bonnet or similar hot chiles)."
    )

    # Condiment URL evidence presence (critical)
    evaluator.add_custom_node(
        result=has_any_url(condiment.condiment_urls),
        id="condiment_url_evidence",
        desc="Provide a URL reference describing the condiment and confirming its ingredients.",
        parent=cond_phase,
        critical=True
    )


async def build_pepper_phase(evaluator: Evaluator, parent, pepper: PepperExtraction) -> None:
    pep_phase = evaluator.add_parallel(
        id="pepper_phase",
        desc="Identify the pepper variety essential to the condiment and provide the requested attributes and required URL references.",
        parent=parent,
        critical=False
    )

    # Common name exists (critical)
    evaluator.add_custom_node(
        result=bool(pepper.pepper_common_name and pepper.pepper_common_name.strip()),
        id="pepper_common_name",
        desc="Provide the common name of the specific Caribbean hot pepper variety identified as essential to the condiment.",
        parent=pep_phase,
        critical=True
    )

    # Origin requirement must match Jamaica (critical)
    origin_text = (pepper.origin_country or "").strip().lower()
    matches_jamaica = "jamaica" in origin_text
    evaluator.add_custom_node(
        result=matches_jamaica,
        id="pepper_origin_requirement",
        desc="State the primary Caribbean country where the pepper variety originated and is traditionally grown, and ensure it matches the constraint (Jamaica).",
        parent=pep_phase,
        critical=True
    )

    # Choose sources for pepper facts
    heat_sources = pepper.pepper_heat_urls if has_any_url(pepper.pepper_heat_urls) else pepper.pepper_urls
    origin_sources = pepper.pepper_origin_urls if has_any_url(pepper.pepper_origin_urls) else pepper.pepper_urls
    general_sources = pepper.pepper_urls

    # SHU requirement (critical) - verify within 100,000 to 350,000 SHU
    shu_node = evaluator.add_leaf(
        id="pepper_shu_requirement",
        desc="Provide the Scoville Heat Unit (SHU) range and ensure it matches the constraint (between 100,000 and 350,000 SHU).",
        parent=pep_phase,
        critical=True
    )
    pepper_name_disp = pepper.pepper_common_name or "the pepper"
    claim_shu = f"{pepper_name_disp} has a Scoville Heat Unit range that falls between 100,000 and 350,000 SHU."
    await evaluator.verify(
        claim=claim_shu,
        node=shu_node,
        sources=heat_sources if has_any_url(heat_sources) else None,
        additional_instruction=(
            "Confirm that authoritative sources list this pepper's SHU range overlapping 100,000–350,000. "
            "Accept formatting variations (commas, en dashes)."
        )
    )

    # Flavor requirement (critical)
    flavor_node = evaluator.add_leaf(
        id="pepper_flavor_requirement",
        desc="Describe the pepper's flavor profile and ensure it matches the constraint (distinctive fruity and sweet flavor profile).",
        parent=pep_phase,
        critical=True
    )
    claim_flavor = f"{pepper_name_disp} has a distinctive fruity and sweet flavor profile (often described as tropical and aromatic)."
    await evaluator.verify(
        claim=claim_flavor,
        node=flavor_node,
        sources=general_sources if has_any_url(general_sources) else None,
        additional_instruction="Look for descriptors like fruity, tropical fruit, sweet, and aromatic when describing the pepper."
    )

    # Traditional use requirement (critical)
    use_node = evaluator.add_leaf(
        id="pepper_traditional_use_requirement",
        desc="Confirm the pepper is traditionally used in Caribbean dishes including jerk seasoning and pikliz.",
        parent=pep_phase,
        critical=True
    )
    claim_use = f"{pepper_name_disp} is traditionally used in Caribbean dishes including jerk seasoning and Haitian pikliz."
    await evaluator.verify(
        claim=claim_use,
        node=use_node,
        sources=general_sources if has_any_url(general_sources) else None,
        additional_instruction="Verify mentions of jerk seasoning and, where applicable, its use in pikliz or Haitian pickled relishes."
    )

    # Botanical classification (critical)
    bot_node = evaluator.add_leaf(
        id="pepper_botanical_classification_requirement",
        desc="Confirm the pepper is botanically classified as Capsicum chinense.",
        parent=pep_phase,
        critical=True
    )
    claim_bot = f"{pepper_name_disp} is botanically classified as Capsicum chinense (C. chinense)."
    await evaluator.verify(
        claim=claim_bot,
        node=bot_node,
        sources=general_sources if has_any_url(general_sources) else None,
        additional_instruction="Accept 'Capsicum chinense' or abbreviated 'C. chinense' as equivalent."
    )

    # Climate requirement (critical)
    climate_node = evaluator.add_leaf(
        id="pepper_climate_requirement",
        desc="Confirm the pepper thrives in hot, humid tropical climates.",
        parent=pep_phase,
        critical=True
    )
    claim_climate = f"{pepper_name_disp} thrives in hot, humid tropical climates."
    await evaluator.verify(
        claim=claim_climate,
        node=climate_node,
        sources=general_sources if has_any_url(general_sources) else None,
        additional_instruction="Look for cultivation notes indicating tropical heat and humidity are ideal."
    )

    # Heat URL evidence presence (critical)
    evaluator.add_custom_node(
        result=has_any_url(pepper.pepper_heat_urls) or has_any_url(pepper.pepper_urls),
        id="pepper_heat_url_evidence",
        desc="Provide a URL reference supporting the pepper variety's SHU range.",
        parent=pep_phase,
        critical=True
    )

    # Origin URL evidence presence (critical)
    evaluator.add_custom_node(
        result=has_any_url(pepper.pepper_origin_urls) or has_any_url(pepper.pepper_urls),
        id="pepper_origin_url_evidence",
        desc="Provide a URL reference supporting the pepper variety's origin/traditional growing region (Jamaica).",
        parent=pep_phase,
        critical=True
    )


# ----------------------------
# Main evaluation entry point
# ----------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Parallel extraction
    chef_task = evaluator.extract(
        prompt=prompt_extract_chef(),
        template_class=ChefExtraction,
        extraction_name="chef_extraction"
    )
    condiment_task = evaluator.extract(
        prompt=prompt_extract_condiment(),
        template_class=CondimentExtraction,
        extraction_name="condiment_extraction"
    )
    pepper_task = evaluator.extract(
        prompt=prompt_extract_pepper(),
        template_class=PepperExtraction,
        extraction_name="pepper_extraction"
    )

    chef_ex, cond_ex, pep_ex = await asyncio.gather(chef_task, condiment_task, pepper_task)

    # Build task tree: task_completion with sequential phases
    task_completion = evaluator.add_sequential(
        id="task_completion",
        desc="Complete the end-to-end task: identify chef, identify condiment, then identify and characterize the pepper, with required URL references.",
        parent=root,
        critical=False
    )

    # Chef phase
    await build_chef_phase(evaluator, task_completion, chef_ex)

    # Condiment phase
    await build_condiment_phase(evaluator, task_completion, cond_ex)

    # Pepper phase (parallel)
    await build_pepper_phase(evaluator, task_completion, pep_ex)

    # Optional: record constraints as ground truth guidance (non-evaluative)
    evaluator.add_ground_truth({
        "condiment_required_name": "pikliz",
        "pepper_origin_required": "Jamaica",
        "pepper_shu_required_range": "100,000–350,000 SHU",
        "pepper_flavor_required": "fruity and sweet"
    }, gt_type="constraints")

    return evaluator.get_summary()