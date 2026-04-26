import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "national_gingerbread_competition_2025"
TASK_DESCRIPTION = (
    "What are the first-place winners (name, location, and entry title) in all four age categories of the 2025 "
    "National Gingerbread House Competition held at the Omni Grove Park Inn in Asheville, North Carolina? "
    "Additionally, what are the official competition rules regarding age categories, edible materials requirements, "
    "and base size specifications? Finally, identify one professional certification program for holiday decorators "
    "that requires a minimum of 6 weeks of training and covers multiple decorating topics."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class WinnerItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    entry_title: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class WinnersExtraction(BaseModel):
    child: Optional[WinnerItem] = None
    youth: Optional[WinnerItem] = None
    teen: Optional[WinnerItem] = None
    adult: Optional[WinnerItem] = None


class RulesExtraction(BaseModel):
    rules_urls: List[str] = Field(default_factory=list)
    age_categories_text: Optional[str] = None
    edible_materials_text: Optional[str] = None
    base_size_text: Optional[str] = None


class EventContextExtraction(BaseModel):
    event_name: Optional[str] = None  # e.g., "National Gingerbread House Competition"
    year: Optional[str] = None        # e.g., "2025"
    venue: Optional[str] = None       # e.g., "Omni Grove Park Inn"
    city: Optional[str] = None        # "Asheville"
    state: Optional[str] = None       # "North Carolina" or "NC"
    sources: List[str] = Field(default_factory=list)


class CertificationExtraction(BaseModel):
    name: Optional[str] = None
    duration_text: Optional[str] = None  # e.g., "8 weeks", "12-week intensive"
    urls: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_event_context() -> str:
    return """
    From the answer, extract the event identity details for the referenced competition.
    Fields to extract:
    - event_name: The proper event name as stated in the answer (e.g., "National Gingerbread House Competition").
    - year: The year of the event (e.g., "2025").
    - venue: The venue/location name (e.g., "Omni Grove Park Inn").
    - city: The host city (e.g., "Asheville").
    - state: The host state (e.g., "North Carolina" or its abbreviation "NC").
    - sources: All URLs cited in the answer that pertain to the event identity (official site, announcements, rules, etc.).
    If an item is not mentioned, set it to null (or empty list for URLs).
    """


def prompt_extract_winners() -> str:
    return """
    Extract the first-place winners for each of the four age categories of the 2025 competition as stated in the answer.
    For each category (child ages 5–8, youth ages 9–12, teen ages 13–17, adult ages 18+), extract:
    - name: Winner name (or team name if applicable)
    - city: Winner's city
    - state: Winner's state (full name or postal abbreviation)
    - entry_title: The title of the first-place entry
    - urls: All URLs the answer cites that support the winner info for this category
    Return a JSON object with keys: child, youth, teen, adult. Each key maps to an object with the above fields.
    If a category isn't mentioned, set it to null. If a field is missing, set it to null; if no URLs, return an empty list.
    """


def prompt_extract_rules() -> str:
    return """
    Extract the official rules details for the competition as presented in the answer.
    Fields to extract:
    - rules_urls: All URLs in the answer that are official rules pages or documents for the 2025 competition.
    - age_categories_text: The rule text or summary that specifies the four age categories (Child 5–8, Youth 9–12, Teen 13–17, Adult 18+), as quoted or paraphrased in the answer.
    - edible_materials_text: The rule text or summary about decorations being 100% edible materials (except base/board if so specified), as quoted or paraphrased in the answer.
    - base_size_text: The rule text or summary that states the maximum base size specification (e.g., "maximum base size 24×24 inches"), as quoted or paraphrased in the answer.
    If any field is not mentioned, set it to null (or empty list for URLs).
    """


def prompt_extract_certification() -> str:
    return """
    Extract the professional certification program for holiday decorators mentioned in the answer.
    Fields to extract:
    - name: The certification program name
    - duration_text: The training duration as stated (e.g., "6+ weeks", "8 weeks", "12-week program"). If a range is given, include it.
    - topics: A list of decorating topics the program covers (e.g., wreaths, centerpieces, tree decorating, garlands).
    - urls: All URLs the answer cites for this certification program.
    If no such program is mentioned, return null for name and duration_text, an empty topics list, and an empty urls list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _loc_str(city: Optional[str], state: Optional[str]) -> str:
    parts = [p.strip() for p in [city or "", state or ""] if p and p.strip()]
    return ", ".join(parts)


def _nonempty_str(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_full_winner_info(item: Optional[WinnerItem]) -> bool:
    if item is None:
        return False
    return all([
        _nonempty_str(item.name),
        _nonempty_str(item.city),
        _nonempty_str(item.state),
        _nonempty_str(item.entry_title),
    ])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_event_context_node(
    evaluator: Evaluator,
    parent,
    event_info: EventContextExtraction,
) -> None:
    """
    competition_event_context (leaf)
    - Verify that the answer states the event identity: 2025 National Gingerbread House Competition,
      held at the Omni Grove Park Inn in Asheville, North Carolina.
    """
    node = evaluator.add_leaf(
        id="competition_event_context",
        desc="States the event identity: 2025 National Gingerbread House Competition, held at the Omni Grove Park Inn, Asheville, North Carolina.",
        parent=parent,
        critical=True,
    )
    claim = (
        "The answer identifies the event as the 2025 National Gingerbread House Competition "
        "held at the Omni Grove Park Inn in Asheville, North Carolina."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="Judge only whether the answer explicitly states this full event identity.",
    )


async def build_competition_winners_node(
    evaluator: Evaluator,
    parent,
    winners: WinnersExtraction,
) -> None:
    """
    competition_winners (parallel, critical)
    - all_four_age_categories_covered (leaf via custom-node check)
    - For each category (child, youth, teen, adult): parallel critical node with 3 critical leaves
      * name
      * location
      * entry title
    """
    winners_node = evaluator.add_parallel(
        id="competition_winners",
        desc="Identify first-place winners for all four age categories and provide required details for each.",
        parent=parent,
        critical=True,
    )

    # Coverage check (must have all 4 categories with full details)
    coverage_ok = all([
        _has_full_winner_info(winners.child),
        _has_full_winner_info(winners.youth),
        _has_full_winner_info(winners.teen),
        _has_full_winner_info(winners.adult),
    ])
    evaluator.add_custom_node(
        result=coverage_ok,
        id="all_four_age_categories_covered",
        desc="The response includes first-place winner information for all four required age categories: Child (5–8), Youth (9–12), Teen (13–17), Adult (18+).",
        parent=winners_node,
        critical=True,
    )

    async def _verify_one_category(
        cat_key: str,
        cat_label: str,
        age_range_text: str,
        item: Optional[WinnerItem],
    ):
        cat_node = evaluator.add_parallel(
            id=f"{cat_key}_category_winner",
            desc=f"{cat_label} category first-place winner details (name, city/state, entry title).",
            parent=winners_node,
            critical=True,
        )
        # Prepare fields
        name = (item.name if item else None) or ""
        city = (item.city if item else None) or ""
        state = (item.state if item else None) or ""
        entry_title = (item.entry_title if item else None) or ""
        urls = (item.urls if item else []) if item else []

        # Name leaf
        name_node = evaluator.add_leaf(
            id=f"{cat_key}_winner_name",
            desc=f"Provide the {cat_label}-category first-place winner's full name.",
            parent=cat_node,
            critical=True,
        )
        name_claim = (
            f"In the 2025 National Gingerbread House Competition at the Omni Grove Park Inn, "
            f"the first-place winner in the {cat_label} ({age_range_text}) category is {name}."
        )
        await evaluator.verify(
            claim=name_claim,
            node=name_node,
            sources=urls,
            additional_instruction="Use official winners announcements or credible event pages for 2025. Allow minor formatting variations in the winner's name.",
        )

        # Location leaf
        loc_node = evaluator.add_leaf(
            id=f"{cat_key}_winner_location",
            desc=f"Provide the {cat_label}-category first-place winner's city and state.",
            parent=cat_node,
            critical=True,
        )
        loc_text = _loc_str(city, state)
        loc_claim = (
            f"The listed location for the {cat_label} ({age_range_text}) first-place winner is {loc_text}."
        )
        await evaluator.verify(
            claim=loc_claim,
            node=loc_node,
            sources=urls,
            additional_instruction="Verify the city and state associated with the named first-place winner for the 2025 competition.",
        )

        # Entry title leaf
        title_node = evaluator.add_leaf(
            id=f"{cat_key}_entry_title",
            desc=f"Provide the {cat_label}-category first-place entry title.",
            parent=cat_node,
            critical=True,
        )
        title_claim = (
            f"The first-place entry title in the {cat_label} ({age_range_text}) category is '{entry_title}'."
        )
        await evaluator.verify(
            claim=title_claim,
            node=title_node,
            sources=urls,
            additional_instruction="Check the official 2025 winners list or equivalent credible source. Allow minor punctuation/casing variations for the title.",
        )

    await _verify_one_category("child", "Child", "ages 5–8", winners.child)
    await _verify_one_category("youth", "Youth", "ages 9–12", winners.youth)
    await _verify_one_category("teen", "Teen", "ages 13–17", winners.teen)
    await _verify_one_category("adult", "Adult", "ages 18+", winners.adult)


async def build_competition_rules_node(
    evaluator: Evaluator,
    parent,
    rules: RulesExtraction,
) -> None:
    """
    competition_rules (parallel, critical)
    - rules_are_official_for_this_competition (leaf)
    - age_categories_rule (leaf)
    - edible_materials_rule (leaf)
    - base_size_rule (leaf)
    """
    rules_node = evaluator.add_parallel(
        id="competition_rules",
        desc="State the official competition rules regarding age categories, edible materials requirement, and base size specifications.",
        parent=parent,
        critical=True,
    )

    rules_urls = rules.rules_urls if rules and rules.rules_urls else []
    age_txt = rules.age_categories_text or ""
    edible_txt = rules.edible_materials_text or ""
    base_txt = rules.base_size_text or ""

    # Official attribution
    official_node = evaluator.add_leaf(
        id="rules_are_official_for_this_competition",
        desc="The rules are explicitly attributed to the 2025 National Gingerbread House Competition at the Omni Grove Park Inn (not a different contest).",
        parent=rules_node,
        critical=True,
    )
    official_claim = (
        "These URLs are official rules for the 2025 National Gingerbread House Competition held at the Omni Grove Park Inn in Asheville, North Carolina (not a different contest or year)."
    )
    await evaluator.verify(
        claim=official_claim,
        node=official_node,
        sources=rules_urls,
        additional_instruction="Confirm the page clearly indicates official rules for the 2025 competition at Omni Grove Park Inn.",
    )

    # Age categories rule
    ages_node = evaluator.add_leaf(
        id="age_categories_rule",
        desc="Rules specify the four age categories: Child (5–8), Youth (9–12), Teen (13–17), Adult (18+).",
        parent=rules_node,
        critical=True,
    )
    ages_claim = (
        "The official rules specify four age categories: Child (ages 5–8), Youth (ages 9–12), Teen (ages 13–17), and Adult (ages 18+)."
    )
    await evaluator.verify(
        claim=ages_claim,
        node=ages_node,
        sources=rules_urls,
        additional_instruction="Small punctuation/typography differences (e.g., hyphen vs en-dash) are acceptable.",
    )

    # Edible materials rule
    edible_node = evaluator.add_leaf(
        id="edible_materials_rule",
        desc="Rules specify that decorations are 100% edible materials.",
        parent=rules_node,
        critical=True,
    )
    edible_claim = (
        "The official rules require that all decorations are made of 100% edible materials (exceptions such as the base/board may be noted in the rules)."
    )
    await evaluator.verify(
        claim=edible_claim,
        node=edible_node,
        sources=rules_urls,
        additional_instruction="Confirm the rule text asserts an all-edible requirement for decorations. Base/board exceptions are acceptable.",
    )

    # Base size rule
    base_node = evaluator.add_leaf(
        id="base_size_rule",
        desc="Rules specify a maximum base size and provide the size specification (e.g., 20×20 or 24×24 inches).",
        parent=rules_node,
        critical=True,
    )
    base_claim = (
        f"The official rules specify a maximum base size: {base_txt}."
        if base_txt.strip()
        else "The official rules specify a maximum base size in inches for the entry's base."
    )
    await evaluator.verify(
        claim=base_claim,
        node=base_node,
        sources=rules_urls,
        additional_instruction="If an exact dimension is provided (e.g., 24×24 inches), verify that dimension against the rules page.",
    )


async def build_professional_certification_node(
    evaluator: Evaluator,
    parent,
    cert: CertificationExtraction,
) -> None:
    """
    professional_certification (parallel, critical)
    - cert_program_name (leaf)
    - cert_program_duration (leaf)
    - cert_program_topics (leaf)
    """
    cert_node = evaluator.add_parallel(
        id="professional_certification",
        desc="Identify one professional certification program for holiday decorators meeting the stated duration and curriculum constraints.",
        parent=parent,
        critical=True,
    )

    prog_name = (cert.name or "").strip()
    duration_text = (cert.duration_text or "").strip()
    topics = cert.topics if cert and cert.topics else []
    urls = cert.urls if cert and cert.urls else []

    # Program name
    name_node = evaluator.add_leaf(
        id="cert_program_name",
        desc="Provide the certification program name.",
        parent=cert_node,
        critical=True,
    )
    name_claim = f"There is a professional certification program for holiday decorators called '{prog_name}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_node,
        sources=urls,
        additional_instruction="Confirm that the referenced program is a professional certification for holiday decorators (not a generic unrelated certification).",
    )

    # Duration (minimum 6 weeks)
    dur_node = evaluator.add_leaf(
        id="cert_program_duration",
        desc="Program requires a minimum of 6 weeks of training.",
        parent=cert_node,
        critical=True,
    )
    dur_claim = (
        f"The '{prog_name}' certification program requires at least six weeks of training."
    )
    await evaluator.verify(
        claim=dur_claim,
        node=dur_node,
        sources=urls,
        additional_instruction="Verify that the program duration is six weeks or longer based on the program page.",
    )

    # Topics (multiple)
    topics_node = evaluator.add_leaf(
        id="cert_program_topics",
        desc="Program covers multiple holiday decorating topics (such as wreaths, centerpieces, or tree decorating).",
        parent=cert_node,
        critical=True,
    )
    topics_list_text = ", ".join(topics[:5]) if topics else "multiple holiday decorating topics"
    topics_claim = (
        f"The '{prog_name}' certification program covers multiple holiday decorating topics, such as {topics_list_text}."
    )
    await evaluator.verify(
        claim=topics_claim,
        node=topics_node,
        sources=urls,
        additional_instruction="Confirm that the curriculum spans more than one distinct holiday decorating topic.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate the agent's answer for the 2025 National Gingerbread House Competition task.
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level aggregation
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

    # Run extractions (in parallel)
    event_extract_task = evaluator.extract(
        prompt=prompt_extract_event_context(),
        template_class=EventContextExtraction,
        extraction_name="event_context",
    )
    winners_extract_task = evaluator.extract(
        prompt=prompt_extract_winners(),
        template_class=WinnersExtraction,
        extraction_name="winners",
    )
    rules_extract_task = evaluator.extract(
        prompt=prompt_extract_rules(),
        template_class=RulesExtraction,
        extraction_name="rules",
    )
    cert_extract_task = evaluator.extract(
        prompt=prompt_extract_certification(),
        template_class=CertificationExtraction,
        extraction_name="certification",
    )

    event_info, winners_info, rules_info, cert_info = await asyncio.gather(
        event_extract_task, winners_extract_task, rules_extract_task, cert_extract_task
    )

    # Build the critical top-level node
    top_node = evaluator.add_parallel(
        id="national_gingerbread_competition_documentation",
        desc="Provide (a) first-place winners for all four age categories in the 2025 National Gingerbread House Competition at the Omni Grove Park Inn (Asheville, NC), (b) official rules on age categories, edible-materials, and base-size, and (c) one qualifying holiday-decorator certification program.",
        parent=root,
        critical=True,
    )

    # competition_event_context (leaf)
    await build_event_context_node(evaluator, top_node, event_info)

    # competition_winners
    await build_competition_winners_node(evaluator, top_node, winners_info)

    # competition_rules
    await build_competition_rules_node(evaluator, top_node, rules_info)

    # professional_certification
    await build_professional_certification_node(evaluator, top_node, cert_info)

    # Return evaluation summary
    return evaluator.get_summary()