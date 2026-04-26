import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "fashion_milestones_2024"
TASK_DESCRIPTION = """
In the fashion industry during 2024, several professionals achieved notable milestones and recognition. Identify four specific individuals or groups based on the following criteria:

1. Fashion Awards 2024 Designer: Identify the designer who won the Designer of the Year award at the Fashion Awards 2024 (organized by the British Fashion Council in December) for the second consecutive year. Provide the designer's full name, both fashion brands they lead as creative director, confirmation of consecutive wins in 2023 and 2024, the name of the organizing body, and the specific date and venue of the 2024 ceremony.

2. L'Oréal Paris Long-Term Ambassador: Identify the L'Oréal Paris global ambassador who celebrated their 20-year partnership anniversary with the brand in 2025. Provide their full name, confirmation of the 20-year milestone in 2025, evidence of their participation in the L'Oréal Paris Fashion Week runway show in September 2024 (including the show's official name, date, and specific location in Paris), and confirmation of their official ambassador role.

3. Los Angeles Styling Duo: Identify the Los Angeles-based styling duo known for styling an acclaimed actor who had standout red carpet fashion appearances in 2024. Provide both individual names in the duo, their professional/brand name as a duo, the full name of the specific acclaimed actor they style (who was known for exceptional red carpet looks in 2024), confirmation of their Los Angeles base, and the names of at least two other celebrity clients they style.

4. Historic Model Achievement: Identify the model who made history by becoming the first transgender person to win the Model of the Year award at the Fashion Awards 2024. Provide the model's full name, the exact award title, confirmation that this was the first time a transgender model won this specific award, confirmation that the award was presented at the Fashion Awards 2024, and additional career details including their age or birth year at the time of the award and at least one major fashion campaign or brand they've worked with.

For each individual or group, provide supporting URL references that verify the information.
"""


# -----------------------------
# Pydantic models for extraction
# -----------------------------
class BrandRole(BaseModel):
    brand: Optional[str] = None
    role: Optional[str] = None  # e.g., "creative director", "artistic director"


class Item1Designer(BaseModel):
    designer_full_name: Optional[str] = None
    brands_led: List[BrandRole] = Field(default_factory=list)
    organizer: Optional[str] = None
    ceremony_date: Optional[str] = None
    ceremony_venue: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class Item2Loreal(BaseModel):
    ambassador_full_name: Optional[str] = None
    milestone_year: Optional[str] = None  # expect "2025" if mentioned
    official_ambassador_role: Optional[str] = None  # e.g., "global ambassador"
    runway_show_official_name: Optional[str] = None  # "Le Défilé L'Oréal Paris – Walk Your Worth"
    runway_show_date: Optional[str] = None  # e.g., "September 23, 2024"
    runway_show_location: Optional[str] = None  # e.g., "Place de l'Opéra, Paris"
    supporting_urls: List[str] = Field(default_factory=list)


class Item3StylingDuo(BaseModel):
    duo_member_names: List[str] = Field(default_factory=list)  # two names
    duo_professional_name: Optional[str] = None  # e.g., "Wayman + Micah"
    base_location: Optional[str] = None  # expect includes "Los Angeles"
    primary_actor_full_name: Optional[str] = None
    additional_celebrity_clients: List[str] = Field(default_factory=list)  # at least two
    supporting_urls: List[str] = Field(default_factory=list)


class Item4ModelAchievement(BaseModel):
    model_full_name: Optional[str] = None
    exact_award_title: Optional[str] = None  # expect "Model of the Year"
    age_in_2024: Optional[str] = None  # e.g., "21"
    birth_year: Optional[str] = None  # e.g., "2003"
    major_campaign_or_brand: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class FashionMilestonesExtraction(BaseModel):
    item_1: Optional[Item1Designer] = None
    item_2: Optional[Item2Loreal] = None
    item_3: Optional[Item3StylingDuo] = None
    item_4: Optional[Item4ModelAchievement] = None


# -----------------------------
# Extraction prompt
# -----------------------------
def prompt_extract_all() -> str:
    return """
Extract the following structured information from the answer text. Do not invent any information; only extract what is explicitly provided. If a field is missing, set it to null (or empty list for URL arrays).

Return a JSON matching this schema:

{
  "item_1": {
    "designer_full_name": string|null,
    "brands_led": [{"brand": string|null, "role": string|null}, ...],   // include up to 2 entries if provided
    "organizer": string|null,
    "ceremony_date": string|null,        // e.g., "December 2, 2024"
    "ceremony_venue": string|null,       // e.g., "Royal Albert Hall, London" or "Royal Albert Hall"
    "supporting_urls": [string, ...]     // all URLs cited for this item
  },
  "item_2": {
    "ambassador_full_name": string|null,
    "milestone_year": string|null,       // year text, e.g., "2025"
    "official_ambassador_role": string|null, // e.g., "global ambassador"
    "runway_show_official_name": string|null, // e.g., "Le Défilé L'Oréal Paris – Walk Your Worth"
    "runway_show_date": string|null,     // e.g., "September 23, 2024"
    "runway_show_location": string|null, // e.g., "Place de l'Opéra, Paris"
    "supporting_urls": [string, ...]
  },
  "item_3": {
    "duo_member_names": [string, ...],   // list of individual names in the duo
    "duo_professional_name": string|null,
    "base_location": string|null,        // e.g., must mention "Los Angeles" if provided
    "primary_actor_full_name": string|null,
    "additional_celebrity_clients": [string, ...], // at least two if provided
    "supporting_urls": [string, ...]
  },
  "item_4": {
    "model_full_name": string|null,
    "exact_award_title": string|null,    // expect "Model of the Year" if provided
    "age_in_2024": string|null,          // "21" if explicitly given
    "birth_year": string|null,           // e.g., "2003" if given
    "major_campaign_or_brand": string|null, // at least one brand/campaign name
    "supporting_urls": [string, ...]
  }
}

Special rules:
- Extract all URLs related to each item into the respective "supporting_urls" array (do not mix across items).
- For brands in item_1, if roles like "creative director", "artistic director" are mentioned, include them in "role".
- Keep original capitalization and formatting of names and titles as presented in the answer.
"""


# -----------------------------
# Helper utilities
# -----------------------------
def _non_empty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


def _has_min_distinct(items: List[str], k: int) -> bool:
    return len({x.strip() for x in items if _non_empty(x)}) >= k


async def _verify_leaf_with_sources(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    leaf_desc: str,
    claim: str,
    sources: Optional[List[str]],
    additional_instruction: str = "None",
    critical: bool = True,
    require_sources: bool = True
):
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=leaf_desc,
        parent=parent,
        critical=critical
    )
    # Enforce source grounding when required
    if require_sources and (not sources or len(sources) == 0):
        node.score = 0.0
        node.status = "failed"
        return False

    # Delegate to the verifier
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources if sources and len(sources) > 0 else None,
        additional_instruction=additional_instruction
    )
    return node.score == 1.0


# -----------------------------
# Verification routines per item
# -----------------------------
async def verify_item_1(evaluator: Evaluator, root, item: Optional[Item1Designer]) -> None:
    item_node = evaluator.add_parallel(
        id="item_1_fashion_awards_designer",
        desc="Fashion Awards 2024 Designer: Designer of the Year winner with a consecutive win (2023 & 2024), including organizer and 2024 ceremony details",
        parent=root,
        critical=False
    )
    urls = item.supporting_urls if item else []

    # designer_full_name
    if not (item and _non_empty(item.designer_full_name)):
        node = evaluator.add_leaf(
            id="designer_full_name",
            desc="Provide the designer’s full name",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "designer_full_name",
            "Provide the designer’s full name",
            claim=f"The Designer of the Year at The Fashion Awards 2024 was {item.designer_full_name}.",
            sources=urls,
            additional_instruction="Check the page explicitly names the 2024 'Designer of the Year' winner."
        )

    # two_brands_led_as_creative_director
    brand_pairs = item.brands_led if item else []
    b1 = brand_pairs[0] if len(brand_pairs) >= 1 else None
    b2 = brand_pairs[1] if len(brand_pairs) >= 2 else None
    if not (b1 and _non_empty(b1.brand) and b2 and _non_empty(b2.brand)):
        node = evaluator.add_leaf(
            id="two_brands_led_as_creative_director",
            desc="Provide the names of the two distinct fashion brands the designer leads as creative director (and indicate the creative-director role for both)",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        role1 = b1.role or "creative director"
        role2 = b2.role or "creative director"
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "two_brands_led_as_creative_director",
            "Provide the names of the two distinct fashion brands the designer leads as creative director (and indicate the creative-director role for both)",
            claim=f"{item.designer_full_name} serves as {role1} at {b1.brand} and as {role2} at {b2.brand}.",
            sources=urls,
            additional_instruction="Confirm both brand appointments; role titles like 'creative director' or 'artistic director' are acceptable equivalents."
        )

    # consecutive_designer_of_year_wins_2023_2024
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "consecutive_designer_of_year_wins_2023_2024",
        "Verify the designer won the Designer of the Year award at the Fashion Awards in both 2023 and 2024 (a consecutive win)",
        claim=f"{item.designer_full_name if item else 'The designer'} won the 'Designer of the Year' at The Fashion Awards in both 2023 and 2024 (consecutive years).",
        sources=urls,
        additional_instruction="Verify that reputable sources explicitly confirm wins in both 2023 and 2024."
    )

    # organizing_body_is_british_fashion_council
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "organizing_body_is_british_fashion_council",
        "State and verify that the organizing body was the British Fashion Council",
        claim="The Fashion Awards are organized by the British Fashion Council.",
        sources=urls,
        additional_instruction="Look for explicit mention that the British Fashion Council organizes The Fashion Awards."
    )

    # ceremony_date_is_dec_2_2024 (fixed expected date)
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "ceremony_date_is_dec_2_2024",
        "State and verify that the 2024 ceremony date was December 2, 2024",
        claim="The 2024 Fashion Awards ceremony took place on December 2, 2024.",
        sources=urls,
        additional_instruction="Confirm date is 2 December 2024 (allow DD Month YYYY or Month DD, YYYY formats)."
    )

    # ceremony_venue_is_royal_albert_hall_london (fixed expected venue)
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "ceremony_venue_is_royal_albert_hall_london",
        "State and verify that the 2024 ceremony venue was the Royal Albert Hall in London",
        claim="The 2024 Fashion Awards ceremony was held at the Royal Albert Hall in London.",
        sources=urls,
        additional_instruction="Confirm venue explicitly as Royal Albert Hall, London."
    )

    # supporting_urls_item_1_reliable
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "supporting_urls_item_1_reliable",
        "Provide URL reference(s) from reliable sources that directly support the above claims for this item",
        claim="This page is a reputable/official source that reports on The Fashion Awards 2024 winners or event details.",
        sources=urls,
        additional_instruction="Treat official British Fashion Council pages or renowned outlets (e.g., Vogue, BBC, The Guardian, BOF, WWD) as reliable. Verify the page discusses The Fashion Awards 2024 and relevant winner/event details."
    )


async def verify_item_2(evaluator: Evaluator, root, item: Optional[Item2Loreal]) -> None:
    item_node = evaluator.add_parallel(
        id="item_2_loreal_ambassador",
        desc="L’Oréal Paris Long-Term Ambassador: global ambassador with a 20-year partnership anniversary in 2025 and participation in the specified September 2024 runway show",
        parent=root,
        critical=False
    )
    urls = item.supporting_urls if item else []

    # ambassador_full_name
    if not (item and _non_empty(item.ambassador_full_name)):
        node = evaluator.add_leaf(
            id="ambassador_full_name",
            desc="Provide the ambassador’s full name",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "ambassador_full_name",
            "Provide the ambassador’s full name",
            claim=f"The L'Oréal Paris global ambassador in question is {item.ambassador_full_name}.",
            sources=urls,
            additional_instruction="Confirm that the page names the ambassador associated with the 20-year partnership and runway appearance."
        )

    # 20_year_partnership_anniversary_in_2025
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "20_year_partnership_anniversary_in_2025",
        "Verify they celebrated a 20-year partnership anniversary with L’Oréal Paris in 2025",
        claim=f"In 2025, {item.ambassador_full_name if item else 'the ambassador'} celebrated a 20-year partnership with L'Oréal Paris.",
        sources=urls,
        additional_instruction="Look for explicit wording like '20 years' or '20th anniversary' tied to the year 2025."
    )

    # official_global_ambassador_role
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "official_global_ambassador_role",
        "Verify they are an official L’Oréal Paris global ambassador",
        claim=f"{item.ambassador_full_name if item else 'The person'} is an official L'Oréal Paris global ambassador.",
        sources=urls,
        additional_instruction="Confirm official ambassador status, ideally on L'Oréal Paris channels or credible media."
    )

    # runway_show_participation_sept_2024
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "runway_show_participation_sept_2024",
        "Verify their participation in the L’Oréal Paris Fashion Week runway show occurring in September 2024",
        claim=f"{item.ambassador_full_name if item else 'The ambassador'} participated in the L'Oréal Paris runway show during Paris Fashion Week in September 2024.",
        sources=urls,
        additional_instruction="Verify attendance/participation in the 2024 Paris Fashion Week L’Oréal Paris show."
    )

    # runway_show_official_name_matches_constraint
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "runway_show_official_name_matches_constraint",
        "State and verify the show’s official name was “Le Défilé L'Oréal Paris – Walk Your Worth”",
        claim="The show's official name was 'Le Défilé L'Oréal Paris – Walk Your Worth'.",
        sources=urls,
        additional_instruction="Allow minor punctuation/hyphen/quote variations but name must clearly match."
    )

    # runway_show_date_matches_constraint
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "runway_show_date_matches_constraint",
        "State and verify the show date was September 23, 2024",
        claim="The show took place on September 23, 2024.",
        sources=urls,
        additional_instruction="Confirm the exact date; allow formats like '23 September 2024'."
    )

    # runway_show_location_matches_constraint
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "runway_show_location_matches_constraint",
        "State and verify the show location was Place de l'Opéra in Paris",
        claim="The show was staged at Place de l'Opéra in Paris.",
        sources=urls,
        additional_instruction="Confirm mention of Place de l'Opéra, Paris (Opéra Garnier square)."
    )

    # supporting_urls_item_2_reliable
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "supporting_urls_item_2_reliable",
        "Provide URL reference(s) from reliable sources that directly support the above claims for this item",
        claim="This page is an official or reputable source about L'Oréal Paris ambassadors or the 2024/2025 events described.",
        sources=urls,
        additional_instruction="Treat official brand channels or renowned outlets (e.g., Vogue, Harper's Bazaar, BOF, WWD, Reuters) as reliable. Verify relevance to the claims."
    )


async def verify_item_3(evaluator: Evaluator, root, item: Optional[Item3StylingDuo]) -> None:
    item_node = evaluator.add_parallel(
        id="item_3_styling_duo",
        desc="Los Angeles Styling Duo: LA-based duo with a duo brand name, known for styling an acclaimed actor with standout 2024 red carpet looks, plus at least two additional celebrity clients",
        parent=root,
        critical=False
    )
    urls = item.supporting_urls if item else []

    # duo_member_names
    if not (item and len([n for n in (item.duo_member_names or []) if _non_empty(n)]) >= 2):
        node = evaluator.add_leaf(
            id="duo_member_names",
            desc="Provide both individual names of the styling duo",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        n1, n2 = [n for n in item.duo_member_names if _non_empty(n)][:2]
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "duo_member_names",
            "Provide both individual names of the styling duo",
            claim=f"The styling duo comprises {n1} and {n2}.",
            sources=urls,
            additional_instruction="Verify that these two individuals are publicly known as a styling duo."
        )

    # duo_professional_name
    if not (item and _non_empty(item.duo_professional_name)):
        node = evaluator.add_leaf(
            id="duo_professional_name",
            desc="Provide the professional/brand name used by the duo",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "duo_professional_name",
            "Provide the professional/brand name used by the duo",
            claim=f"The duo operate under the professional/brand name '{item.duo_professional_name}'.",
            sources=urls,
            additional_instruction="Confirm the duo's brand/professional moniker as used in reputable sources."
        )

    # los_angeles_based
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "los_angeles_based",
        "Verify the duo is based in Los Angeles",
        claim="The styling duo is based in Los Angeles.",
        sources=urls,
        additional_instruction="Look for bio or profile text indicating 'Los Angeles-based' or similar wording."
    )

    # primary_actor_full_name
    if not (item and _non_empty(item.primary_actor_full_name)):
        node = evaluator.add_leaf(
            id="primary_actor_full_name",
            desc="Provide the full name of the specific acclaimed actor the duo styles",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "primary_actor_full_name",
            "Provide the full name of the specific acclaimed actor the duo styles",
            claim=f"The duo styles the actor {item.primary_actor_full_name}.",
            sources=urls,
            additional_instruction="Confirm that this actor is a client of the duo."
        )

    # evidence_duo_styles_primary_actor
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "evidence_duo_styles_primary_actor",
        "Provide evidence (via sources) that the duo styles the specified primary actor",
        claim=f"The styling duo works with (styles) {item.primary_actor_full_name if item else 'the actor'}.",
        sources=urls,
        additional_instruction="Verify explicit statements that the duo styles this actor, such as stylist credits, profiles, or interviews."
    )

    # evidence_primary_actor_standout_2024_red_carpet
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "evidence_primary_actor_standout_2024_red_carpet",
        "Provide evidence (via sources) that the specified actor was described as having standout/notable red-carpet fashion appearances in 2024",
        claim=f"In 2024, {item.primary_actor_full_name if item else 'the actor'} was widely described as having standout or notable red-carpet fashion.",
        sources=urls,
        additional_instruction="Look for credible awards-season or fashion coverage in 2024 describing the actor's standout red carpet looks."
    )

    # two_additional_celebrity_clients
    additional_clients = item.additional_celebrity_clients if item else []
    if not (_has_min_distinct(additional_clients, 2)):
        node = evaluator.add_leaf(
            id="two_additional_celebrity_clients",
            desc="Provide names of at least two other celebrity clients the duo styles",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        c = [x for x in additional_clients if _non_empty(x)]
        c1, c2 = c[0], c[1]
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "two_additional_celebrity_clients",
            "Provide names of at least two other celebrity clients the duo styles",
            claim=f"The duo also styles {c1} and {c2}.",
            sources=urls,
            additional_instruction="Verify both names are listed as clients in reputable profiles or interviews."
        )

    # supporting_urls_item_3_reliable
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "supporting_urls_item_3_reliable",
        "Provide URL reference(s) from reliable sources that directly support the above claims for this item",
        claim="This page is a reputable source about the styling duo and their clients.",
        sources=urls,
        additional_instruction="Treat outlets like The Hollywood Reporter, Vogue, Vanity Fair, WWD, or official websites/socials with press mentions as reliable if they explicitly confirm claims."
    )


async def verify_item_4(evaluator: Evaluator, root, item: Optional[Item4ModelAchievement]) -> None:
    item_node = evaluator.add_parallel(
        id="item_4_historic_model_achievement",
        desc="Historic Model Achievement: first transgender Model of the Year winner at Fashion Awards 2024, plus age/birth year at time and at least one major campaign/brand",
        parent=root,
        critical=False
    )
    urls = item.supporting_urls if item else []

    # model_full_name
    if not (item and _non_empty(item.model_full_name)):
        node = evaluator.add_leaf(
            id="model_full_name",
            desc="Provide the model’s full name",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "model_full_name",
            "Provide the model’s full name",
            claim=f"The 'Model of the Year' winner at The Fashion Awards 2024 was {item.model_full_name}.",
            sources=urls,
            additional_instruction="Confirm the model's full name as the 2024 Model of the Year winner."
        )

    # exact_award_title_model_of_the_year
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "exact_award_title_model_of_the_year",
        "Provide the exact award title (Model of the Year)",
        claim="The exact award title is 'Model of the Year'.",
        sources=urls,
        additional_instruction="Verify the exact phrase 'Model of the Year' is used for the award."
    )

    # presented_at_fashion_awards_2024
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "presented_at_fashion_awards_2024",
        "Verify the Model of the Year award was presented at the Fashion Awards 2024",
        claim="The 'Model of the Year' award was presented at The Fashion Awards 2024.",
        sources=urls,
        additional_instruction="Confirm that this award is part of The Fashion Awards 2024 ceremony."
    )

    # first_transgender_winner_claim
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "first_transgender_winner_claim",
        "Verify this was the first time a transgender person won this specific award",
        claim=f"{item.model_full_name if item else 'The winner'} was the first transgender person to win the 'Model of the Year' award.",
        sources=urls,
        additional_instruction="Look for clear wording like 'first transgender model/person to win the award'."
    )

    # age_matches_constraint_21_in_2024
    # Build a robust claim incorporating either age or birth year if available
    if item and _non_empty(item.age_in_2024) and _non_empty(item.birth_year):
        claim_age = f"In 2024, {item.model_full_name} was {item.age_in_2024} years old (born in {item.birth_year})."
    elif item and _non_empty(item.age_in_2024):
        claim_age = f"In 2024, {item.model_full_name} was {item.age_in_2024} years old."
    elif item and _non_empty(item.birth_year):
        claim_age = f"In 2024, {item.model_full_name} was 21 years old (born in {item.birth_year})."
    else:
        claim_age = f"In 2024, {item.model_full_name if item else 'the model'} was 21 years old."
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "age_matches_constraint_21_in_2024",
        "State and verify the model was 21 years old at the time of the award in 2024",
        claim=claim_age,
        sources=urls,
        additional_instruction="Confirm the model's age in 2024 is 21, or confirm the birth year consistent with age 21 in 2024 (e.g., 2003)."
    )

    # major_campaign_or_brand
    if not (item and _non_empty(item.major_campaign_or_brand)):
        node = evaluator.add_leaf(
            id="major_campaign_or_brand",
            desc="Provide at least one major fashion campaign or brand the model has worked with",
            parent=item_node,
            critical=True
        )
        node.score = 0.0
        node.status = "failed"
    else:
        await _verify_leaf_with_sources(
            evaluator,
            item_node,
            "major_campaign_or_brand",
            "Provide at least one major fashion campaign or brand the model has worked with",
            claim=f"{item.model_full_name} has worked with {item.major_campaign_or_brand}.",
            sources=urls,
            additional_instruction="Verify a major campaign or brand collaboration (e.g., luxury houses, major campaigns)."
        )

    # supporting_urls_item_4_reliable
    await _verify_leaf_with_sources(
        evaluator,
        item_node,
        "supporting_urls_item_4_reliable",
        "Provide URL reference(s) from reliable sources that directly support the above claims for this item",
        claim="This page is a reputable/official source that reports on The Fashion Awards 2024 or the model's achievements.",
        sources=urls,
        additional_instruction="Treat the British Fashion Council or top-tier media (Vogue, WWD, BOF, BBC, The Guardian) as reliable; verify relevance."
    )


# -----------------------------
# Main evaluation entrypoint
# -----------------------------
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
        prompt=prompt_extract_all(),
        template_class=FashionMilestonesExtraction,
        extraction_name="fashion_milestones_extraction"
    )

    # Build and verify four items in parallel structure
    await verify_item_1(evaluator, root, extracted.item_1)
    await verify_item_2(evaluator, root, extracted.item_2)
    await verify_item_3(evaluator, root, extracted.item_3)
    await verify_item_4(evaluator, root, extracted.item_4)

    return evaluator.get_summary()