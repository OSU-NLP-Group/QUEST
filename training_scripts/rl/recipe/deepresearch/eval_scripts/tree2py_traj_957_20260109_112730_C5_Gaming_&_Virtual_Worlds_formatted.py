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
TASK_ID = "gaming_vr_entities_2025"
TASK_DESCRIPTION = """Identify the following four entities from the gaming and virtual reality industry, each meeting specific criteria:

1. VR Arcade Chain: Name a VR arcade entertainment company operating in the United States that had more than 50 locations globally as of 2024-2025 and announced franchise expansion plans or significant growth during 2024.

2. Gaming Industry Conference: Name a major gaming industry conference that took place in March 2025 at the Moscone Center in San Francisco, California.

3. VR Headset: Name a virtual reality headset that was released in October 2023 and features a display resolution of 2064x2208 pixels per eye.

4. Game Development Studio: Name a PlayStation game development studio that is based in Tokyo, Japan, was originally formed in 2012, and developed a game that won Game of the Year at The Game Awards 2024.

For each entity, provide the name and at least one reference URL that verifies the information."""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VRArcadeChain(BaseModel):
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class GamingEvent(BaseModel):
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VRHeadset(BaseModel):
    model: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class GameStudio(BaseModel):
    name: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class EntitiesExtraction(BaseModel):
    vr_arcade_chain: Optional[VRArcadeChain] = None
    gaming_event: Optional[GamingEvent] = None
    vr_headset: Optional[VRHeadset] = None
    game_studio: Optional[GameStudio] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_entities() -> str:
    return """
    Extract the four entities the answer provides for this task. For each category, return the entity's name/model and all reference URLs that the answer cites for that entity.

    Output JSON structure:
    {
      "vr_arcade_chain": {
        "name": string | null,
        "source_urls": string[]    // all URLs explicitly present in the answer for this chain
      },
      "gaming_event": {
        "name": string | null,
        "source_urls": string[]    // all URLs explicitly present in the answer for this event
      },
      "vr_headset": {
        "model": string | null,
        "source_urls": string[]    // all URLs explicitly present in the answer for this headset
      },
      "game_studio": {
        "name": string | null,
        "source_urls": string[]    // all URLs explicitly present in the answer for this studio
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - For URLs, include only valid URLs; if a protocol is missing, prepend http://
    - If the answer provides multiple URLs for an entity, include all of them in source_urls.
    - If a field is not provided in the answer, return null (for name/model) or [] (for source_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    out = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # If missing protocol, prepend http:// (Extractor also attempts this)
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_vr_arcade_chain_checks(evaluator: Evaluator, parent_node, data: Optional[VRArcadeChain]) -> None:
    node = evaluator.add_parallel(
        id="VR_Arcade_Chain",
        desc="Identify a VR arcade entertainment company operating in the US with >50 global locations (2024–2025) and 2024 expansion announcement",
        parent=parent_node,
        critical=False
    )

    name = (data.name if data else None) or ""
    sources = _norm_urls(data.source_urls if data else [])

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="vr_arcade_chain_Chain_Name",
        desc="Provide the name of the VR arcade chain",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sources),
        id="vr_arcade_chain_Reference_URL",
        desc="Provide at least one reference URL that substantiates the required VR arcade chain criteria",
        parent=node,
        critical=True
    )

    # Verification leaves (critical)
    us_op_node = evaluator.add_leaf(
        id="vr_arcade_chain_US_Operation",
        desc="Verify the VR arcade chain operates in the United States",
        parent=node,
        critical=True
    )
    loc_cnt_node = evaluator.add_leaf(
        id="vr_arcade_chain_Location_Count",
        desc="Verify the chain operated more than 50 locations globally as of 2024–2025",
        parent=node,
        critical=True
    )
    expansion_node = evaluator.add_leaf(
        id="vr_arcade_chain_Expansion_2024",
        desc="Verify the chain announced franchise expansion plans or significant growth during 2024",
        parent=node,
        critical=True
    )

    # Batch verify to avoid cross-sibling skip due to auto preconditions
    claims = [
        (
            f"The VR arcade company '{name}' operates in the United States.",
            sources,
            us_op_node,
            "Confirm that the company has U.S. locations or is explicitly described as operating in the United States. Accept 'US', 'U.S.', or 'United States' variants."
        ),
        (
            f"As of 2024 or 2025, the VR arcade company '{name}' operated more than 50 locations globally.",
            sources,
            loc_cnt_node,
            "Verify that the source indicates the company had over 50 locations globally in the 2024–2025 timeframe. Accept phrases like '50+', 'more than 50', 'over 50'."
        ),
        (
            f"In 2024, the VR arcade company '{name}' announced franchise expansion plans or significant growth.",
            sources,
            expansion_node,
            "Look for 2024 press releases or credible articles mentioning a franchise program, expansion plans, significant growth, or new openings announced in 2024."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_gaming_event_checks(evaluator: Evaluator, parent_node, data: Optional[GamingEvent]) -> None:
    node = evaluator.add_parallel(
        id="Gaming_Industry_Event",
        desc="Identify a major gaming industry conference that took place in March 2025 at Moscone Center, San Francisco",
        parent=parent_node,
        critical=False
    )

    name = (data.name if data else None) or ""
    sources = _norm_urls(data.source_urls if data else [])

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="gaming_event_Event_Name",
        desc="Provide the name of the gaming industry conference",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sources),
        id="gaming_event_Reference_URL",
        desc="Provide at least one reference URL that substantiates the required conference criteria",
        parent=node,
        critical=True
    )

    # Verification leaves (critical)
    major_node = evaluator.add_leaf(
        id="gaming_event_Major_Status",
        desc="Verify the conference is characterized as a major gaming industry conference",
        parent=node,
        critical=True
    )
    date_node = evaluator.add_leaf(
        id="gaming_event_Event_Date",
        desc="Verify the event took place in March 2025",
        parent=node,
        critical=True
    )
    location_node = evaluator.add_leaf(
        id="gaming_event_Event_Location",
        desc="Verify the event was held at the Moscone Center in San Francisco, California",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"The conference '{name}' is a major gaming industry conference.",
            sources,
            major_node,
            "Accept descriptors like 'major', 'largest', 'leading', 'flagship', 'premier', or equivalent language in the provided sources."
        ),
        (
            f"The conference '{name}' took place in March 2025.",
            sources,
            date_node,
            "Confirm that the event dates include March 2025. Multi‑day ranges that include March 2025 are acceptable."
        ),
        (
            f"The conference '{name}' was held at the Moscone Center in San Francisco, California.",
            sources,
            location_node,
            "Confirm the venue is Moscone Center, San Francisco, CA."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_vr_headset_checks(evaluator: Evaluator, parent_node, data: Optional[VRHeadset]) -> None:
    node = evaluator.add_parallel(
        id="VR_Headset_Technology",
        desc="Identify a VR headset released in October 2023 with 2064×2208 pixels per‑eye resolution",
        parent=parent_node,
        critical=False
    )

    model_name = (data.model if data else None) or ""
    sources = _norm_urls(data.source_urls if data else [])

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(model_name.strip()),
        id="vr_headset_Headset_Model",
        desc="Provide the model name of the VR headset",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sources),
        id="vr_headset_Reference_URL",
        desc="Provide at least one reference URL that substantiates the required headset criteria",
        parent=node,
        critical=True
    )

    # Verification leaves (critical)
    release_node = evaluator.add_leaf(
        id="vr_headset_Release_Date",
        desc="Verify the headset was released in October 2023",
        parent=node,
        critical=True
    )
    resolution_node = evaluator.add_leaf(
        id="vr_headset_Display_Resolution",
        desc="Verify the headset features a display resolution of 2064×2208 pixels per eye",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"The VR headset '{model_name}' was released in October 2023.",
            sources,
            release_node,
            "Accept 'released', 'launched', or 'went on sale' in October 2023. Exact day is not required as long as the month/year is October 2023."
        ),
        (
            f"The VR headset '{model_name}' has a per‑eye display resolution of 2064 × 2208 pixels.",
            sources,
            resolution_node,
            "Allow minor formatting variants such as '2064x2208', '2064 × 2208', or '2064 by 2208'. Ensure it's per‑eye resolution."
        ),
    ]
    await evaluator.batch_verify(claims)


async def build_game_studio_checks(evaluator: Evaluator, parent_node, data: Optional[GameStudio]) -> None:
    node = evaluator.add_parallel(
        id="Game_Studio_Information",
        desc="Identify a PlayStation studio based in Tokyo (formed in 2012) that developed a TGA 2024 Game of the Year winner",
        parent=parent_node,
        critical=False
    )

    name = (data.name if data else None) or ""
    sources = _norm_urls(data.source_urls if data else [])

    # Existence checks (critical)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id="studio_Studio_Name",
        desc="Provide the name of the game development studio",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(sources),
        id="studio_Reference_URL",
        desc="Provide at least one reference URL that substantiates the required studio criteria",
        parent=node,
        critical=True
    )

    # Verification leaves (critical)
    loc_node = evaluator.add_leaf(
        id="studio_Studio_Location",
        desc="Verify the studio is based in Tokyo, Japan",
        parent=node,
        critical=True
    )
    year_node = evaluator.add_leaf(
        id="studio_Formation_Year",
        desc="Verify the studio was originally formed in 2012",
        parent=node,
        critical=True
    )
    ps_aff_node = evaluator.add_leaf(
        id="studio_PlayStation_Affiliation",
        desc="Verify the studio is a PlayStation Studios developer",
        parent=node,
        critical=True
    )
    award_node = evaluator.add_leaf(
        id="studio_Award_Win",
        desc="Verify the studio developed a game that won Game of the Year at The Game Awards 2024",
        parent=node,
        critical=True
    )

    claims = [
        (
            f"The studio '{name}' is based in Tokyo, Japan.",
            sources,
            loc_node,
            "Look for explicit mention that the studio is based in Tokyo, Japan. Headquarters or primary office in Tokyo is acceptable."
        ),
        (
            f"The studio '{name}' was originally formed in 2012.",
            sources,
            year_node,
            "Accept synonyms like 'formed', 'founded', 'established' in 2012. If multiple dates are given, confirm the original formation year is 2012."
        ),
        (
            f"The studio '{name}' is a PlayStation Studios developer.",
            sources,
            ps_aff_node,
            "Verify that the studio is part of PlayStation Studios/first‑party under Sony Interactive Entertainment, or is explicitly labeled as such."
        ),
        (
            f"The studio '{name}' developed a game that won 'Game of the Year' at The Game Awards 2024.",
            sources,
            award_node,
            "Confirm the game won the 'Game of the Year' award at The Game Awards 2024 and that this studio is credited as the developer (co‑developer acceptable)."
        ),
    ]
    await evaluator.batch_verify(claims)


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
    Evaluate an answer for the four-entity gaming/VR identification task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Evaluate four categories independently
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

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_entities(),
        template_class=EntitiesExtraction,
        extraction_name="entities_extraction",
    )

    # 2) Build verification subtrees per category
    await build_vr_arcade_chain_checks(evaluator, root, extracted.vr_arcade_chain if extracted else None)
    await build_gaming_event_checks(evaluator, root, extracted.gaming_event if extracted else None)
    await build_vr_headset_checks(evaluator, root, extracted.vr_headset if extracted else None)
    await build_game_studio_checks(evaluator, root, extracted.game_studio if extracted else None)

    # 3) Return summary
    return evaluator.get_summary()