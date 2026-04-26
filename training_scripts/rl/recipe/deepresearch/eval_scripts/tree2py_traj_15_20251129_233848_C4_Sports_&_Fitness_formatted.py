import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cfb_player_georgia_rb_2024"
TASK_DESCRIPTION = (
    "Identify a college football player who meets ALL of the following criteria for the 2024-2025 season: "
    "(1) Plays the position of running back, (2) Currently plays for the Georgia Bulldogs, "
    "(3) Competes in the Southeastern Conference (SEC), (4) Is from the state of California (born or raised in California), "
    "(5) Attended Mater Dei High School, (6) Is classified as a sophomore during the 2024-2025 academic year. "
    "Provide the player's full name and include the URL of their official university athletics profile page as a reference."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class PlayerExtraction(BaseModel):
    """Information about the identified player as extracted from the answer."""
    full_name: Optional[str] = None
    position: Optional[str] = None
    team: Optional[str] = None
    conference: Optional[str] = None
    origin_detail: Optional[str] = None  # e.g., "Santa Ana, CA" or "Born in Los Angeles, California"
    origin_state: Optional[str] = None   # e.g., "California", "Calif.", or "CA"
    high_school: Optional[str] = None
    class_year: Optional[str] = None     # e.g., "Sophomore", "So.", "RS-Sophomore"
    profile_url: Optional[str] = None    # official roster/profile page on georgiadogs.com
    additional_urls: List[str] = Field(default_factory=list)  # any other URLs included by the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_player_info() -> str:
    return """
    Extract the identified college football player's information exactly as stated in the answer. Do not invent or infer details that are not explicitly provided in the answer.

    Fields to extract (use null if a field is not present in the answer):
    - full_name: The player's full name (not a nickname alone).
    - position: The player's position as written (e.g., "Running Back", "RB").
    - team: The college team name (e.g., "Georgia Bulldogs", "Georgia", "University of Georgia").
    - conference: The athletic conference (e.g., "SEC", "Southeastern Conference").
    - origin_detail: The origin text describing birthplace or where the player is from, as stated (e.g., "Santa Ana, CA", "born in Los Angeles, California", "from Anaheim, Calif.").
    - origin_state: The state string exactly as mentioned (e.g., "California", "Calif.", "CA"). Do not infer from city if the state is not explicitly mentioned.
    - high_school: The player's high school name as stated (e.g., "Mater Dei High School", "Mater Dei").
    - class_year: The class standing as stated for the 2024 season (e.g., "Sophomore", "So.", "RS-Sophomore").
    - profile_url: The URL to the player's official university athletics profile page (if provided). Prefer the official athletics site (e.g., georgiadogs.com). If multiple URLs are provided, pick the one that is the official athletics player profile.
    - additional_urls: Any other URLs present in the answer related to the player.

    Return a single JSON object with these fields exactly. Do not add fields. Do not infer missing information.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def build_all_urls(extracted: PlayerExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.profile_url and isinstance(extracted.profile_url, str) and extracted.profile_url.strip():
        urls.append(extracted.profile_url.strip())
    if extracted.additional_urls:
        urls.extend([u for u in extracted.additional_urls if isinstance(u, str) and u.strip()])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped


def is_valid_georgia_profile_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        # Official Georgia Bulldogs athletics domain
        if "georgiadogs.com" not in host:
            return False
        # Typical football roster/player profile path usually includes /sports/football and roster or player segments
        # Allow flexible patterns: roster, players, bio, or typical roster-profile URLs
        valid_path = ("/sports/football" in path) and ("roster" in path or "player" in path or "bio" in path)
        return valid_path
    except Exception:
        return False


def name_is_full(name: Optional[str]) -> bool:
    if not name or not isinstance(name, str):
        return False
    tokens = [t for t in name.strip().split() if t]
    return len(tokens) >= 2


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def verify_player_criteria(
    evaluator: Evaluator,
    root_parent,
    extracted: PlayerExtraction
) -> None:
    """
    Build the verification tree as specified by the rubric and run all verifications.
    Root is a critical parallel node; all children are critical leaves/custom nodes.
    """
    root = evaluator.add_parallel(
        id="Player_Identification",
        desc="Identify a college football player who meets all specified criteria for the 2024-2025 season and provide the required reference information",
        parent=root_parent,
        critical=True
    )

    # Prepare URLs for evidence-based checks
    all_urls = build_all_urls(extracted)
    profile_url = extracted.profile_url if extracted.profile_url else None
    player_name = extracted.full_name or "the identified player"

    # 1) Full_Name_Provided (custom existence check)
    evaluator.add_custom_node(
        result=name_is_full(extracted.full_name),
        id="Full_Name_Provided",
        desc="The response provides the player's full name (not only a nickname or partial name)",
        parent=root,
        critical=True
    )

    # 2) Position_Verification (verify by official profile if available)
    node_pos = evaluator.add_leaf(
        id="Position_Verification",
        desc="The identified player plays the position of running back (RB)",
        parent=root,
        critical=True
    )
    claim_pos = f"On the linked official player profile page for {player_name}, the listed position is Running Back (RB) (allow 'Running Back', 'RB', or similar)."
    await evaluator.verify(
        claim=claim_pos,
        node=node_pos,
        sources=all_urls if all_urls else None,
        additional_instruction="Accept synonyms and abbreviations such as 'RB', 'Running Back', 'Tailback'. Minor case or punctuation differences are acceptable."
    )

    # 3) Team_Verification
    node_team = evaluator.add_leaf(
        id="Team_Verification",
        desc="The identified player currently plays for the Georgia Bulldogs football team",
        parent=root,
        critical=True
    )
    claim_team = (
        "The linked profile page is the official University of Georgia (Georgia Bulldogs) athletics player profile, "
        "and it indicates the player is on the Georgia Bulldogs football team (UGA)."
    )
    await evaluator.verify(
        claim=claim_team,
        node=node_team,
        sources=all_urls if all_urls else None,
        additional_instruction="Verify the site branding (georgiadogs.com) and football team affiliation on the page. Accept 'Georgia', 'UGA', or 'Georgia Bulldogs' as equivalent team identifiers."
    )

    # 4) Conference_Verification (simple verification from the answer context)
    node_conf = evaluator.add_leaf(
        id="Conference_Verification",
        desc="The identified player competes in the Southeastern Conference (SEC)",
        parent=root,
        critical=True
    )
    claim_conf = "The Georgia Bulldogs football team competes in the Southeastern Conference (SEC)."
    await evaluator.verify(
        claim=claim_conf,
        node=node_conf,
        sources=None,
        additional_instruction="Judge primarily based on the provided answer text and task context; accept if the answer explicitly states or clearly implies SEC membership. Do not require the profile page to explicitly mention SEC."
    )

    # 5) California_Origin_Verification
    node_origin = evaluator.add_leaf(
        id="California_Origin_Verification",
        desc="The identified player is from the state of California (born or raised in California)",
        parent=root,
        critical=True
    )
    claim_origin = (
        f"On the linked official profile, {player_name}'s hometown or birthplace is in the state of California. "
        "Accept 'California', 'Calif.', or 'CA' as indicating California."
    )
    await evaluator.verify(
        claim=claim_origin,
        node=node_origin,
        sources=all_urls if all_urls else None,
        additional_instruction="Look for 'Hometown' or biographical text indicating a California location. Treat 'Calif.' or 'CA' as California. If multiple locations are listed (birth vs hometown), either being in California satisfies the condition."
    )

    # 6) High_School_Verification
    node_hs = evaluator.add_leaf(
        id="High_School_Verification",
        desc="The identified player attended Mater Dei High School",
        parent=root,
        critical=True
    )
    claim_hs = (
        f"The linked official player profile for {player_name} indicates the player attended Mater Dei High School "
        "(accept 'Mater Dei' or 'Mater Dei HS' as equivalent)."
    )
    await evaluator.verify(
        claim=claim_hs,
        node=node_hs,
        sources=all_urls if all_urls else None,
        additional_instruction="Check the bio or 'Hometown/High School' field for 'Mater Dei'. Minor variations like 'Mater Dei HS' are acceptable."
    )

    # 7) Class_Standing_Verification
    node_class = evaluator.add_leaf(
        id="Class_Standing_Verification",
        desc="The identified player is classified as a sophomore during the 2024-2025 academic year",
        parent=root,
        critical=True
    )
    claim_class = (
        f"On the linked official player profile for {player_name}, the player's class for the 2024 season is Sophomore. "
        "Accept 'So.', 'Sophomore', or 'RS-Sophomore/Redshirt Sophomore' as valid Sophomore classifications."
    )
    await evaluator.verify(
        claim=claim_class,
        node=node_class,
        sources=all_urls if all_urls else None,
        additional_instruction="Treat roster year 2024 as the 2024-2025 academic year. Consider 'So.' and 'Redshirt Sophomore' as satisfying 'Sophomore'."
    )

    # 8) Official_Profile_URL_Verification (custom format/domain presence check)
    evaluator.add_custom_node(
        result=is_valid_georgia_profile_url(profile_url),
        id="Official_Profile_URL_Verification",
        desc="The response includes a valid URL to the player's official university athletics profile page (on the official university athletics website)",
        parent=root,
        critical=True
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Georgia Bulldogs RB 2024-2025 player identification task.
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
        default_model=model
    )

    # Extract structured player information from the answer
    extracted: PlayerExtraction = await evaluator.extract(
        prompt=prompt_extract_player_info(),
        template_class=PlayerExtraction,
        extraction_name="player_extraction"
    )

    # Build verification tree and run checks
    await verify_player_criteria(evaluator, root, extracted)

    # Return the unified summary
    return evaluator.get_summary()