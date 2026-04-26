import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "bigten_public_aacsb_2023_round1_university"
TASK_DESCRIPTION = """I'm researching universities that have strong academic credentials combined with recent success in Division I men's basketball. Specifically, I need to identify a public university that meets all of the following criteria:

1. The university must have an AACSB-accredited business school
2. The university must be a member of the Big Ten Conference
3. The university's men's basketball team must have participated in the 2023 NCAA Division I Men's Basketball Tournament
4. In the 2023 tournament, the team must have been seeded between #8 and #12 (inclusive)
5. The team must have won their first-round game in the 2023 tournament
6. This first-round victory must have occurred specifically on March 16, 2023
7. In that first-round game, the university's team must have defeated an opponent that was seeded higher than them (i.e., the opponent had a lower seed number)

Please identify this university and provide the following information:
- The university's name
- Confirmation of its public university status with a reference
- Confirmation of AACSB accreditation for its business school with a reference
- Confirmation of Big Ten Conference membership with a reference
- The team's seed number in the 2023 NCAA tournament with a reference
- The opponent they defeated in the first round, including the opponent's seed number
- The final score of that first-round game
- Reference documentation for the first-round game details

What university meets all these criteria?"""


# ---------------------------
# Data models for extraction
# ---------------------------
class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None

    # References for institutional facts
    public_status_sources: List[str] = Field(default_factory=list)
    aacsb_sources: List[str] = Field(default_factory=list)
    big_ten_sources: List[str] = Field(default_factory=list)

    # NCAA participation references
    ncaa_participation_sources: List[str] = Field(default_factory=list)

    # Seeding info
    seed_number_2023: Optional[str] = None
    seed_sources: List[str] = Field(default_factory=list)

    # First round game details
    first_round_opponent: Optional[str] = None
    first_round_opponent_seed: Optional[str] = None
    first_round_score: Optional[str] = None
    first_round_game_sources: List[str] = Field(default_factory=list)


# ---------------------------
# Extraction prompt
# ---------------------------
def prompt_extract_university_info() -> str:
    return """
Extract the university identification and all referenced evidence from the answer.

Return a JSON object with the following fields:
- university_name: The explicit name of the identified university.
- public_status_sources: An array of all URLs cited in the answer that support the university being a public institution (e.g., official university page, Wikipedia, state system pages). If none are cited, return an empty array.
- aacsb_sources: An array of URLs cited that support AACSB accreditation of the university’s business school (e.g., AACSB directory page, official college page). If none are cited, return an empty array.
- big_ten_sources: An array of URLs cited that support Big Ten Conference membership. If none are cited, return an empty array.
- ncaa_participation_sources: An array of URLs cited that support participation in the 2023 NCAA Division I Men's Basketball Tournament (e.g., NCAA bracket, team page, reputable news). If none are cited, return an empty array.
- seed_number_2023: The team's 2023 NCAA tournament seed as stated in the answer (include any leading '#' if present). If not provided, return null.
- seed_sources: An array of URLs cited that specifically support the team’s 2023 seed. If none are cited, return an empty array.
- first_round_opponent: The name of the first-round opponent as stated in the answer. If not provided, return null.
- first_round_opponent_seed: The opponent’s seed as stated in the answer (include any leading '#' if present). If not provided, return null.
- first_round_score: The final score of the first-round game as stated in the answer (preserve punctuation like hyphen/en-dash). If not provided, return null.
- first_round_game_sources: An array of URLs cited that provide details for the first-round game (date, opponent, seeds, score, result). If none are cited, return an empty array.

Rules:
- Only extract information explicitly present in the answer text.
- For URLs, extract the actual URLs (including those in markdown links).
- Do not invent URLs.
- If any requested information is missing from the answer, return null for the field (or empty list for URL arrays).
    """.strip()


# ---------------------------
# Helper functions
# ---------------------------
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def ensure_seed_prefix(seed: Optional[str]) -> Optional[str]:
    if not seed:
        return seed
    s = seed.strip()
    if not s:
        return None
    # Add leading '#' if missing and seed looks like a number
    if not s.startswith("#"):
        # If it already contains '#', keep it
        if "#" in s:
            return s
        # If it starts with digits
        if s[0].isdigit():
            return f"#{s}"
    return s


# ---------------------------
# Verification builder
# ---------------------------
async def build_verification_tree(evaluator: Evaluator, extraction: UniversityExtraction):
    # Create main critical sequential node (reflecting rubric root)
    main_node = evaluator.add_sequential(
        id="University_Identification_and_Verification",
        desc="Identifies a university and verifies it meets all specified criteria related to institutional type, accreditation, athletic conference membership, and 2023 NCAA tournament performance",
        parent=evaluator.root,
        critical=True
    )

    # 1) University named (critical)
    uni_named = evaluator.add_custom_node(
        result=bool(extraction.university_name and extraction.university_name.strip()),
        id="University_Named",
        desc="A specific university is clearly identified by name",
        parent=main_node,
        critical=True
    )

    # 2) Criteria verification (critical parallel)
    criteria_node = evaluator.add_parallel(
        id="Criteria_Verification",
        desc="The identified university meets all required institutional, accreditation, athletic, and tournament performance criteria",
        parent=main_node,
        critical=True
    )

    uni_name = extraction.university_name or ""

    # Public University Status (critical leaf)
    public_leaf = evaluator.add_leaf(
        id="Public_University_Status",
        desc="The university is confirmed as a public institution with supporting reference documentation",
        parent=criteria_node,
        critical=True
    )
    public_claim = f"{uni_name} is a public university (or public/state-related institution)."
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=extraction.public_status_sources,
        additional_instruction="Verify that the referenced page explicitly indicates the institution is public; allow variants such as 'public', 'public state-related', or 'public land-grant'."
    )

    # AACSB Accreditation (critical leaf)
    aacsb_leaf = evaluator.add_leaf(
        id="AACSB_Accreditation",
        desc="The university's business school holds AACSB accreditation with supporting reference documentation",
        parent=criteria_node,
        critical=True
    )
    aacsb_claim = f"The business school at {uni_name} is accredited by AACSB."
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_leaf,
        sources=extraction.aacsb_sources,
        additional_instruction="Look for explicit AACSB accreditation on the provided source pages (e.g., AACSB directory or official business school accreditation page)."
    )

    # Big Ten Membership (critical leaf)
    bigten_leaf = evaluator.add_leaf(
        id="Big_Ten_Membership",
        desc="The university is a member of the Big Ten Conference with supporting reference documentation",
        parent=criteria_node,
        critical=True
    )
    bigten_claim = f"{uni_name} is a member of the Big Ten Conference."
    await evaluator.verify(
        claim=bigten_claim,
        node=bigten_leaf,
        sources=extraction.big_ten_sources,
        additional_instruction="Confirm explicit Big Ten membership on the referenced page(s)."
    )

    # NCAA 2023 Participation (critical leaf)
    participation_leaf = evaluator.add_leaf(
        id="NCAA_2023_Tournament_Participation",
        desc="The university's men's basketball team participated in the 2023 NCAA Division I Tournament with supporting reference documentation",
        parent=criteria_node,
        critical=True
    )
    participation_claim = f"The men's basketball team of {uni_name} participated in the 2023 NCAA Division I Men's Basketball Tournament."
    part_sources = combine_sources(extraction.ncaa_participation_sources, extraction.first_round_game_sources, extraction.seed_sources)
    await evaluator.verify(
        claim=participation_claim,
        node=participation_leaf,
        sources=part_sources,
        additional_instruction="Accept evidence from NCAA bracket pages, reputable game recaps, or official athletic pages that explicitly show the team in the 2023 tournament."
    )

    # Tournament Seeding between #8 and #12 inclusive (critical leaf)
    seed_leaf = evaluator.add_leaf(
        id="Tournament_Seeding",
        desc="The university's team was seeded between #8 and #12 (inclusive) in the 2023 tournament with supporting reference documentation",
        parent=criteria_node,
        critical=True
    )
    team_seed = ensure_seed_prefix(extraction.seed_number_2023)
    if team_seed:
        seed_claim = f"In the 2023 NCAA tournament, {uni_name}'s men's basketball team was seeded {team_seed}, which is between #8 and #12 inclusive."
    else:
        seed_claim = f"In the 2023 NCAA tournament, {uni_name}'s men's basketball team had a seed between #8 and #12 inclusive."
    seed_sources = combine_sources(extraction.seed_sources, extraction.first_round_game_sources, extraction.ncaa_participation_sources)
    await evaluator.verify(
        claim=seed_claim,
        node=seed_leaf,
        sources=seed_sources,
        additional_instruction="Verify the tournament seed on the provided source(s); allow equivalent formatting like 'No. 10 seed' or '10-seed'."
    )

    # First Round Performance (critical parallel group)
    fr_group = evaluator.add_parallel(
        id="First_Round_Performance",
        desc="Verifies specific details of the university's first-round tournament game and outcome",
        parent=criteria_node,
        critical=True
    )

    # First Round Victory (critical leaf)
    fr_victory_leaf = evaluator.add_leaf(
        id="First_Round_Victory",
        desc="The university's team won their first-round game in the 2023 NCAA tournament with supporting reference documentation",
        parent=fr_group,
        critical=True
    )
    fr_victory_claim = f"{uni_name} won its first-round game in the 2023 NCAA tournament."
    fr_sources = combine_sources(extraction.first_round_game_sources, extraction.ncaa_participation_sources, extraction.seed_sources)
    await evaluator.verify(
        claim=fr_victory_claim,
        node=fr_victory_leaf,
        sources=fr_sources,
        additional_instruction="Confirm that the referenced page clearly labels the game as first round (Round of 64, not First Four) and shows a win."
    )

    # Game Date March 16, 2023 (critical leaf)
    fr_date_leaf = evaluator.add_leaf(
        id="Game_Date",
        desc="The first-round victory occurred on March 16, 2023 with supporting reference documentation",
        parent=fr_group,
        critical=True
    )
    fr_date_claim = "The first-round victory occurred on March 16, 2023."
    await evaluator.verify(
        claim=fr_date_claim,
        node=fr_date_leaf,
        sources=fr_sources,
        additional_instruction="Verify the game date shown on the referenced page(s). Minor timezone label differences are acceptable if the page clearly indicates March 16, 2023."
    )

    # Opponent Seeding Higher (critical leaf)
    fr_opp_seed_leaf = evaluator.add_leaf(
        id="Opponent_Seeding",
        desc="The defeated opponent was seeded higher (lower seed number) than the university's team with supporting reference documentation",
        parent=fr_group,
        critical=True
    )
    opp_name = extraction.first_round_opponent or ""
    opp_seed = ensure_seed_prefix(extraction.first_round_opponent_seed)
    if team_seed and opp_seed and opp_name:
        opp_seed_claim = f"In the first-round game, {uni_name} ({team_seed}) defeated {opp_name} ({opp_seed}), where the opponent had a lower seed number (i.e., was higher seeded) than {uni_name}."
    elif opp_name:
        opp_seed_claim = f"In the first-round game, {uni_name} defeated a higher-seeded opponent, {opp_name}."
    else:
        opp_seed_claim = f"In the first-round game, {uni_name} defeated a higher-seeded opponent."
    await evaluator.verify(
        claim=opp_seed_claim,
        node=fr_opp_seed_leaf,
        sources=fr_sources,
        additional_instruction="Confirm that the opponent’s seed number is lower than the team's seed number, indicating the opponent was higher seeded."
    )

    # Game Score (critical leaf)
    fr_score_leaf = evaluator.add_leaf(
        id="Game_Score",
        desc="The final score of the first-round game is documented and verifiable with supporting reference documentation",
        parent=fr_group,
        critical=True
    )
    score_text = extraction.first_round_score or ""
    if opp_name and score_text:
        fr_score_claim = f"The final score of the first-round game was {score_text}, with {uni_name} defeating {opp_name}."
    elif score_text:
        fr_score_claim = f"The final score of the first-round game was {score_text}, with {uni_name} winning."
    else:
        fr_score_claim = f"The final score of the first-round game is documented on the cited source(s) for {uni_name}."
    await evaluator.verify(
        claim=fr_score_claim,
        node=fr_score_leaf,
        sources=fr_sources,
        additional_instruction="Verify the exact final score (allow minor punctuation variations like hyphen vs en dash) and that it corresponds to the first-round game described."
    )


# ---------------------------
# Main evaluation entry
# ---------------------------
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
        strategy=AggregationStrategy.SEQUENTIAL,
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_extraction"
    )

    await build_verification_tree(evaluator, extraction)

    return evaluator.get_summary()