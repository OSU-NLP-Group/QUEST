import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_politics_early_2026"
TASK_DESCRIPTION = """In early 2026, two significant political developments occurred in the United States:

First, a special election runoff was held on January 31, 2026, for a Texas state senate seat in a district that had traditionally voted Republican. A Democratic candidate won this election.

Second, in February 2026, the U.S. House of Representatives passed a voter registration bill that requires individuals to provide documentary proof of U.S. citizenship when registering to vote and to show photo identification when voting.

Please provide the following information:

1. Texas Special Election (January 31, 2026):
   - The name of the winning Democratic candidate
   - The political party affiliation of the winner
   - The exact number of votes the winner received in the runoff
   - The specific Texas Senate district number
   - A reference URL documenting these election results

2. Federal Voter Registration Legislation (passed by U.S. House in February 2026):
   - The bill number (with H.R. designation)
   - The full official name of the act
   - Confirmation that the bill requires documentary proof of U.S. citizenship at voter registration
   - Confirmation that the bill requires photo identification at the time of voting
   - A reference URL documenting the bill and its requirements

3. Congressional Veto Override Requirements:
   - The constitutional threshold required in each chamber (expressed as a fraction or percentage)
   - The exact number of votes needed in the U.S. House of Representatives to override a presidential veto, assuming all 435 members are present and voting
   - The exact number of votes needed in the U.S. Senate to override a presidential veto, assuming all 100 members are present and voting
   - Clarification that both chambers must independently achieve this threshold to override a veto
   - A reference URL documenting these constitutional requirements
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TexasElection(BaseModel):
    winner_name: Optional[str] = None
    winner_party: Optional[str] = None
    vote_count: Optional[str] = None
    district: Optional[str] = None
    election_date: Optional[str] = None
    source_url: Optional[str] = None


class FederalBill(BaseModel):
    bill_number: Optional[str] = None
    bill_name: Optional[str] = None
    citizenship_proof_requirement: Optional[str] = None
    photo_id_requirement: Optional[str] = None
    passage_timing: Optional[str] = None
    source_url: Optional[str] = None


class VetoOverride(BaseModel):
    constitutional_threshold: Optional[str] = None
    house_vote_count: Optional[str] = None
    senate_vote_count: Optional[str] = None
    both_chambers_requirement: Optional[str] = None
    source_url: Optional[str] = None


class TaskExtraction(BaseModel):
    texas_election: Optional[TexasElection] = None
    federal_bill: Optional[FederalBill] = None
    veto_override: Optional[VetoOverride] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract structured information from the answer for three sections: (1) Texas special election runoff (2) Federal voter registration bill passed by the House (3) Constitutional veto override requirements.
    
    Return a JSON object with keys: "texas_election", "federal_bill", and "veto_override".
    
    1) texas_election:
       - winner_name: the name of the winning Democratic candidate, exactly as stated in the answer
       - winner_party: the party affiliation of the winner (e.g., "Democrat"), exactly as stated in the answer
       - vote_count: the exact number of votes the winner received in the runoff (keep commas or formatting as shown in the answer)
       - district: the Texas Senate district number or label (e.g., "District 30" or "SD-30"), exactly as stated in the answer
       - election_date: the runoff election date, exactly as stated in the answer
       - source_url: a single URL provided in the answer that documents the election results; if multiple URLs are provided, pick the most authoritative or the first clearly relevant one
    
    2) federal_bill:
       - bill_number: the H.R. bill number (e.g., "H.R. 1234"), exactly as stated in the answer
       - bill_name: the full official name of the act, exactly as stated in the answer
       - citizenship_proof_requirement: copy the answer's phrasing that confirms the bill requires documentary proof of U.S. citizenship at voter registration (e.g., "requires proof of U.S. citizenship")
       - photo_id_requirement: copy the answer's phrasing that confirms the bill requires photo identification at the time of voting
       - passage_timing: the statement in the answer confirming the bill was passed by the House in February 2026 (e.g., "passed in February 2026")
       - source_url: a single URL provided in the answer that documents the bill and its requirements; if multiple URLs are provided, pick the most authoritative or the first clearly relevant one
    
    3) veto_override:
       - constitutional_threshold: the constitutional threshold statement (e.g., "two-thirds") exactly as given in the answer
       - house_vote_count: the exact number stated for House votes needed (expect "290" when all 435 members vote), exactly as in the answer
       - senate_vote_count: the exact number stated for Senate votes needed (expect "67" when all 100 members vote), exactly as in the answer
       - both_chambers_requirement: the answer's statement clarifying that both chambers must independently achieve the threshold
       - source_url: a single URL provided in the answer documenting these constitutional requirements; if multiple URLs are provided, pick the most authoritative or the first clearly relevant one
    
    Rules:
    - Extract only what is explicitly present in the answer text.
    - If any field is missing in the answer, set it to null.
    - For URLs, extract the actual URL string.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _str(x: Optional[str]) -> str:
    return x or ""


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_texas_election(evaluator: Evaluator, parent_node, info: Optional[TexasElection]) -> None:
    node = evaluator.add_parallel(
        id="texas_election_identification",
        desc="Identify and provide details about the Texas state senate special election runoff held on January 31, 2026",
        parent=parent_node,
        critical=True
    )

    # Existence check: all required fields present
    exist = evaluator.add_custom_node(
        result=(
            info is not None and
            bool(_str(info.winner_name).strip()) and
            bool(_str(info.winner_party).strip()) and
            bool(_str(info.vote_count).strip()) and
            bool(_str(info.district).strip()) and
            bool(_str(info.election_date).strip()) and
            bool(_str(info.source_url).strip())
        ),
        id="texas_election_required_fields",
        desc="Texas election: required fields (winner name, party, vote count, district, date, source URL) are provided",
        parent=node,
        critical=True
    )

    src = _str(info.source_url) if info else None

    # winner_name
    leaf_winner_name = evaluator.add_leaf(
        id="winner_name",
        desc="Correctly identify the name of the winning candidate",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning candidate's name is '{_str(info.winner_name)}'.",
        node=leaf_winner_name,
        sources=src,
        additional_instruction="Verify the page explicitly names the runoff winner for the Texas Senate special election."
    )

    # winner_party
    leaf_winner_party = evaluator.add_leaf(
        id="winner_party",
        desc="Correctly identify the winning candidate's party affiliation as Democrat",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning candidate {_str(info.winner_name)} is a Democrat.",
        node=leaf_winner_party,
        sources=src,
        additional_instruction="Confirm the winner's party is Democratic (allow 'Democrat' or 'Democratic Party' variants)."
    )

    # vote_count
    leaf_vote_count = evaluator.add_leaf(
        id="vote_count",
        desc="Provide the exact number of votes received by the winning candidate",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The winning candidate received exactly '{_str(info.vote_count)}' votes.",
        node=leaf_vote_count,
        sources=src,
        additional_instruction="Match the exact vote total; allow minor formatting differences such as commas or spaces."
    )

    # district
    leaf_district = evaluator.add_leaf(
        id="district",
        desc="Correctly identify the specific Texas Senate district number",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The special election runoff was for Texas Senate {_str(info.district)}.",
        node=leaf_district,
        sources=src,
        additional_instruction="Confirm the district designation (e.g., 'District 30', 'SD-30'); allow minor format variants."
    )

    # election_date
    leaf_date = evaluator.add_leaf(
        id="election_date",
        desc="Confirm the runoff election date as January 31, 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The runoff election occurred on January 31, 2026.",
        node=leaf_date,
        sources=src,
        additional_instruction="Confirm the election date on the source page."
    )

    # source_url validity
    leaf_src = evaluator.add_leaf(
        id="source_url",
        desc="Provide a reliable reference URL documenting the election results",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page documents the January 31, 2026 Texas Senate special election runoff results for {_str(info.district)}, including the winner and vote totals.",
        node=leaf_src,
        sources=src,
        additional_instruction="The page should credibly report or present official results of the specified runoff."
    )


async def verify_federal_legislation(evaluator: Evaluator, parent_node, info: Optional[FederalBill]) -> None:
    node = evaluator.add_parallel(
        id="federal_legislation_identification",
        desc="Identify and provide details about the voter registration bill passed by the U.S. House in February 2026",
        parent=parent_node,
        critical=True
    )

    # Existence check: essential fields present (bill_number and source_url)
    exist = evaluator.add_custom_node(
        result=(
            info is not None and
            bool(_str(info.bill_number).strip()) and
            bool(_str(info.source_url).strip())
        ),
        id="federal_legislation_required_fields",
        desc="Federal legislation: required fields (bill number and source URL) are provided",
        parent=node,
        critical=True
    )

    src = _str(info.source_url) if info else None

    # bill_number
    leaf_bill_number = evaluator.add_leaf(
        id="bill_number",
        desc="Correctly identify the bill number (H.R. designation)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The bill number is '{_str(info.bill_number)}'.",
        node=leaf_bill_number,
        sources=src,
        additional_instruction="Verify the exact H.R. bill number on the page."
    )

    # bill_name (set as critical to satisfy critical-parent constraint)
    leaf_bill_name = evaluator.add_leaf(
        id="bill_name",
        desc="Provide the full official name of the act",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The full official title of this act is '{_str(info.bill_name)}'.",
        node=leaf_bill_name,
        sources=src,
        additional_instruction="Match the official act title shown on the page; allow minor punctuation variants."
    )

    # citizenship_proof_requirement
    leaf_citizenship = evaluator.add_leaf(
        id="citizenship_proof_requirement",
        desc="Confirm the bill requires documentary proof of U.S. citizenship at voter registration",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This bill requires documentary proof of U.S. citizenship when registering to vote.",
        node=leaf_citizenship,
        sources=src,
        additional_instruction="Confirm language that clearly mandates documentary proof of citizenship at registration; allow synonymous phrasing."
    )

    # photo_id_requirement
    leaf_photo_id = evaluator.add_leaf(
        id="photo_id_requirement",
        desc="Confirm the bill requires photo ID at the time of voting",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This bill requires voters to present photo identification at the time of voting.",
        node=leaf_photo_id,
        sources=src,
        additional_instruction="Confirm a requirement to show photo ID when voting; allow synonymous phrasing."
    )

    # passage_timing
    leaf_passage = evaluator.add_leaf(
        id="passage_timing",
        desc="Confirm the bill was passed by the House in February 2026",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The U.S. House of Representatives passed this bill in February 2026.",
        node=leaf_passage,
        sources=src,
        additional_instruction="Confirm the page indicates House passage occurring in February 2026."
    )

    # source_url validity
    leaf_src = evaluator.add_leaf(
        id="source_url",
        desc="Provide a reliable reference URL documenting the bill and its requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"This page documents the bill {_str(info.bill_number)} and its requirements (citizenship proof and photo ID).",
        node=leaf_src,
        sources=src,
        additional_instruction="The page should credibly describe the bill’s identity and the stated requirements."
    )


async def verify_veto_override(evaluator: Evaluator, parent_node, info: Optional[VetoOverride]) -> None:
    node = evaluator.add_parallel(
        id="veto_override_calculation",
        desc="Determine and explain the constitutional requirements for congressional veto override",
        parent=parent_node,
        critical=True
    )

    # Existence check: source URL present
    exist = evaluator.add_custom_node(
        result=(info is not None and bool(_str(info.source_url).strip())),
        id="veto_override_required_fields",
        desc="Veto override: source URL is provided",
        parent=node,
        critical=True
    )

    src = _str(info.source_url) if info else None

    # constitutional_threshold
    leaf_threshold = evaluator.add_leaf(
        id="constitutional_threshold",
        desc="Correctly state that a two-thirds vote is required in each chamber",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="A two-thirds vote in each chamber of Congress is required to override a presidential veto.",
        node=leaf_threshold,
        sources=src,
        additional_instruction="Confirm constitutional language (Article I, Section 7) or authoritative source indicating a two-thirds requirement."
    )

    # house_vote_calculation
    leaf_house_calc = evaluator.add_leaf(
        id="house_vote_calculation",
        desc="Correctly calculate that 290 votes are needed in the House when all 435 members vote",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="When all 435 members vote, 290 votes are needed in the House to override a presidential veto.",
        node=leaf_house_calc,
        sources=src,
        additional_instruction="Confirm an explicit statement or a calculation (2/3 of 435 = 290) for the House."
    )

    # senate_vote_calculation
    leaf_senate_calc = evaluator.add_leaf(
        id="senate_vote_calculation",
        desc="Correctly calculate that 67 votes are needed in the Senate when all 100 members vote",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="When all 100 senators vote, 67 votes are needed in the Senate to override a presidential veto.",
        node=leaf_senate_calc,
        sources=src,
        additional_instruction="Confirm an explicit statement or a calculation (2/3 of 100 = 67) for the Senate."
    )

    # both_chambers_requirement
    leaf_both = evaluator.add_leaf(
        id="both_chambers_requirement",
        desc="Clearly state that both chambers must achieve the two-thirds threshold",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Both the U.S. House and the U.S. Senate must each independently achieve the two-thirds threshold to override a presidential veto.",
        node=leaf_both,
        sources=src,
        additional_instruction="Confirm that override requires separate two-thirds votes in both chambers."
    )

    # source_url validity
    leaf_src = evaluator.add_leaf(
        id="source_url",
        desc="Provide a reliable reference URL documenting the constitutional override requirements",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page documents the constitutional requirements and vote thresholds to override a presidential veto.",
        node=leaf_src,
        sources=src,
        additional_instruction="The page should credibly describe the override process and thresholds."
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

    # Extract all info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=TaskExtraction,
        extraction_name="structured_extraction"
    )

    # Build and verify subtrees
    await verify_texas_election(evaluator, root, extraction.texas_election)
    await verify_federal_legislation(evaluator, root, extraction.federal_bill)
    await verify_veto_override(evaluator, root, extraction.veto_override)

    return evaluator.get_summary()