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
TASK_ID = "maduro_briefing_jan2026"
TASK_DESCRIPTION = """You are a legal policy analyst preparing a comprehensive briefing document on the January 2026 capture and legal proceedings of Venezuelan President Nicolás Maduro for a congressional oversight committee. Your briefing must include:

1. Federal Court Information: Identify the specific U.S. federal district court handling the criminal case, including the complete street address of the courthouse where proceedings are taking place, the date of Maduro's initial arraignment, and the currently scheduled next hearing date.

2. State Department Reward History: Document the complete chronological progression of the State Department's reward offer for Maduro's capture, including all three stages with specific amounts and dates: the initial reward (amount and year), the first increase (amount and specific date), and the second increase (amount, specific date, and the Treasury Department designation that triggered this increase, including both the entity designated and the date of that designation).

3. Supreme Court Legal Precedents: Identify and describe at least two relevant U.S. Supreme Court cases that legal experts have cited as applicable to Maduro's situation: (a) one case establishing that unlawful foreign abduction does not bar prosecution in U.S. courts, and (b) one case relevant to either head-of-state immunity or presidential authority to recognize foreign governments. For each case, provide the complete case name, year decided, and a brief explanation of its legal principle.

4. International Law Framework: Document the international law context, including: (a) the specific UN Charter article that governs the use of force between nations, (b) the UN Secretary-General's characterization of the U.S. operation, and (c) mention of U.S. Congressional responses under the War Powers Resolution.

5. Operation Timeline: Provide a chronological timeline of key dates, including the date of the U.S. military operation, the date of Maduro's first court appearance, and the year of the original indictment.

For each major section, provide at least one URL reference to an official government source or credible news outlet that supports the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SectionCourt(BaseModel):
    urls: List[str] = Field(default_factory=list)
    district_court: Optional[str] = None
    courthouse_address: Optional[str] = None
    initial_arraignment_date: Optional[str] = None
    next_hearing_date: Optional[str] = None


class SectionReward(BaseModel):
    urls: List[str] = Field(default_factory=list)
    initial_reward_amount: Optional[str] = None
    initial_reward_year_or_date: Optional[str] = None
    first_increase_amount: Optional[str] = None
    first_increase_date: Optional[str] = None
    second_increase_amount: Optional[str] = None
    second_increase_date: Optional[str] = None
    treasury_designation_entity: Optional[str] = None
    treasury_designation_date: Optional[str] = None
    treasury_designation_type: Optional[str] = None
    treasury_designation_url: Optional[str] = None
    nrp_first_over_25m: Optional[bool] = None
    nrp_first_over_25m_claim_text: Optional[str] = None


class CaseInfo(BaseModel):
    case_name: Optional[str] = None
    year: Optional[str] = None
    principle: Optional[str] = None


class SectionPrecedents(BaseModel):
    urls: List[str] = Field(default_factory=list)
    abduction: Optional[CaseInfo] = None
    immunity_or_recognition: Optional[CaseInfo] = None


class SectionInternational(BaseModel):
    urls: List[str] = Field(default_factory=list)
    un_charter_article: Optional[str] = None
    unsg_characterization: Optional[str] = None
    war_powers_mention: Optional[str] = None


class SectionTimeline(BaseModel):
    urls: List[str] = Field(default_factory=list)
    operation_date: Optional[str] = None
    first_court_appearance_date: Optional[str] = None
    original_indictment_date_or_year: Optional[str] = None


class BriefingExtraction(BaseModel):
    court: SectionCourt = Field(default_factory=SectionCourt)
    reward: SectionReward = Field(default_factory=SectionReward)
    precedents: SectionPrecedents = Field(default_factory=SectionPrecedents)
    international: SectionInternational = Field(default_factory=SectionInternational)
    timeline: SectionTimeline = Field(default_factory=SectionTimeline)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_briefing() -> str:
    return """
Extract the requested structured information from the answer for the five required sections. Return a single JSON object with the following schema.

court:
- urls: an array of URLs cited in the federal court section (official government sites or credible outlets if present)
- district_court: the name of the U.S. federal district court handling the criminal case, exactly as stated in the answer
- courthouse_address: the street address stated in the answer for the courthouse where proceedings are taking place
- initial_arraignment_date: the date of Maduro's initial arraignment as written in the answer (keep the format used by the answer)
- next_hearing_date: the currently scheduled next hearing date as written in the answer

reward:
- urls: an array of URLs cited for the reward progression
- initial_reward_amount: the initial reward amount as written (e.g., "$15 million")
- initial_reward_year_or_date: the year or specific date for the initial reward (e.g., "2020")
- first_increase_amount: amount for the first increase (e.g., "$25 million")
- first_increase_date: date for the first increase (e.g., "January 10, 2025")
- second_increase_amount: amount for the second increase (e.g., "$50 million")
- second_increase_date: date for the second increase (e.g., "August 7, 2025")
- treasury_designation_entity: the designated entity that triggered the second increase (e.g., "Cartel of the Suns")
- treasury_designation_date: the date of that Treasury designation (e.g., "July 25, 2025")
- treasury_designation_type: the designation type (e.g., "Specially Designated Global Terrorist")
- treasury_designation_url: a URL (if provided) that documents that Treasury designation
- nrp_first_over_25m: boolean if the answer asserts that Maduro is the first Narcotics Rewards Program target exceeding $25 million
- nrp_first_over_25m_claim_text: the exact phrasing of that claim if present

precedents:
- urls: an array of URLs cited for Supreme Court precedents
- abduction: object with:
  - case_name: case name as written (e.g., "United States v. Alvarez-Machain")
  - year: year decided (e.g., "1992")
  - principle: a short explanation of the legal rule in the answer
- immunity_or_recognition: object with:
  - case_name: the case name provided for head-of-state immunity or presidential recognition authority (e.g., "Schooner Exchange v. McFaddon" or "Zivotofsky v. Kerry")
  - year: year decided (e.g., "1812" or "2015")
  - principle: a short explanation of the legal rule in the answer

international:
- urls: an array of URLs cited for the international-law context
- un_charter_article: the specific UN Charter article number named (e.g., "Article 2(4)")
- unsg_characterization: the quoted or paraphrased language used by the UN Secretary-General
- war_powers_mention: text describing Congress's War Powers Resolution response mentioned in the answer

timeline:
- urls: an array of URLs cited for the timeline
- operation_date: the date of the U.S. military operation as stated in the answer
- first_court_appearance_date: the date of Maduro's first court appearance as stated in the answer
- original_indictment_date_or_year: the original indictment date or year as stated in the answer

Rules:
- Extract only what appears in the answer; do not invent.
- For dates, keep the format used in the answer.
- For URLs, extract the full URL strings exactly as they appear (including markdown links).
- If a field is missing, return null for that field; if a URL list is missing, return an empty array.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*args: Optional[List[str] | str]) -> List[str]:
    urls: List[str] = []
    for item in args:
        if isinstance(item, list):
            for u in item:
                if isinstance(u, str):
                    u2 = u.strip()
                    if u2 and u2 not in urls:
                        urls.append(u2)
        elif isinstance(item, str):
            u2 = item.strip()
            if u2 and u2 not in urls:
                urls.append(u2)
    return urls


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_court_details(evaluator: Evaluator, parent_node, court: SectionCourt) -> None:
    node = evaluator.add_parallel(
        id="Federal_Court_Details",
        desc="Federal court identification, courthouse address, initial arraignment date, and next hearing date, with a supporting URL.",
        parent=parent_node,
        critical=True
    )

    # URL presence (critical gating)
    evaluator.add_custom_node(
        result=bool(court and court.urls and len(court.urls) > 0),
        id="Court_Info_URL",
        desc="Provide ≥1 URL from an official government source or a credible news outlet supporting the federal court information section.",
        parent=node,
        critical=True
    )

    urls = court.urls if court else []

    # District court assertion (expected SDNY)
    district_leaf = evaluator.add_leaf(
        id="District_Court",
        desc="Identify the criminal case as being in the U.S. District Court for the Southern District of New York (SDNY).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The federal criminal case is being handled by the U.S. District Court for the Southern District of New York (SDNY).",
        node=district_leaf,
        sources=urls,
        additional_instruction="Verify the specific federal district named for the case. Allow minor naming variations (e.g., 'S.D.N.Y.'), but it must be the Southern District of New York."
    )

    # Courthouse address (expected 500 Pearl Street)
    address_leaf = evaluator.add_leaf(
        id="Courthouse_Address",
        desc="State the courthouse street address as 500 Pearl Street, New York, NY 10007.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The courthouse address for the proceedings is 500 Pearl Street, New York, NY 10007.",
        node=address_leaf,
        sources=urls,
        additional_instruction="Check whether the page states that the proceedings are taking place at 500 Pearl Street, New York, NY 10007. Accept minor formatting variants (e.g., 'St.' vs 'Street')."
    )

    # Initial arraignment date (expected Jan 5, 2026)
    arraign_leaf = evaluator.add_leaf(
        id="Initial_Arraignment_Date_In_Court_Section",
        desc="In the federal-court section, state the initial arraignment date as January 5, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Maduro's initial arraignment date in federal court was January 5, 2026.",
        node=arraign_leaf,
        sources=urls,
        additional_instruction="Confirm that the initial arraignment occurred on January 5, 2026. Allow reasonable date formatting variants (e.g., 'Jan. 5, 2026')."
    )

    # Next hearing date (expected Mar 26, 2026)
    next_hearing_leaf = evaluator.add_leaf(
        id="Next_Hearing_Date",
        desc="State the currently scheduled next hearing date as March 26, 2026.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The next scheduled hearing date is March 26, 2026.",
        node=next_hearing_leaf,
        sources=urls,
        additional_instruction="Verify that at least one cited page clearly lists the next hearing for March 26, 2026."
    )


async def verify_reward_history(evaluator: Evaluator, parent_node, reward: SectionReward) -> None:
    node = evaluator.add_parallel(
        id="Reward_History",
        desc="Chronological progression of the State Department reward offer (3 stages) including the trigger linkage, plus a supporting URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(reward and reward.urls and len(reward.urls) > 0),
        id="Reward_Info_URL",
        desc="Provide ≥1 URL from an official government source or a credible news outlet supporting the reward-history section.",
        parent=node,
        critical=True
    )

    urls = reward.urls if reward else []
    all_reward_sources = _combine_sources(urls, reward.treasury_designation_url if reward else None)

    # Combined progression + trigger
    progression_leaf = evaluator.add_leaf(
        id="Reward_Progression_All_Stages_And_Trigger",
        desc="Document the complete 3-stage reward progression with amounts/dates and the Treasury designation trigger details.",
        parent=node,
        critical=True
    )
    claim_progression = (
        "The State Department reward for Nicolás Maduro progressed as follows: "
        "initial $15 million in 2020; increased to $25 million on January 10, 2025; "
        "increased to $50 million on August 7, 2025; and the second increase was linked to the Treasury Department's "
        "designation of the 'Cartel of the Suns' as a Specially Designated Global Terrorist (SDGT) on July 25, 2025."
    )
    await evaluator.verify(
        claim=claim_progression,
        node=progression_leaf,
        sources=all_reward_sources,
        additional_instruction="Confirm each amount with its specified date in sequence, and verify that the second increase is explicitly linked to the listed Treasury SDGT designation (entity and designation date)."
    )

    # NRP historical significance
    nrp_leaf = evaluator.add_leaf(
        id="NRP_Historical_Significance",
        desc="State that Maduro is the first Narcotics Rewards Program target exceeding $25 million.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Maduro is the first target within the State Department’s Narcotics Rewards Program to have a reward exceeding $25 million.",
        node=nrp_leaf,
        sources=urls,
        additional_instruction="Verify that the cited page(s) explicitly or clearly support the 'first to exceed $25 million' claim for the Narcotics Rewards Program."
    )


async def verify_precedents(evaluator: Evaluator, parent_node, precedents: SectionPrecedents) -> None:
    node = evaluator.add_parallel(
        id="Supreme_Court_Precedents",
        desc="At least two relevant U.S. Supreme Court cases with case name, year, and brief legal principle, supported by URL(s).",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(precedents and precedents.urls and len(precedents.urls) > 0),
        id="Precedents_URL",
        desc="Provide ≥1 URL from an official government source or a credible news outlet supporting the Supreme Court precedents section.",
        parent=node,
        critical=True
    )

    urls = precedents.urls if precedents else []

    # Abduction precedent (expected Alvarez-Machain 1992)
    abduct_leaf = evaluator.add_leaf(
        id="Abduction_Precedent",
        desc="Identify United States v. Alvarez-Machain (1992) and explain that unlawful foreign abduction does not bar prosecution in U.S. courts.",
        parent=node,
        critical=True
    )
    claim_abduction = (
        "United States v. Alvarez-Machain (1992) holds that unlawful foreign abduction of a defendant "
        "does not bar prosecution in U.S. courts."
    )
    await evaluator.verify(
        claim=claim_abduction,
        node=abduct_leaf,
        sources=urls,
        additional_instruction="Verify both the case name and year and the principle that abduction does not bar prosecution."
    )

    # Immunity or recognition precedent: prefer what the answer cited if it’s one of the two allowed
    chosen_case_name = None
    if precedents and precedents.immunity_or_recognition and precedents.immunity_or_recognition.case_name:
        nm = precedents.immunity_or_recognition.case_name.lower()
        if "schooner exchange" in nm:
            chosen_case_name = "Schooner Exchange v. McFaddon (1812)"
            claim_text = ("Schooner Exchange v. McFaddon (1812) recognizes foreign sovereign immunity, "
                          "establishing that U.S. courts generally lack jurisdiction over foreign sovereigns absent consent.")
        elif "zivotofsky" in nm:
            chosen_case_name = "Zivotofsky v. Kerry (2015)"
            claim_text = ("Zivotofsky v. Kerry (2015) holds that the President has exclusive authority to recognize "
                          "foreign governments and territorial boundaries.")
    if chosen_case_name is None:
        # Default to Zivotofsky
        chosen_case_name = "Zivotofsky v. Kerry (2015)"
        claim_text = ("Zivotofsky v. Kerry (2015) holds that the President has exclusive authority to recognize "
                      "foreign governments and territorial boundaries.")

    immunity_leaf = evaluator.add_leaf(
        id="Immunity_Or_Recognition_Precedent",
        desc="Identify Schooner Exchange v. McFaddon (1812) OR Zivotofsky v. Kerry (2015) with a relevant legal principle.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=claim_text,
        node=immunity_leaf,
        sources=urls,
        additional_instruction="Confirm the identified case, its year, and the principle relevant to head-of-state immunity or presidential recognition authority."
    )


async def verify_international(evaluator: Evaluator, parent_node, intl: SectionInternational) -> None:
    node = evaluator.add_parallel(
        id="International_Law_Framework",
        desc="International law context and U.S. congressional response, supported by a URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(intl and intl.urls and len(intl.urls) > 0),
        id="International_Law_URL",
        desc="Provide ≥1 URL from an official government source or a credible news outlet supporting the international-law section.",
        parent=node,
        critical=True
    )

    urls = intl.urls if intl else []

    # UN Charter Article 2(4)
    un_article_leaf = evaluator.add_leaf(
        id="UN_Charter_Article",
        desc="Identify Article 2(4) of the UN Charter as governing the use of force between states.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Article 2(4) of the UN Charter governs the inter-state use of force, prohibiting the threat or use of force against the territorial integrity or political independence of any state.",
        node=un_article_leaf,
        sources=urls,
        additional_instruction="Verify that the cited material identifies Article 2(4) in this role. Paraphrases are acceptable."
    )

    # UN Secretary-General characterization: "dangerous precedent"
    unsg_leaf = evaluator.add_leaf(
        id="UNSG_Characterization",
        desc="Document that the UN Secretary-General characterized the operation as setting a 'dangerous precedent'.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The UN Secretary‑General characterized the operation as setting a 'dangerous precedent'.",
        node=unsg_leaf,
        sources=urls,
        additional_instruction="Confirm that the quoted or closely paraphrased phrasing appears in the cited source(s)."
    )

    # War Powers mention
    war_powers_leaf = evaluator.add_leaf(
        id="War_Powers_Mention",
        desc="Mention that Congress considered War Powers Resolution resolutions in response to the operation.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="Members of the U.S. Congress considered or introduced War Powers Resolution measures in response to the operation.",
        node=war_powers_leaf,
        sources=urls,
        additional_instruction="Verify that at least one cited source mentions congressional consideration of War Powers Resolution measures; specific resolution numbers are not required."
    )


async def verify_timeline(evaluator: Evaluator, parent_node, timeline: SectionTimeline) -> None:
    node = evaluator.add_parallel(
        id="Operation_Timeline",
        desc="Chronological timeline including required key dates, supported by a URL.",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(timeline and timeline.urls and len(timeline.urls) > 0),
        id="Timeline_URL",
        desc="Provide ≥1 URL from an official government source or a credible news outlet supporting the operation timeline section.",
        parent=node,
        critical=True
    )

    urls = timeline.urls if timeline else []

    timeline_leaf = evaluator.add_leaf(
        id="Timeline_Includes_All_Required_Dates",
        desc="Provide a timeline that includes: the U.S. military operation date (Jan 3, 2026), the date of Maduro's first court appearance (Jan 5, 2026), and the original indictment date (March 2020).",
        parent=node,
        critical=True
    )
    claim_timeline = (
        "The timeline includes all of the following: "
        "the U.S. military operation on January 3, 2026; "
        "Maduro's first court appearance on January 5, 2026; "
        "and the original indictment in March 2020."
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=timeline_leaf,
        sources=urls,
        additional_instruction="Verify that the cited source(s) explicitly include those dates or closely equivalent formulations."
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
    # Initialize evaluator
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_briefing(),
        template_class=BriefingExtraction,
        extraction_name="briefing_extraction"
    )

    # Critical wrapper node for the entire report
    report_node = evaluator.add_parallel(
        id="Legal_Research_Report",
        desc="Comprehensive briefing covering court details, reward history, Supreme Court precedents, international law context, and key dates, with ≥1 supporting URL per major section.",
        parent=root,
        critical=True
    )

    # Optional ground truth-like expectations (for transparency only)
    evaluator.add_ground_truth({
        "expected_court": {
            "district": "U.S. District Court for the Southern District of New York (SDNY)",
            "address": "500 Pearl Street, New York, NY 10007",
            "initial_arraignment": "January 5, 2026",
            "next_hearing": "March 26, 2026"
        },
        "expected_reward_history": [
            "Initial $15M in 2020",
            "Increase to $25M on January 10, 2025",
            "Increase to $50M on August 7, 2025",
            "Second increase linked to Treasury SDGT designation of 'Cartel of the Suns' on July 25, 2025"
        ],
        "expected_precedents": [
            "United States v. Alvarez-Machain (1992) – abduction does not bar prosecution",
            "Schooner Exchange v. McFaddon (1812) – foreign sovereign immunity OR Zivotofsky v. Kerry (2015) – presidential recognition"
        ],
        "expected_international": [
            "UN Charter Article 2(4) governs use of force",
            "UNSG called operation a 'dangerous precedent'",
            "Congress considered War Powers Resolution responses"
        ],
        "expected_timeline": [
            "Operation: January 3, 2026",
            "First court appearance: January 5, 2026",
            "Original indictment: March 2020"
        ]
    }, gt_type="rubric_expectations")

    # Build and run verifications
    await verify_court_details(evaluator, report_node, extraction.court)
    await verify_reward_history(evaluator, report_node, extraction.reward)
    await verify_precedents(evaluator, report_node, extraction.precedents)
    await verify_international(evaluator, report_node, extraction.international)
    await verify_timeline(evaluator, report_node, extraction.timeline)

    # Return structured summary
    return evaluator.get_summary()