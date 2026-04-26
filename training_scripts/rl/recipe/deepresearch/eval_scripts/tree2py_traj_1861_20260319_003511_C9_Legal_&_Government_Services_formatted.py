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
TASK_ID = "hr7744_legislative_analysis_2026"
TASK_DESCRIPTION = (
    "As of March 19, 2026, the Department of Homeland Security has been operating under a partial shutdown for over one "
    "month while Congress debates HR 7744, the Department of Homeland Security Appropriations Act, 2026. Provide a comprehensive "
    "legislative analysis addressing the following: (1) Current Legislative Status: Verify the current status of HR 7744 in both "
    "the House and Senate as of March 19, 2026, including the date the House passed the bill and the Senate's current position. "
    "(2) Senate Passage Requirements: Explain the procedural vote threshold required for the Senate to advance HR 7744, including "
    "how many votes are needed, how many votes recent attempts have received, and calculate the vote gap that must be overcome. "
    "(3) Presidential Action Scenarios: Analyze what happens if the President vetoes the bill after Congressional passage, "
    "including the constitutional requirements for a veto override (two-thirds threshold, quorum requirements, recorded vote "
    "requirements, and bicameral requirements), and calculate the specific number of votes required for an override in the House "
    "(out of 435 members) and in the Senate (out of 100 members). (4) Shutdown Impact Analysis: Document the operational impacts "
    "of the ongoing DHS shutdown, including the shutdown start date and current duration as of March 19, 2026, specific impacts on "
    "TSA operations (personnel status, security line delays, and the TSA PreCheck Touchless ID expansion program), specific impacts "
    "on CBP operations (Global Entry processing status), industry stakeholder responses (airline CEOs, travel industry, U.S. Chamber "
    "of Commerce), and impacts on travelers during the Spring Break period. For each factual claim, provide supporting URL references "
    "from reliable sources."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClaimWithSources(BaseModel):
    claim_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class HouseStatusExtraction(BaseModel):
    passage_date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SenateStatusExtraction(BaseModel):
    status_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ClotureExtraction(BaseModel):
    threshold_text: Optional[str] = None
    threshold_number: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class VoteAttempt(BaseModel):
    yeas: Optional[str] = None
    nays: Optional[str] = None
    vote_text: Optional[str] = None
    date: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class SenateVotesExtraction(BaseModel):
    attempts: List[VoteAttempt] = Field(default_factory=list)


class VoteGapExtraction(BaseModel):
    base_votes: Optional[str] = None   # e.g., "51"
    threshold: Optional[str] = None    # e.g., "60"
    computed_gap: Optional[str] = None # e.g., "9"


class OverrideExtraction(BaseModel):
    two_thirds_requirement: ClaimWithSources = Field(default_factory=ClaimWithSources)
    quorum_requirement: ClaimWithSources = Field(default_factory=ClaimWithSources)
    recorded_roll_call: ClaimWithSources = Field(default_factory=ClaimWithSources)
    bicameral_requirement: ClaimWithSources = Field(default_factory=ClaimWithSources)
    house_override_number: ClaimWithSources = Field(default_factory=ClaimWithSources)  # expect "290"
    senate_override_number: ClaimWithSources = Field(default_factory=ClaimWithSources) # expect "67"


class ShutdownExtraction(BaseModel):
    shutdown_start_date: ClaimWithSources = Field(default_factory=ClaimWithSources)                 # expect "February 15, 2026"
    shutdown_duration_as_of_2026_03_19: ClaimWithSources = Field(default_factory=ClaimWithSources) # expect "over one month"
    tsa_officers_unpaid: ClaimWithSources = Field(default_factory=ClaimWithSources)
    tsa_security_line_delays: ClaimWithSources = Field(default_factory=ClaimWithSources)
    tsa_precheck_touchless_id_expansion: ClaimWithSources = Field(default_factory=ClaimWithSources) # expect "65 airports by Spring 2026"
    cbp_global_entry_halted: ClaimWithSources = Field(default_factory=ClaimWithSources)             # expect "halted as of Feb 22, 2026"
    industry_airline_ceos: ClaimWithSources = Field(default_factory=ClaimWithSources)
    industry_travel_industry: ClaimWithSources = Field(default_factory=ClaimWithSources)
    industry_us_chamber: ClaimWithSources = Field(default_factory=ClaimWithSources)
    spring_break_traveler_impacts: ClaimWithSources = Field(default_factory=ClaimWithSources)


class LegislativeAnalysisExtraction(BaseModel):
    house_status: HouseStatusExtraction = Field(default_factory=HouseStatusExtraction)
    senate_status: SenateStatusExtraction = Field(default_factory=SenateStatusExtraction)
    cloture: ClotureExtraction = Field(default_factory=ClotureExtraction)
    senate_votes: SenateVotesExtraction = Field(default_factory=SenateVotesExtraction)
    vote_gap: VoteGapExtraction = Field(default_factory=VoteGapExtraction)
    override: OverrideExtraction = Field(default_factory=OverrideExtraction)
    shutdown: ShutdownExtraction = Field(default_factory=ShutdownExtraction)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_legislative_analysis() -> str:
    return """
    Extract structured information from the answer for the HR 7744 legislative analysis. Only extract what is explicitly stated in the answer, and only extract URLs that are explicitly present in the answer text (including plain URLs or markdown links). Do not invent content. If something is missing, set the field to null or an empty list as specified.

    Return a JSON object with the following structure and fields:

    {
      "house_status": {
        "passage_date": string | null,      // Example: "March 5, 2026"
        "urls": string[]                    // URLs that support the House passage date
      },
      "senate_status": {
        "status_text": string | null,       // Example: "Cloture failed; the bill is stalled", ideally phrased as of March 19, 2026
        "urls": string[]                    // URLs that support the Senate status
      },
      "cloture": {
        "threshold_text": string | null,    // Example: "60 votes to invoke cloture"
        "threshold_number": string | null,  // Example: "60"
        "urls": string[]                    // URLs that support the cloture threshold
      },
      "senate_votes": {
        "attempts": [                       // Include all attempts mentioned
          {
            "yeas": string | null,          // Example: "51"
            "nays": string | null,          // Example: "46"
            "vote_text": string | null,     // Any descriptive text of the vote
            "date": string | null,          // Date of the vote if present
            "urls": string[]                // URLs that report this vote tally
          }
        ]
      },
      "vote_gap": {
        "base_votes": string | null,        // The reported recent yes votes used for gap calc, e.g., "51"
        "threshold": string | null,         // The target threshold used, e.g., "60"
        "computed_gap": string | null       // The answer's computed gap (e.g., "9")
      },
      "override": {
        "two_thirds_requirement": { "claim_text": string | null, "urls": string[] },
        "quorum_requirement":     { "claim_text": string | null, "urls": string[] },
        "recorded_roll_call":     { "claim_text": string | null, "urls": string[] },
        "bicameral_requirement":  { "claim_text": string | null, "urls": string[] },
        "house_override_number":  { "claim_text": string | null, "urls": string[] }, // Expect "290" if all 435 vote
        "senate_override_number": { "claim_text": string | null, "urls": string[] }  // Expect "67" if all 100 vote
      },
      "shutdown": {
        "shutdown_start_date":                 { "claim_text": string | null, "urls": string[] }, // expect "February 15, 2026"
        "shutdown_duration_as_of_2026_03_19":  { "claim_text": string | null, "urls": string[] }, // expect "over one month"
        "tsa_officers_unpaid":                 { "claim_text": string | null, "urls": string[] },
        "tsa_security_line_delays":            { "claim_text": string | null, "urls": string[] },
        "tsa_precheck_touchless_id_expansion": { "claim_text": string | null, "urls": string[] }, // expect "65 airports by Spring 2026"
        "cbp_global_entry_halted":             { "claim_text": string | null, "urls": string[] }, // expect "halted as of Feb 22, 2026"
        "industry_airline_ceos":               { "claim_text": string | null, "urls": string[] },
        "industry_travel_industry":            { "claim_text": string | null, "urls": string[] },
        "industry_us_chamber":                 { "claim_text": string | null, "urls": string[] },
        "spring_break_traveler_impacts":       { "claim_text": string | null, "urls": string[] }
      }
    }

    Special instructions:
    - For every 'urls' field, list ONLY URLs explicitly present in the answer. If the answer mentions a source by name but no URL, do not include it.
    - Keep dates as strings in the format they appear in the answer (e.g., "Mar. 5, 2026" or "March 5, 2026").
    - For Senate recent votes, extract each distinct vote attempt with its own URLs if available.
    - If multiple URLs are provided for a single claim, include them all.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    clean: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            # extractor may have omitted protocol; prepend http:// per framework suggestion
            u = "http://" + u if "://" not in u else u
        if u not in seen:
            seen.add(u)
            clean.append(u)
    return clean


def union_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        for u in sanitize_urls(lst):
            if u not in merged:
                merged.append(u)
    return merged


def has_attempt_with_tally(attempts: List[VoteAttempt], yeas: str, nays: str) -> bool:
    y = yeas.strip()
    n = nays.strip()
    for a in attempts:
        if (a.yeas or "").strip() == y and (a.nays or "").strip() == n:
            return True
        # Also allow match if vote_text contains "51-46" style
        vt = (a.vote_text or "").replace(" ", "")
        if f"{y}-{n}" in vt or f"{y}–{n}" in vt or f"{y}/{n}" in vt:
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_part_1(evaluator: Evaluator, parent_node, data: LegislativeAnalysisExtraction) -> None:
    p1 = evaluator.add_parallel(
        id="Part_1_Current_Legislative_Status",
        desc="Status in House and Senate as of March 19, 2026, including House passage date and Senate position, with sources.",
        parent=parent_node,
        critical=True,
    )

    # 1A. House passage date (expect March 5, 2026) + reliable source
    n1 = evaluator.add_parallel(
        id="House_Passage_Date_March_5_2026_With_Reliable_Source",
        desc="States the House passed HR 7744 on March 5, 2026 AND provides a reliable supporting URL.",
        parent=p1,
        critical=True,
    )
    hp_date_present = evaluator.add_custom_node(
        result=bool((data.house_status.passage_date or "").strip()),
        id="house_passage_date_present",
        desc="House passage date is stated in the answer",
        parent=n1,
        critical=True,
    )
    hp_sources_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.house_status.urls)) > 0,
        id="house_passage_sources_present",
        desc="House passage date has at least one supporting URL",
        parent=n1,
        critical=True,
    )
    # date correctness simple check
    date_correct_node = evaluator.add_leaf(
        id="house_passage_date_correct_march_5_2026",
        desc="Stated House passage date equals March 5, 2026",
        parent=n1,
        critical=True,
    )
    await evaluator.verify(
        claim=f'The date "{(data.house_status.passage_date or "").strip()}" is the same calendar date as "March 5, 2026".',
        node=date_correct_node,
        additional_instruction="Evaluate whether the two dates denote the same day. Allow variations like 'Mar. 5, 2026' vs 'March 5, 2026'.",
    )
    # source-supported claim
    hp_supported_node = evaluator.add_leaf(
        id="house_passage_supported_by_urls",
        desc="House passage on March 5, 2026 is supported by the cited URLs",
        parent=n1,
        critical=True,
    )
    await evaluator.verify(
        claim="The U.S. House of Representatives passed H.R. 7744 on March 5, 2026.",
        node=hp_supported_node,
        sources=sanitize_urls(data.house_status.urls),
        additional_instruction="Confirm the date and passage of H.R. 7744 on the cited page(s).",
    )

    # 1B. Senate status as of March 19, 2026 + reliable source
    n2 = evaluator.add_parallel(
        id="Senate_Status_As_Of_March_19_2026_With_Reliable_Source",
        desc="Accurately states the Senate's current position/status on HR 7744 as of March 19, 2026 AND provides a reliable supporting URL.",
        parent=p1,
        critical=True,
    )
    sen_status_present = evaluator.add_custom_node(
        result=bool((data.senate_status.status_text or "").strip()),
        id="senate_status_text_present",
        desc="Senate status text is provided",
        parent=n2,
        critical=True,
    )
    sen_sources_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.senate_status.urls)) > 0,
        id="senate_status_sources_present",
        desc="Senate status has at least one supporting URL",
        parent=n2,
        critical=True,
    )
    sen_supported_node = evaluator.add_leaf(
        id="senate_status_supported_by_urls",
        desc="Senate status (as of March 19, 2026) is supported by the cited URLs",
        parent=n2,
        critical=True,
    )
    await evaluator.verify(
        claim=f'As of March 19, 2026, the U.S. Senate\'s position/status on H.R. 7744 is: "{(data.senate_status.status_text or "").strip()}".',
        node=sen_supported_node,
        sources=sanitize_urls(data.senate_status.urls),
        additional_instruction="Verify that the cited page(s) substantiate the described Senate status (e.g., stalled, cloture failed, pending).",
    )


async def verify_part_2(evaluator: Evaluator, parent_node, data: LegislativeAnalysisExtraction) -> None:
    p2 = evaluator.add_parallel(
        id="Part_2_Senate_Passage_Requirements",
        desc="Senate procedural threshold to advance HR 7744; include recent attempt vote counts; compute vote gap; with sources where the claims are factual.",
        parent=parent_node,
        critical=True,
    )

    # 2A. Cloture threshold 60 + reliable source
    n1 = evaluator.add_parallel(
        id="Cloture_Threshold_60_With_Reliable_Source",
        desc="States the Senate procedural threshold is 60 votes to invoke cloture/advance HR 7744 AND provides a reliable supporting URL.",
        parent=p2,
        critical=True,
    )
    cloture_sources_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.cloture.urls)) > 0,
        id="cloture_sources_present",
        desc="Cloture threshold claim has at least one supporting URL",
        parent=n1,
        critical=True,
    )
    cloture_supported_node = evaluator.add_leaf(
        id="cloture_threshold_60_supported_by_urls",
        desc="Senate requires 60 votes to invoke cloture (supported by URLs)",
        parent=n1,
        critical=True,
    )
    await evaluator.verify(
        claim="Advancing H.R. 7744 in the Senate requires 60 votes to invoke cloture (to end debate or proceed).",
        node=cloture_supported_node,
        sources=sanitize_urls(data.cloture.urls),
        additional_instruction="Accept explanations that 60 votes are needed to overcome a filibuster or invoke cloture on most legislation.",
    )

    # 2B. Recent Senate attempt votes 51-46 and 51-45 + reliable source(s)
    n2 = evaluator.add_parallel(
        id="Recent_Senate_Attempt_Votes_51_46_And_51_45_With_Reliable_Source",
        desc="Reports the recent Senate procedural votes/attempts as 51-46 and 51-45 AND provides reliable supporting URL(s).",
        parent=p2,
        critical=True,
    )
    all_attempt_urls = []
    for a in data.senate_votes.attempts:
        all_attempt_urls.extend(sanitize_urls(a.urls))
    attempts_urls = sanitize_urls(all_attempt_urls)

    attempts_sources_present = evaluator.add_custom_node(
        result=len(attempts_urls) > 0,
        id="recent_attempts_sources_present",
        desc="Recent Senate attempt votes have at least one supporting URL",
        parent=n2,
        critical=True,
    )

    # presence checks within extracted attempts
    has_5146_node = evaluator.add_custom_node(
        result=has_attempt_with_tally(data.senate_votes.attempts, "51", "46"),
        id="has_attempt_51_46_in_answer",
        desc="The answer reports a recent Senate attempt with a 51-46 tally",
        parent=n2,
        critical=True,
    )
    has_5145_node = evaluator.add_custom_node(
        result=has_attempt_with_tally(data.senate_votes.attempts, "51", "45"),
        id="has_attempt_51_45_in_answer",
        desc="The answer reports a recent Senate attempt with a 51-45 tally",
        parent=n2,
        critical=True,
    )

    # URL-supported checks for both tallies
    vote_5146_node = evaluator.add_leaf(
        id="vote_51_46_supported_by_urls",
        desc="A recent Senate procedural vote on H.R. 7744 was 51-46 (supported by URLs)",
        parent=n2,
        critical=True,
    )
    await evaluator.verify(
        claim="A recent Senate procedural vote on H.R. 7744 had a tally of 51-46.",
        node=vote_5146_node,
        sources=attempts_urls,
        additional_instruction="Confirm any credible page that reports a 51-46 procedural vote related to H.R. 7744.",
    )

    vote_5145_node = evaluator.add_leaf(
        id="vote_51_45_supported_by_urls",
        desc="A recent Senate procedural vote on H.R. 7744 was 51-45 (supported by URLs)",
        parent=n2,
        critical=True,
    )
    await evaluator.verify(
        claim="A recent Senate procedural vote on H.R. 7744 had a tally of 51-45.",
        node=vote_5145_node,
        sources=attempts_urls,
        additional_instruction="Confirm any credible page that reports a 51-45 procedural vote related to H.R. 7744.",
    )

    # 2C. Vote gap calculation: 60 - 51 = 9 (pure arithmetic)
    gap_leaf = evaluator.add_leaf(
        id="Vote_Gap_Calculation_60_minus_51_equals_9",
        desc="Correctly calculates the vote gap to reach 60 given 51 votes as 9.",
        parent=p2,
        critical=True,
    )
    await evaluator.verify(
        claim="60 - 51 = 9.",
        node=gap_leaf,
        additional_instruction="This is a simple arithmetic check. Confirm that 60 minus 51 equals 9.",
    )


async def verify_part_3(evaluator: Evaluator, parent_node, data: LegislativeAnalysisExtraction) -> None:
    p3 = evaluator.add_parallel(
        id="Part_3_Presidential_Veto_Override_Scenario",
        desc="If the President vetoes after passage: explain constitutional override requirements and compute override vote numbers for full attendance (House 435; Senate 100), with sources for factual/legal claims.",
        parent=parent_node,
        critical=True,
    )

    # 3A. Two-thirds requirement with reliable source
    n1 = evaluator.add_parallel(
        id="Override_Two_Thirds_Requirement_With_Reliable_Source",
        desc="States a veto override requires a two-thirds vote of Members voting in each chamber AND provides a reliable supporting URL.",
        parent=p3,
        critical=True,
    )
    src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.two_thirds_requirement.urls)) > 0,
        id="override_two_thirds_sources_present",
        desc="Two-thirds override requirement has at least one supporting URL",
        parent=n1,
        critical=True,
    )
    leaf = evaluator.add_leaf(
        id="override_two_thirds_supported_by_urls",
        desc="Two-thirds requirement supported by URLs",
        parent=n1,
        critical=True,
    )
    await evaluator.verify(
        claim="Overriding a presidential veto requires a two-thirds vote in each chamber of Congress of members present and voting.",
        node=leaf,
        sources=sanitize_urls(data.override.two_thirds_requirement.urls),
        additional_instruction="Accept authoritative sources (e.g., Constitution, House/Senate rules, CRS, official congressional websites).",
    )

    # 3B. Quorum requirement with reliable source
    n2 = evaluator.add_parallel(
        id="Override_Quorum_Requirement_With_Reliable_Source",
        desc="States a quorum must be present for override votes AND provides a reliable supporting URL.",
        parent=p3,
        critical=True,
    )
    q_src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.quorum_requirement.urls)) > 0,
        id="override_quorum_sources_present",
        desc="Quorum requirement has at least one supporting URL",
        parent=n2,
        critical=True,
    )
    q_leaf = evaluator.add_leaf(
        id="override_quorum_supported_by_urls",
        desc="Quorum requirement for override votes supported by URLs",
        parent=n2,
        critical=True,
    )
    await evaluator.verify(
        claim="A quorum must be present in each chamber for conducting business, including veto override votes.",
        node=q_leaf,
        sources=sanitize_urls(data.override.quorum_requirement.urls),
        additional_instruction="Accept references to Article I quorum rules, chamber rules, or authoritative procedural guides.",
    )

    # 3C. Recorded roll-call (yeas and nays) with reliable source
    n3 = evaluator.add_parallel(
        id="Override_Recorded_Roll_Call_Yeas_And_Nays_With_Reliable_Source",
        desc="States override votes must be recorded roll-call votes (Yeas and Nays entered in the journal) AND provides a reliable supporting URL.",
        parent=p3,
        critical=True,
    )
    rr_src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.recorded_roll_call.urls)) > 0,
        id="override_recorded_roll_sources_present",
        desc="Recorded roll-call requirement has at least one supporting URL",
        parent=n3,
        critical=True,
    )
    rr_leaf = evaluator.add_leaf(
        id="override_recorded_roll_supported_by_urls",
        desc="Recorded roll-call (yeas and nays) requirement supported by URLs",
        parent=n3,
        critical=True,
    )
    await evaluator.verify(
        claim="Veto override votes are taken by recorded roll call, with yeas and nays entered in the Journal in each chamber.",
        node=rr_leaf,
        sources=sanitize_urls(data.override.recorded_roll_call.urls),
        additional_instruction="Accept references to constitutional requirements that veto votes are recorded (yeas and nays).",
    )

    # 3D. Bicameral separate approval with reliable source
    n4 = evaluator.add_parallel(
        id="Override_Bicameral_Separate_Approval_With_Reliable_Source",
        desc="States both chambers must separately approve the override AND provides a reliable supporting URL.",
        parent=p3,
        critical=True,
    )
    bi_src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.bicameral_requirement.urls)) > 0,
        id="override_bicameral_sources_present",
        desc="Bicameral separate approval has at least one supporting URL",
        parent=n4,
        critical=True,
    )
    bi_leaf = evaluator.add_leaf(
        id="override_bicameral_supported_by_urls",
        desc="Both chambers must each vote to override (supported by URLs)",
        parent=n4,
        critical=True,
    )
    await evaluator.verify(
        claim="To override a veto, both the House and the Senate must each separately approve the override by the constitutional two‑thirds threshold.",
        node=bi_leaf,
        sources=sanitize_urls(data.override.bicameral_requirement.urls),
        additional_instruction="Accept authoritative civics/procedure sources confirming bicameral override.",
    )

    # 3E. House override votes = 290 of 435 (with reliable source for basis)
    n5 = evaluator.add_parallel(
        id="House_Override_Votes_290_Out_Of_435_With_Reliable_Source",
        desc="Computes/states that 290 House votes are required for an override if all 435 vote AND provides a reliable supporting URL (or authoritative reference) for the calculation basis.",
        parent=p3,
        critical=True,
    )
    ho_src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.house_override_number.urls)) > 0,
        id="house_override_number_sources_present",
        desc="House override calculation basis has at least one supporting URL",
        parent=n5,
        critical=True,
    )
    ho_math_leaf = evaluator.add_leaf(
        id="house_override_two_thirds_math_check",
        desc="Two-thirds of 435 equals 290",
        parent=n5,
        critical=True,
    )
    await evaluator.verify(
        claim="Two thirds of 435 equals 290.",
        node=ho_math_leaf,
        additional_instruction="Compute 2/3 × 435. The correct whole number needed to meet or exceed two-thirds is 290.",
    )

    # 3F. Senate override votes = 67 of 100 (with reliable source for basis)
    n6 = evaluator.add_parallel(
        id="Senate_Override_Votes_67_Out_Of_100_With_Reliable_Source",
        desc="Computes/states that 67 Senate votes are required for an override if all 100 vote AND provides a reliable supporting URL (or authoritative reference) for the calculation basis.",
        parent=p3,
        critical=True,
    )
    so_src_present = evaluator.add_custom_node(
        result=len(sanitize_urls(data.override.senate_override_number.urls)) > 0,
        id="senate_override_number_sources_present",
        desc="Senate override calculation basis has at least one supporting URL",
        parent=n6,
        critical=True,
    )
    so_math_leaf = evaluator.add_leaf(
        id="senate_override_two_thirds_math_check",
        desc="Two-thirds of 100 equals 67",
        parent=n6,
        critical=True,
    )
    await evaluator.verify(
        claim="Two thirds of 100 equals 67.",
        node=so_math_leaf,
        additional_instruction="Compute 2/3 × 100. The correct whole number needed to meet or exceed two-thirds is 67.",
    )


async def verify_part_4(evaluator: Evaluator, parent_node, data: LegislativeAnalysisExtraction) -> None:
    p4 = evaluator.add_parallel(
        id="Part_4_Shutdown_Impact_Analysis",
        desc="DHS shutdown start/duration and specified TSA/CBP impacts and stakeholder/traveler impacts, with reliable supporting URLs for each required factual claim.",
        parent=parent_node,
        critical=True,
    )

    # Helper to add a two-leaf pattern: (1) sources-present (2) supported-by-urls
    async def two_step_url_check(
        base_id: str,
        desc: str,
        claim: str,
        urls: List[str],
        additional_instruction: str = "None"
    ):
        node = evaluator.add_parallel(
            id=base_id,
            desc=desc,
            parent=p4,
            critical=True,
        )
        src_present = evaluator.add_custom_node(
            result=len(urls) > 0,
            id=f"{base_id}_sources_present",
            desc=f"At least one supporting URL is provided for {base_id}",
            parent=node,
            critical=True,
        )
        leaf = evaluator.add_leaf(
            id=f"{base_id}_supported_by_urls",
            desc=f"{desc} (supported by cited URLs)",
            parent=node,
            critical=True,
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=additional_instruction,
        )

    # 4A. Shutdown start date (expect Feb 15, 2026)
    await two_step_url_check(
        base_id="Shutdown_Start_Date_Feb_15_2026_With_Reliable_Source",
        desc="States the partial DHS shutdown began February 15, 2026 AND provides a reliable supporting URL.",
        claim="The partial DHS shutdown began on February 15, 2026.",
        urls=sanitize_urls(data.shutdown.shutdown_start_date.urls),
        additional_instruction="Confirm the start date for the DHS shutdown on the cited page(s).",
    )

    # 4B. Shutdown duration over one month as of Mar 19, 2026
    await two_step_url_check(
        base_id="Shutdown_Duration_Over_One_Month_As_Of_Mar_19_2026_With_Reliable_Source",
        desc="States that as of March 19, 2026, the shutdown has lasted over one month AND provides a reliable supporting URL.",
        claim="As of March 19, 2026, the DHS partial shutdown has lasted over one month.",
        urls=sanitize_urls(data.shutdown.shutdown_duration_as_of_2026_03_19.urls),
        additional_instruction="Confirm the reported duration as of March 19, 2026 (e.g., articles around that date stating 'over a month').",
    )

    # 4C. TSA officers working without pay
    await two_step_url_check(
        base_id="TSA_Officers_Working_Without_Pay_With_Reliable_Source",
        desc="States TSA officers are working without pay during the shutdown AND provides a reliable supporting URL.",
        claim="During the DHS partial shutdown, TSA officers are working without pay.",
        urls=sanitize_urls(data.shutdown.tsa_officers_unpaid.urls),
        additional_instruction="Look for references to TSA officers being excepted/essential and working without paychecks.",
    )

    # 4D. TSA security line delays
    await two_step_url_check(
        base_id="TSA_Security_Line_Delays_With_Reliable_Source",
        desc="Documents TSA security line delays during the shutdown AND provides a reliable supporting URL.",
        claim="The DHS partial shutdown led to TSA security line delays.",
        urls=sanitize_urls(data.shutdown.tsa_security_line_delays.urls),
        additional_instruction="Any credible reporting of longer wait times, delays, or staffing constraints at TSA checkpoints suffices.",
    )

    # 4E. TSA PreCheck Touchless ID expansion to 65 airports by Spring 2026
    await two_step_url_check(
        base_id="Touchless_ID_Expansion_To_65_Airports_By_Spring_2026_With_Reliable_Source",
        desc="States TSA PreCheck Touchless ID is expanding to 65 airports by Spring 2026 AND provides a reliable supporting URL.",
        claim="TSA PreCheck Touchless ID is expanding to 65 airports by Spring 2026.",
        urls=sanitize_urls(data.shutdown.tsa_precheck_touchless_id_expansion.urls),
        additional_instruction="Confirm announcements or coverage specifying 65 airports by Spring 2026.",
    )

    # 4F. CBP Global Entry arrival processing halted as of Feb 22, 2026
    await two_step_url_check(
        base_id="CBP_Global_Entry_Arrival_Processing_Halted_As_Of_Feb_22_2026_With_Reliable_Source",
        desc="States CBP halted Global Entry arrival processing as of February 22, 2026 AND provides a reliable supporting URL.",
        claim="CBP halted Global Entry arrival processing as of February 22, 2026.",
        urls=sanitize_urls(data.shutdown.cbp_global_entry_halted.urls),
        additional_instruction="Confirm service status notices or reporting matching the described halt date.",
    )

    # 4G. Industry response: Airline CEOs
    await two_step_url_check(
        base_id="Industry_Response_Airline_CEOs_With_Reliable_Source",
        desc="Documents airline CEO responses to the shutdown AND provides a reliable supporting URL.",
        claim="Airline CEOs publicly responded to the DHS shutdown, expressing concerns and/or urging resolution.",
        urls=sanitize_urls(data.shutdown.industry_airline_ceos.urls),
        additional_instruction="Confirm statements, letters, or press coverage from major airline CEOs referencing the shutdown.",
    )

    # 4H. Industry response: Travel industry
    await two_step_url_check(
        base_id="Industry_Response_Travel_Industry_With_Reliable_Source",
        desc="Documents broader travel-industry response to the shutdown AND provides a reliable supporting URL.",
        claim="The broader travel industry publicly responded to the DHS shutdown, expressing concerns and/or urging resolution.",
        urls=sanitize_urls(data.shutdown.industry_travel_industry.urls),
        additional_instruction="Confirm statements or coverage from trade groups, airports, or travel associations.",
    )

    # 4I. Industry response: U.S. Chamber of Commerce
    await two_step_url_check(
        base_id="Industry_Response_US_Chamber_Of_Commerce_With_Reliable_Source",
        desc="Documents the U.S. Chamber of Commerce response to the shutdown AND provides a reliable supporting URL.",
        claim="The U.S. Chamber of Commerce publicly responded to the DHS shutdown, expressing concerns and/or urging resolution.",
        urls=sanitize_urls(data.shutdown.industry_us_chamber.urls),
        additional_instruction="Confirm statements, letters, or press releases by the U.S. Chamber related to the shutdown.",
    )

    # 4J. Spring Break traveler impacts
    await two_step_url_check(
        base_id="Spring_Break_Traveler_Impacts_With_Reliable_Source",
        desc="Documents impacts on travelers during the Spring Break period AND provides a reliable supporting URL.",
        claim="During the Spring Break period, travelers were impacted by the DHS shutdown (e.g., longer lines, delays, disruptions).",
        urls=sanitize_urls(data.shutdown.spring_break_traveler_impacts.urls),
        additional_instruction="Confirm travel-impact reporting around Spring Break (mid-March) tied to the shutdown.",
    )


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
    """
    Evaluate an answer against the HR 7744 Legislative Analysis rubric using the obj_task_eval framework.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # JSON root is parallel aggregation
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

    # Create the rubric's top-level analysis node (critical)
    analysis_root = evaluator.add_parallel(
        id="HR_7744_Legislative_Analysis",
        desc="Comprehensive legislative analysis for HR 7744 covering: (1) legislative status, (2) Senate procedural threshold + vote math, (3) veto override requirements + vote calculations, and (4) DHS shutdown impacts, with reliable supporting URLs for each required factual claim.",
        parent=root,
        critical=True,
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_legislative_analysis(),
        template_class=LegislativeAnalysisExtraction,
        extraction_name="legislative_analysis_extraction",
    )

    # Record a small custom info block (optional diagnostics)
    evaluator.add_custom_info(
        info={
            "attempt_counts": len(extracted.senate_votes.attempts),
            "house_passage_urls_count": len(extracted.house_status.urls),
            "senate_status_urls_count": len(extracted.senate_status.urls),
        },
        info_type="diagnostic",
        info_name="extraction_diagnostics",
    )

    # Build and verify the tree parts
    await verify_part_1(evaluator, analysis_root, extracted)
    await verify_part_2(evaluator, analysis_root, extracted)
    await verify_part_3(evaluator, analysis_root, extracted)
    await verify_part_4(evaluator, analysis_root, extracted)

    # Return structured evaluation summary
    return evaluator.get_summary()