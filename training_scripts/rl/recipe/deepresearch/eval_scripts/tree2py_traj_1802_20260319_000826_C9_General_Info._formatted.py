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
TASK_ID = "multi_professionals_march_2026"
TASK_DESCRIPTION = """
Identify four professionals across different fields who meet ALL of the following criteria as of March 2026:

Professional A (NBA Player):
- Was part of the Los Angeles Lakers roster that won the NBA championship in 2020
- Was selected in the first round of the 2017 NBA draft
- As of March 2026, plays for an NBA team whose home arena has a basketball seating capacity between 17,000 and 18,000 (inclusive)

Professional B (Dancer/Entertainment Professional):
- Competed in Season 10 of "So You Think You Can Dance"
- Finished in the top 3 females in that season
- Is originally from Kansas
- Later became a troupe member on "Dancing with the Stars"

Professional C (Business Journalist):
- Holds a bachelor's degree from an institution within the Harvard system that was founded in the 1600s
- Holds an MBA from an institution within the Harvard system that was founded in the 1900s and offered the world's first MBA program
- Previously worked at McKinsey & Company
- Worked at Fox Business before joining CBS News
- Joined CBS News in 2024

Professional D (Tennis Player):
- Represents Chile in professional ATP tennis
- Achieved a career-high ATP singles ranking in the top 20 during the year 2024
- Has won exactly 3 ATP career titles (as of March 2026)
- Plays left-handed

For each professional (A, B, C, D), provide:
1. Their full name
2. At least one URL reference that confirms their identity and key qualifying criteria
"""


# --------------------------------------------------------------------------- #
# Data models for extracted structured info                                   #
# --------------------------------------------------------------------------- #
class ProfessionalA(BaseModel):
    name: Optional[str] = None
    current_team: Optional[str] = None
    home_arena: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    championship_urls: List[str] = Field(default_factory=list)
    draft_urls: List[str] = Field(default_factory=list)
    arena_urls: List[str] = Field(default_factory=list)


class ProfessionalB(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    sytycd_urls: List[str] = Field(default_factory=list)
    top3_urls: List[str] = Field(default_factory=list)
    kansas_urls: List[str] = Field(default_factory=list)
    dwts_urls: List[str] = Field(default_factory=list)


class ProfessionalC(BaseModel):
    name: Optional[str] = None
    ba_institution: Optional[str] = None
    mba_institution: Optional[str] = None
    joined_cbs_year: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    ba_urls: List[str] = Field(default_factory=list)
    mba_urls: List[str] = Field(default_factory=list)
    mckinsey_urls: List[str] = Field(default_factory=list)
    fox_business_urls: List[str] = Field(default_factory=list)
    cbs_urls: List[str] = Field(default_factory=list)


class ProfessionalD(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    representation_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)
    titles_urls: List[str] = Field(default_factory=list)
    handedness_urls: List[str] = Field(default_factory=list)


class AllProfessionalsExtraction(BaseModel):
    professional_a: Optional[ProfessionalA] = None
    professional_b: Optional[ProfessionalB] = None
    professional_c: Optional[ProfessionalC] = None
    professional_d: Optional[ProfessionalD] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract from the answer the structured information for four professionals labeled A, B, C, and D. For each, extract ONLY what is explicitly stated in the answer. If something is missing, return null or an empty list accordingly. Extract URLs exactly as they appear in the answer (plain or markdown). Do not fabricate anything.

Required JSON structure:
{
  "professional_a": {
    "name": string or null,
    "current_team": string or null,
    "home_arena": string or null,
    "sources": [urls...],
    "championship_urls": [urls...],    // URLs that support being on the 2020 Lakers title roster
    "draft_urls": [urls...],           // URLs that support 2017 first round draft selection
    "arena_urls": [urls...]            // URLs that support current team and/or arena capacity
  },
  "professional_b": {
    "name": string or null,
    "sources": [urls...],
    "sytycd_urls": [urls...],          // URLs that support Season 10 participation
    "top3_urls": [urls...],            // URLs that support top 3 female placement
    "kansas_urls": [urls...],          // URLs that support Kansas origin
    "dwts_urls": [urls...]             // URLs that support DWTS troupe membership
  },
  "professional_c": {
    "name": string or null,
    "ba_institution": string or null,  // e.g., "Harvard College"
    "mba_institution": string or null, // e.g., "Harvard Business School"
    "joined_cbs_year": string or null, // as written in the answer, e.g., "2024"
    "sources": [urls...],
    "ba_urls": [urls...],              // URLs that support BA from Harvard College and/or its founding century
    "mba_urls": [urls...],             // URLs that support MBA from HBS and/or its founding/first-MBA claim
    "mckinsey_urls": [urls...],        // URLs that support McKinsey employment
    "fox_business_urls": [urls...],    // URLs that support Fox Business employment
    "cbs_urls": [urls...]              // URLs that support CBS News employment and 2024 join date
  },
  "professional_d": {
    "name": string or null,
    "sources": [urls...],
    "representation_urls": [urls...],  // URLs that support representing Chile in ATP tennis
    "ranking_urls": [urls...],         // URLs that support top-20 career-high during 2024
    "titles_urls": [urls...],          // URLs that support exactly 3 ATP career titles as of March 2026
    "handedness_urls": [urls...]       // URLs that support playing left-handed
  }
}

Rules:
- URLs: extract only valid URLs that appear in the answer (plain or inside markdown links). If none are provided for a field, leave the list empty.
- Do not normalize names or institutions; copy as-is from the answer.
- If the answer mentions multiple candidates per role, take the first one presented for each (A, B, C, D).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            u_stripped = u.strip()
            if not u_stripped:
                continue
            if u_stripped not in seen:
                seen.add(u_stripped)
                merged.append(u_stripped)
    return merged


def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return any(bool((u or "").strip()) for u in (urls or []))


def pick_sources(preferred: List[str], fallback: List[str]) -> List[str]:
    return preferred if nonempty_urls(preferred) else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_professional_a(evaluator: Evaluator, parent_node, a: ProfessionalA) -> None:
    # Professional A top-level node (Parallel)
    pro_a_node = evaluator.add_parallel(
        id="Professional_A_NBA_Player",
        desc="Identify the NBA player who meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Gating: name and at least one source overall
    all_a_sources = merge_urls(a.sources, a.championship_urls, a.draft_urls, a.arena_urls)
    evaluator.add_custom_node(
        result=(a is not None and (a.name or "").strip() != "" and nonempty_urls(all_a_sources)),
        id="A_Identity_Provided",
        desc="Professional A: Name is provided and at least one source URL exists",
        parent=pro_a_node,
        critical=True
    )

    # A.1 Championship 2020
    a_champ_node = evaluator.add_parallel(
        id="A_Championship_2020",
        desc="Verify player was on 2020 Lakers championship roster",
        parent=pro_a_node,
        critical=True
    )
    champ_urls = pick_sources(a.championship_urls, all_a_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(champ_urls),
        id="A_Championship_URL",
        desc="URL reference confirming 2020 Lakers championship roster membership",
        parent=a_champ_node,
        critical=True
    )

    leaf_champ = evaluator.add_leaf(
        id="A_Championship_Fact",
        desc="Player was part of the Los Angeles Lakers roster that won the 2020 NBA championship",
        parent=a_champ_node,
        critical=True
    )
    champ_claim = f"{a.name} was on the Los Angeles Lakers roster that won the 2020 NBA championship."
    await evaluator.verify(
        claim=champ_claim,
        node=leaf_champ,
        sources=champ_urls,
        additional_instruction="Accept roster documents, team season summaries, or credible news coverage explicitly listing this player on the 2019–20 Lakers championship team."
    )

    # A.2 Draft 2017 first round
    a_draft_node = evaluator.add_parallel(
        id="A_Draft_2017",
        desc="Verify player was selected in first round of 2017 NBA draft",
        parent=pro_a_node,
        critical=True
    )
    draft_urls = pick_sources(a.draft_urls, all_a_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(draft_urls),
        id="A_Draft_URL",
        desc="URL reference confirming 2017 first round draft selection",
        parent=a_draft_node,
        critical=True
    )

    leaf_draft = evaluator.add_leaf(
        id="A_Draft_Fact",
        desc="Player was selected in the first round of the 2017 NBA draft",
        parent=a_draft_node,
        critical=True
    )
    draft_claim = f"{a.name} was selected in the first round of the 2017 NBA Draft."
    await evaluator.verify(
        claim=draft_claim,
        node=leaf_draft,
        sources=draft_urls,
        additional_instruction="Verify that the player was a Round 1 pick in the 2017 NBA Draft (any pick number within round 1 qualifies)."
    )

    # A.3 Current team arena capacity between 17,000 and 18,000 inclusive
    a_arena_node = evaluator.add_parallel(
        id="A_Current_Arena",
        desc="Verify current team's arena capacity is between 17,000 and 18,000",
        parent=pro_a_node,
        critical=True
    )
    arena_urls = pick_sources(a.arena_urls, all_a_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(arena_urls),
        id="A_Arena_URL",
        desc="URL reference confirming current team and arena capacity",
        parent=a_arena_node,
        critical=True
    )

    leaf_arena = evaluator.add_leaf(
        id="A_Arena_Capacity_Fact",
        desc="Current team's home arena has basketball seating capacity between 17,000 and 18,000 (inclusive)",
        parent=a_arena_node,
        critical=True
    )
    team_hint = f" Team hint: {a.current_team}." if (a.current_team or "").strip() else ""
    arena_hint = f" Arena hint: {a.home_arena}." if (a.home_arena or "").strip() else ""
    arena_claim = (
        f"As of March 2026, the NBA team for which {a.name} plays has a home arena with a basketball "
        f"seating capacity between 17,000 and 18,000 (inclusive)."
    )
    await evaluator.verify(
        claim=arena_claim,
        node=leaf_arena,
        sources=arena_urls,
        additional_instruction="Confirm the player's then-current NBA team and check the official or credible arena capacity for basketball events to be within 17,000–18,000 inclusive."
    )


async def verify_professional_b(evaluator: Evaluator, parent_node, b: ProfessionalB) -> None:
    # Professional B top-level node (Parallel)
    pro_b_node = evaluator.add_parallel(
        id="Professional_B_Dancer",
        desc="Identify the dancer/entertainment professional who meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Gating: name and at least one source overall
    all_b_sources = merge_urls(b.sources, b.sytycd_urls, b.top3_urls, b.kansas_urls, b.dwts_urls)
    evaluator.add_custom_node(
        result=(b is not None and (b.name or "").strip() != "" and nonempty_urls(all_b_sources)),
        id="B_Identity_Provided",
        desc="Professional B: Name is provided and at least one source URL exists",
        parent=pro_b_node,
        critical=True
    )

    # B.1 SYTYCD Season 10
    b_sytycd_node = evaluator.add_parallel(
        id="B_SYTYCD_Season_10",
        desc="Verify competed in Season 10 of So You Think You Can Dance",
        parent=pro_b_node,
        critical=True
    )
    sytycd_urls = pick_sources(b.sytycd_urls, all_b_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(sytycd_urls),
        id="B_Season_10_URL",
        desc="URL reference confirming SYTYCD Season 10 participation",
        parent=b_sytycd_node,
        critical=True
    )

    leaf_sytycd = evaluator.add_leaf(
        id="B_Season_10_Fact",
        desc="Competed in Season 10 of So You Think You Can Dance",
        parent=b_sytycd_node,
        critical=True
    )
    sytycd_claim = f"{b.name} competed in Season 10 of So You Think You Can Dance."
    await evaluator.verify(
        claim=sytycd_claim,
        node=leaf_sytycd,
        sources=sytycd_urls,
        additional_instruction="Season 10 aired in 2013; allow synonyms like 'So You Think You Can Dance (U.S.) Season 10'."
    )

    # B.2 Top 3 females in Season 10
    b_top3_node = evaluator.add_parallel(
        id="B_Top_3_Females",
        desc="Verify finished in top 3 females in Season 10",
        parent=pro_b_node,
        critical=True
    )
    top3_urls = pick_sources(b.top3_urls, all_b_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(top3_urls),
        id="B_Top_3_URL",
        desc="URL reference confirming top 3 female placement",
        parent=b_top3_node,
        critical=True
    )

    leaf_top3 = evaluator.add_leaf(
        id="B_Top_3_Fact",
        desc="Finished in the top 3 females in Season 10",
        parent=b_top3_node,
        critical=True
    )
    top3_claim = f"{b.name} finished in the top 3 among female contestants in Season 10 of So You Think You Can Dance."
    await evaluator.verify(
        claim=top3_claim,
        node=leaf_top3,
        sources=top3_urls,
        additional_instruction="Accept equivalent phrasing such as 'Top 3 girls' or similar gendered category ranking."
    )

    # B.3 Originally from Kansas
    b_ks_node = evaluator.add_parallel(
        id="B_Kansas_Origin",
        desc="Verify originally from Kansas",
        parent=pro_b_node,
        critical=True
    )
    ks_urls = pick_sources(b.kansas_urls, all_b_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(ks_urls),
        id="B_Kansas_URL",
        desc="URL reference confirming Kansas origin",
        parent=b_ks_node,
        critical=True
    )

    leaf_ks = evaluator.add_leaf(
        id="B_Kansas_Fact",
        desc="Originally from Kansas",
        parent=b_ks_node,
        critical=True
    )
    ks_claim = f"{b.name} is originally from the U.S. state of Kansas."
    await evaluator.verify(
        claim=ks_claim,
        node=leaf_ks,
        sources=ks_urls,
        additional_instruction="Look for birthplace, hometown, or origin statements. Accept phrasing like 'from Kansas' or 'born in [city], Kansas'."
    )

    # B.4 DWTS troupe membership
    b_dwts_node = evaluator.add_parallel(
        id="B_DWTS_Troupe",
        desc="Verify became troupe member on Dancing with the Stars",
        parent=pro_b_node,
        critical=True
    )
    dwts_urls = pick_sources(b.dwts_urls, all_b_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(dwts_urls),
        id="B_DWTS_URL",
        desc="URL reference confirming DWTS troupe membership",
        parent=b_dwts_node,
        critical=True
    )

    leaf_dwts = evaluator.add_leaf(
        id="B_DWTS_Fact",
        desc="Became a troupe member on Dancing with the Stars",
        parent=b_dwts_node,
        critical=True
    )
    dwts_claim = f"{b.name} became a troupe member on Dancing with the Stars."
    await evaluator.verify(
        claim=dwts_claim,
        node=leaf_dwts,
        sources=dwts_urls,
        additional_instruction="The 'troupe' is distinct from 'pro' partners; confirm specific wording indicating troupe membership."
    )


async def verify_professional_c(evaluator: Evaluator, parent_node, c: ProfessionalC) -> None:
    # Professional C top-level node (Parallel)
    pro_c_node = evaluator.add_parallel(
        id="Professional_C_Journalist",
        desc="Identify the business journalist who meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Gating: name and at least one source overall
    all_c_sources = merge_urls(
        c.sources, c.ba_urls, c.mba_urls, c.mckinsey_urls, c.fox_business_urls, c.cbs_urls
    )
    evaluator.add_custom_node(
        result=(c is not None and (c.name or "").strip() != "" and nonempty_urls(all_c_sources)),
        id="C_Identity_Provided",
        desc="Professional C: Name is provided and at least one source URL exists",
        parent=pro_c_node,
        critical=True
    )

    # C.1 BA from Harvard College + founded 1600s
    c_ba_node = evaluator.add_parallel(
        id="C_Harvard_College_BA",
        desc="Verify bachelor's degree from Harvard College (founded in 1600s)",
        parent=pro_c_node,
        critical=True
    )
    ba_urls = pick_sources(c.ba_urls, all_c_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(ba_urls),
        id="C_BA_URL",
        desc="URL reference confirming Harvard College degree and founding date",
        parent=c_ba_node,
        critical=True
    )

    leaf_ba_degree = evaluator.add_leaf(
        id="C_BA_Degree_Fact",
        desc="Earned bachelor's degree from Harvard College",
        parent=c_ba_node,
        critical=True
    )
    ba_inst = c.ba_institution or "Harvard College"
    ba_claim = f"{c.name} holds a bachelor's degree from Harvard College (the undergraduate college of Harvard University)."
    await evaluator.verify(
        claim=ba_claim,
        node=leaf_ba_degree,
        sources=ba_urls,
        additional_instruction=f"Accept bios that say 'Harvard University' for the AB/BA as equivalent to Harvard College. Institution hint: {ba_inst}."
    )

    leaf_ba_founded = evaluator.add_leaf(
        id="C_College_Founded_1600s_Fact",
        desc="Harvard College was founded in the 1600s",
        parent=c_ba_node,
        critical=True
    )
    ba_founded_claim = "Harvard College (Harvard University) was founded in the 1600s (1636)."
    await evaluator.verify(
        claim=ba_founded_claim,
        node=leaf_ba_founded,
        sources=ba_urls,
        additional_instruction="This is a general historical fact; verify that the founding year 1636 (17th century) or general '1600s' is supported by the provided sources."
    )

    # C.2 MBA from HBS + founded 1900s and first MBA program
    c_mba_node = evaluator.add_parallel(
        id="C_Harvard_Business_School_MBA",
        desc="Verify MBA from Harvard Business School (founded in 1900s, first MBA program)",
        parent=pro_c_node,
        critical=True
    )
    mba_urls = pick_sources(c.mba_urls, all_c_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(mba_urls),
        id="C_MBA_URL",
        desc="URL reference confirming Harvard Business School MBA and founding date/first MBA program",
        parent=c_mba_node,
        critical=True
    )

    leaf_mba_degree = evaluator.add_leaf(
        id="C_MBA_Degree_Fact",
        desc="Earned MBA from Harvard Business School",
        parent=c_mba_node,
        critical=True
    )
    mba_inst = c.mba_institution or "Harvard Business School"
    mba_claim = f"{c.name} holds an MBA from Harvard Business School."
    await evaluator.verify(
        claim=mba_claim,
        node=leaf_mba_degree,
        sources=mba_urls,
        additional_instruction=f"Accept equivalents like 'MBA from Harvard' that clearly refer to HBS. Institution hint: {mba_inst}."
    )

    leaf_hbs_history = evaluator.add_leaf(
        id="C_HBS_Founded_1900s_First_MBA_Fact",
        desc="Harvard Business School was founded in the 1900s and offered the world's first MBA program",
        parent=c_mba_node,
        critical=True
    )
    hbs_hist_claim = "Harvard Business School was founded in the 1900s and offered the world's first MBA program (established in 1908)."
    await evaluator.verify(
        claim=hbs_hist_claim,
        node=leaf_hbs_history,
        sources=mba_urls,
        additional_instruction="Look for references indicating HBS was founded in 1908 and that it launched the first MBA program."
    )

    # C.3 McKinsey experience
    c_mck_node = evaluator.add_parallel(
        id="C_McKinsey_Experience",
        desc="Verify previously worked at McKinsey & Company",
        parent=pro_c_node,
        critical=True
    )
    mck_urls = pick_sources(c.mckinsey_urls, all_c_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(mck_urls),
        id="C_McKinsey_URL",
        desc="URL reference confirming McKinsey employment",
        parent=c_mck_node,
        critical=True
    )

    leaf_mck = evaluator.add_leaf(
        id="C_McKinsey_Fact",
        desc="Previously worked at McKinsey & Company",
        parent=c_mck_node,
        critical=True
    )
    mck_claim = f"{c.name} previously worked at McKinsey & Company."
    await evaluator.verify(
        claim=mck_claim,
        node=leaf_mck,
        sources=mck_urls,
        additional_instruction="Accept bios, LinkedIn profiles, or credible news bios explicitly listing McKinsey experience."
    )

    # C.4 Fox Business experience
    c_fox_node = evaluator.add_parallel(
        id="C_Fox_Business_Experience",
        desc="Verify worked at Fox Business before CBS News",
        parent=pro_c_node,
        critical=True
    )
    fox_urls = pick_sources(c.fox_business_urls, all_c_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(fox_urls),
        id="C_Fox_URL",
        desc="URL reference confirming Fox Business employment",
        parent=c_fox_node,
        critical=True
    )

    leaf_fox = evaluator.add_leaf(
        id="C_Fox_Fact",
        desc="Worked at Fox Business",
        parent=c_fox_node,
        critical=True
    )
    fox_claim = f"{c.name} worked at Fox Business (prior to joining CBS News)."
    await evaluator.verify(
        claim=fox_claim,
        node=leaf_fox,
        sources=fox_urls,
        additional_instruction="Look for employment history specifically mentioning Fox Business (Fox Business Network)."
    )

    # C.5 CBS News join year 2024
    c_cbs_node = evaluator.add_parallel(
        id="C_CBS_News_2024",
        desc="Verify joined CBS News in 2024",
        parent=pro_c_node,
        critical=True
    )
    cbs_urls = pick_sources(c.cbs_urls, all_c_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(cbs_urls),
        id="C_CBS_URL",
        desc="URL reference confirming CBS News employment and 2024 join date",
        parent=c_cbs_node,
        critical=True
    )

    leaf_cbs = evaluator.add_leaf(
        id="C_CBS_Fact",
        desc="Joined CBS News in 2024",
        parent=c_cbs_node,
        critical=True
    )
    join_hint = f" Join year hint from answer: {c.joined_cbs_year}." if (c.joined_cbs_year or "").strip() else ""
    cbs_claim = f"{c.name} joined CBS News in 2024."
    await evaluator.verify(
        claim=cbs_claim,
        node=leaf_cbs,
        sources=cbs_urls,
        additional_instruction="Confirm both that this person is with CBS News and that their join date is in 2024. If multiple dates appear, prioritize the official CBS profile or credible announcements."
    )


async def verify_professional_d(evaluator: Evaluator, parent_node, d: ProfessionalD) -> None:
    # Professional D top-level node (Parallel)
    pro_d_node = evaluator.add_parallel(
        id="Professional_D_Tennis_Player",
        desc="Identify the tennis player who meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Gating: name and at least one source overall
    all_d_sources = merge_urls(
        d.sources, d.representation_urls, d.ranking_urls, d.titles_urls, d.handedness_urls
    )
    evaluator.add_custom_node(
        result=(d is not None and (d.name or "").strip() != "" and nonempty_urls(all_d_sources)),
        id="D_Identity_Provided",
        desc="Professional D: Name is provided and at least one source URL exists",
        parent=pro_d_node,
        critical=True
    )

    # D.1 Represents Chile
    d_chile_node = evaluator.add_parallel(
        id="D_Chile_Representation",
        desc="Verify represents Chile in professional ATP tennis",
        parent=pro_d_node,
        critical=True
    )
    chile_urls = pick_sources(d.representation_urls, all_d_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(chile_urls),
        id="D_Chile_URL",
        desc="URL reference confirming Chilean representation",
        parent=d_chile_node,
        critical=True
    )

    leaf_chile = evaluator.add_leaf(
        id="D_Chile_Fact",
        desc="Represents Chile in professional ATP tennis",
        parent=d_chile_node,
        critical=True
    )
    chile_claim = f"{d.name} represents Chile on the ATP Tour."
    await evaluator.verify(
        claim=chile_claim,
        node=leaf_chile,
        sources=chile_urls,
        additional_instruction="Accept ATP profile pages or credible bios explicitly stating nationality/representation as Chile."
    )

    # D.2 Top 20 career-high during 2024
    d_rank_node = evaluator.add_parallel(
        id="D_Top_20_Ranking_2024",
        desc="Verify achieved career-high ATP ranking in top 20 during 2024",
        parent=pro_d_node,
        critical=True
    )
    ranking_urls = pick_sources(d.ranking_urls, all_d_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(ranking_urls),
        id="D_Ranking_URL",
        desc="URL reference confirming career-high ranking in top 20 during 2024",
        parent=d_rank_node,
        critical=True
    )

    leaf_rank = evaluator.add_leaf(
        id="D_Ranking_Fact",
        desc="Achieved career-high ATP singles ranking in the top 20 during the year 2024",
        parent=d_rank_node,
        critical=True
    )
    rank_claim = f"{d.name} achieved a career-high ATP singles ranking that was inside the top 20 at some point during the year 2024."
    await evaluator.verify(
        claim=rank_claim,
        node=leaf_rank,
        sources=ranking_urls,
        additional_instruction="Look for phrasing like 'career-high ranking No. X (month 2024)'; any ranking 1–20 during 2024 satisfies the condition."
    )

    # D.3 Exactly 3 ATP titles as of March 2026
    d_titles_node = evaluator.add_parallel(
        id="D_Three_ATP_Titles",
        desc="Verify has won exactly 3 ATP career titles",
        parent=pro_d_node,
        critical=True
    )
    titles_urls = pick_sources(d.titles_urls, all_d_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(titles_urls),
        id="D_Titles_URL",
        desc="URL reference confirming 3 career ATP titles",
        parent=d_titles_node,
        critical=True
    )

    leaf_titles = evaluator.add_leaf(
        id="D_Titles_Fact",
        desc="Has won exactly 3 ATP career titles",
        parent=d_titles_node,
        critical=True
    )
    titles_claim = f"As of March 2026, {d.name} has exactly three ATP Tour-level singles titles (excluding Challenger and doubles)."
    await evaluator.verify(
        claim=titles_claim,
        node=leaf_titles,
        sources=titles_urls,
        additional_instruction="Count only ATP Tour-level singles titles up to March 2026. Exclude doubles and Challenger titles."
    )

    # D.4 Left-handed
    d_hand_node = evaluator.add_parallel(
        id="D_Left_Handed",
        desc="Verify plays left-handed",
        parent=pro_d_node,
        critical=True
    )
    hand_urls = pick_sources(d.handedness_urls, all_d_sources)

    evaluator.add_custom_node(
        result=nonempty_urls(hand_urls),
        id="D_Handedness_URL",
        desc="URL reference confirming left-handed playing style",
        parent=d_hand_node,
        critical=True
    )

    leaf_hand = evaluator.add_leaf(
        id="D_Handedness_Fact",
        desc="Plays left-handed",
        parent=d_hand_node,
        critical=True
    )
    hand_claim = f"{d.name} plays left-handed."
    await evaluator.verify(
        claim=hand_claim,
        node=leaf_hand,
        sources=hand_urls,
        additional_instruction="Accept profile bios or ATP pages that list 'plays: left-handed' or similar phrasing."
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
    # Initialize evaluator (root is non-critical parallel to allow partial credit)
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

    # Extract all professionals' info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AllProfessionalsExtraction,
        extraction_name="professionals_extraction"
    )

    # Build rubric tree according to the provided JSON (parallel top-level)
    # Professional A
    if extracted.professional_a is None:
        extracted.professional_a = ProfessionalA()
    await verify_professional_a(evaluator, root, extracted.professional_a)

    # Professional B
    if extracted.professional_b is None:
        extracted.professional_b = ProfessionalB()
    await verify_professional_b(evaluator, root, extracted.professional_b)

    # Professional C
    if extracted.professional_c is None:
        extracted.professional_c = ProfessionalC()
    await verify_professional_c(evaluator, root, extracted.professional_c)

    # Professional D
    if extracted.professional_d is None:
        extracted.professional_d = ProfessionalD()
    await verify_professional_d(evaluator, root, extracted.professional_d)

    # Return final structured evaluation summary
    return evaluator.get_summary()