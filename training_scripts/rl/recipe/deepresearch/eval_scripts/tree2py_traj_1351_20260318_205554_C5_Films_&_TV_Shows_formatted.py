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
TASK_ID = "gatiss_1946_uk_be_co_pro_series"
TASK_DESCRIPTION = """
I am researching British period crime drama series that premiered in 2025 and I'm particularly interested in series where the creator also stars as the lead character. I've heard about a series set in post-war 1946 London that was an international UK-Belgium co-production. Can you identify this series and provide comprehensive information including: (1) Series Basic Information: the name of the series, the creator who also stars as the lead character and the character name they play, confirmation of the setting (year and location), the genre classification, and the UK broadcaster and premiere date in 2025; (2) Production Details: the primary UK production company, the Belgian co-production company, the actual filming location (country), the director's name, both writers' names (the creator and the co-writer), and all four executive producers; (3) Cast Information: the lead actor and character name (same as creator), and three key supporting cast members and their character names; (4) Technical Specifications: the number of episodes in Season 1, the approximate runtime per episode, and the episode structure format (how cases are distributed across episodes); (5) Broadcast and Release Information: the exact UK premiere date (specific date in 2025), the US broadcaster and US premiere date, and information about whether the series was renewed for Season 2 and when that renewal was announced. For each piece of information, please provide a reference URL from an official or reliable source that verifies the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BasicInfo(BaseModel):
    series_name: FieldWithSources = Field(default_factory=FieldWithSources)
    genre_classification: FieldWithSources = Field(default_factory=FieldWithSources)
    creator_name: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_actor_name: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_character_name: FieldWithSources = Field(default_factory=FieldWithSources)
    setting_year: FieldWithSources = Field(default_factory=FieldWithSources)
    setting_location: FieldWithSources = Field(default_factory=FieldWithSources)
    coproduction_uk_belgium: FieldWithSources = Field(default_factory=FieldWithSources)
    ensemble_supporting_cast: FieldWithSources = Field(default_factory=FieldWithSources)
    uk_broadcaster: FieldWithSources = Field(default_factory=FieldWithSources)
    uk_premiere_date: FieldWithSources = Field(default_factory=FieldWithSources)


class ProductionDetails(BaseModel):
    primary_uk_production_company: FieldWithSources = Field(default_factory=FieldWithSources)
    belgian_coproduction_company: FieldWithSources = Field(default_factory=FieldWithSources)
    filming_location_country: FieldWithSources = Field(default_factory=FieldWithSources)
    director_name: FieldWithSources = Field(default_factory=FieldWithSources)
    writer_creator_name: FieldWithSources = Field(default_factory=FieldWithSources)
    co_writer_name: FieldWithSources = Field(default_factory=FieldWithSources)
    executive_producers: FieldWithSources = Field(default_factory=FieldWithSources)  # comma-separated list is fine


class CastInfo(BaseModel):
    lead_actor_name: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_character_name: FieldWithSources = Field(default_factory=FieldWithSources)
    polly_walker_character: FieldWithSources = Field(default_factory=FieldWithSources)      # expect "Trottie Book"
    elliot_levey_character: FieldWithSources = Field(default_factory=FieldWithSources)      # expect "Inspector Bliss"
    connor_finch_character: FieldWithSources = Field(default_factory=FieldWithSources)      # expect "Jack"


class TechnicalSpecs(BaseModel):
    season1_episode_count: FieldWithSources = Field(default_factory=FieldWithSources)        # expect "6"
    approx_runtime_per_episode: FieldWithSources = Field(default_factory=FieldWithSources)   # expect around "48 minutes"
    episode_structure: FieldWithSources = Field(default_factory=FieldWithSources)            # expect "3 cases across 6 eps (two-parters)"


class ReleaseInfo(BaseModel):
    uk_broadcaster: FieldWithSources = Field(default_factory=FieldWithSources)
    uk_premiere_date: FieldWithSources = Field(default_factory=FieldWithSources)             # expect "July 16, 2025"
    us_broadcaster: FieldWithSources = Field(default_factory=FieldWithSources)               # expect "PBS"
    us_premiere_date: FieldWithSources = Field(default_factory=FieldWithSources)
    season2_renewed_status: FieldWithSources = Field(default_factory=FieldWithSources)       # expect "renewed" / "picked up"
    season2_renewal_announcement_date: FieldWithSources = Field(default_factory=FieldWithSources)


class SeriesExtraction(BaseModel):
    basic: BasicInfo = Field(default_factory=BasicInfo)
    production: ProductionDetails = Field(default_factory=ProductionDetails)
    cast: CastInfo = Field(default_factory=CastInfo)
    technical: TechnicalSpecs = Field(default_factory=TechnicalSpecs)
    release: ReleaseInfo = Field(default_factory=ReleaseInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_series_all() -> str:
    return """
Extract structured information from the answer about a British period crime drama series set in post‑war 1946 London, where the creator also stars as the lead. For EACH requested fact, also extract a list of explicit URL sources cited in the answer that verify the specific fact. Only include URLs that are explicitly present in the answer (plain URL or a markdown link target). If any value is missing in the answer text, set it to null and return an empty sources list.

Return JSON with this exact structure:

{
  "basic": {
    "series_name": {"value": string|null, "sources": [urls...]},
    "genre_classification": {"value": string|null, "sources": [urls...]},              // e.g., "British period crime drama"
    "creator_name": {"value": string|null, "sources": [urls...]},
    "lead_actor_name": {"value": string|null, "sources": [urls...]},
    "lead_character_name": {"value": string|null, "sources": [urls...]},
    "setting_year": {"value": string|null, "sources": [urls...]},                      // e.g., "1946"
    "setting_location": {"value": string|null, "sources": [urls...]},                  // e.g., "London"
    "coproduction_uk_belgium": {"value": string|null, "sources": [urls...]},          // e.g., "UK–Belgium co‑production"
    "ensemble_supporting_cast": {"value": string|null, "sources": [urls...]},         // copy short phrase such as "ensemble supporting cast" if present
    "uk_broadcaster": {"value": string|null, "sources": [urls...]},                    // e.g., "U&Alibi"
    "uk_premiere_date": {"value": string|null, "sources": [urls...]}                   // exact UK premiere date in 2025 as written in the answer
  },
  "production": {
    "primary_uk_production_company": {"value": string|null, "sources": [urls...]},     // e.g., "Eagle Eye Drama"
    "belgian_coproduction_company": {"value": string|null, "sources": [urls...]},      // e.g., "Happy Duck Films"
    "filming_location_country": {"value": string|null, "sources": [urls...]},          // country actually used for filming
    "director_name": {"value": string|null, "sources": [urls...]},                     // e.g., "Carolina Giammetta"
    "writer_creator_name": {"value": string|null, "sources": [urls...]},               // creator as a credited writer
    "co_writer_name": {"value": string|null, "sources": [urls...]},                    // e.g., "Matthew Sweet"
    "executive_producers": {"value": string|null, "sources": [urls...]}                // comma-separated list with all names if present
  },
  "cast": {
    "lead_actor_name": {"value": string|null, "sources": [urls...]},
    "lead_character_name": {"value": string|null, "sources": [urls...]},
    "polly_walker_character": {"value": string|null, "sources": [urls...]},            // expected: "Trottie Book"
    "elliot_levey_character": {"value": string|null, "sources": [urls...]},            // expected: "Inspector Bliss"
    "connor_finch_character": {"value": string|null, "sources": [urls...]}             // expected: "Jack"
  },
  "technical": {
    "season1_episode_count": {"value": string|null, "sources": [urls...]},             // expected around "6"
    "approx_runtime_per_episode": {"value": string|null, "sources": [urls...]},        // expected around "48 minutes"
    "episode_structure": {"value": string|null, "sources": [urls...]}                  // expected: "three cases across six episodes (two-part per case)" or equivalent
  },
  "release": {
    "uk_broadcaster": {"value": string|null, "sources": [urls...]},
    "uk_premiere_date": {"value": string|null, "sources": [urls...]},
    "us_broadcaster": {"value": string|null, "sources": [urls...]},                    // expected: "PBS"
    "us_premiere_date": {"value": string|null, "sources": [urls...]},
    "season2_renewed_status": {"value": string|null, "sources": [urls...]},            // e.g., "renewed"
    "season2_renewal_announcement_date": {"value": string|null, "sources": [urls...]}  // exact calendar date if provided
  }
}

Rules:
- Do NOT invent URLs. Only include URLs explicitly present in the answer.
- Keep values as strings; do not normalize or reformat—extract exactly as written.
- If a section is not present, return null/empty accordingly.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u2 = u.strip()
        if u2 and u2 not in seen:
            out.append(u2)
            seen.add(u2)
    return out


def _merge_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(_clean_urls(lst))
    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for u in merged:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _with_citation_instruction(base: str, urls: Optional[List[str]]) -> str:
    if urls and len(urls) > 0:
        return base
    # Explicitly penalize missing citations per source-grounding policy
    return base + "\nIMPORTANT: No valid source URLs were provided in the answer for this claim. You must judge this claim as NOT SUPPORTED."


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identification_and_core(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Identify_and_Confirm_Series",
        desc="Identify the series and confirm it matches the core identifying constraints.",
        parent=parent,
        critical=True,
    )

    claims_and_nodes: List[tuple] = []

    # 1) Series name
    leaf = evaluator.add_leaf(
        id="Series_Name_with_Citation",
        desc="Provide the series name and at least one reference URL verifying the name.",
        parent=node,
        critical=True,
    )
    series_name = data.basic.series_name.value or ""
    series_name_src = _clean_urls(data.basic.series_name.sources)
    claim = f"The series name is '{series_name}'."
    add_ins = _with_citation_instruction(
        "Verify that the provided page(s) explicitly give the title of the series as stated. Minor punctuation/casing variations are acceptable.",
        series_name_src,
    )
    claims_and_nodes.append((claim, series_name_src, leaf, add_ins))

    # 2) British period crime drama
    leaf = evaluator.add_leaf(
        id="British_Period_Crime_Drama_with_Citation",
        desc="Confirm the series is a British period crime drama and provide a reference URL verifying this classification.",
        parent=node,
        critical=True,
    )
    genre_src = _clean_urls(data.basic.genre_classification.sources)
    claim = "This series is a British period crime drama."
    add_ins = _with_citation_instruction(
        "Treat 'period crime drama', 'crime mystery period drama', or equivalent wording as acceptable. The classification must be explicit on the page(s).",
        genre_src,
    )
    claims_and_nodes.append((claim, genre_src, leaf, add_ins))

    # 3) Creator is Mark Gatiss
    leaf = evaluator.add_leaf(
        id="Creator_Is_Mark_Gatiss_with_Citation",
        desc="Confirm the creator is Mark Gatiss and provide a reference URL verifying this.",
        parent=node,
        critical=True,
    )
    creator_src = _merge_sources(data.basic.creator_name.sources, data.production.writer_creator_name.sources)
    claim = "The creator of the series is Mark Gatiss."
    add_ins = _with_citation_instruction(
        "Accept phrasings like 'created by Mark Gatiss' or 'from creator Mark Gatiss'.",
        creator_src,
    )
    claims_and_nodes.append((claim, creator_src, leaf, add_ins))

    # 4) Mark Gatiss is lead actor
    leaf = evaluator.add_leaf(
        id="Mark_Gatiss_Is_Lead_Actor_with_Citation",
        desc="Confirm Mark Gatiss stars as the lead actor and provide a reference URL verifying he is the lead.",
        parent=node,
        critical=True,
    )
    lead_actor_src = _merge_sources(data.basic.lead_actor_name.sources, data.cast.lead_actor_name.sources)
    claim = "Mark Gatiss stars as the lead actor in the series."
    add_ins = _with_citation_instruction(
        "Accept wording such as 'starring Mark Gatiss' or 'Mark Gatiss leads the cast'.",
        lead_actor_src,
    )
    claims_and_nodes.append((claim, lead_actor_src, leaf, add_ins))

    # 5) Lead character name played by Gatiss
    leaf = evaluator.add_leaf(
        id="Lead_Character_Name_Played_By_Gatiss_with_Citation",
        desc="Provide the lead character name played by Mark Gatiss and provide a reference URL verifying the character name.",
        parent=node,
        critical=True,
    )
    lead_char = data.basic.lead_character_name.value or data.cast.lead_character_name.value or ""
    lead_char_src = _merge_sources(data.basic.lead_character_name.sources, data.cast.lead_character_name.sources)
    claim = f"Mark Gatiss plays the lead character named '{lead_char}'."
    add_ins = _with_citation_instruction(
        "Allow for reasonable title or nickname variations (e.g., with/without 'Mr.', 'Inspector', etc.) when clearly the same character.",
        lead_char_src,
    )
    claims_and_nodes.append((claim, lead_char_src, leaf, add_ins))

    # 6) Setting: 1946 London
    leaf = evaluator.add_leaf(
        id="Setting_1946_London_with_Citation",
        desc="Confirm the setting is 1946 London and provide a reference URL verifying the setting year and location.",
        parent=node,
        critical=True,
    )
    set_src = _merge_sources(data.basic.setting_year.sources, data.basic.setting_location.sources)
    claim = "The series is set in London in 1946 (post‑war)."
    add_ins = _with_citation_instruction(
        "Treat 'post‑war 1946 London' or equivalent wording as acceptable. Both the year (1946) and London should be supported.",
        set_src,
    )
    claims_and_nodes.append((claim, set_src, leaf, add_ins))

    # 7) UK–Belgium co‑production
    leaf = evaluator.add_leaf(
        id="UK_Belgium_CoProduction_with_Citation",
        desc="Confirm the series is a UK–Belgium co-production and provide a reference URL verifying the co-production nature.",
        parent=node,
        critical=True,
    )
    coprod_src = _merge_sources(
        data.basic.coproduction_uk_belgium.sources,
        data.production.primary_uk_production_company.sources,
        data.production.belgian_coproduction_company.sources,
    )
    claim = "The series is a UK–Belgium co‑production."
    add_ins = _with_citation_instruction(
        "Pages may reference both a UK company and a Belgian co‑production partner; that is acceptable as evidence.",
        coprod_src,
    )
    claims_and_nodes.append((claim, coprod_src, leaf, add_ins))

    # 8) Ensemble supporting cast
    leaf = evaluator.add_leaf(
        id="Ensemble_Supporting_Cast_with_Citation",
        desc="Confirm the series features an ensemble supporting cast and provide a reference URL verifying this.",
        parent=node,
        critical=True,
    )
    ensemble_src = _clean_urls(data.basic.ensemble_supporting_cast.sources)
    claim = "The series features an ensemble supporting cast."
    add_ins = _with_citation_instruction(
        "Accept explicit use of 'ensemble' or an equivalent description that clearly indicates an ensemble supporting cast.",
        ensemble_src,
    )
    claims_and_nodes.append((claim, ensemble_src, leaf, add_ins))

    # 9) UK broadcaster is U&Alibi
    leaf = evaluator.add_leaf(
        id="UK_Broadcaster_UAndAlibi_with_Citation",
        desc="Confirm the UK broadcaster is U&Alibi and provide a reference URL verifying this.",
        parent=node,
        critical=True,
    )
    uk_broad_src = _merge_sources(data.basic.uk_broadcaster.sources, data.release.uk_broadcaster.sources)
    claim = "The UK broadcaster for the series is U&Alibi."
    add_ins = _with_citation_instruction(
        "Accept minor formatting variants such as 'U & Alibi' or 'U& ALIBI'. If the brand appears as part of UKTV, that is acceptable.",
        uk_broad_src,
    )
    claims_and_nodes.append((claim, uk_broad_src, leaf, add_ins))

    # 10) UK premiere date: July 16, 2025
    leaf = evaluator.add_leaf(
        id="UK_Premiere_Date_July_16_2025_with_Citation",
        desc="Provide the exact UK premiere date (July 16, 2025) and a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    uk_prem_src = _merge_sources(data.basic.uk_premiere_date.sources, data.release.uk_premiere_date.sources)
    claim = "The UK premiere date is July 16, 2025 (accept '16 July 2025' as equivalent)."
    add_ins = _with_citation_instruction(
        "Date formats like '16 July 2025' or '16/07/2025' should be treated as equivalent to 'July 16, 2025'.",
        uk_prem_src,
    )
    claims_and_nodes.append((claim, uk_prem_src, leaf, add_ins))

    # Run all parallel verifications together
    await evaluator.batch_verify(claims_and_nodes)


async def verify_production_details(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Production_Details",
        desc="Provide production company and key creative personnel details as requested.",
        parent=parent,
        critical=True,
    )

    items: List[tuple] = []

    # Primary UK production company - Eagle Eye Drama
    leaf = evaluator.add_leaf(
        id="Primary_UK_Production_Company_Eagle_Eye_Drama_with_Citation",
        desc="State the primary UK production company is Eagle Eye Drama and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    ukco_src = _clean_urls(data.production.primary_uk_production_company.sources)
    claim = "The primary UK production company is Eagle Eye Drama."
    add_ins = _with_citation_instruction(
        "Look for production credits. Accept 'produced by Eagle Eye Drama' or equivalent wording.",
        ukco_src,
    )
    items.append((claim, ukco_src, leaf, add_ins))

    # Belgian co-production company - Happy Duck Films
    leaf = evaluator.add_leaf(
        id="Belgian_CoProduction_Company_Happy_Duck_Films_with_Citation",
        desc="State the Belgian co-production company is Happy Duck Films and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    beco_src = _clean_urls(data.production.belgian_coproduction_company.sources)
    claim = "The Belgian co‑production company is Happy Duck Films."
    add_ins = _with_citation_instruction(
        "Look for co‑production credits that explicitly mention a Belgian company 'Happy Duck Films'.",
        beco_src,
    )
    items.append((claim, beco_src, leaf, add_ins))

    # Filmed in Belgium
    leaf = evaluator.add_leaf(
        id="Filmed_Primarily_in_Belgium_with_Citation",
        desc="State the actual filming location country is Belgium and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    film_src = _clean_urls(data.production.filming_location_country.sources)
    claim = "Principal filming took place in Belgium."
    add_ins = _with_citation_instruction(
        "Accept statements like 'filmed in Belgium' or mentions of Belgian locations used for filming.",
        film_src,
    )
    items.append((claim, film_src, leaf, add_ins))

    # Director - Carolina Giammetta
    leaf = evaluator.add_leaf(
        id="Director_Carolina_Giammetta_with_Citation",
        desc="State the director is Carolina Giammetta and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    dir_src = _clean_urls(data.production.director_name.sources)
    claim = "The director of the series is Carolina Giammetta."
    add_ins = _with_citation_instruction(
        "Look for 'director' credit. If multiple directors are listed, ensure Carolina Giammetta is included.",
        dir_src,
    )
    items.append((claim, dir_src, leaf, add_ins))

    # Writer - Mark Gatiss (creator)
    leaf = evaluator.add_leaf(
        id="Writer_Mark_Gatiss_with_Citation",
        desc="State Mark Gatiss is a credited writer and provide a reference URL verifying the writing credit.",
        parent=node,
        critical=True,
    )
    w1_src = _merge_sources(data.production.writer_creator_name.sources, data.basic.creator_name.sources)
    claim = "Mark Gatiss is credited as a writer on the series."
    add_ins = _with_citation_instruction(
        "Accept 'written by Mark Gatiss' or 'created and written by Mark Gatiss'.",
        w1_src,
    )
    items.append((claim, w1_src, leaf, add_ins))

    # Co-writer - Matthew Sweet
    leaf = evaluator.add_leaf(
        id="CoWriter_Matthew_Sweet_with_Citation",
        desc="State Matthew Sweet is the co-writer and provide a reference URL verifying the co-writing credit.",
        parent=node,
        critical=True,
    )
    w2_src = _clean_urls(data.production.co_writer_name.sources)
    claim = "Matthew Sweet is credited as a co‑writer on the series."
    add_ins = _with_citation_instruction(
        "Look for explicit co‑writing credit naming Matthew Sweet.",
        w2_src,
    )
    items.append((claim, w2_src, leaf, add_ins))

    # Executive producers - all four names
    leaf = evaluator.add_leaf(
        id="Executive_Producers_All_Four_Names_with_Citation",
        desc="List all four executive producers (Mark Gatiss, Carolina Giammetta, Jo McGrath, Walter Iuzzolino) and provide a reference URL verifying them.",
        parent=node,
        critical=True,
    )
    ep_src = _clean_urls(data.production.executive_producers.sources)
    claim = "The executive producers are Mark Gatiss, Carolina Giammetta, Jo McGrath, and Walter Iuzzolino."
    add_ins = _with_citation_instruction(
        "Pass only if ALL FOUR names are present on the provided page(s) explicitly as 'executive producer(s)'.",
        ep_src,
    )
    items.append((claim, ep_src, leaf, add_ins))

    await evaluator.batch_verify(items)


async def verify_cast_information(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Cast_Information",
        desc="Provide the three requested supporting cast members and their character names with citations.",
        parent=parent,
        critical=True,
    )

    items: List[tuple] = []

    # Polly Walker -> Trottie Book
    leaf = evaluator.add_leaf(
        id="Polly_Walker_Plays_Trottie_Book_with_Citation",
        desc="State Polly Walker plays Trottie Book and provide a reference URL verifying the actor-character pairing.",
        parent=node,
        critical=True,
    )
    pw_src = _clean_urls(data.cast.polly_walker_character.sources)
    claim = "Polly Walker plays Trottie Book."
    add_ins = _with_citation_instruction(
        "Accept minor title variants (e.g., 'Lady Trottie Book') if clearly the same character.",
        pw_src,
    )
    items.append((claim, pw_src, leaf, add_ins))

    # Elliot Levey -> Inspector Bliss
    leaf = evaluator.add_leaf(
        id="Elliot_Levey_Plays_Inspector_Bliss_with_Citation",
        desc="State Elliot Levey plays Inspector Bliss and provide a reference URL verifying the actor-character pairing.",
        parent=node,
        critical=True,
    )
    el_src = _clean_urls(data.cast.elliot_levey_character.sources)
    claim = "Elliot Levey plays Inspector Bliss."
    add_ins = _with_citation_instruction(
        "Look for cast lists crediting Elliot Levey as Inspector Bliss.",
        el_src,
    )
    items.append((claim, el_src, leaf, add_ins))

    # Connor Finch -> Jack
    leaf = evaluator.add_leaf(
        id="Connor_Finch_Plays_Jack_with_Citation",
        desc="State Connor Finch plays Jack and provide a reference URL verifying the actor-character pairing.",
        parent=node,
        critical=True,
    )
    cf_src = _clean_urls(data.cast.connor_finch_character.sources)
    claim = "Connor Finch plays a character named Jack."
    add_ins = _with_citation_instruction(
        "If a surname appears (e.g., 'Jack Surname'), that still satisfies the requirement as long as 'Jack' is the given name.",
        cf_src,
    )
    items.append((claim, cf_src, leaf, add_ins))

    await evaluator.batch_verify(items)


async def verify_technical_specs(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Technical_Specifications",
        desc="Provide the requested episode/format technical specifications.",
        parent=parent,
        critical=True,
    )

    items: List[tuple] = []

    # Season 1 episode count: 6
    leaf = evaluator.add_leaf(
        id="Season_1_Episode_Count_6_with_Citation",
        desc="State Season 1 consists of 6 episodes and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    epc_src = _clean_urls(data.technical.season1_episode_count.sources)
    claim = "Season 1 consists of 6 episodes."
    add_ins = _with_citation_instruction(
        "Explicit confirmation of '6x' or '6 episodes' on the page(s) is required.",
        epc_src,
    )
    items.append((claim, epc_src, leaf, add_ins))

    # Approx runtime ~48 minutes
    leaf = evaluator.add_leaf(
        id="Approx_Runtime_48_Minutes_with_Citation",
        desc="State the approximate runtime per episode is ~48 minutes and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    rt_src = _clean_urls(data.technical.approx_runtime_per_episode.sources)
    claim = "The approximate runtime per episode is about 48 minutes."
    add_ins = _with_citation_instruction(
        "Allow reasonable approximations (e.g., 45–50 minutes) as equivalent to ~48 minutes.",
        rt_src,
    )
    items.append((claim, rt_src, leaf, add_ins))

    # Three cases across six episodes (two-part per case)
    leaf = evaluator.add_leaf(
        id="Three_Cases_Across_Six_Episodes_TwoPart_Per_Case_with_Citation",
        desc="Describe the episode structure as three cases told across six episodes (two-part per case) and provide a reference URL verifying this structure.",
        parent=node,
        critical=True,
    )
    struct_src = _clean_urls(data.technical.episode_structure.sources)
    claim = "The series structure comprises three cases told across six episodes, with each case spanning two parts."
    add_ins = _with_citation_instruction(
        "Accept phrasing like 'three two‑part stories across six episodes' as equivalent.",
        struct_src,
    )
    items.append((claim, struct_src, leaf, add_ins))

    await evaluator.batch_verify(items)


async def verify_broadcast_and_release(evaluator: Evaluator, parent, data: SeriesExtraction) -> None:
    node = evaluator.add_parallel(
        id="Broadcast_and_Release_Information",
        desc="Provide US broadcast info and Season 2 renewal info (including announcement timing) with citations.",
        parent=parent,
        critical=True,
    )

    items: List[tuple] = []

    # US broadcaster is PBS
    leaf = evaluator.add_leaf(
        id="US_Broadcaster_PBS_with_Citation",
        desc="State the US broadcaster is PBS and provide a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    us_broad_src = _clean_urls(data.release.us_broadcaster.sources)
    claim = "In the United States, the broadcaster is PBS (e.g., PBS, PBS Masterpiece/Mystery!)."
    add_ins = _with_citation_instruction(
        "Accept branding variants like 'PBS Masterpiece' or 'Masterpiece on PBS' as PBS distribution.",
        us_broad_src,
    )
    items.append((claim, us_broad_src, leaf, add_ins))

    # US premiere date (as provided in the answer)
    leaf = evaluator.add_leaf(
        id="US_Premiere_Date_Provided_with_Citation",
        desc="Provide the US premiere date (exact calendar date) and a reference URL verifying it.",
        parent=node,
        critical=True,
    )
    us_prem_val = data.release.us_premiere_date.value or ""
    us_prem_src = _clean_urls(data.release.us_premiere_date.sources)
    claim = f"The U.S. premiere date is {us_prem_val}."
    add_ins = _with_citation_instruction(
        "Verify the exact U.S. premiere calendar date on the page(s). Allow different date formats as long as the date is the same.",
        us_prem_src,
    )
    items.append((claim, us_prem_src, leaf, add_ins))

    # Season 2 renewed status
    leaf = evaluator.add_leaf(
        id="Season_2_Renewed_Status_with_Citation",
        desc="State whether the series was renewed for Season 2 and provide a reference URL verifying the renewal status.",
        parent=node,
        critical=True,
    )
    s2_status_src = _clean_urls(data.release.season2_renewed_status.sources)
    claim = "The series was renewed for Season 2."
    add_ins = _with_citation_instruction(
        "Look for explicit wording like 'renewed for Season 2' or a recommission announcement.",
        s2_status_src,
    )
    items.append((claim, s2_status_src, leaf, add_ins))

    # Season 2 renewal announcement date (as provided)
    leaf = evaluator.add_leaf(
        id="Season_2_Renewal_Announcement_Date_with_Citation",
        desc="Provide the Season 2 renewal announcement date and a reference URL verifying that date.",
        parent=node,
        critical=True,
    )
    s2_date_val = data.release.season2_renewal_announcement_date.value or ""
    s2_date_src = _clean_urls(data.release.season2_renewal_announcement_date.sources)
    claim = f"The Season 2 renewal was announced on {s2_date_val}."
    add_ins = _with_citation_instruction(
        "Verify the exact announcement date on the page(s). Allow standard date format variations.",
        s2_date_src,
    )
    items.append((claim, s2_date_src, leaf, add_ins))

    # Season 2 renewal announced before Season 1 aired (UK)
    leaf = evaluator.add_leaf(
        id="Season_2_Renewal_Announced_Before_Season_1_Aired_with_Citation",
        desc="Confirm the Season 2 renewal announcement occurred before Season 1 aired and provide a reference URL verifying the timing relationship.",
        parent=node,
        critical=True,
    )
    uk_prem_val = data.basic.uk_premiere_date.value or data.release.uk_premiere_date.value or "July 16, 2025"
    combo_src = _merge_sources(
        data.release.season2_renewal_announcement_date.sources,
        data.basic.uk_premiere_date.sources,
        data.release.uk_premiere_date.sources,
    )
    claim = f"The Season 2 renewal announcement occurred before the Season 1 UK premiere date of {uk_prem_val}."
    add_ins = _with_citation_instruction(
        "Use the announcement date source and the UK premiere date source to determine ordering. Pass only if the announcement date is earlier than the UK premiere date.",
        combo_src,
    )
    items.append((claim, combo_src, leaf, add_ins))

    await evaluator.batch_verify(items)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Wrapper root
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

    # Build the top-level (as described by rubric) under the wrapper root
    main = evaluator.add_sequential(
        id="Series_Identification_and_Comprehensive_Information",
        desc="Evaluate that the response identifies the intended series and provides all requested information with per-fact reference URLs.",
        parent=root,
        critical=True,
    )

    # 1) Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_series_all(),
        template_class=SeriesExtraction,
        extraction_name="series_extraction"
    )

    # 2) Identify and confirm the series (parallel)
    await verify_identification_and_core(evaluator, main, extracted)

    # 3) Provide requested details (parallel block with 4 parallel subgroups)
    provide_details = evaluator.add_parallel(
        id="Provide_Requested_Details",
        desc="Provide the requested production, cast, technical, and release/renewal details, each with its own verifying URL.",
        parent=main,
        critical=True,
    )

    # 3.1 Production
    await verify_production_details(evaluator, provide_details, extracted)

    # 3.2 Cast
    await verify_cast_information(evaluator, provide_details, extracted)

    # 3.3 Technical specifications
    await verify_technical_specs(evaluator, provide_details, extracted)

    # 3.4 Broadcast and release
    await verify_broadcast_and_release(evaluator, provide_details, extracted)

    return evaluator.get_summary()