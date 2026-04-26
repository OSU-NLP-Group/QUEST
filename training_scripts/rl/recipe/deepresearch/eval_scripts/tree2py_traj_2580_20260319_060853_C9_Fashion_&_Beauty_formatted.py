import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "luxury_beauty_ambassadors_2024"
TASK_DESCRIPTION = """
Identify three music artists who became beauty brand ambassadors for luxury brands offering makeup products in 2024, where each artist also appeared in fashion brand campaigns during the same year. The three artists must meet artist-specific timing and age constraints, and each response must include the artist name, the beauty brand, the exact announcement month and year, at least one 2024 fashion campaign, evidence of music activity in 2024, and reference URLs supporting each required claim.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampaignInfo(BaseModel):
    brand: Optional[str] = None
    campaign_name_or_collection: Optional[str] = None
    season: Optional[str] = None  # e.g., "FW24", "Fall/Winter 2024"
    year: Optional[str] = None    # e.g., "2024"
    urls: List[str] = Field(default_factory=list)


class ArtistEntry(BaseModel):
    # Core identity
    name: Optional[str] = None

    # Beauty ambassador info
    beauty_brand: Optional[str] = None
    announcement_month_year: Optional[str] = None  # e.g., "March 2024"
    official_designation_title: Optional[str] = None  # e.g., "beauty ambassador", "global partner"
    designation_scope: Optional[str] = None  # e.g., "global", "international"
    ambassador_announcement_urls: List[str] = Field(default_factory=list)
    major_publication_urls: List[str] = Field(default_factory=list)

    # Brand-level checks
    brand_is_luxury_urls: List[str] = Field(default_factory=list)
    brand_offers_makeup_urls: List[str] = Field(default_factory=list)

    # Music identity and activity
    profile_urls: List[str] = Field(default_factory=list)  # bio pages (e.g., Wikipedia, official site)
    music_activity_statement: Optional[str] = None
    music_urls: List[str] = Field(default_factory=list)

    # Second-artist specific (hit releases)
    hit_releases_statement: Optional[str] = None
    hit_releases_urls: List[str] = Field(default_factory=list)
    first_or_new_designation_urls: List[str] = Field(default_factory=list)  # "first celebrity ambassador" evidence

    # Third-artist specific (beauty division; major album/chart)
    beauty_division_urls: List[str] = Field(default_factory=list)
    major_album_or_chart_statement: Optional[str] = None
    major_album_or_chart_urls: List[str] = Field(default_factory=list)

    # Fashion campaigns (2024)
    campaigns: List[CampaignInfo] = Field(default_factory=list)
    campaign_brand_info_urls: List[str] = Field(default_factory=list)  # brand-recognition support
    luxury_or_premium_campaign_urls: List[str] = Field(default_factory=list)  # second-artist fashion luxury/premium support

    # Age / DOB evidence
    dob: Optional[str] = None
    age_or_dob_urls: List[str] = Field(default_factory=list)

    # Misc
    notes: Optional[str] = None


class ArtistsExtraction(BaseModel):
    first: Optional[ArtistEntry] = None
    second: Optional[ArtistEntry] = None
    third: Optional[ArtistEntry] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_artists() -> str:
    return """
Extract structured information for exactly three artists described in the answer. If the answer contains more than three, take only the first three that best match the requested roles (first/second/third as ordered in the answer). If fewer are present, still extract as many as available and leave missing fields as null or empty arrays.

For each of the three artists, extract the following JSON fields:

Common fields for all artists:
- name: The artist's name (string).
- beauty_brand: The beauty brand they became ambassador for (string).
- announcement_month_year: The exact announcement month and year as written in the answer (e.g., "March 2024", "September 2024", "October 2024").
- official_designation_title: The official title used (e.g., "beauty ambassador", "global partner", etc.).
- designation_scope: The scope modifier if present (e.g., "global", "international"); null if not stated.
- ambassador_announcement_urls: URLs that directly support the ambassador announcement (list of URLs).
- major_publication_urls: URLs from major fashion/beauty publications that reported the announcement (list of URLs).
- brand_is_luxury_urls: URLs supporting that the beauty brand is a luxury brand (list of URLs).
- brand_offers_makeup_urls: URLs supporting that the brand offers makeup products (list of URLs).
- profile_urls: URLs that establish the person is primarily a music artist (e.g., Wikipedia, label/official, Spotify/About) (list of URLs).
- music_activity_statement: A brief string capturing 2024 music activity evidence as stated (e.g., releases, tours, performances); null if missing.
- music_urls: URLs supporting 2024 music activity (list of URLs).
- campaigns: An array of objects showing 2024 fashion campaign participation. Each object:
  - brand: The fashion brand (string; null if missing)
  - campaign_name_or_collection: The campaign or collection name (string; null if missing)
  - season: The season label if mentioned (e.g., "FW24", "Fall/Winter 2024") (string; null if missing)
  - year: The campaign year if present (string; null if missing)
  - urls: URLs supporting this campaign (list of URLs)
- campaign_brand_info_urls: Extra URLs proving the fashion brand(s) are real/recognized brands (brand sites or reputable publications) (list of URLs).
- dob: Date of birth string if stated (e.g., "Jan 12, 2001") (string; null if missing).
- age_or_dob_urls: URLs that can verify DOB and thus age (list of URLs).
- notes: Any short notes you think the answer explicitly states that are relevant; otherwise null.

Extra fields (only populate if applicable for that artist type):
- For the second artist:
  - hit_releases_statement: A short statement about hit releases in 2024 (string; null if missing).
  - hit_releases_urls: URLs supporting those hit releases (list of URLs).
  - first_or_new_designation_urls: URLs explicitly stating this was the brand's first celebrity ambassador/global partner or a notable new partnership designation (list of URLs).
  - luxury_or_premium_campaign_urls: URLs supporting that the fashion campaign brand is luxury/premium (list of URLs).
- For the third artist:
  - beauty_division_urls: URLs supporting that the beauty brand is the beauty division of a luxury fashion house (list of URLs).
  - major_album_or_chart_statement: Short statement about a major album or significant chart success in 2024 (string; null if missing).
  - major_album_or_chart_urls: URLs supporting that album or chart success (list of URLs).

Rules:
- Extract only what is explicitly present in the answer; do not invent missing data.
- For every URL field, extract only valid, complete URLs that are explicitly present. If none are present, return an empty list.
- If any field is missing, set it to null (for strings) or [] (for URL lists/arrays).
- Make sure months and years are preserved exactly as they appear in the answer text.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _join_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        if lst:
            merged.extend(lst)
    return _dedup_urls(merged)


def _campaigns_2024(entry: Optional[ArtistEntry]) -> List[CampaignInfo]:
    if not entry or not entry.campaigns:
        return []
    out = []
    for c in entry.campaigns:
        if (c.year and "2024" in c.year) or (c.season and "24" in c.season):
            out.append(c)
    return out


def _first_campaign_2024(entry: Optional[ArtistEntry]) -> Optional[CampaignInfo]:
    cands = _campaigns_2024(entry)
    return cands[0] if cands else (entry.campaigns[0] if entry and entry.campaigns else None)


def _distinct_campaign_brands_2024(entry: Optional[ArtistEntry]) -> List[str]:
    cands = _campaigns_2024(entry)
    brands = []
    seen = set()
    for c in cands:
        b = (c.brand or "").strip()
        if b and b.lower() not in seen:
            seen.add(b.lower())
            brands.append(b)
    return brands


def _collect_campaign_urls_2024(entry: Optional[ArtistEntry]) -> List[str]:
    cands = _campaigns_2024(entry)
    urls: List[str] = []
    for c in cands:
        urls.extend(c.urls or [])
    return _dedup_urls(urls)


def _collect_fw24_campaign_urls(entry: Optional[ArtistEntry]) -> Tuple[bool, List[str], Optional[str]]:
    fw_terms = ["fw24", "fall/winter 2024", "fall winter 2024", "f/w 2024", "fall-winter 2024"]
    matched_urls: List[str] = []
    matched_label: Optional[str] = None
    if not entry or not entry.campaigns:
        return False, [], None
    for c in entry.campaigns:
        season = (c.season or "").lower()
        if any(t in season for t in fw_terms):
            matched_urls.extend(c.urls or [])
            matched_label = c.season
    return (len(matched_urls) > 0), _dedup_urls(matched_urls), matched_label


def _has_at_least_n_campaign_brands_2024(entry: Optional[ArtistEntry], n: int) -> bool:
    return len(_distinct_campaign_brands_2024(entry)) >= n


# --------------------------------------------------------------------------- #
# Verification builders (per-artist)                                          #
# --------------------------------------------------------------------------- #
async def verify_first_artist(evaluator: Evaluator, parent, entry: Optional[ArtistEntry]) -> None:
    node = evaluator.add_parallel(
        id="first_artist",
        desc="First qualifying artist (March 2024; international/global beauty ambassador; age 20–30).",
        parent=parent,
        critical=False
    )

    # ----- Output completeness and citations (presence checks) -----
    oc = evaluator.add_parallel(
        id="first_output_completeness_and_citations",
        desc="First artist output includes all required fields and URLs sufficient to verify each required claim.",
        parent=node,
        critical=True
    )
    # Presence leaves (custom nodes)
    evaluator.add_custom_node(bool(entry and entry.name and entry.name.strip()),
                              "first_field_artist_name_present",
                              "Artist name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.beauty_brand and entry.beauty_brand.strip()),
                              "first_field_beauty_brand_present",
                              "Beauty brand name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.announcement_month_year and entry.announcement_month_year.strip()),
                              "first_field_announcement_date_present",
                              "Announcement date (month and year) is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(_first_campaign_2024(entry) is not None),
                              "first_field_fashion_campaign_present",
                              "At least one fashion campaign (name/identifier) is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and ((entry.music_activity_statement and entry.music_activity_statement.strip()) or entry.music_urls)),
                              "first_field_music_activity_evidence_present",
                              "A music-activity evidence statement for 2024 is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.ambassador_announcement_urls),
                              "first_url_supports_ambassador_announcement",
                              "Provides URL(s) supporting the ambassador announcement timing and official designation/scope.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.major_publication_urls),
                              "first_url_supports_major_publication_coverage",
                              "Provides URL(s) showing the announcement was covered by at least one major fashion or beauty publication.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.brand_is_luxury_urls),
                              "first_url_supports_brand_is_luxury",
                              "Provides URL(s) supporting that the beauty brand is a luxury brand.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.brand_offers_makeup_urls),
                              "first_url_supports_brand_offers_makeup",
                              "Provides URL(s) supporting that the beauty brand offers makeup products.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and (_collect_campaign_urls_2024(entry) or entry.campaign_brand_info_urls)),
                              "first_url_supports_fashion_campaign",
                              "Provides URL(s) supporting the 2024 fashion campaign participation.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and (entry.music_urls)),
                              "first_url_supports_music_activity_2024",
                              "Provides URL(s) supporting music activity in 2024.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.age_or_dob_urls),
                              "first_url_supports_age_or_dob",
                              "Provides URL(s) sufficient to verify the artist's age at the time of announcement (e.g., DOB source).",
                              parent=oc, critical=True)

    # ----- Beauty ambassador criteria -----
    bac = evaluator.add_parallel(
        id="first_beauty_ambassador_criteria",
        desc="First artist meets beauty-ambassador constraints.",
        parent=node,
        critical=True
    )
    # 1) Announcement occurred in March 2024
    leaf_1 = evaluator.add_leaf("first_announcement_month",
                                "Announcement occurred in March 2024.",
                                parent=bac, critical=True)
    claim = f"The announcement that {entry.name if entry else 'the artist'} became a beauty brand ambassador for {entry.beauty_brand if entry else 'the brand'} occurred in March 2024."
    await evaluator.verify(claim=claim, node=leaf_1,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Verify month/year. The source should clearly indicate the announcement timing as March 2024.")
    # 2) Scope global/international
    leaf_2 = evaluator.add_leaf("first_scope_international_or_global",
                                "Announcement explicitly indicates international or global beauty ambassador status (or equivalent scope).",
                                parent=bac, critical=True)
    claim = f"The announcement indicates {entry.name if entry else 'the artist'} was appointed as an international or global beauty ambassador for {entry.beauty_brand if entry else 'the brand'} (or equivalent scope)."
    await evaluator.verify(claim=claim, node=leaf_2,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Look for terms like 'global', 'international', or equivalents in the announcement sources.")
    # 3) Brand is luxury
    leaf_3 = evaluator.add_leaf("first_brand_is_luxury",
                                "Beauty brand is a luxury brand.",
                                parent=bac, critical=True)
    claim = f"The beauty brand {entry.beauty_brand if entry else 'the brand'} is a luxury brand."
    await evaluator.verify(claim=claim, node=leaf_3,
                           sources=(entry.brand_is_luxury_urls if entry else []),
                           additional_instruction="Accept reputable fashion/beauty publications or industry references that classify the brand as luxury.")
    # 4) Brand offers makeup
    leaf_4 = evaluator.add_leaf("first_brand_offers_makeup",
                                "Beauty brand offers makeup products as part of its portfolio.",
                                parent=bac, critical=True)
    claim = f"The beauty brand {entry.beauty_brand if entry else 'the brand'} offers makeup products."
    await evaluator.verify(claim=claim, node=leaf_4,
                           sources=(entry.brand_offers_makeup_urls if entry else []),
                           additional_instruction="Confirm that the brand's beauty line sells makeup (e.g., lipsticks, foundations, etc.).")
    # 5) Official designation
    leaf_5 = evaluator.add_leaf("first_official_designation",
                                "Partnership is officially designated as ambassador/beauty ambassador (or equivalent official title).",
                                parent=bac, critical=True)
    claim = f"The partnership designates {entry.name if entry else 'the artist'} as an ambassador or beauty ambassador of {entry.beauty_brand if entry else 'the brand'}."
    await evaluator.verify(claim=claim, node=leaf_5,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="The source should clearly use 'ambassador' or an equivalent official title.")
    # 6) Major publication coverage
    leaf_6 = evaluator.add_leaf("first_major_publication_coverage",
                                "Announcement covered by at least one major fashion or beauty publication.",
                                parent=bac, critical=True)
    claim = f"The ambassador announcement of {entry.name if entry else 'the artist'} for {entry.beauty_brand if entry else 'the brand'} was covered by a major fashion or beauty publication."
    await evaluator.verify(claim=claim, node=leaf_6,
                           sources=(entry.major_publication_urls if entry else []),
                           additional_instruction="Major publications include Vogue, Harper’s Bazaar, Elle, W, Allure, etc. The page must be about this announcement.")

    # ----- Music criteria -----
    mc = evaluator.add_parallel(
        id="first_music_criteria",
        desc="First artist meets music-artist/activity constraints.",
        parent=node,
        critical=True
    )
    # 1) Primarily music artist
    leaf_m1 = evaluator.add_leaf("first_primarily_music_artist",
                                 "Artist is primarily known as a music artist (singer/musician/recording artist).",
                                 parent=mc, critical=True)
    claim = f"{entry.name if entry else 'The person'} is primarily known as a music artist (singer/musician/recording artist)."
    await evaluator.verify(claim=claim, node=leaf_m1,
                           sources=(entry.profile_urls if entry else []),
                           additional_instruction="Use the provided profile/biography pages to confirm the person is primarily a music artist.")
    # 2) Evidence of music career activity in 2024
    leaf_m2 = evaluator.add_leaf("first_music_activity_2024",
                                 "Evidence of music career activity in 2024 (release and/or public performance).",
                                 parent=mc, critical=True)
    claim = f"In 2024, {entry.name if entry else 'the artist'} had active music career activity (e.g., releases or performances)."
    await evaluator.verify(claim=claim, node=leaf_m2,
                           sources=(entry.music_urls if entry else []),
                           additional_instruction="Look for 2024-dated releases, chart updates, tours, or notable performances.")

    # ----- Fashion criteria -----
    fc = evaluator.add_parallel(
        id="first_fashion_criteria",
        desc="First artist meets 2024 fashion campaign constraints.",
        parent=node,
        critical=True
    )
    # 1) At least one 2024 campaign
    leaf_f1 = evaluator.add_leaf("first_at_least_one_campaign_2024",
                                 "At least one fashion brand campaign appearance in 2024 is identified.",
                                 parent=fc, critical=True)
    first_camp = _first_campaign_2024(entry)
    brandsnippet = (first_camp.brand if first_camp and first_camp.brand else "a fashion brand")
    claim = f"In 2024, {entry.name if entry else 'the artist'} appeared in at least one fashion brand campaign such as {brandsnippet}."
    await evaluator.verify(claim=claim, node=leaf_f1,
                           sources=_collect_campaign_urls_2024(entry),
                           additional_instruction="Verify that at least one of the provided campaign URLs clearly shows the artist in a 2024 campaign.")
    # 2) Campaign brand recognizable
    leaf_f2 = evaluator.add_leaf("first_campaign_brand_recognizable_via_sources",
                                 "The campaign brand is recognizable/established as evidenced by at least one provided reference URL that identifies it as a real fashion brand (e.g., official brand site or reputable publication/source describing the brand/campaign).",
                                 parent=fc, critical=True)
    claim = "The fashion campaign brand is a real, recognized fashion brand according to the provided sources."
    await evaluator.verify(claim=claim, node=leaf_f2,
                           sources=_join_sources(_collect_campaign_urls_2024(entry),
                                                 entry.campaign_brand_info_urls if entry else []),
                           additional_instruction="The evidence can be an official brand site or a reputable fashion publication describing the brand/campaign.")

    # ----- Age requirement -----
    leaf_age = evaluator.add_leaf("first_age_requirement",
                                  "Artist was between 20–30 years old at the time of the March 2024 announcement.",
                                  parent=node, critical=True)
    claim = f"Given the date of birth '{entry.dob if entry else ''}', {entry.name if entry else 'the artist'} was between 20 and 30 years old (inclusive) in March 2024, the announcement month."
    await evaluator.verify(claim=claim, node=leaf_age,
                           sources=_join_sources((entry.age_or_dob_urls if entry else []),
                                                 (entry.ambassador_announcement_urls if entry else [])),
                           additional_instruction="Compute age from DOB at March 2024. If the exact day is near birthday, accept inclusive range if plausible.")


async def verify_second_artist(evaluator: Evaluator, parent, entry: Optional[ArtistEntry]) -> None:
    node = evaluator.add_parallel(
        id="second_artist",
        desc="Second qualifying artist (September 2024; first celebrity ambassador/global partner or notable first/new designation; age 20–30).",
        parent=parent,
        critical=False
    )

    # ----- Output completeness and citations -----
    oc = evaluator.add_parallel(
        id="second_output_completeness_and_citations",
        desc="Second artist output includes all required fields and URLs sufficient to verify each required claim.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(bool(entry and entry.name and entry.name.strip()),
                              "second_field_artist_name_present",
                              "Artist name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.beauty_brand and entry.beauty_brand.strip()),
                              "second_field_beauty_brand_present",
                              "Beauty brand name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.announcement_month_year and entry.announcement_month_year.strip()),
                              "second_field_announcement_date_present",
                              "Announcement date (month and year) is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(_first_campaign_2024(entry) is not None),
                              "second_field_fashion_campaign_present",
                              "At least one fashion campaign (name/identifier) is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and ((entry.hit_releases_statement and entry.hit_releases_statement.strip()) or entry.hit_releases_urls)),
                              "second_field_music_activity_evidence_present",
                              "A 2024 hit-release/music-activity evidence statement is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.ambassador_announcement_urls),
                              "second_url_supports_ambassador_announcement",
                              "Provides URL(s) supporting the ambassador announcement timing and official designation.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.major_publication_urls),
                              "second_url_supports_major_publication_coverage",
                              "Provides URL(s) showing the announcement was covered by at least one major fashion or beauty publication.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.first_or_new_designation_urls),
                              "second_url_supports_first_or_new_designation",
                              "Provides URL(s) supporting the 'first celebrity ambassador/global partner' (or notable first/new designation) claim.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.brand_is_luxury_urls),
                              "second_url_supports_brand_is_luxury",
                              "Provides URL(s) supporting that the beauty brand is a luxury brand.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.brand_offers_makeup_urls),
                              "second_url_supports_brand_offers_makeup",
                              "Provides URL(s) supporting that the beauty brand offers makeup products.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and (_collect_campaign_urls_2024(entry) or entry.campaign_brand_info_urls)),
                              "second_url_supports_fashion_campaign",
                              "Provides URL(s) supporting the 2024 fashion campaign participation (and that it is luxury/premium if that is not already established by the same URL).",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.hit_releases_urls),
                              "second_url_supports_hit_releases_2024",
                              "Provides URL(s) supporting the 2024 hit songs/albums claim.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.age_or_dob_urls),
                              "second_url_supports_age_or_dob",
                              "Provides URL(s) sufficient to verify the artist's age at the time of announcement (e.g., DOB source).",
                              parent=oc, critical=True)

    # ----- Beauty ambassador criteria -----
    bac = evaluator.add_parallel(
        id="second_beauty_ambassador_criteria",
        desc="Second artist meets beauty-ambassador constraints.",
        parent=node,
        critical=True
    )
    # 1) September 2024 announcement
    leaf_1 = evaluator.add_leaf("second_announcement_month",
                                "Announcement occurred in September 2024.",
                                parent=bac, critical=True)
    claim = f"The announcement that {entry.name if entry else 'the artist'} became a beauty brand ambassador/global partner for {entry.beauty_brand if entry else 'the brand'} occurred in September 2024."
    await evaluator.verify(claim=claim, node=leaf_1,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Verify month/year. The source should clearly indicate September 2024.")
    # 2) Brand is luxury
    leaf_2 = evaluator.add_leaf("second_brand_is_luxury",
                                "Beauty brand is a luxury brand.",
                                parent=bac, critical=True)
    claim = f"The beauty brand {entry.beauty_brand if entry else 'the brand'} is a luxury brand."
    await evaluator.verify(claim=claim, node=leaf_2,
                           sources=(entry.brand_is_luxury_urls if entry else []),
                           additional_instruction="Accept reputable sources that classify the brand as luxury.")
    # 3) Brand offers makeup
    leaf_3 = evaluator.add_leaf("second_brand_offers_makeup",
                                "Beauty brand offers makeup products as part of its portfolio.",
                                parent=bac, critical=True)
    claim = f"The beauty brand {entry.beauty_brand if entry else 'the brand'} offers makeup products."
    await evaluator.verify(claim=claim, node=leaf_3,
                           sources=(entry.brand_offers_makeup_urls if entry else []),
                           additional_instruction="Confirm the brand’s beauty line includes makeup items.")
    # 4) Official designation ambassador/global partner
    leaf_4 = evaluator.add_leaf("second_official_designation",
                                "Partnership is officially designated as ambassador/global partner (or equivalent official title).",
                                parent=bac, critical=True)
    claim = f"The partnership designates {entry.name if entry else 'the artist'} as an ambassador or global partner of {entry.beauty_brand if entry else 'the brand'}."
    await evaluator.verify(claim=claim, node=leaf_4,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Look for 'ambassador' or 'global partner' or equivalent in the sources.")
    # 5) First/new notable designation
    leaf_5 = evaluator.add_leaf("second_first_or_notable_new_designation",
                                "Partnership is explicitly stated as the brand's first celebrity ambassador/global partner or otherwise a notable first/new partnership designation (as constrained).",
                                parent=bac, critical=True)
    claim = f"The partnership is explicitly the brand's first celebrity ambassador/global partner, or otherwise a notable first/new designation."
    await evaluator.verify(claim=claim, node=leaf_5,
                           sources=(entry.first_or_new_designation_urls if entry else []),
                           additional_instruction="The page should clearly say 'first celebrity ambassador', 'first-ever', or note a new/first designation.")
    # 6) Major publication coverage
    leaf_6 = evaluator.add_leaf("second_major_publication_coverage",
                                "Announcement covered by at least one major fashion or beauty publication.",
                                parent=bac, critical=True)
    claim = f"The ambassador/global partner announcement for {entry.name if entry else 'the artist'} with {entry.beauty_brand if entry else 'the brand'} was covered by a major fashion/beauty publication."
    await evaluator.verify(claim=claim, node=leaf_6,
                           sources=(entry.major_publication_urls if entry else []),
                           additional_instruction="Accept publications like Vogue, Elle, Harper’s Bazaar, W, Allure, etc.")

    # ----- Music criteria -----
    mc = evaluator.add_parallel(
        id="second_music_criteria",
        desc="Second artist meets music-artist and 2024 hit-release constraint.",
        parent=node,
        critical=True
    )
    # 1) Primarily music artist
    leaf_m1 = evaluator.add_leaf("second_primarily_music_artist",
                                 "Artist is primarily known as a music artist.",
                                 parent=mc, critical=True)
    claim = f"{entry.name if entry else 'The person'} is primarily known as a music artist."
    await evaluator.verify(claim=claim, node=leaf_m1,
                           sources=(entry.profile_urls if entry else []),
                           additional_instruction="Use profile/bio pages to confirm music-artist identity.")
    # 2) Hit releases in 2024
    leaf_m2 = evaluator.add_leaf("second_hit_releases_2024",
                                 "Evidence the artist released hit songs or albums in 2024.",
                                 parent=mc, critical=True)
    claim = f"In 2024, {entry.name if entry else 'the artist'} released hit songs or a hit album."
    await evaluator.verify(claim=claim, node=leaf_m2,
                           sources=(entry.hit_releases_urls if entry else []),
                           additional_instruction="Look for reputable coverage, chart mentions, or official sources indicating hit status.")

    # ----- Fashion criteria -----
    fc = evaluator.add_parallel(
        id="second_fashion_criteria",
        desc="Second artist meets 2024 fashion campaign constraints (luxury/premium).",
        parent=node,
        critical=True
    )
    # 1) At least one campaign in 2024
    leaf_f1 = evaluator.add_leaf("second_at_least_one_campaign_2024",
                                 "At least one fashion brand campaign appearance in 2024 is identified.",
                                 parent=fc, critical=True)
    first_camp = _first_campaign_2024(entry)
    brandsnippet = (first_camp.brand if first_camp and first_camp.brand else "a fashion brand")
    claim = f"In 2024, {entry.name if entry else 'the artist'} appeared in at least one fashion brand campaign such as {brandsnippet}."
    await evaluator.verify(claim=claim, node=leaf_f1,
                           sources=_collect_campaign_urls_2024(entry),
                           additional_instruction="Verify a 2024-dated campaign appearance.")
    # 2) Luxury or premium brand
    leaf_f2 = evaluator.add_leaf("second_luxury_or_premium_brand",
                                 "Fashion campaign is for a luxury or premium brand (as required for the second artist).",
                                 parent=fc, critical=True)
    claim = "At least one of the 2024 fashion campaigns was for a luxury or premium fashion brand."
    await evaluator.verify(claim=claim, node=leaf_f2,
                           sources=_join_sources(_collect_campaign_urls_2024(entry),
                                                 entry.luxury_or_premium_campaign_urls if entry else []),
                           additional_instruction="Confirm that the brand is positioned as luxury/premium via reputable sources or brand positioning.")
    # 3) Campaign brand recognizable
    leaf_f3 = evaluator.add_leaf("second_campaign_brand_recognizable_via_sources",
                                 "The campaign brand is recognizable/established as evidenced by at least one provided reference URL that identifies it as a real fashion brand (e.g., official brand site or reputable publication/source describing the brand/campaign).",
                                 parent=fc, critical=True)
    claim = "The fashion campaign brand is a real, recognized fashion brand according to the provided sources."
    await evaluator.verify(claim=claim, node=leaf_f3,
                           sources=_join_sources(_collect_campaign_urls_2024(entry),
                                                 entry.campaign_brand_info_urls if entry else []),
                           additional_instruction="Accept brand official sites or reputable fashion publications describing the brand/campaign.")

    # ----- Age requirement -----
    leaf_age = evaluator.add_leaf("second_age_requirement",
                                  "Artist was between 20–30 years old at the time of the September 2024 announcement.",
                                  parent=node, critical=True)
    claim = f"Given the date of birth '{entry.dob if entry else ''}', {entry.name if entry else 'the artist'} was between 20 and 30 years old (inclusive) in September 2024, the announcement month."
    await evaluator.verify(claim=claim, node=leaf_age,
                           sources=_join_sources((entry.age_or_dob_urls if entry else []),
                                                 (entry.ambassador_announcement_urls if entry else [])),
                           additional_instruction="Compute age from DOB at September 2024; allow inclusive bounds where birthday proximity makes it plausible.")


async def verify_third_artist(evaluator: Evaluator, parent, entry: Optional[ArtistEntry]) -> None:
    node = evaluator.add_parallel(
        id="third_artist",
        desc="Third qualifying artist (October 2024; luxury fashion house beauty division; ≥2 fashion brands incl. ≥1 Fall/Winter 2024; age 25–35).",
        parent=parent,
        critical=False
    )

    # ----- Output completeness and citations -----
    oc = evaluator.add_parallel(
        id="third_output_completeness_and_citations",
        desc="Third artist output includes all required fields and URLs sufficient to verify each required claim.",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(bool(entry and entry.name and entry.name.strip()),
                              "third_field_artist_name_present",
                              "Artist name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.beauty_brand and entry.beauty_brand.strip()),
                              "third_field_beauty_brand_present",
                              "Beauty brand name is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.announcement_month_year and entry.announcement_month_year.strip()),
                              "third_field_announcement_date_present",
                              "Announcement date (month and year) is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(_has_at_least_n_campaign_brands_2024(entry, 2)),
                              "third_field_fashion_campaigns_present",
                              "Fashion campaign details are provided sufficient to evaluate ≥2 brands and ≥1 FW24 campaign.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and ((entry.major_album_or_chart_statement and entry.major_album_or_chart_statement.strip()) or entry.major_album_or_chart_urls)),
                              "third_field_music_activity_evidence_present",
                              "A 2024 album/success evidence statement is provided.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.ambassador_announcement_urls),
                              "third_url_supports_ambassador_announcement",
                              "Provides URL(s) supporting the ambassador announcement timing and official designation.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.major_publication_urls),
                              "third_url_supports_major_publication_coverage",
                              "Provides URL(s) showing the announcement was covered by at least one major fashion or beauty publication.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.beauty_division_urls),
                              "third_url_supports_beauty_division_claim",
                              "Provides URL(s) supporting that the beauty brand is the beauty division of a luxury fashion house.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.brand_offers_makeup_urls),
                              "third_url_supports_brand_offers_makeup",
                              "Provides URL(s) supporting that the beauty brand offers makeup products.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and _has_at_least_n_campaign_brands_2024(entry, 2) and _collect_campaign_urls_2024(entry)),
                              "third_url_supports_two_distinct_brand_campaigns",
                              "Provides URL(s) supporting campaign participation for at least two different fashion brands in 2024.",
                              parent=oc, critical=True)
    has_fw24, fw24_urls, _ = _collect_fw24_campaign_urls(entry)
    evaluator.add_custom_node(bool(has_fw24 and fw24_urls),
                              "third_url_supports_fw24_campaign",
                              "Provides URL(s) supporting that at least one identified campaign is for a Fall/Winter 2024 collection.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.major_album_or_chart_urls),
                              "third_url_supports_music_success_2024",
                              "Provides URL(s) supporting the major album release or significant chart success in 2024.",
                              parent=oc, critical=True)
    evaluator.add_custom_node(bool(entry and entry.age_or_dob_urls),
                              "third_url_supports_age_or_dob",
                              "Provides URL(s) sufficient to verify the artist's age at the time of announcement (e.g., DOB source).",
                              parent=oc, critical=True)

    # ----- Beauty ambassador criteria -----
    bac = evaluator.add_parallel(
        id="third_beauty_ambassador_criteria",
        desc="Third artist meets beauty-ambassador constraints.",
        parent=node,
        critical=True
    )
    # 1) October 2024 announcement
    leaf_1 = evaluator.add_leaf("third_announcement_month",
                                "Announcement occurred in October 2024.",
                                parent=bac, critical=True)
    claim = f"The announcement that {entry.name if entry else 'the artist'} became a beauty brand ambassador for {entry.beauty_brand if entry else 'the brand'} occurred in October 2024."
    await evaluator.verify(claim=claim, node=leaf_1,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Verify month/year as October 2024.")
    # 2) Brand is beauty division of luxury fashion house
    leaf_2 = evaluator.add_leaf("third_brand_is_luxury_fashion_house_beauty_division",
                                "Beauty brand is the beauty division of a luxury fashion house.",
                                parent=bac, critical=True)
    claim = f"The beauty brand {entry.beauty_brand if entry else 'the brand'} is the beauty division of a luxury fashion house."
    await evaluator.verify(claim=claim, node=leaf_2,
                           sources=(entry.beauty_division_urls if entry else []),
                           additional_instruction="The source should explicitly link the beauty brand to a luxury fashion house as its beauty division.")
    # 3) Brand offers makeup
    leaf_3 = evaluator.add_leaf("third_brand_offers_makeup",
                                "Beauty brand offers makeup products as part of its portfolio.",
                                parent=bac, critical=True)
    claim = f"The brand {entry.beauty_brand if entry else 'the brand'} offers makeup products."
    await evaluator.verify(claim=claim, node=leaf_3,
                           sources=(entry.brand_offers_makeup_urls if entry else []),
                           additional_instruction="Confirm makeup items are part of the beauty line.")
    # 4) Official designation
    leaf_4 = evaluator.add_leaf("third_official_designation",
                                "Partnership is officially designated as ambassador/beauty ambassador (or equivalent official title).",
                                parent=bac, critical=True)
    claim = f"The partnership designates {entry.name if entry else 'the artist'} as an ambassador or beauty ambassador."
    await evaluator.verify(claim=claim, node=leaf_4,
                           sources=(entry.ambassador_announcement_urls if entry else []),
                           additional_instruction="Look for 'ambassador' or an equivalent official designation.")
    # 5) Major publication coverage
    leaf_5 = evaluator.add_leaf("third_major_publication_coverage",
                                "Announcement covered by at least one major fashion or beauty publication.",
                                parent=bac, critical=True)
    claim = f"The ambassador announcement for {entry.name if entry else 'the artist'} was covered by a major fashion/beauty publication."
    await evaluator.verify(claim=claim, node=leaf_5,
                           sources=(entry.major_publication_urls if entry else []),
                           additional_instruction="Accept coverage from recognized major publications (Vogue, Elle, Harper’s Bazaar, etc.).")

    # ----- Music criteria -----
    mc = evaluator.add_parallel(
        id="third_music_criteria",
        desc="Third artist meets music-artist and 2024 success constraint.",
        parent=node,
        critical=True
    )
    # 1) Primarily music artist
    leaf_m1 = evaluator.add_leaf("third_primarily_music_artist",
                                 "Artist is primarily known as a music artist.",
                                 parent=mc, critical=True)
    claim = f"{entry.name if entry else 'The person'} is primarily known as a music artist."
    await evaluator.verify(claim=claim, node=leaf_m1,
                           sources=(entry.profile_urls if entry else []),
                           additional_instruction="Use biography/official sources showing the primary identity as a music artist.")
    # 2) Major album or chart success in 2024
    leaf_m2 = evaluator.add_leaf("third_major_album_or_chart_success_2024",
                                 "Evidence of a major album release or significant chart success in 2024.",
                                 parent=mc, critical=True)
    claim = f"In 2024, {entry.name if entry else 'the artist'} released a major album or achieved significant chart success."
    await evaluator.verify(claim=claim, node=leaf_m2,
                           sources=(entry.major_album_or_chart_urls if entry else []),
                           additional_instruction="Look for official/industry sources (chart listings, labels, reputable press) indicating 2024 success.")

    # ----- Fashion criteria -----
    fc = evaluator.add_parallel(
        id="third_fashion_criteria",
        desc="Third artist meets multi-campaign constraints in 2024 (≥2 brands; includes FW24).",
        parent=node,
        critical=True
    )
    # 1) At least two distinct brands in 2024
    leaf_f1 = evaluator.add_leaf("third_two_distinct_brands_2024",
                                 "Appeared in campaigns for at least two different fashion brands in 2024.",
                                 parent=fc, critical=True)
    brands = _distinct_campaign_brands_2024(entry)
    brand_list_text = ", ".join(brands[:3]) if brands else "multiple brands"
    claim = f"In 2024, {entry.name if entry else 'the artist'} appeared in campaigns for at least two different fashion brands (e.g., {brand_list_text})."
    await evaluator.verify(claim=claim, node=leaf_f1,
                           sources=_collect_campaign_urls_2024(entry),
                           additional_instruction="Confirm that at least two distinct brand campaigns in 2024 feature the artist.")
    # 2) At least one FW24
    leaf_f2 = evaluator.add_leaf("third_at_least_one_fw24_campaign",
                                 "At least one identified campaign is for a Fall/Winter 2024 collection.",
                                 parent=fc, critical=True)
    has_fw24, fw24_urls, fw_label = _collect_fw24_campaign_urls(entry)
    claim = f"At least one campaign for {entry.name if entry else 'the artist'} is for a Fall/Winter 2024 collection."
    await evaluator.verify(claim=claim, node=leaf_f2,
                           sources=fw24_urls,
                           additional_instruction="Look for 'FW24', 'Fall/Winter 2024', or equivalent season labeling on the campaign page.")
    # 3) Campaign brands recognizable
    leaf_f3 = evaluator.add_leaf("third_campaign_brands_recognizable_via_sources",
                                 "The campaign brands are recognizable/established as evidenced by provided reference URL(s) that identify each brand as a real fashion brand (e.g., official brand sites or reputable publication/source describing the brand/campaign).",
                                 parent=fc, critical=True)
    claim = "The campaign brands are real, recognized fashion brands according to the provided sources."
    await evaluator.verify(claim=claim, node=leaf_f3,
                           sources=_join_sources(_collect_campaign_urls_2024(entry),
                                                 entry.campaign_brand_info_urls if entry else []),
                           additional_instruction="Accept brand official sites or reputable fashion publications describing the brands/campaigns.")

    # ----- Age requirement -----
    leaf_age = evaluator.add_leaf("third_age_requirement",
                                  "Artist was between 25–35 years old at the time of the October 2024 announcement.",
                                  parent=node, critical=True)
    claim = f"Given the date of birth '{entry.dob if entry else ''}', {entry.name if entry else 'the artist'} was between 25 and 35 years old (inclusive) in October 2024, the announcement month."
    await evaluator.verify(claim=claim, node=leaf_age,
                           sources=_join_sources((entry.age_or_dob_urls if entry else []),
                                                 (entry.ambassador_announcement_urls if entry else [])),
                           additional_instruction="Compute age from DOB at October 2024; allow inclusive bounds near birthdays.")


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
    """
    Evaluate an answer for the 'luxury_beauty_ambassadors_2024' task.
    """
    # Initialize evaluator (root is non-critical by design to avoid conflicting critical hierarchy)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Artists evaluated in parallel
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

    # Extract structured information
    artists = await evaluator.extract(
        prompt=prompt_extract_artists(),
        template_class=ArtistsExtraction,
        extraction_name="artists_extraction"
    )

    # Build artist subtrees and run verifications
    await verify_first_artist(evaluator, root, artists.first)
    await verify_second_artist(evaluator, root, artists.second)
    await verify_third_artist(evaluator, root, artists.third)

    # Return summary
    return evaluator.get_summary()