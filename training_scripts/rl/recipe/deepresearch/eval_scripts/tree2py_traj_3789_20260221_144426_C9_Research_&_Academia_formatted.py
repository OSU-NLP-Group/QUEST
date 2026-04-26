import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "harvard_cfa_interstellar_researcher"
TASK_DESCRIPTION = (
    "Identify a researcher who meets ALL of the following criteria:\n"
    "- Is affiliated with Harvard University\n"
    "- Holds a professorship-level position (not assistant professor level)\n"
    "- Works in the field of astronomy or astrophysics\n"
    "- Served as a department chair, with that service ending in 2020\n"
    "- Currently directs an institute or research center\n"
    "- Is associated with the Harvard-Smithsonian Center for Astrophysics\n"
    "- Has published research articles about the interstellar object 3I/ATLAS\n"
    "- Published these articles in 2025 or 2026\n\n"
    "Once you have identified this researcher, provide the following information with a supporting URL for each:\n"
    "1. The researcher's full name\n"
    "2. Their specific endowed chair or professorship title\n"
    "3. The full name of the institute or center they currently direct\n"
    "4. The year they began directing this institute (format: YYYY)\n"
    "5. The complete time period they served as department chair (format: YYYY-YYYY)\n"
    "6. The public platform where they publish articles about interstellar objects (e.g., blog, website name)\n"
    "7. The specific month and year when 3I/ATLAS's discovery was announced (format: Month YYYY)\n"
    "8. The approximate number of articles they published about 3I/ATLAS (provide a specific count or reasonable range)\n"
    "9. The date when 3I/ATLAS reached perihelion (closest approach to the Sun) in 2025 (format: Month DD, YYYY)\n"
    "10. The date when 3I/ATLAS made its closest approach to Earth in 2025 (format: Month DD, YYYY)\n"
    "11. The designation number confirming 3I/ATLAS as the nth interstellar object discovered (provide the number)\n"
    "12. The names of the two interstellar objects discovered before 3I/ATLAS\n"
    "Each piece of information must include at least one supporting reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ResearcherExtraction(BaseModel):
    # Core identity and affiliations
    full_name: Optional[str] = None
    output_1_sources: List[str] = Field(default_factory=list)

    harvard_affiliation_urls: List[str] = Field(default_factory=list)

    professorship_title: Optional[str] = None
    professorship_urls: List[str] = Field(default_factory=list)

    field: Optional[str] = None
    field_urls: List[str] = Field(default_factory=list)

    department_chair_period: Optional[str] = None  # e.g., "2011-2020"
    department_chair_urls: List[str] = Field(default_factory=list)

    director_institute_name: Optional[str] = None
    director_start_year: Optional[str] = None  # YYYY
    director_role_urls: List[str] = Field(default_factory=list)

    cfa_association_urls: List[str] = Field(default_factory=list)

    interstellar_publications_platform: Optional[str] = None
    interstellar_publication_article_urls: List[str] = Field(default_factory=list)

    # Outputs-specific sources (ensure each output has ≥1 URL)
    output_2_sources: List[str] = Field(default_factory=list)
    output_3_sources: List[str] = Field(default_factory=list)
    output_4_sources: List[str] = Field(default_factory=list)
    output_5_sources: List[str] = Field(default_factory=list)
    output_6_sources: List[str] = Field(default_factory=list)

    # 3I/ATLAS facts for outputs 7–12
    atlas_discovery_announcement_month_year: Optional[str] = None  # e.g., "July 2025"
    output_7_sources: List[str] = Field(default_factory=list)

    atlas_article_count_or_range: Optional[str] = None  # e.g., "5–7" or "6"
    output_8_sources: List[str] = Field(default_factory=list)

    atlas_perihelion_date_2025: Optional[str] = None  # e.g., "September 15, 2025"
    output_9_sources: List[str] = Field(default_factory=list)

    atlas_earth_approach_date_2025: Optional[str] = None  # e.g., "October 20, 2025"
    output_10_sources: List[str] = Field(default_factory=list)

    atlas_nth_designation_number: Optional[str] = None  # e.g., "3"
    output_11_sources: List[str] = Field(default_factory=list)

    previous_interstellar_objects: List[str] = Field(default_factory=list)  # e.g., ["1I/‘Oumuamua", "2I/Borisov"]
    output_12_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_researcher_info() -> str:
    return """
    Extract all information listed below strictly from the provided answer text. If any field is not explicitly present, set it to null (or an empty array for lists). For any field that requires supporting URLs, extract all URLs that the answer explicitly provides for that field. Do not invent URLs.

    Required fields to extract:

    Core researcher identification and qualification:
    - full_name (string)
    - harvard_affiliation_urls (array of URLs supporting Harvard affiliation)
    - professorship_title (string; professorship-level, not assistant)
    - professorship_urls (array of URLs supporting the professorship title)
    - field (string; should indicate astronomy or astrophysics)
    - field_urls (array of URLs supporting that field)
    - department_chair_period (string, format like "YYYY-YYYY")
    - department_chair_urls (array of URLs supporting the chair history; should show service ended in 2020)
    - director_institute_name (string; full name of the institute or center they currently direct)
    - director_start_year (string, 4 digits YYYY)
    - director_role_urls (array of URLs supporting they currently direct that institute and the start year)
    - cfa_association_urls (array of URLs supporting association with the Harvard-Smithsonian Center for Astrophysics)
    - interstellar_publications_platform (string; public platform name where they publish interstellar object articles)
    - interstellar_publication_article_urls (array of URLs to articles about 3I/ATLAS; preferably from years 2025 or 2026)

    Outputs-specific fields (each must have ≥1 supporting URL extracted from the answer):
    - output_1_sources (array of URLs supporting the full name)
    - output_2_sources (array of URLs supporting the endowed chair/professorship title)
    - output_3_sources (array of URLs supporting the institute/center name they direct)
    - output_4_sources (array of URLs supporting the directorship start year)
    - output_5_sources (array of URLs supporting the department-chair time period)
    - output_6_sources (array of URLs supporting the public platform name)

    3I/ATLAS facts for outputs 7–12 (each with its own sources):
    - atlas_discovery_announcement_month_year (string, format "Month YYYY")
    - output_7_sources (array of URLs supporting the discovery announcement month/year)
    - atlas_article_count_or_range (string representing a specific count or a reasonable range, e.g., "5–7")
    - output_8_sources (array of URLs supporting the count/range)
    - atlas_perihelion_date_2025 (string, format "Month DD, YYYY")
    - output_9_sources (array of URLs supporting the perihelion date)
    - atlas_earth_approach_date_2025 (string, format "Month DD, YYYY")
    - output_10_sources (array of URLs supporting the closest Earth approach date)
    - atlas_nth_designation_number (string; e.g., "3")
    - output_11_sources (array of URLs supporting the designation number)
    - previous_interstellar_objects (array of exactly two strings, the two interstellar objects discovered before 3I/ATLAS)
    - output_12_sources (array of URLs supporting the names of the two previous interstellar objects)

    IMPORTANT RULES:
    - Extract only what is explicitly present in the answer; do not infer or invent.
    - For all URL arrays, include only valid URLs that appear in the answer, in any reasonable format (plain or markdown).
    - If a URL is missing a protocol, prepend "http://".
    - If the answer provides more articles than needed, include them all in the 'interstellar_publication_article_urls' list.

    Return the result as a JSON object that matches exactly the schema of the ResearcherExtraction model.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _is_nonempty_str(s: Optional[str]) -> bool:
    return bool(s) and str(s).strip() != ""


def _ends_in_2020(period: Optional[str]) -> bool:
    if not _is_nonempty_str(period):
        return False
    import re
    years = re.findall(r"(\d{4})", period)
    if years:
        return years[-1] == "2020"
    return "2020" in period


def _is_four_digit_year(s: Optional[str]) -> bool:
    if not _is_nonempty_str(s):
        return False
    import re
    return bool(re.fullmatch(r"\d{4}", s.strip()))


def _designation_is_3(s: Optional[str]) -> bool:
    if not _is_nonempty_str(s):
        return False
    import re
    digits = re.findall(r"\d+", s)
    return any(d == "3" for d in digits)


def _month_year_is_july_2025(s: Optional[str]) -> bool:
    if not _is_nonempty_str(s):
        return False
    return s.strip().lower() == "july 2025"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_researcher_identification_nodes(evaluator: Evaluator, parent_node, info: ResearcherExtraction) -> None:
    """
    Build the 'researcher_identification' parallel critical node and its children.
    Each child is a sequential critical node with:
      - existence/custom checks
      - URL-supported verification leaf
    """
    ident_node = evaluator.add_parallel(
        id="researcher_identification",
        desc="Identify one researcher who satisfies all stated researcher-identification criteria (with supporting URLs for these claims).",
        parent=parent_node,
        critical=True
    )

    # 1) Harvard affiliation
    harvard_node = evaluator.add_sequential(
        id="harvard_affiliation",
        desc="Researcher is affiliated with Harvard University (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(info.harvard_affiliation_urls),
        id="harvard_affiliation_sources_provided",
        desc="At least one URL supporting Harvard affiliation is provided.",
        parent=harvard_node,
        critical=True
    )
    harvard_verify = evaluator.add_leaf(
        id="harvard_affiliation_supported",
        desc="Harvard affiliation is supported by cited URLs.",
        parent=harvard_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} is affiliated with Harvard University.",
        node=harvard_verify,
        sources=info.harvard_affiliation_urls,
        additional_instruction="Verify the page explicitly indicates Harvard affiliation (faculty page, department profile, etc.). Allow reasonable name variations."
    )

    # 2) Professorship rank (not assistant)
    prof_node = evaluator.add_sequential(
        id="professorship_rank",
        desc="Researcher holds a professorship-level position (not assistant professor level) (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.professorship_title) and _has_sources(info.professorship_urls),
        id="professorship_title_present_with_sources",
        desc="Professorship title present and ≥1 supporting URL provided.",
        parent=prof_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(_is_nonempty_str(info.professorship_title) and ("assistant" not in info.professorship_title.lower())),
        id="professorship_not_assistant",
        desc="Title is not 'Assistant Professor' level.",
        parent=prof_node,
        critical=True
    )
    prof_verify = evaluator.add_leaf(
        id="professorship_supported_by_urls",
        desc="Professorship-level position is supported by cited URLs.",
        parent=prof_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} holds the professorship-level title '{info.professorship_title or ''}'.",
        node=prof_verify,
        sources=info.professorship_urls,
        additional_instruction="Confirm the page shows a professorship-level title (e.g., Professor, Associate Professor, endowed chair). It must not be an Assistant Professor."
    )

    # 3) Astronomy or astrophysics field
    field_node = evaluator.add_sequential(
        id="astronomy_field",
        desc="Researcher works in astronomy or astrophysics (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(info.field_urls),
        id="field_sources_provided",
        desc="At least one URL supporting astronomy/astrophysics field is provided.",
        parent=field_node,
        critical=True
    )
    field_verify = evaluator.add_leaf(
        id="field_supported_by_urls",
        desc="Astronomy/astrophysics field is supported by cited URLs.",
        parent=field_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} works in astronomy or astrophysics.",
        node=field_verify,
        sources=info.field_urls,
        additional_instruction="Confirm the page indicates the researcher's field is astronomy or astrophysics (department affiliation, research interests, etc.)."
    )

    # 4) Department chair history ending in 2020
    chair_node = evaluator.add_sequential(
        id="department_chair_history",
        desc="Researcher served as a department chair, with that service ending in 2020 (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.department_chair_period) and _has_sources(info.department_chair_urls),
        id="chair_period_present_with_sources",
        desc="Chair service period is present and ≥1 supporting URL provided.",
        parent=chair_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_ends_in_2020(info.department_chair_period),
        id="chair_period_ends_2020",
        desc="Chair service period ends in 2020.",
        parent=chair_node,
        critical=True
    )
    chair_verify = evaluator.add_leaf(
        id="chair_history_supported_by_urls",
        desc="Chair service period ending in 2020 is supported by cited URLs.",
        parent=chair_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} served as department chair during {info.department_chair_period or ''}, with service ending in 2020.",
        node=chair_verify,
        sources=info.department_chair_urls,
        additional_instruction="Confirm the page states the researcher served as department chair and that the end of service was in 2020."
    )

    # 5) Current director role (institute/center)
    director_node = evaluator.add_sequential(
        id="current_director_role",
        desc="Researcher currently directs an institute or research center (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.director_institute_name) and _is_four_digit_year(info.director_start_year) and _has_sources(info.director_role_urls),
        id="director_role_present_with_sources",
        desc="Director institute name, start year (YYYY), and ≥1 supporting URL provided.",
        parent=director_node,
        critical=True
    )
    director_verify = evaluator.add_leaf(
        id="director_role_supported_by_urls",
        desc="Current director role supported by cited URLs.",
        parent=director_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} currently directs '{info.director_institute_name or ''}' and began in {info.director_start_year or ''}.",
        node=director_verify,
        sources=info.director_role_urls,
        additional_instruction="Confirm that the page states they currently direct the named institute/center and gives (or implies) the start year."
    )

    # 6) CfA association
    cfa_node = evaluator.add_sequential(
        id="cfa_association",
        desc="Researcher is associated with the Harvard-Smithsonian Center for Astrophysics (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(info.cfa_association_urls),
        id="cfa_sources_provided",
        desc="At least one URL supporting CfA association is provided.",
        parent=cfa_node,
        critical=True
    )
    cfa_verify = evaluator.add_leaf(
        id="cfa_association_supported_by_urls",
        desc="CfA association supported by cited URLs.",
        parent=cfa_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} is associated with the Harvard-Smithsonian Center for Astrophysics.",
        node=cfa_verify,
        sources=info.cfa_association_urls,
        additional_instruction="Confirm the page indicates association with the CfA (Harvard-Smithsonian Center for Astrophysics)."
    )

    # 7) Published research articles about 3I/ATLAS
    atlas_pub_node = evaluator.add_sequential(
        id="interstellar_research_3i_atlas",
        desc="Researcher has published research articles about the interstellar object 3I/ATLAS (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(info.interstellar_publication_article_urls),
        id="atlas_article_urls_provided",
        desc="At least one URL to 3I/ATLAS articles is provided.",
        parent=atlas_pub_node,
        critical=True
    )
    atlas_pub_verify = evaluator.add_leaf(
        id="atlas_articles_supported_by_urls",
        desc="3I/ATLAS publications supported by cited URLs.",
        parent=atlas_pub_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{info.full_name or 'The researcher'} has published research articles about the interstellar object 3I/ATLAS.",
        node=atlas_pub_verify,
        sources=info.interstellar_publication_article_urls,
        additional_instruction="Confirm that the linked article(s) explicitly discuss 3I/ATLAS."
    )

    # 8) Publication years 2025 or 2026
    yrs_node = evaluator.add_sequential(
        id="publication_years_2025_2026",
        desc="The 3I/ATLAS research articles are published in 2025 or 2026 (supported by ≥1 URL).",
        parent=ident_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(info.interstellar_publication_article_urls),
        id="atlas_article_urls_exist_for_year_check",
        desc="At least one article URL provided for year verification.",
        parent=yrs_node,
        critical=True
    )
    yrs_verify = evaluator.add_leaf(
        id="atlas_articles_year_2025_or_2026_supported_by_urls",
        desc="3I/ATLAS article publication year is 2025 or 2026, supported by cited URLs.",
        parent=yrs_node,
        critical=True
    )
    await evaluator.verify(
        claim="The linked article(s) show publication dates in 2025 or 2026.",
        node=yrs_verify,
        sources=info.interstellar_publication_article_urls,
        additional_instruction="Check the article publication dates on the linked pages; at least one should be in 2025 or 2026."
    )


async def build_outputs_verification_nodes(evaluator: Evaluator, parent_node, info: ResearcherExtraction) -> None:
    """
    Build the 'required_outputs_with_evidence' parallel critical node and its 12 required outputs.
    Each output is a sequential critical node with:
      - existence/custom checks
      - URL-supported verification leaf
    """
    outputs_node = evaluator.add_parallel(
        id="required_outputs_with_evidence",
        desc="Provide all requested output fields (1–12). Each field must include at least one supporting reference URL.",
        parent=parent_node,
        critical=True
    )

    # Output 1: Full name
    out1 = evaluator.add_sequential(
        id="output_1_full_name",
        desc="Provide the researcher's full name + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.full_name) and _has_sources(info.output_1_sources),
        id="output_1_value_and_sources_present",
        desc="Full name present and ≥1 supporting URL provided.",
        parent=out1,
        critical=True
    )
    out1_verify = evaluator.add_leaf(
        id="output_1_name_supported",
        desc="Full name is supported by cited URLs.",
        parent=out1,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher's full name is '{info.full_name or ''}'.",
        node=out1_verify,
        sources=info.output_1_sources,
        additional_instruction="Confirm the page shows the person's full name. Allow minor variations (middle initials, accents)."
    )

    # Output 2: Endowed title / professorship
    out2 = evaluator.add_sequential(
        id="output_2_endowed_title",
        desc="Provide the specific endowed chair or professorship title + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.professorship_title) and _has_sources(info.output_2_sources),
        id="output_2_value_and_sources_present",
        desc="Professorship/endowed title present and ≥1 supporting URL provided.",
        parent=out2,
        critical=True
    )
    out2_verify = evaluator.add_leaf(
        id="output_2_title_supported",
        desc="Endowed chair/professorship title is supported by cited URLs.",
        parent=out2,
        critical=True
    )
    await evaluator.verify(
        claim=f"The endowed chair/professorship title is '{info.professorship_title or ''}'.",
        node=out2_verify,
        sources=info.output_2_sources,
        additional_instruction="Confirm the page explicitly states the endowed chair or professorship title."
    )

    # Output 3: Directed institute/center name
    out3 = evaluator.add_sequential(
        id="output_3_directed_institute_name",
        desc="Provide the full name of the institute/center they currently direct + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.director_institute_name) and _has_sources(info.output_3_sources),
        id="output_3_value_and_sources_present",
        desc="Institute/center name present and ≥1 supporting URL provided.",
        parent=out3,
        critical=True
    )
    out3_verify = evaluator.add_leaf(
        id="output_3_institute_supported",
        desc="Directed institute/center name is supported by cited URLs.",
        parent=out3,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher currently directs the institute/center named '{info.director_institute_name or ''}'.",
        node=out3_verify,
        sources=info.output_3_sources,
        additional_instruction="Confirm the page states they currently direct the named institute/center."
    )

    # Output 4: Directorship start year (YYYY)
    out4 = evaluator.add_sequential(
        id="output_4_directorship_start_year",
        desc="Provide the year (YYYY) they began directing this institute/center + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_four_digit_year(info.director_start_year) and _has_sources(info.output_4_sources),
        id="output_4_value_and_sources_present",
        desc="Directorship start year (YYYY) present and ≥1 supporting URL provided.",
        parent=out4,
        critical=True
    )
    out4_verify = evaluator.add_leaf(
        id="output_4_start_year_supported",
        desc="Directorship start year is supported by cited URLs.",
        parent=out4,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher began directing the institute/center in {info.director_start_year or ''}.",
        node=out4_verify,
        sources=info.output_4_sources,
        additional_instruction="Confirm the page provides or implies the start year for the director role."
    )

    # Output 5: Chair service period (YYYY-YYYY)
    out5 = evaluator.add_sequential(
        id="output_5_chair_service_period",
        desc="Provide the complete department-chair time period (YYYY-YYYY) + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.department_chair_period) and _has_sources(info.output_5_sources),
        id="output_5_value_and_sources_present",
        desc="Chair service period present and ≥1 supporting URL provided.",
        parent=out5,
        critical=True
    )
    out5_verify = evaluator.add_leaf(
        id="output_5_period_supported",
        desc="Chair service period is supported by cited URLs.",
        parent=out5,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher served as department chair during {info.department_chair_period or ''}.",
        node=out5_verify,
        sources=info.output_5_sources,
        additional_instruction="Confirm the page states the full chair service period."
    )

    # Output 6: Public platform name
    out6 = evaluator.add_sequential(
        id="output_6_public_platform_name",
        desc="Provide the public platform where they publish articles about interstellar objects (e.g., blog/site name) + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.interstellar_publications_platform) and _has_sources(info.output_6_sources),
        id="output_6_value_and_sources_present",
        desc="Public platform name present and ≥1 supporting URL provided.",
        parent=out6,
        critical=True
    )
    out6_verify = evaluator.add_leaf(
        id="output_6_platform_supported",
        desc="Public platform name is supported by cited URLs.",
        parent=out6,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher publishes interstellar object articles on '{info.interstellar_publications_platform or ''}'.",
        node=out6_verify,
        sources=info.output_6_sources,
        additional_instruction="Confirm the page indicates the public platform name (blog/site)."
    )

    # Output 7: Discovery announcement month/year (must be July 2025)
    out7 = evaluator.add_sequential(
        id="output_7_discovery_announcement_month_year",
        desc="Provide the month and year when 3I/ATLAS's discovery was announced (Month YYYY), and it must be July 2025 + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.atlas_discovery_announcement_month_year) and _has_sources(info.output_7_sources),
        id="output_7_value_and_sources_present",
        desc="Discovery announcement month/year present and ≥1 supporting URL provided.",
        parent=out7,
        critical=True
    )
    evaluator.add_custom_node(
        result=_month_year_is_july_2025(info.atlas_discovery_announcement_month_year),
        id="output_7_value_is_july_2025",
        desc="Discovery announcement month/year equals 'July 2025'.",
        parent=out7,
        critical=True
    )
    out7_verify = evaluator.add_leaf(
        id="output_7_discovery_supported",
        desc="Discovery announcement month/year (July 2025) is supported by cited URLs.",
        parent=out7,
        critical=True
    )
    await evaluator.verify(
        claim="The discovery of 3I/ATLAS was announced in July 2025.",
        node=out7_verify,
        sources=info.output_7_sources,
        additional_instruction="Confirm the page states the discovery announcement of 3I/ATLAS occurred in July 2025."
    )

    # Output 8: Approximate number of articles about 3I/ATLAS
    out8 = evaluator.add_sequential(
        id="output_8_article_count_or_range",
        desc="Provide an approximate number (specific count or range) of articles they published about 3I/ATLAS + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.atlas_article_count_or_range) and _has_sources(info.output_8_sources),
        id="output_8_value_and_sources_present",
        desc="Article count/range present and ≥1 supporting URL provided.",
        parent=out8,
        critical=True
    )
    out8_verify = evaluator.add_leaf(
        id="output_8_count_supported",
        desc="Approximate number of 3I/ATLAS articles is supported by cited URLs.",
        parent=out8,
        critical=True
    )
    await evaluator.verify(
        claim=f"The researcher published approximately {info.atlas_article_count_or_range or ''} articles about 3I/ATLAS.",
        node=out8_verify,
        sources=info.output_8_sources,
        additional_instruction="Confirm the page supports the stated count or reasonable range of 3I/ATLAS articles."
    )

    # Output 9: Perihelion date (2025)
    out9 = evaluator.add_sequential(
        id="output_9_perihelion_date_2025",
        desc="Provide the date when 3I/ATLAS reached perihelion in 2025 (Month DD, YYYY) + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.atlas_perihelion_date_2025) and _has_sources(info.output_9_sources),
        id="output_9_value_and_sources_present",
        desc="Perihelion date present and ≥1 supporting URL provided.",
        parent=out9,
        critical=True
    )
    out9_verify = evaluator.add_leaf(
        id="output_9_perihelion_supported",
        desc="Perihelion date (2025) is supported by cited URLs.",
        parent=out9,
        critical=True
    )
    await evaluator.verify(
        claim=f"3I/ATLAS reached perihelion on {info.atlas_perihelion_date_2025 or ''}.",
        node=out9_verify,
        sources=info.output_9_sources,
        additional_instruction="Confirm the page states the perihelion date of 3I/ATLAS in 2025."
    )

    # Output 10: Closest Earth approach date (2025)
    out10 = evaluator.add_sequential(
        id="output_10_closest_earth_approach_date_2025",
        desc="Provide the date when 3I/ATLAS made its closest approach to Earth in 2025 (Month DD, YYYY) + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.atlas_earth_approach_date_2025) and _has_sources(info.output_10_sources),
        id="output_10_value_and_sources_present",
        desc="Closest Earth approach date present and ≥1 supporting URL provided.",
        parent=out10,
        critical=True
    )
    out10_verify = evaluator.add_leaf(
        id="output_10_earth_approach_supported",
        desc="Closest Earth approach date (2025) is supported by cited URLs.",
        parent=out10,
        critical=True
    )
    await evaluator.verify(
        claim=f"3I/ATLAS made its closest approach to Earth on {info.atlas_earth_approach_date_2025 or ''}.",
        node=out10_verify,
        sources=info.output_10_sources,
        additional_instruction="Confirm the page states the closest approach date to Earth for 3I/ATLAS in 2025."
    )

    # Output 11: Nth interstellar designation number (must be 3)
    out11 = evaluator.add_sequential(
        id="output_11_nth_interstellar_designation_number",
        desc="Provide the designation number confirming 3I/ATLAS as the nth interstellar object discovered, and it must be 3 + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_is_nonempty_str(info.atlas_nth_designation_number) and _has_sources(info.output_11_sources),
        id="output_11_value_and_sources_present",
        desc="Designation number present and ≥1 supporting URL provided.",
        parent=out11,
        critical=True
    )
    evaluator.add_custom_node(
        result=_designation_is_3(info.atlas_nth_designation_number),
        id="output_11_value_is_3",
        desc="Designation number equals '3'.",
        parent=out11,
        critical=True
    )
    out11_verify = evaluator.add_leaf(
        id="output_11_designation_supported",
        desc="Designation number '3' is supported by cited URLs.",
        parent=out11,
        critical=True
    )
    await evaluator.verify(
        claim="3I/ATLAS is the 3rd interstellar object discovered (designation number 3).",
        node=out11_verify,
        sources=info.output_11_sources,
        additional_instruction="Confirm that authoritative sources identify 3I/ATLAS as the third interstellar object."
    )

    # Output 12: Two previous interstellar objects
    out12 = evaluator.add_sequential(
        id="output_12_two_previous_interstellar_objects",
        desc="Provide the names of the two interstellar objects discovered before 3I/ATLAS + ≥1 supporting URL.",
        parent=outputs_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=(info.previous_interstellar_objects is not None and len(info.previous_interstellar_objects) >= 2) and _has_sources(info.output_12_sources),
        id="output_12_values_and_sources_present",
        desc="Two previous interstellar object names present and ≥1 supporting URL provided.",
        parent=out12,
        critical=True
    )
    prev_objs_str = ", ".join(info.previous_interstellar_objects[:2]) if info.previous_interstellar_objects else ""
    out12_verify = evaluator.add_leaf(
        id="output_12_prev_objects_supported",
        desc="Names of the two previous interstellar objects are supported by cited URLs.",
        parent=out12,
        critical=True
    )
    await evaluator.verify(
        claim=f"The two interstellar objects discovered before 3I/ATLAS are {prev_objs_str}.",
        node=out12_verify,
        sources=info.output_12_sources,
        additional_instruction="Confirm the page identifies the two interstellar objects that precede 3I/ATLAS."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an agent's answer for the Harvard/CfA interstellar researcher task.
    Builds a critical sequential root:
      1) Researcher identification criteria (parallel critical)
      2) Required outputs with evidence (parallel critical)
    """
    # Initialize evaluator with critical sequential root
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
        default_model=model,
    )

    # IMPORTANT: Mark root critical and ensure children are also critical
    root.critical = True

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_researcher_info(),
        template_class=ResearcherExtraction,
        extraction_name="researcher_info",
    )

    # Build researcher identification subtree
    await build_researcher_identification_nodes(evaluator, root, extracted_info)

    # Build outputs verification subtree
    await build_outputs_verification_nodes(evaluator, root, extracted_info)

    # Return standardized evaluation summary
    return evaluator.get_summary()