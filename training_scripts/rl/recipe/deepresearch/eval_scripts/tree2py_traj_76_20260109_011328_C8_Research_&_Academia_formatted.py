import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "conference_2025_us_cvml"
TASK_DESCRIPTION = (
    "Identify a major computer vision or machine learning conference taking place in the United States between June 1 and July 31, 2025. "
    "For the identified conference, provide the following 12 specific pieces of information, each supported by a reference URL from the official conference website: "
    "(1) Full official conference name and standard acronym, "
    "(2) Exact dates (both start date and end date) of the main conference, "
    "(3) Official venue name and host city, "
    "(4) Maximum page limit for the main paper body (excluding references), "
    "(5) Acceptance rate percentage for the 2025 edition, "
    "(6) Confirmation of whether double-blind peer review is used, "
    "(7) Confirmation of whether supplementary materials are permitted, "
    "(8) Author registration policy (whether at least one author must register), "
    "(9) Specific dates of the author rebuttal or response period, "
    "(10) Policy on whether one registration can cover multiple accepted papers (and if so, how many), "
    "(11) Venue capacity or actual number of attendees for 2025, "
    "(12) Availability of virtual attendance options. "
    "Each piece of information must be accompanied by the URL of the official conference page where this information can be verified."
)

DATE_WINDOW_START = "June 1, 2025"
DATE_WINDOW_END = "July 31, 2025"


class ConferenceExtraction(BaseModel):
    conference_name: Optional[str] = None
    acronym: Optional[str] = None
    conference_name_urls: List[str] = Field(default_factory=list)

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dates_urls: List[str] = Field(default_factory=list)

    venue_name: Optional[str] = None
    host_city: Optional[str] = None
    conference_country: Optional[str] = None
    venue_urls: List[str] = Field(default_factory=list)

    page_limit_main_body: Optional[str] = None
    page_limit_urls: List[str] = Field(default_factory=list)

    acceptance_rate_2025: Optional[str] = None
    acceptance_rate_urls: List[str] = Field(default_factory=list)

    double_blind_review_used: Optional[str] = None  # yes/no or descriptive text
    double_blind_urls: List[str] = Field(default_factory=list)

    supplementary_materials_permitted: Optional[str] = None  # yes/no or descriptive text
    supplementary_urls: List[str] = Field(default_factory=list)

    author_registration_required: Optional[str] = None  # yes/no or descriptive text
    author_registration_urls: List[str] = Field(default_factory=list)

    rebuttal_start_date: Optional[str] = None
    rebuttal_end_date: Optional[str] = None
    rebuttal_urls: List[str] = Field(default_factory=list)

    multiple_papers_one_registration_allowed: Optional[str] = None  # yes/no
    max_papers_per_registration: Optional[str] = None  # number or textual limit
    multiple_papers_urls: List[str] = Field(default_factory=list)

    venue_capacity_or_attendees_2025: Optional[str] = None
    capacity_attendance_urls: List[str] = Field(default_factory=list)

    virtual_attendance_available: Optional[str] = None  # yes/no or descriptive text
    virtual_attendance_urls: List[str] = Field(default_factory=list)

    website_root_domain: Optional[str] = None  # if the answer mentions an official domain


def prompt_extract_conference() -> str:
    return (
        "Extract details for exactly ONE identified conference from the answer. If multiple conferences are mentioned, extract the first qualifying one. "
        "Return a JSON object with the following fields, using strings for all values and arrays for URLs:\n"
        "1. conference_name: Full official conference name\n"
        "2. acronym: Standard acronym\n"
        "3. conference_name_urls: Array of official conference website URLs that verify the name/acronym\n"
        "4. start_date: Exact start date of the MAIN conference in the 2025 edition (not workshops). Keep text as given in the answer.\n"
        "5. end_date: Exact end date of the MAIN conference in the 2025 edition. Keep text as given in the answer.\n"
        "6. dates_urls: Array of official URLs that verify the start/end dates\n"
        "7. venue_name: Official venue name (e.g., convention center/hotel)\n"
        "8. host_city: Host city (include state abbreviation if present)\n"
        "9. conference_country: Country of the conference (e.g., 'United States') if provided\n"
        "10. venue_urls: Array of official URLs that verify the venue and city\n"
        "11. page_limit_main_body: Maximum page limit for the main paper body (excluding references). Keep text as given (e.g., '8 pages').\n"
        "12. page_limit_urls: Array of official URLs that verify the page limit\n"
        "13. acceptance_rate_2025: Acceptance rate percentage for the 2025 edition (e.g., '24%'); keep text as given.\n"
        "14. acceptance_rate_urls: Array of official URLs that verify the acceptance rate\n"
        "15. double_blind_review_used: 'yes' or 'no' (or an equivalent textual statement) indicating whether double-blind peer review is used.\n"
        "16. double_blind_urls: Array of official URLs that verify the review policy\n"
        "17. supplementary_materials_permitted: 'yes' or 'no' (or equivalent textual statement)\n"
        "18. supplementary_urls: Array of official URLs that verify the supplementary materials policy\n"
        "19. author_registration_required: 'yes' or 'no' (or equivalent textual statement) indicating if at least one author must register.\n"
        "20. author_registration_urls: Array of official URLs that verify the author registration requirement\n"
        "21. rebuttal_start_date: Start date of the rebuttal/author response period. Keep text as given.\n"
        "22. rebuttal_end_date: End date of the rebuttal/author response period. Keep text as given.\n"
        "23. rebuttal_urls: Array of official URLs that verify the rebuttal/response period dates\n"
        "24. multiple_papers_one_registration_allowed: 'yes' or 'no' indicating whether a single registration can cover multiple accepted papers.\n"
        "25. max_papers_per_registration: If 'yes', provide the number or textual limit (e.g., 'up to 2 papers'). If missing, return null.\n"
        "26. multiple_papers_urls: Array of official URLs that verify the multiple-papers-per-registration policy\n"
        "27. venue_capacity_or_attendees_2025: Provide either venue capacity or the actual number of attendees for 2025, as text.\n"
        "28. capacity_attendance_urls: Array of official URLs that verify the capacity or attendee count\n"
        "29. virtual_attendance_available: 'yes' or 'no' (or textual statement) indicating if virtual attendance is available.\n"
        "30. virtual_attendance_urls: Array of official URLs that verify virtual attendance availability\n"
        "31. website_root_domain: If the answer explicitly mentions the official site domain (e.g., 'icml.cc'), extract it; otherwise null.\n\n"
        "URL extraction rules:\n"
        "- Extract only URLs explicitly present in the answer (plain or markdown).\n"
        "- Prefer official conference website pages; however, extract whatever URLs the answer provides.\n"
        "- If a required URL is not present in the answer, return an empty array for that URL field.\n"
        "- Do not invent or infer URLs."
    )


def _yn(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in {"yes", "y", "true"}:
        return "yes"
    if v in {"no", "n", "false"}:
        return "no"
    # allow phrases like "uses double-blind", "single-blind"
    if "double" in v and "blind" in v:
        return "yes"
    if "single-blind" in v or "not double-blind" in v:
        return "no"
    return v


def _has_urls(urls: List[str]) -> bool:
    return bool(urls) and len(urls) > 0 and any(isinstance(u, str) and u.strip() for u in urls)


async def _verify_official_url_presence(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    conf_name: Optional[str],
    urls: List[str],
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )
    claim = (
        f"At least one of the provided URLs is an official page of the {conf_name or 'conference'}."
    )
    add_ins = (
        "Judge officialness by domain and site branding/logos and top-level navigation. "
        "Accept official conference domains or their subdomains (e.g., icml.cc, iclr.cc, cvpr.thecvf.com, neurips.cc, eccv2024.ecva.net). "
        "Reject third-party sites (e.g., openreview.net, easychair.org, news/blogs) unless they are clearly part of the official conference site. "
        "If no URL is provided, conclude NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if _has_urls(urls) else None,
        additional_instruction=add_ins,
    )


async def _verify_content_with_official_urls(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    claim: str,
    urls: List[str],
    extra_instruction: Optional[str] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )
    base_ins = (
        "Verify the claim strictly against the provided official conference website page(s). "
        "If the URLs are not official pages or no URL is provided, conclude NOT SUPPORTED. "
        "Allow minor formatting variations (e.g., capitalization, abbreviations) but the substantive information must match exactly."
    )
    add_ins = base_ins if not extra_instruction else (base_ins + " " + extra_instruction)
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls if _has_urls(urls) else None,
        additional_instruction=add_ins,
    )


async def _build_eligibility_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    elig = evaluator.add_parallel(
        id="Conference_Eligibility",
        desc="The identified conference satisfies all eligibility constraints (US, date window, major CV/ML).",
        parent=parent_node,
        critical=True,
    )
    # US location
    us_leaf = evaluator.add_leaf(
        id="Eligibility_US",
        desc="The conference is held in the United States (based on the stated host city/venue location).",
        parent=elig,
        critical=True,
    )
    claim_us = (
        f"The conference is held in the United States."
    )
    add_ins_us = (
        "Use the venue/city official page(s) to verify that the location is in the United States. "
        "City/state indicators (e.g., 'Seattle, WA') or explicit 'United States' suffice."
        "If no official venue/city URL is provided, conclude NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=claim_us,
        node=us_leaf,
        sources=ext.venue_urls if _has_urls(ext.venue_urls) else None,
        additional_instruction=add_ins_us,
    )

    # Date window
    date_window_leaf = evaluator.add_leaf(
        id="Eligibility_Date_Window",
        desc="The main conference start and end dates fall between June 1 and July 31, 2025 (inclusive).",
        parent=elig,
        critical=True,
    )
    claim_dw = (
        "The MAIN conference start and end dates fall between June 1 and July 31, 2025 (inclusive)."
    )
    add_ins_dw = (
        "Check the specific dates shown on the official conference schedule page(s). "
        "Focus on the MAIN conference dates (not tutorials/workshops). "
        "If the dates are outside the window or unclear, or no official date URL is provided, conclude NOT SUPPORTED."
    )
    await evaluator.verify(
        claim=claim_dw,
        node=date_window_leaf,
        sources=ext.dates_urls if _has_urls(ext.dates_urls) else None,
        additional_instruction=add_ins_dw,
    )

    # Major CV/ML recognition
    major_leaf = evaluator.add_leaf(
        id="Eligibility_Major_CVML",
        desc="The conference qualifies as a recognized major computer vision or machine learning conference (evaluatable as true/false by the grader based on common domain recognition).",
        parent=elig,
        critical=True,
    )
    conf_display = (ext.conference_name or "").strip()
    acr = (ext.acronym or "").strip()
    claim_major = (
        f"The conference {conf_display} ({acr}) is a recognized major computer vision or machine learning conference."
    )
    add_ins_major = (
        "Use general domain knowledge to decide whether this is a widely recognized major CV/ML conference. "
        "Examples include CVPR, ICML, NeurIPS, ICLR, ECCV and similar. "
        "Reject niche workshops or small regional events."
    )
    await evaluator.verify(
        claim=claim_major,
        node=major_leaf,
        sources=None,
        additional_instruction=add_ins_major,
    )


async def _build_name_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Conference_Name",
        desc="Provides (1) full official conference name and standard acronym, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Conference_Name_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the stated name/acronym.",
        ext.conference_name,
        ext.conference_name_urls,
    )
    claim = (
        f"The official conference name is '{ext.conference_name or ''}' and its standard acronym is '{ext.acronym or ''}'."
    )
    extra = "Allow minor variants (e.g., 'IEEE/CVF Conference on Computer Vision and Pattern Recognition' vs 'CVPR')."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Conference_Name_Content",
        "Full official conference name AND standard acronym are both provided and match the official conference website.",
        claim,
        ext.conference_name_urls,
        extra_instruction=extra,
    )


async def _build_dates_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Conference_Dates",
        desc="Provides (2) exact start and end dates of the main conference, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Conference_Dates_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the start/end dates.",
        ext.conference_name,
        ext.dates_urls,
    )
    claim = (
        f"The MAIN conference takes place from {ext.start_date or '[missing start date]'} to {ext.end_date or '[missing end date]'}."
    )
    extra = "Focus on the MAIN conference dates (exclude separate workshop/tutorial days if distinct)."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Conference_Dates_Content",
        "Both exact start date AND exact end date of the main conference are provided and match the official conference website.",
        claim,
        ext.dates_urls,
        extra_instruction=extra,
    )


async def _build_venue_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Venue_and_City",
        desc="Provides (3) official venue name and host city, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Venue_and_City_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the venue and city.",
        ext.conference_name,
        ext.venue_urls,
    )
    country_text = ext.conference_country or "United States"
    claim = (
        f"The official venue is '{ext.venue_name or ''}', located in '{ext.host_city or ''}', {country_text}."
    )
    extra = "Accept city/state abbreviations (e.g., 'Seattle, WA')."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Venue_and_City_Content",
        "Official venue name AND host city are provided and match the official conference website.",
        claim,
        ext.venue_urls,
        extra_instruction=extra,
    )


async def _build_page_limit_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Page_Limit",
        desc="Provides (4) maximum page limit for main paper body excluding references, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Page_Limit_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the stated page limit (main body excluding references).",
        ext.conference_name,
        ext.page_limit_urls,
    )
    claim = (
        f"The maximum page limit for the main paper body (excluding references) is {ext.page_limit_main_body or '[missing page limit]'}."
    )
    extra = "Check official author guidelines for 2025; ensure limit excludes references and appendices."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Page_Limit_Content",
        "The maximum page limit for the main paper body (excluding references) is stated and matches the official author/submission guidelines for the identified conference edition.",
        claim,
        ext.page_limit_urls,
        extra_instruction=extra,
    )


async def _build_acceptance_rate_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Acceptance_Rate",
        desc="Provides (5) acceptance rate percentage for the 2025 edition, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Acceptance_Rate_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the 2025 acceptance rate.",
        ext.conference_name,
        ext.acceptance_rate_urls,
    )
    claim = f"The acceptance rate for the 2025 edition is {ext.acceptance_rate_2025 or '[missing acceptance rate]'}."
    extra = "If the official site does not provide an acceptance rate, conclude NOT SUPPORTED."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Acceptance_Rate_Content",
        "An acceptance rate percentage for the 2025 edition is provided and matches what is published on the official conference website.",
        claim,
        ext.acceptance_rate_urls,
        extra_instruction=extra,
    )


async def _build_double_blind_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Double_Blind_Review",
        desc="Provides (6) whether double-blind peer review is used, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Double_Blind_Review_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the double-blind review policy.",
        ext.conference_name,
        ext.double_blind_urls,
    )
    ynv = _yn(ext.double_blind_review_used)
    if ynv == "yes":
        claim = "The conference uses double-blind peer review."
    elif ynv == "no":
        claim = "The conference does not use double-blind peer review."
    else:
        claim = f"The double-blind peer review policy is: {ext.double_blind_review_used or '[missing]'}."
    extra = "Check the review/submission policy for the 2025 edition."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Double_Blind_Review_Content",
        "A clear yes/no statement about double-blind peer review is provided and matches the official review/submission policy for the identified conference edition.",
        claim,
        ext.double_blind_urls,
        extra_instruction=extra,
    )


async def _build_supplementary_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Supplementary_Materials",
        desc="Provides (7) whether supplementary materials are permitted, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Supplementary_Materials_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the supplementary materials policy.",
        ext.conference_name,
        ext.supplementary_urls,
    )
    ynv = _yn(ext.supplementary_materials_permitted)
    if ynv == "yes":
        claim = "Supplementary materials are permitted."
    elif ynv == "no":
        claim = "Supplementary materials are not permitted."
    else:
        claim = f"The supplementary materials policy is: {ext.supplementary_materials_permitted or '[missing]'}."
    extra = "Check the submission policy for 2025; consider allowed formats and deadlines as corroborating context."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Supplementary_Materials_Content",
        "A clear yes/no statement about whether supplementary materials are permitted is provided and matches the official submission policy for the identified conference edition.",
        claim,
        ext.supplementary_urls,
        extra_instruction=extra,
    )


async def _build_author_registration_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Author_Registration",
        desc="Provides (8) author registration policy (whether at least one author must register), with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Author_Registration_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the author registration requirement.",
        ext.conference_name,
        ext.author_registration_urls,
    )
    ynv = _yn(ext.author_registration_required)
    if ynv == "yes":
        claim = "At least one author must register for the conference."
    elif ynv == "no":
        claim = "It is not required that at least one author register for the conference."
    else:
        claim = f"The author registration requirement is: {ext.author_registration_required or '[missing]'}."
    extra = "Look for official registration or 'presenter attendance' policies that specify whether an author must register."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Author_Registration_Content",
        "States whether at least one author must register (yes/no) and matches the official registration policy for the identified conference edition.",
        claim,
        ext.author_registration_urls,
        extra_instruction=extra,
    )


async def _build_rebuttal_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Rebuttal_Period",
        desc="Provides (9) specific dates of the author rebuttal/response period, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Rebuttal_Period_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the rebuttal/response period dates.",
        ext.conference_name,
        ext.rebuttal_urls,
    )
    claim = (
        f"The author rebuttal/response period is from {ext.rebuttal_start_date or '[missing start]'} to {ext.rebuttal_end_date or '[missing end]'}."
    )
    extra = "Check the official timeline; ensure the stage is explicitly labelled 'rebuttal' or 'author response'."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Rebuttal_Period_Content",
        "Specific rebuttal/response period dates (start and end) are provided and match the official timeline for the identified conference edition.",
        claim,
        ext.rebuttal_urls,
        extra_instruction=extra,
    )


async def _build_multi_papers_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Multiple_Papers_Policy",
        desc="Provides (10) whether one registration can cover multiple accepted papers (and if so, how many), with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Multiple_Papers_Policy_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the multiple-paper-per-registration policy (including the number if applicable).",
        ext.conference_name,
        ext.multiple_papers_urls,
    )
    ynv = _yn(ext.multiple_papers_one_registration_allowed)
    if ynv == "yes":
        if ext.max_papers_per_registration:
            claim = f"One registration can cover multiple accepted papers, up to {ext.max_papers_per_registration}."
        else:
            claim = "One registration can cover multiple accepted papers, but the exact allowed number is unspecified."
    elif ynv == "no":
        claim = "A single registration cannot cover multiple accepted papers."
    else:
        claim = (
            f"The policy on covering multiple papers with one registration is: {ext.multiple_papers_one_registration_allowed or '[missing]'}; "
            f"number limit: {ext.max_papers_per_registration or '[missing]'}."
        )
    extra = "If 'yes', verify the exact number allowed; if not stated on the official page, conclude NOT SUPPORTED."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Multiple_Papers_Policy_Content",
        "States whether one registration can cover multiple papers; if yes, states the allowed number; matches the official registration policy for the identified conference edition.",
        claim,
        ext.multiple_papers_urls,
        extra_instruction=extra,
    )


async def _build_capacity_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Venue_Capacity_Attendance",
        desc="Provides (11) venue capacity OR actual number of attendees for 2025, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Venue_Capacity_Attendance_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the stated capacity or 2025 attendee count.",
        ext.conference_name,
        ext.capacity_attendance_urls,
    )
    claim = f"The venue capacity or 2025 attendee count is {ext.venue_capacity_or_attendees_2025 or '[missing]'}."
    extra = "Verify the number strictly on the official conference website; if the official site does not provide such a number, conclude NOT SUPPORTED."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Venue_Capacity_Attendance_Content",
        "Provides either (a) venue capacity or (b) actual attendee count for the 2025 edition; the value matches what is published on the official conference website.",
        claim,
        ext.capacity_attendance_urls,
        extra_instruction=extra,
    )


async def _build_virtual_nodes(evaluator: Evaluator, parent_node, ext: ConferenceExtraction) -> None:
    node = evaluator.add_parallel(
        id="Virtual_Attendance",
        desc="Provides (12) availability of virtual attendance options, with official-site URL proof.",
        parent=parent_node,
        critical=True,
    )
    await _verify_official_url_presence(
        evaluator,
        node,
        "Virtual_Attendance_Official_URL",
        "At least one URL is provided on the official conference website that directly verifies the virtual attendance availability/format.",
        ext.conference_name,
        ext.virtual_attendance_urls,
    )
    ynv = _yn(ext.virtual_attendance_available)
    if ynv == "yes":
        claim = "Virtual attendance options are available."
    elif ynv == "no":
        claim = "Virtual attendance options are not available."
    else:
        claim = f"Virtual attendance availability: {ext.virtual_attendance_available or '[missing]'}."
    extra = "Verify official statements about hybrid/virtual formats or livestreams for 2025."
    await _verify_content_with_official_urls(
        evaluator,
        node,
        "Virtual_Attendance_Content",
        "Clearly states whether virtual attendance is available (yes/no) and matches the official attendance format information for the identified conference edition.",
        claim,
        ext.virtual_attendance_urls,
        extra_instruction=extra,
    )


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
        strategy=AggregationStrategy.PARALLEL,
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

    ext: ConferenceExtraction = await evaluator.extract(
        prompt=prompt_extract_conference(),
        template_class=ConferenceExtraction,
        extraction_name="conference_extraction",
    )

    evaluator.add_custom_info(
        info={
            "required_time_window": {"start": DATE_WINDOW_START, "end": DATE_WINDOW_END},
            "eligibility_country": "United States",
            "task_focus": "Major CV/ML conferences only",
        },
        info_type="constraints",
        info_name="task_constraints",
    )

    task_node = evaluator.add_parallel(
        id="Conference_Identification_Task",
        desc="Evaluate whether the response identifies one qualifying conference and provides all 12 required pieces of information, each verified by an official-conference-website URL.",
        parent=root,
        critical=True,
    )

    await _build_eligibility_nodes(evaluator, task_node, ext)

    await _build_name_nodes(evaluator, task_node, ext)
    await _build_dates_nodes(evaluator, task_node, ext)
    await _build_venue_nodes(evaluator, task_node, ext)
    await _build_page_limit_nodes(evaluator, task_node, ext)
    await _build_acceptance_rate_nodes(evaluator, task_node, ext)
    await _build_double_blind_nodes(evaluator, task_node, ext)
    await _build_supplementary_nodes(evaluator, task_node, ext)
    await _build_author_registration_nodes(evaluator, task_node, ext)
    await _build_rebuttal_nodes(evaluator, task_node, ext)
    await _build_multi_papers_nodes(evaluator, task_node, ext)
    await _build_capacity_nodes(evaluator, task_node, ext)
    await _build_virtual_nodes(evaluator, task_node, ext)

    return evaluator.get_summary()