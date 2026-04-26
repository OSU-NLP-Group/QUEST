import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "us_municipal_statutes"
TASK_DESCRIPTION = """For municipalities in the United States, provide specific statutory procedural requirements for the following governance actions:

1. Florida Municipality — Bond Referendum: For a municipality seeking to issue general obligation bonds that require voter approval: (a) What is the voter approval threshold? (b) What is the statutory citation? (c) Provide a reference URL from an official Florida government source.

2. California Municipality — Regular City Council Meetings: For posting agendas for regular meetings of a city council: (a) What is the minimum advance notice period (in hours)? (b) What are the required posting locations? (c) What is the statutory citation? (d) Provide a reference URL from an official California government source.

3. New Jersey Municipality — Annual Budget Adoption: For a municipality on a calendar-year fiscal year: (a) What is the statutory deadline for final budget adoption (month and day)? (b) How many days in advance must the budget be published before the public hearing? (c) What is the statutory citation? (d) Provide a reference URL from an official New Jersey government source.

4. Kansas City — Annexation by Consent: When a property owner consents to annexation of adjoining property: (a) Is a public hearing required? (b) What is the statutory citation? (c) Provide a reference URL from an official Kansas government source.
"""

# Ground-truth style expectations (used only for claims/instructions; not hard-coded scoring)
GT = {
    "florida": {
        "threshold": "a majority of votes cast (more than 50%)",
        "citation": "Florida Statutes § 100.201 (F.S. 100.201)",
    },
    "california": {
        "notice_period": "72 hours before regular meetings",
        "posting_locations": "posted in a location freely accessible to the public and on the agency’s website if the agency has one",
        "citation": "California Government Code § 54954.2",
    },
    "new_jersey": {
        "deadline": "no later than March 20 (calendar fiscal year)",
        "publication": "at least 10 days before the public hearing",
        "citation": "N.J.S.A. 40A:4-10",
    },
    "kansas": {
        "hearing_requirement": "public hearing is not required when property owners consent to annexation of adjoining property",
        "citation": "K.S.A. 12-520a and/or 12-520b",
    }
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FloridaBond(BaseModel):
    voter_threshold: Optional[str] = None
    voter_threshold_urls: List[str] = Field(default_factory=list)
    statute_citation: Optional[str] = None
    statute_urls: List[str] = Field(default_factory=list)


class CaliforniaAgenda(BaseModel):
    notice_period_hours: Optional[str] = None
    notice_urls: List[str] = Field(default_factory=list)
    posting_locations: Optional[str] = None
    posting_urls: List[str] = Field(default_factory=list)
    statute_citation: Optional[str] = None
    statute_urls: List[str] = Field(default_factory=list)


class NewJerseyBudget(BaseModel):
    adoption_deadline: Optional[str] = None
    deadline_urls: List[str] = Field(default_factory=list)
    publication_days: Optional[str] = None
    publication_urls: List[str] = Field(default_factory=list)
    statute_citation: Optional[str] = None
    statute_urls: List[str] = Field(default_factory=list)


class KansasAnnexation(BaseModel):
    hearing_required: Optional[str] = None  # e.g., "not required", "no", "required"
    hearing_urls: List[str] = Field(default_factory=list)
    statute_citation: Optional[str] = None
    statute_urls: List[str] = Field(default_factory=list)


class MunicipalProceduresExtraction(BaseModel):
    florida: Optional[FloridaBond] = None
    california: Optional[CaliforniaAgenda] = None
    new_jersey: Optional[NewJerseyBudget] = None
    kansas: Optional[KansasAnnexation] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
    Extract from the answer the specific information requested for each of the four scenarios below. Return exactly and only what the answer states (do not infer or invent). If any item is not present in the answer, return null (for strings) or an empty list (for URL lists).

    For each scenario, extract the following fields:

    1) Florida Municipality — Bond Referendum:
       - voter_threshold: string exactly as stated in the answer describing the voter approval threshold.
       - voter_threshold_urls: array of all URLs cited in the answer that support the threshold requirement.
       - statute_citation: string of the statute citation exactly as written in the answer (e.g., "F.S. 100.201", "Florida Statutes § 100.201").
       - statute_urls: array of all URLs cited for the statute.

    2) California Municipality — Regular City Council Meetings (Brown Act):
       - notice_period_hours: string as stated in the answer for minimum advance notice (e.g., "72 hours", "at least 72 hours").
       - notice_urls: array of URLs cited that support the notice period requirement.
       - posting_locations: string summarizing the posting locations as the answer states (e.g., "publicly accessible location and agency website if available").
       - posting_urls: array of URLs cited that support the posting locations requirement.
       - statute_citation: string of the statute citation as written in the answer (e.g., "Gov. Code § 54954.2").
       - statute_urls: array of all URLs cited for the statute.

    3) New Jersey Municipality — Annual Budget Adoption:
       - adoption_deadline: string as stated in the answer for the final adoption deadline (month and day, e.g., "March 20").
       - deadline_urls: array of URLs cited that support the adoption deadline.
       - publication_days: string as stated in the answer for how many days in advance the budget must be published before the hearing (e.g., "at least 10 days").
       - publication_urls: array of URLs cited that support the publication requirement.
       - statute_citation: string of the statute citation as written in the answer (e.g., "N.J.S.A. 40A:4-10").
       - statute_urls: array of all URLs cited for the statute.

    4) Kansas City — Annexation by Consent:
       - hearing_required: string as stated in the answer indicating whether a public hearing is required when all property owners consent (e.g., "not required", "no", "required").
       - hearing_urls: array of URLs cited that support the hearing requirement determination.
       - statute_citation: string of the statute citation as written in the answer (e.g., "K.S.A. 12-520a", "K.S.A. 12-520b").
       - statute_urls: array of all URLs cited for the statute.

    URL extraction rules:
    - Extract only URLs explicitly present in the answer (including markdown links). Do not invent URLs.
    - Include all URLs provided for each requirement, even if they are not official government sources.
    - Do not deduplicate; keep order as they appear in the answer if possible.

    Return a JSON object with keys: florida, california, new_jersey, kansas. Each key maps to an object as defined.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip() != "")


def _official_source_hint(state: str) -> str:
    hints = {
        "Florida": "Treat as official if the domain is statutes.leg.state.fl.us, flsenate.gov, leg.state.fl.us, myflorida.com, or ends with .fl.us/.state.fl.us.",
        "California": "Treat as official if the domain is leginfo.legislature.ca.gov or any *.ca.gov domain (e.g., oag.ca.gov, ca.gov).",
        "New Jersey": "Treat as official if the domain is nj.gov, state.nj.us, njleg.state.nj.us, or pub.njleg.gov.",
        "Kansas": "Treat as official if the domain is kslegislature.org or a *.ks.gov domain (e.g., ag.ks.gov, sos.ks.gov).",
    }
    return hints.get(state, "Use only official government domains (.gov, .state.xx.us), legislative sites, or clearly official state portals.")


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_florida_tree(evaluator: Evaluator, parent, data: Optional[FloridaBond]) -> None:
    scenario = evaluator.add_parallel(
        id="florida_bond_referendum_scenario",
        desc="Florida municipality bond referendum requirements - all required information provided",
        parent=parent,
        critical=False,
    )

    # Voter approval threshold requirement
    vt_group = evaluator.add_parallel(
        id="florida_voter_threshold_requirement",
        desc="Voter approval threshold requirement",
        parent=scenario,
        critical=True,
    )

    vt_val_exists = evaluator.add_custom_node(
        result=_has_text(data.voter_threshold) if data else False,
        id="florida_voter_threshold_value_exists",
        desc="Florida voter threshold value is provided in the answer",
        parent=vt_group,
        critical=True,
    )

    vt_value = evaluator.add_leaf(
        id="florida_voter_threshold_value",
        desc="Correct identification that voter approval requires majority of votes cast (more than 50% of votes cast)",
        parent=vt_group,
        critical=True,
    )
    vt_claim = (
        f"The answer states the voter approval threshold as '{(data.voter_threshold or '')}'. "
        f"This must match Florida's rule for municipal bond referenda requiring voter approval: {GT['florida']['threshold']}."
        f" Confirm using the provided Florida source(s)."
    )
    await evaluator.verify(
        claim=vt_claim,
        node=vt_value,
        sources=(data.voter_threshold_urls if data else []),
        additional_instruction=(
            "Consider 'majority,' 'simple majority,' 'more than half,' and '>50%' as equivalent. "
            "If the answer claims any supermajority (e.g., 60%), that is incorrect. "
            f"{_official_source_hint('Florida')}"
        ),
    )

    vt_url_exists = evaluator.add_custom_node(
        result=bool(data and data.voter_threshold_urls),
        id="florida_voter_threshold_url_exists",
        desc="At least one Florida voter threshold supporting URL is provided",
        parent=vt_group,
        critical=True,
    )

    vt_url = evaluator.add_leaf(
        id="florida_voter_threshold_url",
        desc="Valid URL reference from official Florida government source supporting the voter threshold requirement",
        parent=vt_group,
        critical=True,
    )
    vt_url_claim = (
        "This webpage is an official Florida government source and it supports that municipal bond referenda requiring voter approval are approved by a majority of votes cast."
    )
    await evaluator.verify(
        claim=vt_url_claim,
        node=vt_url,
        sources=(data.voter_threshold_urls if data else []),
        additional_instruction=_official_source_hint("Florida"),
    )

    # Statutory citation requirement
    fl_cit_group = evaluator.add_parallel(
        id="florida_statutory_citation_requirement",
        desc="Statutory citation requirement",
        parent=scenario,
        critical=True,
    )

    fl_cit_exists = evaluator.add_custom_node(
        result=_has_text(data.statute_citation) if data else False,
        id="florida_citation_value_exists",
        desc="Florida statutory citation is provided in the answer",
        parent=fl_cit_group,
        critical=True,
    )

    fl_cit_value = evaluator.add_leaf(
        id="florida_citation_value",
        desc="Correct statutory citation (F.S. 100.201 or Florida Statutes Section 100.201)",
        parent=fl_cit_group,
        critical=True,
    )
    fl_cit_claim = (
        f"The answer cites the statute as '{(data.statute_citation or '')}'. "
        "This must correspond to Florida Statutes Section 100.201 (also expressible as 'F.S. 100.201' or '§ 100.201, Fla. Stat.'). "
        "Minor formatting differences are acceptable."
    )
    await evaluator.verify(
        claim=fl_cit_claim,
        node=fl_cit_value,
        additional_instruction="Accept equivalent notations such as 'F.S. 100.201', 'Florida Statutes § 100.201', or '§ 100.201, Fla. Stat.'.",
    )

    fl_cit_url_exists = evaluator.add_custom_node(
        result=bool(data and data.statute_urls),
        id="florida_citation_url_exists",
        desc="At least one Florida statute URL is provided",
        parent=fl_cit_group,
        critical=True,
    )

    fl_cit_url = evaluator.add_leaf(
        id="florida_citation_url",
        desc="Valid URL reference from official Florida government source for the statute",
        parent=fl_cit_group,
        critical=True,
    )
    fl_cit_url_claim = "This webpage is an official Florida government source that shows Florida Statutes Section 100.201 governing bond referenda."
    await evaluator.verify(
        claim=fl_cit_url_claim,
        node=fl_cit_url,
        sources=(data.statute_urls if data else []),
        additional_instruction=_official_source_hint("Florida"),
    )


async def build_california_tree(evaluator: Evaluator, parent, data: Optional[CaliforniaAgenda]) -> None:
    scenario = evaluator.add_parallel(
        id="california_meeting_agenda_scenario",
        desc="California city council regular meeting agenda posting requirements - all required information provided",
        parent=parent,
        critical=False,
    )

    # Minimum advance notice period
    ca_notice_group = evaluator.add_parallel(
        id="california_advance_notice_requirement",
        desc="Minimum advance notice period requirement",
        parent=scenario,
        critical=True,
    )

    ca_notice_exists = evaluator.add_custom_node(
        result=_has_text(data.notice_period_hours) if data else False,
        id="california_notice_period_value_exists",
        desc="California notice period value is provided in the answer",
        parent=ca_notice_group,
        critical=True,
    )

    ca_notice_value = evaluator.add_leaf(
        id="california_notice_period_value",
        desc="Correct identification that agendas must be posted at least 72 hours before regular meetings",
        parent=ca_notice_group,
        critical=True,
    )
    ca_notice_claim = (
        f"The answer states the minimum advance notice period as '{(data.notice_period_hours or '')}'. "
        f"This must match the Brown Act requirement for regular meetings: {GT['california']['notice_period']}. "
        "Confirm with the provided California source(s)."
    )
    await evaluator.verify(
        claim=ca_notice_claim,
        node=ca_notice_value,
        sources=(data.notice_urls if data else []),
        additional_instruction=(
            "Focus on Government Code § 54954.2(a)(1) for regular meetings. "
            "Accept phrasing like 'at least 72 hours' or '72 hours in advance'. "
            f"{_official_source_hint('California')}"
        ),
    )

    ca_notice_url_exists = evaluator.add_custom_node(
        result=bool(data and data.notice_urls),
        id="california_notice_period_url_exists",
        desc="At least one California notice period supporting URL is provided",
        parent=ca_notice_group,
        critical=True,
    )

    ca_notice_url = evaluator.add_leaf(
        id="california_notice_period_url",
        desc="Valid URL reference from official California government source supporting the notice period requirement",
        parent=ca_notice_group,
        critical=True,
    )
    ca_notice_url_claim = (
        "This webpage is an official California government source that supports that agendas for regular meetings must be posted at least 72 hours before the meeting."
    )
    await evaluator.verify(
        claim=ca_notice_url_claim,
        node=ca_notice_url,
        sources=(data.notice_urls if data else []),
        additional_instruction=_official_source_hint("California"),
    )

    # Posting locations
    ca_post_group = evaluator.add_parallel(
        id="california_posting_locations_requirement",
        desc="Required posting locations",
        parent=scenario,
        critical=True,
    )

    ca_post_exists = evaluator.add_custom_node(
        result=_has_text(data.posting_locations) if data else False,
        id="california_posting_locations_value_exists",
        desc="California posting locations value is provided in the answer",
        parent=ca_post_group,
        critical=True,
    )

    ca_post_value = evaluator.add_leaf(
        id="california_posting_locations_value",
        desc="Correct identification that posting must be in publicly accessible location and on agency website if available",
        parent=ca_post_group,
        critical=True,
    )
    ca_post_claim = (
        f"The answer states posting locations as '{(data.posting_locations or '')}'. "
        "This must match the Brown Act requirement that agendas be posted in a location freely accessible to the public and on the agency's website, if the agency has one."
    )
    await evaluator.verify(
        claim=ca_post_claim,
        node=ca_post_value,
        sources=(data.posting_urls if data else []),
        additional_instruction=(
            "Look to Gov. Code § 54954.2(a)(1). Accept equivalent phrasing indicating a 'publicly accessible location' and posting on the local agency website if it exists. "
            f"{_official_source_hint('California')}"
        ),
    )

    ca_post_url_exists = evaluator.add_custom_node(
        result=bool(data and data.posting_urls),
        id="california_posting_locations_url_exists",
        desc="At least one California posting locations supporting URL is provided",
        parent=ca_post_group,
        critical=True,
    )

    ca_post_url = evaluator.add_leaf(
        id="california_posting_locations_url",
        desc="Valid URL reference from official California government source supporting the posting locations requirement",
        parent=ca_post_group,
        critical=True,
    )
    ca_post_url_claim = (
        "This webpage is an official California government source that supports the required posting locations (publicly accessible location and on the agency website if the agency has one)."
    )
    await evaluator.verify(
        claim=ca_post_url_claim,
        node=ca_post_url,
        sources=(data.posting_urls if data else []),
        additional_instruction=_official_source_hint("California"),
    )

    # Statutory citation
    ca_cit_group = evaluator.add_parallel(
        id="california_statutory_citation_requirement",
        desc="Statutory citation requirement",
        parent=scenario,
        critical=True,
    )

    ca_cit_exists = evaluator.add_custom_node(
        result=_has_text(data.statute_citation) if data else False,
        id="california_citation_value_exists",
        desc="California statutory citation is provided in the answer",
        parent=ca_cit_group,
        critical=True,
    )

    ca_cit_value = evaluator.add_leaf(
        id="california_citation_value",
        desc="Correct statutory citation (Government Code Section 54954.2 or Gov. Code § 54954.2)",
        parent=ca_cit_group,
        critical=True,
    )
    ca_cit_claim = (
        f"The answer cites the statute as '{(data.statute_citation or '')}', which must correspond to California Government Code § 54954.2 "
        "(allowing minor formatting variants such as 'Gov. Code 54954.2' or 'Government Code Section 54954.2')."
    )
    await evaluator.verify(
        claim=ca_cit_claim,
        node=ca_cit_value,
        additional_instruction="Accept equivalent notations like 'Gov. Code § 54954.2', 'Government Code 54954.2', or '§ 54954.2'.",
    )

    ca_cit_url_exists = evaluator.add_custom_node(
        result=bool(data and data.statute_urls),
        id="california_citation_url_exists",
        desc="At least one California statute URL is provided",
        parent=ca_cit_group,
        critical=True,
    )

    ca_cit_url = evaluator.add_leaf(
        id="california_citation_url",
        desc="Valid URL reference from official California government source for the statute",
        parent=ca_cit_group,
        critical=True,
    )
    ca_cit_url_claim = "This webpage is an official California government source that shows Government Code § 54954.2."
    await evaluator.verify(
        claim=ca_cit_url_claim,
        node=ca_cit_url,
        sources=(data.statute_urls if data else []),
        additional_instruction=_official_source_hint("California"),
    )


async def build_new_jersey_tree(evaluator: Evaluator, parent, data: Optional[NewJerseyBudget]) -> None:
    scenario = evaluator.add_parallel(
        id="new_jersey_budget_scenario",
        desc="New Jersey municipality annual budget adoption requirements - all required information provided",
        parent=parent,
        critical=False,
    )

    # Adoption deadline
    nj_deadline_group = evaluator.add_parallel(
        id="new_jersey_adoption_deadline_requirement",
        desc="Final budget adoption deadline requirement",
        parent=scenario,
        critical=True,
    )

    nj_deadline_exists = evaluator.add_custom_node(
        result=_has_text(data.adoption_deadline) if data else False,
        id="new_jersey_deadline_value_exists",
        desc="New Jersey budget adoption deadline value is provided in the answer",
        parent=nj_deadline_group,
        critical=True,
    )

    nj_deadline_value = evaluator.add_leaf(
        id="new_jersey_deadline_value",
        desc="Correct identification that budget must be adopted no later than March 20 of the calendar fiscal year",
        parent=nj_deadline_group,
        critical=True,
    )
    nj_deadline_claim = (
        f"The answer states the final adoption deadline as '{(data.adoption_deadline or '')}'. "
        f"This must match the statutory baseline: {GT['new_jersey']['deadline']}. Confirm with the provided New Jersey source(s)."
    )
    await evaluator.verify(
        claim=nj_deadline_claim,
        node=nj_deadline_value,
        sources=(data.deadline_urls if data else []),
        additional_instruction=(
            "Focus on N.J.S.A. 40A:4 provisions (specifically § 40A:4-10). "
            "Note: Extensions by the Director may occur administratively, but the statutory baseline is March 20. "
            f"{_official_source_hint('New Jersey')}"
        ),
    )

    nj_deadline_url_exists = evaluator.add_custom_node(
        result=bool(data and data.deadline_urls),
        id="new_jersey_deadline_url_exists",
        desc="At least one New Jersey budget deadline supporting URL is provided",
        parent=nj_deadline_group,
        critical=True,
    )

    nj_deadline_url = evaluator.add_leaf(
        id="new_jersey_deadline_url",
        desc="Valid URL reference from official New Jersey government source supporting the adoption deadline",
        parent=nj_deadline_group,
        critical=True,
    )
    nj_deadline_url_claim = "This webpage is an official New Jersey government source that supports the March 20 final adoption deadline for calendar-year municipal budgets."
    await evaluator.verify(
        claim=nj_deadline_url_claim,
        node=nj_deadline_url,
        sources=(data.deadline_urls if data else []),
        additional_instruction=_official_source_hint("New Jersey"),
    )

    # Publication requirement
    nj_pub_group = evaluator.add_parallel(
        id="new_jersey_publication_requirement",
        desc="Advance publication requirement",
        parent=scenario,
        critical=True,
    )

    nj_pub_exists = evaluator.add_custom_node(
        result=_has_text(data.publication_days) if data else False,
        id="new_jersey_publication_value_exists",
        desc="New Jersey budget publication requirement value is provided in the answer",
        parent=nj_pub_group,
        critical=True,
    )

    nj_pub_value = evaluator.add_leaf(
        id="new_jersey_publication_value",
        desc="Correct identification that budget must be published at least 10 days before the public hearing",
        parent=nj_pub_group,
        critical=True,
    )
    nj_pub_claim = (
        f"The answer states the publication lead time as '{(data.publication_days or '')}'. "
        f"This must match the statutory requirement: {GT['new_jersey']['publication']}."
    )
    await evaluator.verify(
        claim=nj_pub_claim,
        node=nj_pub_value,
        sources=(data.publication_urls if data else []),
        additional_instruction=(
            "Look for New Jersey statutory or authoritative administrative sources confirming at least 10 days' publication before the hearing. "
            f"{_official_source_hint('New Jersey')}"
        ),
    )

    nj_pub_url_exists = evaluator.add_custom_node(
        result=bool(data and data.publication_urls),
        id="new_jersey_publication_url_exists",
        desc="At least one New Jersey budget publication supporting URL is provided",
        parent=nj_pub_group,
        critical=True,
    )

    nj_pub_url = evaluator.add_leaf(
        id="new_jersey_publication_url",
        desc="Valid URL reference from official New Jersey government source supporting the publication requirement",
        parent=nj_pub_group,
        critical=True,
    )
    nj_pub_url_claim = "This webpage is an official New Jersey government source that supports the requirement to publish the budget at least 10 days before the hearing."
    await evaluator.verify(
        claim=nj_pub_url_claim,
        node=nj_pub_url,
        sources=(data.publication_urls if data else []),
        additional_instruction=_official_source_hint("New Jersey"),
    )

    # Statutory citation
    nj_cit_group = evaluator.add_parallel(
        id="new_jersey_statutory_citation_requirement",
        desc="Statutory citation requirement",
        parent=scenario,
        critical=True,
    )

    nj_cit_exists = evaluator.add_custom_node(
        result=_has_text(data.statute_citation) if data else False,
        id="new_jersey_citation_value_exists",
        desc="New Jersey statutory citation is provided in the answer",
        parent=nj_cit_group,
        critical=True,
    )

    nj_cit_value = evaluator.add_leaf(
        id="new_jersey_citation_value",
        desc="Correct statutory citation (N.J.S.A. 40A:4-10 or similar variation)",
        parent=nj_cit_group,
        critical=True,
    )
    nj_cit_claim = (
        f"The answer cites the statute as '{(data.statute_citation or '')}', which must correspond to N.J.S.A. 40A:4-10 "
        "(accepting minor format variations, e.g., 'NJSA 40A:4-10', '§ 40A:4-10')."
    )
    await evaluator.verify(
        claim=nj_cit_claim,
        node=nj_cit_value,
        additional_instruction="Accept variations like 'N.J.S.A. 40A:4-10', 'NJSA 40A:4-10', or '§ 40A:4-10'.",
    )

    nj_cit_url_exists = evaluator.add_custom_node(
        result=bool(data and data.statute_urls),
        id="new_jersey_citation_url_exists",
        desc="At least one New Jersey statute URL is provided",
        parent=nj_cit_group,
        critical=True,
    )

    nj_cit_url = evaluator.add_leaf(
        id="new_jersey_citation_url",
        desc="Valid URL reference from official New Jersey government source for the statute",
        parent=nj_cit_group,
        critical=True,
    )
    nj_cit_url_claim = "This webpage is an official New Jersey government source showing N.J.S.A. 40A:4-10 or the relevant budget statute."
    await evaluator.verify(
        claim=nj_cit_url_claim,
        node=nj_cit_url,
        sources=(data.statute_urls if data else []),
        additional_instruction=_official_source_hint("New Jersey"),
    )


async def build_kansas_tree(evaluator: Evaluator, parent, data: Optional[KansasAnnexation]) -> None:
    scenario = evaluator.add_parallel(
        id="kansas_annexation_scenario",
        desc="Kansas city annexation by consent requirements - all required information provided",
        parent=parent,
        critical=False,
    )

    # Hearing requirement
    ks_hearing_group = evaluator.add_parallel(
        id="kansas_hearing_requirement",
        desc="Public hearing requirement",
        parent=scenario,
        critical=True,
    )

    ks_hearing_exists = evaluator.add_custom_node(
        result=_has_text(data.hearing_required) if data else False,
        id="kansas_hearing_value_exists",
        desc="Kansas hearing requirement value is provided in the answer",
        parent=ks_hearing_group,
        critical=True,
    )

    ks_hearing_value = evaluator.add_leaf(
        id="kansas_hearing_value",
        desc="Correct identification that public hearing is not required when property owners consent to annexation",
        parent=ks_hearing_group,
        critical=True,
    )
    ks_hearing_claim = (
        f"The answer states the hearing requirement as '{(data.hearing_required or '')}'. "
        f"This must match Kansas law for annexation by owner consent: {GT['kansas']['hearing_requirement']}."
    )
    await evaluator.verify(
        claim=ks_hearing_claim,
        node=ks_hearing_value,
        sources=(data.hearing_urls if data else []),
        additional_instruction=(
            "Confirm using K.S.A. 12-520a/12-520b context: consent annexations do not require the unilateral-annexation hearing/service plan process. "
            f"{_official_source_hint('Kansas')}"
        ),
    )

    ks_hearing_url_exists = evaluator.add_custom_node(
        result=bool(data and data.hearing_urls),
        id="kansas_hearing_url_exists",
        desc="At least one Kansas hearing requirement supporting URL is provided",
        parent=ks_hearing_group,
        critical=True,
    )

    ks_hearing_url = evaluator.add_leaf(
        id="kansas_hearing_url",
        desc="Valid URL reference from official Kansas government source supporting the hearing requirement determination",
        parent=ks_hearing_group,
        critical=True,
    )
    ks_hearing_url_claim = (
        "This webpage is an official Kansas government source that supports that a public hearing is not required when all property owners consent to annexation of adjoining property."
    )
    await evaluator.verify(
        claim=ks_hearing_url_claim,
        node=ks_hearing_url,
        sources=(data.hearing_urls if data else []),
        additional_instruction=_official_source_hint("Kansas"),
    )

    # Statutory citation
    ks_cit_group = evaluator.add_parallel(
        id="kansas_statutory_citation_requirement",
        desc="Statutory citation requirement",
        parent=scenario,
        critical=True,
    )

    ks_cit_exists = evaluator.add_custom_node(
        result=_has_text(data.statute_citation) if data else False,
        id="kansas_citation_value_exists",
        desc="Kansas statutory citation is provided in the answer",
        parent=ks_cit_group,
        critical=True,
    )

    ks_cit_value = evaluator.add_leaf(
        id="kansas_citation_value",
        desc="Correct statutory citation (K.S.A. 12-520a, K.S.A. 12-520b, or both)",
        parent=ks_cit_group,
        critical=True,
    )
    ks_cit_claim = (
        f"The answer cites the statute as '{(data.statute_citation or '')}', which must correspond to K.S.A. 12-520a and/or 12-520b "
        "(accepting minor formatting variants like 'KSA 12-520a')."
    )
    await evaluator.verify(
        claim=ks_cit_claim,
        node=ks_cit_value,
        additional_instruction="Accept either 12-520a, 12-520b, or both; minor formatting differences are acceptable.",
    )

    ks_cit_url_exists = evaluator.add_custom_node(
        result=bool(data and data.statute_urls),
        id="kansas_citation_url_exists",
        desc="At least one Kansas statute URL is provided",
        parent=ks_cit_group,
        critical=True,
    )

    ks_cit_url = evaluator.add_leaf(
        id="kansas_citation_url",
        desc="Valid URL reference from official Kansas government source for the statute",
        parent=ks_cit_group,
        critical=True,
    )
    ks_cit_url_claim = "This webpage is an official Kansas government source showing K.S.A. 12-520a and/or 12-520b."
    await evaluator.verify(
        claim=ks_cit_url_claim,
        node=ks_cit_url,
        sources=(data.statute_urls if data else []),
        additional_instruction=_official_source_hint("Kansas"),
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=MunicipalProceduresExtraction,
        extraction_name="municipal_procedures_extraction",
    )

    # Ground truth context (for transparency; not used as hard constraints)
    evaluator.add_ground_truth(
        {
            "florida_expected": {
                "voter_threshold": GT["florida"]["threshold"],
                "citation": GT["florida"]["citation"],
            },
            "california_expected": {
                "notice_period": GT["california"]["notice_period"],
                "posting_locations": GT["california"]["posting_locations"],
                "citation": GT["california"]["citation"],
            },
            "new_jersey_expected": {
                "deadline": GT["new_jersey"]["deadline"],
                "publication": GT["new_jersey"]["publication"],
                "citation": GT["new_jersey"]["citation"],
            },
            "kansas_expected": {
                "hearing_requirement": GT["kansas"]["hearing_requirement"],
                "citation": GT["kansas"]["citation"],
            },
        },
        gt_type="expected_requirements",
    )

    # Build verification subtrees
    await build_florida_tree(evaluator, root, extracted.florida or FloridaBond())
    await build_california_tree(evaluator, root, extracted.california or CaliforniaAgenda())
    await build_new_jersey_tree(evaluator, root, extracted.new_jersey or NewJerseyBudget())
    await build_kansas_tree(evaluator, root, extracted.kansas or KansasAnnexation())

    return evaluator.get_summary()