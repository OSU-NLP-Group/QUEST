import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "mar_5_2026_western_hemisphere_updates"
TASK_DESCRIPTION = (
    "On March 5, 2026, President Donald Trump made several significant announcements regarding U.S. foreign policy "
    "and national security in the Western Hemisphere. Provide a comprehensive description of the major developments "
    "announced that day, including: (1) The name of the new hemispheric security initiative that was announced; "
    "(2) The person appointed to lead this initiative and their previous government position; "
    "(3) The date and location of the inaugural summit for this initiative; "
    "(4) The number of member countries in this initiative according to official sources; "
    "(5) The type and primary purpose of this initiative; "
    "(6) Any other major diplomatic breakthrough announced on the same date. "
    "Your answer should include all relevant dates, names, locations, and key details that can be verified through "
    "official government sources and news reports."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class InitiativeExtraction(BaseModel):
    name: FieldWithSources = Field(default_factory=FieldWithSources)
    announcement_date: FieldWithSources = Field(default_factory=FieldWithSources)
    chairman: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_name: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_appointment_date: FieldWithSources = Field(default_factory=FieldWithSources)
    lead_previous_position: FieldWithSources = Field(default_factory=FieldWithSources)
    dhs_replacement_nominee: FieldWithSources = Field(default_factory=FieldWithSources)
    inaugural_summit_date: FieldWithSources = Field(default_factory=FieldWithSources)
    inaugural_summit_location: FieldWithSources = Field(default_factory=FieldWithSources)
    member_country_count: FieldWithSources = Field(default_factory=FieldWithSources)
    initiative_type: FieldWithSources = Field(default_factory=FieldWithSources)
    initiative_primary_purpose: FieldWithSources = Field(default_factory=FieldWithSources)

    all_urls: List[str] = Field(default_factory=list)
    government_urls: List[str] = Field(default_factory=list)
    news_urls: List[str] = Field(default_factory=list)
    wikipedia_urls: List[str] = Field(default_factory=list)


class BreakthroughExtraction(BaseModel):
    what_happened: FieldWithSources = Field(default_factory=FieldWithSources)
    announcement_date: FieldWithSources = Field(default_factory=FieldWithSources)
    context_burgum_visit: FieldWithSources = Field(default_factory=FieldWithSources)

    all_urls: List[str] = Field(default_factory=list)
    government_urls: List[str] = Field(default_factory=list)
    news_urls: List[str] = Field(default_factory=list)
    wikipedia_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_initiative() -> str:
    return """
Extract from the answer all details about the hemispheric security initiative announced by the U.S. on March 5, 2026.
For each field below, return both the value stated in the answer and ALL source URLs in the answer that directly support that field.

Return a JSON exactly matching this schema (do not add extra fields):
{
  "name": {"value": str|null, "source_urls": [url, ...]},
  "announcement_date": {"value": str|null, "source_urls": [url, ...]},
  "chairman": {"value": str|null, "source_urls": [url, ...]},
  "lead_name": {"value": str|null, "source_urls": [url, ...]},
  "lead_appointment_date": {"value": str|null, "source_urls": [url, ...]},
  "lead_previous_position": {"value": str|null, "source_urls": [url, ...]},
  "dhs_replacement_nominee": {"value": str|null, "source_urls": [url, ...]},
  "inaugural_summit_date": {"value": str|null, "source_urls": [url, ...]},
  "inaugural_summit_location": {"value": str|null, "source_urls": [url, ...]},
  "member_country_count": {"value": str|null, "source_urls": [url, ...]},
  "initiative_type": {"value": str|null, "source_urls": [url, ...]},
  "initiative_primary_purpose": {"value": str|null, "source_urls": [url, ...]},

  "all_urls": [url, ...],
  "government_urls": [url, ...],
  "news_urls": [url, ...],
  "wikipedia_urls": [url, ...]
}

Value guidance (copy exactly what the answer states; do not invent):
- name: Initiative name (e.g., "Shield of the Americas")
- announcement_date: Date the initiative was announced (e.g., "March 5, 2026")
- chairman: Who serves as chairman (e.g., "President Donald Trump")
- lead_name: Appointed lead's name (e.g., "Kristi Noem")
- lead_appointment_date: The appointment date (e.g., "March 5, 2026")
- lead_previous_position: The immediately previous government position of the lead (e.g., "Secretary of Homeland Security")
- dhs_replacement_nominee: Name of the person nominated to replace the lead as DHS Secretary (e.g., "Senator Markwayne Mullin")
- inaugural_summit_date: Date of the inaugural summit (e.g., "March 7, 2026")
- inaugural_summit_location: Location of the inaugural summit (e.g., "Doral, Florida")
- member_country_count: Number of member countries (e.g., "12")
- initiative_type: Short type description (e.g., "multinational military coalition")
- initiative_primary_purpose: Primary purpose (e.g., "combating transnational criminal organizations and drug cartels")

URL classification rules:
- government_urls: Only official U.S. government sites (domains ending in .gov or .mil, or whitehouse.gov).
- news_urls: News outlets (e.g., reuters.com, apnews.com, bloomberg.com, wsj.com, bbc.com, cnn.com, foxnews.com, nytimes.com, etc.).
- wikipedia_urls: Any en.wikipedia.org URLs.
- all_urls: All URLs mentioned anywhere in the answer (deduplicated).

Only include URLs that appear explicitly in the answer text. If a field is not present in the answer, set its "value" to null and "source_urls" to [].
"""


def prompt_extract_breakthrough() -> str:
    return """
Extract from the answer the other major diplomatic breakthrough announced on March 5, 2026, and its key context. 
For each field below, return both the value stated in the answer and ALL source URLs in the answer that directly support that field.

Return a JSON exactly matching this schema:
{
  "what_happened": {"value": str|null, "source_urls": [url, ...]},
  "announcement_date": {"value": str|null, "source_urls": [url, ...]},
  "context_burgum_visit": {"value": str|null, "source_urls": [url, ...]},

  "all_urls": [url, ...],
  "government_urls": [url, ...],
  "news_urls": [url, ...],
  "wikipedia_urls": [url, ...]
}

Value guidance (copy exactly what the answer states; do not invent):
- what_happened: e.g., "The United States and Venezuela agreed to reestablish diplomatic relations."
- announcement_date: e.g., "March 5, 2026"
- context_burgum_visit: e.g., "The announcement came at the end of a two-day visit by U.S. Interior Secretary Doug Burgum."

URL classification rules:
- government_urls: Only official U.S. government sites (domains ending in .gov or .mil, or whitehouse.gov).
- news_urls: News outlets (e.g., reuters.com, apnews.com, bloomberg.com, wsj.com, bbc.com, cnn.com, foxnews.com, nytimes.com, etc.).
- wikipedia_urls: Any en.wikipedia.org URLs.
- all_urls: All URLs mentioned anywhere in the answer (deduplicated).

Only include URLs that appear explicitly in the answer text. If a field is not present in the answer, set its "value" to null and "source_urls" to [].
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_keep_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _is_gov_url(url: str) -> bool:
    if not isinstance(url, str):
        return False
    u = url.lower()
    return u.endswith(".gov") or ".gov/" in u or u.endswith(".mil") or ".mil/" in u or "whitehouse.gov" in u


def _is_wikipedia_url(url: str) -> bool:
    return isinstance(url, str) and "wikipedia.org" in url.lower()


def _build_sources_prefer_field(
    field: FieldWithSources,
    *fallback_lists: List[str],
) -> List[str]:
    result = list(field.source_urls or [])
    for lst in fallback_lists:
        result.extend(lst or [])
    return _unique_keep_order(result)


def _build_sources_prefer_wikipedia(
    field: FieldWithSources,
    wikipedia_urls: List[str],
    *fallback_lists: List[str],
) -> List[str]:
    result = list(wikipedia_urls or [])
    result.extend(field.source_urls or [])
    for lst in fallback_lists:
        result.extend(lst or [])
    return _unique_keep_order(result)


def _additional_instruction_for_claim(base_hint: str, has_sources: bool, extra: Optional[str] = None) -> str:
    prefix = (
        "You must verify strictly against the provided webpage content. "
        "Allow reasonable paraphrasing, casing differences, and minor wording variations for names/titles. "
    )
    if not has_sources:
        # Enforce source-grounding policy: fail if no URL provided for a factual claim.
        suffix = (
            "No URL evidence was provided for this specific claim. "
            "Treat the claim as not supported and return Incorrect."
        )
    else:
        suffix = "Focus on explicit support on the page; do not rely on prior knowledge."
    middle = base_hint if base_hint else ""
    if extra:
        middle = (middle + " " + extra).strip()
    return f"{prefix}{middle} {suffix}".strip()


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_initiative_nodes(evaluator: Evaluator, parent_node, ie: InitiativeExtraction) -> None:
    init_node = evaluator.add_parallel(
        id="Hemispheric_Security_Initiative",
        desc="Provide all required details about the hemispheric security initiative announced on March 5, 2026, consistent with the constraints.",
        parent=parent_node,
        critical=True,
    )

    # Convenience common pools
    all_urls = _unique_keep_order((ie.all_urls or []) + (ie.government_urls or []) + (ie.news_urls or []) + (ie.wikipedia_urls or []))
    gov_urls = _unique_keep_order(ie.government_urls or [u for u in all_urls if _is_gov_url(u)])
    news_urls = _unique_keep_order(ie.news_urls or [u for u in all_urls if not _is_gov_url(u) and not _is_wikipedia_url(u)])
    wiki_urls = _unique_keep_order(ie.wikipedia_urls or [u for u in all_urls if _is_wikipedia_url(u)])

    claims_and_nodes: List[tuple] = []

    # Initiative_Announcement_Date
    leaf = evaluator.add_leaf(
        id="Initiative_Announcement_Date",
        desc="States that the initiative was announced on March 5, 2026.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.announcement_date, ie.name.source_urls, gov_urls, news_urls, all_urls)
    claim = f"The hemispheric security initiative was announced on {ie.announcement_date.value}."
    add_ins = _additional_instruction_for_claim(
        "Check that the announcement date is correctly stated.",
        has_sources=bool(srcs),
        extra="The target correct date is March 5, 2026; the page must explicitly or clearly indicate the date."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Initiative_Name
    leaf = evaluator.add_leaf(
        id="Initiative_Name",
        desc="States the initiative name: Shield of the Americas.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.name, gov_urls, news_urls, all_urls)
    claim = f"The hemispheric security initiative is named '{ie.name.value}'."
    add_ins = _additional_instruction_for_claim(
        "Verify the official or commonly-used name on the page.",
        has_sources=bool(srcs),
        extra="Minor hyphenation or capitalization differences are acceptable if clearly the same initiative."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Initiative_Chairman
    leaf = evaluator.add_leaf(
        id="Initiative_Chairman",
        desc="States that President Donald Trump serves as Chairman of the initiative.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.chairman, gov_urls, news_urls, all_urls)
    initiative_label = ie.name.value or "the initiative"
    claim = f"President Donald Trump serves as Chairman of {initiative_label}."
    add_ins = _additional_instruction_for_claim(
        "Confirm that the page attributes the role of Chairman to President Donald Trump.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Initiative_Lead (includes appointment date)
    leaf = evaluator.add_leaf(
        id="Initiative_Lead",
        desc="Identifies the appointed initiative lead as Kristi Noem and notes the appointment date as March 5, 2026.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.lead_name, ie.lead_appointment_date.source_urls, gov_urls, news_urls, all_urls)
    claim = f"The initiative lead is {ie.lead_name.value} and the appointment was announced on {ie.lead_appointment_date.value}."
    add_ins = _additional_instruction_for_claim(
        "Both the identity of the lead and the appointment date must be supported.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Lead_Previous_Government_Position
    leaf = evaluator.add_leaf(
        id="Lead_Previous_Government_Position",
        desc="States that Kristi Noem’s previous role was Secretary of Homeland Security.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.lead_previous_position, gov_urls, news_urls, all_urls)
    claim = f"Before this appointment, {ie.lead_name.value or 'the lead'} served as {ie.lead_previous_position.value}."
    add_ins = _additional_instruction_for_claim(
        "Verify the immediately previous government position held by the named lead.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # DHS_Replacement_Nomination
    leaf = evaluator.add_leaf(
        id="DHS_Replacement_Nomination",
        desc="States that Senator Markwayne Mullin was nominated to replace Kristi Noem as Secretary of Homeland Security.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.dhs_replacement_nominee, gov_urls, news_urls, all_urls)
    nominee = ie.dhs_replacement_nominee.value or "the nominee"
    claim = f"{nominee} was nominated to replace {ie.lead_name.value or 'the lead'} as Secretary of Homeland Security."
    add_ins = _additional_instruction_for_claim(
        "Confirm the nomination to replace the lead as DHS Secretary and the nominee's identity.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Inaugural_Summit_Date
    leaf = evaluator.add_leaf(
        id="Inaugural_Summit_Date",
        desc="Provides the inaugural summit date as March 7, 2026.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.inaugural_summit_date, gov_urls, news_urls, all_urls)
    claim = f"The inaugural summit is scheduled for {ie.inaugural_summit_date.value}."
    add_ins = _additional_instruction_for_claim(
        "Check that the specific summit date is explicitly stated.",
        has_sources=bool(srcs),
        extra="Target correct date is March 7, 2026."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Inaugural_Summit_Location
    leaf = evaluator.add_leaf(
        id="Inaugural_Summit_Location",
        desc="Provides the inaugural summit location as Doral, Florida.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.inaugural_summit_location, gov_urls, news_urls, all_urls)
    claim = f"The inaugural summit location is {ie.inaugural_summit_location.value}."
    add_ins = _additional_instruction_for_claim(
        "The page should clearly indicate the summit's city and state (or precise venue) location.",
        has_sources=bool(srcs),
        extra="Target correct location is Doral, Florida."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Member_Country_Count (prefer Wikipedia infobox support)
    leaf = evaluator.add_leaf(
        id="Member_Country_Count",
        desc="States the number of member countries as 12 and attributes this count to the Wikipedia infobox (as specified).",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_wikipedia(ie.member_country_count, wiki_urls, gov_urls, news_urls, all_urls)
    claim = f"According to Wikipedia, the initiative has {ie.member_country_count.value} member countries (as shown in the infobox or clearly stated)."
    add_ins = _additional_instruction_for_claim(
        "Use a Wikipedia page (preferably its infobox) to verify the member-country count.",
        has_sources=bool(srcs),
        extra="If a Wikipedia URL is not provided, treat the claim as not supported."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Initiative_Type
    leaf = evaluator.add_leaf(
        id="Initiative_Type",
        desc="Describes the initiative as a multinational military coalition.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.initiative_type, gov_urls, news_urls, wiki_urls, all_urls)
    claim = f"The initiative is described as {ie.initiative_type.value}."
    add_ins = _additional_instruction_for_claim(
        "Confirm the initiative's type/characterization as described on the page.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Initiative_Primary_Purpose
    leaf = evaluator.add_leaf(
        id="Initiative_Primary_Purpose",
        desc="States that the initiative focuses on combating transnational criminal organizations and drug cartels.",
        parent=init_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(ie.initiative_primary_purpose, gov_urls, news_urls, wiki_urls, all_urls)
    claim = f"The initiative's primary purpose is {ie.initiative_primary_purpose.value}."
    add_ins = _additional_instruction_for_claim(
        "Confirm the primary purpose/mission focus as stated on the page. If multiple purposes are listed, focus on the primary/central one.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    await evaluator.batch_verify(claims_and_nodes)


async def build_breakthrough_nodes(evaluator: Evaluator, parent_node, be: BreakthroughExtraction) -> None:
    br_node = evaluator.add_parallel(
        id="Other_Diplomatic_Breakthrough",
        desc="Covers the other major diplomatic breakthrough announced on March 5, 2026, consistent with the constraints.",
        parent=parent_node,
        critical=True,
    )

    all_urls = _unique_keep_order((be.all_urls or []) + (be.government_urls or []) + (be.news_urls or []) + (be.wikipedia_urls or []))
    gov_urls = _unique_keep_order(be.government_urls or [u for u in all_urls if _is_gov_url(u)])
    news_urls = _unique_keep_order(be.news_urls or [u for u in all_urls if not _is_gov_url(u) and not _is_wikipedia_url(u)])

    claims_and_nodes: List[tuple] = []

    # Breakthrough_What_Happened
    leaf = evaluator.add_leaf(
        id="Breakthrough_What_Happened",
        desc="States that the United States and Venezuela agreed to reestablish diplomatic relations.",
        parent=br_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(be.what_happened, gov_urls, news_urls, all_urls)
    claim = f"{be.what_happened.value or 'An agreement'}"
    # If value missing, the claim will be vague; enforce unsupported when no sources.
    add_ins = _additional_instruction_for_claim(
        "Verify that the page explicitly indicates the reestablishment of diplomatic relations between the U.S. and Venezuela.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Breakthrough_Announcement_Date
    leaf = evaluator.add_leaf(
        id="Breakthrough_Announcement_Date",
        desc="States that the diplomatic-relations restoration was announced on March 5, 2026.",
        parent=br_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(be.announcement_date, gov_urls, news_urls, all_urls)
    claim = f"The announcement of the U.S.–Venezuela diplomatic-relations restoration occurred on {be.announcement_date.value}."
    add_ins = _additional_instruction_for_claim(
        "Confirm the announcement date on the page.",
        has_sources=bool(srcs),
        extra="Target correct date is March 5, 2026."
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    # Breakthrough_Context_Burgum_Visit
    leaf = evaluator.add_leaf(
        id="Breakthrough_Context_Burgum_Visit",
        desc="States that the announcement came at the end of a two-day visit by U.S. Interior Secretary Doug Burgum.",
        parent=br_node,
        critical=True,
    )
    srcs = _build_sources_prefer_field(be.context_burgum_visit, gov_urls, news_urls, all_urls)
    claim = f"The announcement came at the end of a two-day visit by U.S. Interior Secretary Doug Burgum: {be.context_burgum_visit.value}."
    add_ins = _additional_instruction_for_claim(
        "Confirm that the timing/context mentions a two-day visit by U.S. Interior Secretary Doug Burgum culminating in the announcement.",
        has_sources=bool(srcs),
    )
    claims_and_nodes.append((claim, srcs, leaf, add_ins))

    await evaluator.batch_verify(claims_and_nodes)


async def build_sourcing_nodes(evaluator: Evaluator, parent_node, ie: InitiativeExtraction, be: BreakthroughExtraction) -> None:
    src_node = evaluator.add_parallel(
        id="Sourcing_Requirement",
        desc="Includes verifiable sourcing as required by the question (official government sources and news reports).",
        parent=parent_node,
        critical=True,
    )

    # Aggregate URLs across initiative and breakthrough
    combined_all = _unique_keep_order(
        (ie.all_urls or []) + (be.all_urls or []) + (ie.government_urls or []) + (be.government_urls or []) +
        (ie.news_urls or []) + (be.news_urls or []) + (ie.wikipedia_urls or []) + (be.wikipedia_urls or [])
    )
    combined_gov = _unique_keep_order((ie.government_urls or []) + (be.government_urls or []) + [u for u in combined_all if _is_gov_url(u)])
    combined_news = _unique_keep_order((ie.news_urls or []) + (be.news_urls or []))

    # Official_Government_Source_Used
    gov_leaf = evaluator.add_leaf(
        id="Official_Government_Source_Used",
        desc="Includes at least one citation/attribution to an official government source supporting one or more key claims.",
        parent=src_node,
        critical=True,
    )
    # Claim: at least one official page that confirms one of the key facts.
    gov_claim = (
        "This is an official U.S. government source (URL domain ends with .gov or .mil, or is whitehouse.gov) "
        "that confirms at least one key fact stated in the answer about either: "
        "the 'Shield of the Americas' initiative (its name, chairman, lead/appointment, dates, location, nature, or purpose) "
        "or the U.S.–Venezuela diplomatic-relations restoration announced on March 5, 2026."
    )
    gov_add_ins = (
        "Consider the URL shown to decide if it is a U.S. government site (.gov/.mil or whitehouse.gov). "
        "Then check whether the page confirms at least one of the listed key facts. "
        "If no government URL is provided or none confirm a listed fact, return Incorrect."
    )
    await evaluator.verify(
        claim=gov_claim,
        node=gov_leaf,
        sources=combined_gov if combined_gov else None,
        additional_instruction=gov_add_ins,
    )

    # News_Report_Used
    news_leaf = evaluator.add_leaf(
        id="News_Report_Used",
        desc="Includes at least one citation/attribution to a news report supporting one or more key claims.",
        parent=src_node,
        critical=True,
    )
    news_claim = (
        "This is a credible news report that confirms at least one key fact stated in the answer about either: "
        "the 'Shield of the Americas' initiative (its announcement/name/lead/summit details/member count/type/purpose) "
        "or the U.S.–Venezuela diplomatic-relations restoration announced on March 5, 2026."
    )
    news_add_ins = (
        "Use only news outlet pages (non-government, non-wikipedia). "
        "The page should explicitly confirm at least one key fact. "
        "If no news URL is provided or none confirm a listed fact, return Incorrect."
    )
    await evaluator.verify(
        claim=news_claim,
        node=news_leaf,
        sources=combined_news if combined_news else None,
        additional_instruction=news_add_ins,
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the March 5, 2026 Western Hemisphere announcements task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root is parallel per rubric
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

    # IMPORTANT: Root is critical per rubric - ensure all descendants are also critical
    root.critical = True

    # Extract initiative and breakthrough info in parallel
    initiative_task = evaluator.extract(
        prompt=prompt_extract_initiative(),
        template_class=InitiativeExtraction,
        extraction_name="initiative_extraction",
    )
    breakthrough_task = evaluator.extract(
        prompt=prompt_extract_breakthrough(),
        template_class=BreakthroughExtraction,
        extraction_name="breakthrough_extraction",
    )
    ie, be = await asyncio.gather(initiative_task, breakthrough_task)

    # Build verification subtrees
    await build_initiative_nodes(evaluator, root, ie)
    await build_breakthrough_nodes(evaluator, root, be)
    await build_sourcing_nodes(evaluator, root, ie, be)

    # Return the standardized evaluation summary
    return evaluator.get_summary()