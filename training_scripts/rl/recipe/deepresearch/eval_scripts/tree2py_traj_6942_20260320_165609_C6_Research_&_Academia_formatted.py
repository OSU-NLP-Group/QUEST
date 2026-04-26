import asyncio
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_2026_spain_university_expeditions"
TASK_DESCRIPTION = """
For the total solar eclipse occurring on August 12, 2026, identify two distinct university-organized expeditions to mainland Spain that meet all of the following criteria:

1. The viewing location must be a specific, named site (not just a city or region) within the path of totality in mainland Spain
2. The Spanish province where the viewing site is located must be identifiable
3. The expected duration of totality at the viewing site must be stated and be at least 1 minute
4. The expedition must be organized by or officially affiliated with an accredited university or college
5. At least one astronomy expert (professor, astronomer, or planetarium director) must be identified as leading or speaking at the expedition
6. The expert's academic credentials or position (professorship, research position, or equivalent) must be specified
7. The expedition must include educational programming such as lectures, astronomy talks, or stargazing activities
8. The expedition dates must include August 12, 2026

For each of the two expeditions, provide:
- The specific viewing site name
- The Spanish province of the viewing site
- The duration of totality at the viewing site
- The organizing or affiliated university/college name
- The name of at least one astronomy expert involved
- The expert's academic credentials or position
- A description of the educational programming included

The two expeditions must be distinct (different viewing sites or different organizing universities).
""".strip()

ECLIPSE_DATE_TEXT = "August 12, 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ExpeditionInfo(BaseModel):
    viewing_site_name: Optional[str] = None
    province: Optional[str] = None
    totality_duration: Optional[str] = None
    university: Optional[str] = None
    expert_name: Optional[str] = None
    expert_credentials: Optional[str] = None
    education_programming: Optional[str] = None
    dates_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class ExpeditionsExtraction(BaseModel):
    expedition1: Optional[ExpeditionInfo] = None
    expedition2: Optional[ExpeditionInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_expeditions() -> str:
    return """
    Extract information for exactly two university-organized eclipse expeditions to mainland Spain mentioned in the answer.
    If more than two are mentioned, select any two distinct expeditions (prefer ones with the most complete information).
    If fewer than two are mentioned, fill the missing expedition with nulls/empties.

    For each expedition, extract the following fields exactly as written in the answer:
    - viewing_site_name: A specific, named viewing site (e.g., a park, campus, stadium, observatory, landmark). Do not put just a city or a broad region here.
    - province: The Spanish province (e.g., "Zaragoza", "Soria"). Prefer the plain province name, without "Province of".
    - totality_duration: The stated expected duration of totality at the viewing site (e.g., "1m 20s", "80 seconds", "01:20").
    - university: The organizing or officially affiliated accredited university/college (institution name).
    - expert_name: The name of at least one astronomy expert (professor, astronomer, or planetarium director) leading or speaking at the expedition.
    - expert_credentials: The expert’s academic credentials or position (e.g., "Professor of Astronomy", "Research Scientist", "Planetarium Director").
    - education_programming: A short phrase describing included educational programming (e.g., "astronomy lectures", "guided stargazing", "public talks").
    - dates_text: The date(s) or date range for the expedition as written in the answer (e.g., "August 10–13, 2026").
    - source_urls: An array of all URLs provided in the answer that are relevant to this expedition. Include any URLs embedded as markdown links or plain links.
      Only include actual URLs; if none are present, return an empty array.

    Return a JSON object with:
    {
      "expedition1": {...},
      "expedition2": {...}
    }
    Do not invent any information not present in the answer text. Use null for missing scalar fields and [] for missing URL lists.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s2 = unicodedata.normalize("NFKD", s)
    s2 = s2.encode("ascii", "ignore").decode("ascii")
    s2 = re.sub(r"[\W_]+", "", s2.lower())
    return s2.strip()


def parse_duration_to_seconds(duration: Optional[str]) -> Optional[int]:
    if not duration:
        return None
    s = duration.strip().lower()

    # Handle HH:MM:SS or MM:SS
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b", s)
    if m:
        parts = [int(x) for x in m.groups(default="0")]
        if m.group(3) is not None:  # HH:MM:SS
            h, mm, ss = parts
            return h * 3600 + mm * 60 + ss
        else:  # MM:SS
            mm, ss = parts[:2]
            return mm * 60 + ss

    # Sum unit-based segments (e.g., "1m 20s", "1.5 minutes", "90 sec")
    total_seconds = 0.0
    found = False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)", s):
        found = True
        val = float(num)
        unit = unit[0]  # ensure string
        if unit.startswith("h"):
            total_seconds += val * 3600
        elif unit.startswith("m"):
            total_seconds += val * 60
        elif unit.startswith("s"):
            total_seconds += val
    if found:
        return int(round(total_seconds))

    # Fallback single number possibly followed by 'sec'/'second' somewhere
    m2 = re.search(r"\b(\d+(?:\.\d+)?)\b\s*(seconds?|secs?)", s)
    if m2:
        return int(round(float(m2.group(1))))

    # e.g., "about 1 minute" without explicit number-unit match earlier
    if "minute" in s and re.search(r"\b1\b", s):
        return 60
    if "minute" in s:
        m3 = re.search(r"\b(\d+(?:\.\d+)?)\b", s)
        if m3:
            return int(round(float(m3.group(1)) * 60))

    return None


def sources_present(urls: Optional[List[str]]) -> bool:
    return bool(urls and len(urls) > 0)


# --------------------------------------------------------------------------- #
# Verification logic per expedition                                           #
# --------------------------------------------------------------------------- #
async def verify_expedition(
    evaluator: Evaluator,
    parent_node,
    exp: ExpeditionInfo,
    index: int,
    other_exp: Optional[ExpeditionInfo] = None,
):
    """
    Build the verification sub-tree for one expedition and run verifications.
    index is 1-based (1 or 2).
    """
    exp_id = f"expedition_{index}"
    exp_node = evaluator.add_parallel(
        id=exp_id,
        desc=("First university-organized eclipse expedition to mainland Spain"
              if index == 1 else
              "Second university-organized eclipse expedition to mainland Spain"),
        parent=parent_node,
        critical=False  # Allow partial across expeditions at root level
    )

    urls = exp.source_urls or []

    # -------------------- Viewing location group ------------------------- #
    vl_node = evaluator.add_parallel(
        id=f"{exp_id}_viewing_location",
        desc=("Viewing location details for the first expedition"
              if index == 1 else "Viewing location details for the second expedition"),
        parent=exp_node,
        critical=True
    )
    # Group-level "sources present" gate (local precondition for this group)
    vl_sources_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.viewing_site_name and exp.viewing_site_name.strip()),
        id=f"{exp_id}_viewing_location_sources_gate",
        desc="Viewing location: source URL(s) provided and specific site name present in answer",
        parent=vl_node,
        critical=True
    )

    # 1) Specific named site (not just city/region)
    specific_site_leaf = evaluator.add_leaf(
        id=f"{exp_id}_specific_site",
        desc="A specific, named viewing site (not just a city or region) is identified and supported by a reference URL",
        parent=vl_node,
        critical=True
    )
    claim = f"The viewing site for the expedition is a specific, named site (not just a city/region): '{exp.viewing_site_name}'."
    await evaluator.verify(
        claim=claim,
        node=specific_site_leaf,
        sources=urls,
        additional_instruction=(
            "Judge whether the named location is a specific venue/landmark (e.g., park, campus, stadium, "
            "observatory, lookout, plaza), not merely a city or broad region. "
            "Confirm this with the provided webpage(s)."
        ),
        extra_prerequisites=[vl_sources_gate]
    )

    # 2) Site confirmed to be in mainland Spain
    spain_loc_leaf = evaluator.add_leaf(
        id=f"{exp_id}_spain_location",
        desc="The viewing site is confirmed to be in mainland Spain and supported by a reference URL",
        parent=vl_node,
        critical=True
    )
    claim = f"The viewing site '{exp.viewing_site_name}' is located in mainland (peninsular) Spain."
    await evaluator.verify(
        claim=claim,
        node=spain_loc_leaf,
        sources=urls,
        additional_instruction=(
            "Treat 'mainland Spain' as the Iberian Peninsula portion (excluding Canary Islands, Balearic Islands, Ceuta, and Melilla). "
            "Confirm the site's location within mainland Spain using the provided webpages."
        ),
        extra_prerequisites=[vl_sources_gate]
    )

    # 3) Province identified
    province_leaf = evaluator.add_leaf(
        id=f"{exp_id}_province",
        desc="The Spanish province where the viewing site is located is identified and supported by a reference URL",
        parent=vl_node,
        critical=True
    )
    claim = f"The viewing site '{exp.viewing_site_name}' is located in the Spanish province of '{exp.province}'."
    await evaluator.verify(
        claim=claim,
        node=province_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the named site lies within the specified Spanish 'provincia'. "
            "Allow minor naming variants (e.g., with/without 'Province of')."
        ),
        extra_prerequisites=[vl_sources_gate]
    )

    # 4) Path of totality
    path_leaf = evaluator.add_leaf(
        id=f"{exp_id}_totality_path",
        desc="The viewing site is confirmed to be within the path of totality and supported by a reference URL",
        parent=vl_node,
        critical=True
    )
    claim = f"The viewing site '{exp.viewing_site_name}' is within the path of totality for the August 12, 2026 total solar eclipse."
    await evaluator.verify(
        claim=claim,
        node=path_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm using the provided webpages (e.g., authoritative eclipse maps such as NASA, timeanddate, or the expedition page) "
            "that this specific site lies within the totality path for the 12 Aug 2026 eclipse."
        ),
        extra_prerequisites=[vl_sources_gate]
    )

    # 5) Totality duration (group)
    dur_node = evaluator.add_sequential(
        id=f"{exp_id}_totality_duration",
        desc="The expected duration of totality at the viewing site is stated and supported by a reference URL",
        parent=vl_node,
        critical=True
    )
    dur_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.totality_duration and exp.totality_duration.strip()),
        id=f"{exp_id}_totality_duration_gate",
        desc="Totality duration: source URL(s) provided and duration text present in answer",
        parent=dur_node,
        critical=True
    )
    duration_leaf = evaluator.add_leaf(
        id=f"{exp_id}_totality_duration_stated",
        desc="The expected duration of totality at the viewing site is stated and supported by a reference URL",
        parent=dur_node,
        critical=True
    )
    claim = f"The expected duration of totality at '{exp.viewing_site_name}' is '{exp.totality_duration}'."
    await evaluator.verify(
        claim=claim,
        node=duration_leaf,
        sources=urls,
        additional_instruction=(
            "Verify that the provided webpages explicitly state this totality duration (allowing small rounding differences). "
            "If the page gives an equivalent value (e.g., 80 seconds vs 1m20s), consider it supported."
        ),
        extra_prerequisites=[dur_gate]
    )
    # Minimum 1 minute (custom)
    seconds = parse_duration_to_seconds(exp.totality_duration)
    min_ok = seconds is not None and seconds >= 60
    evaluator.add_custom_node(
        result=min_ok,
        id=f"{exp_id}_duration_minimum",
        desc="The stated totality duration is at least 1 minute",
        parent=dur_node,
        critical=True
    )

    # ----------------------- University affiliation ---------------------- #
    univ_node = evaluator.add_parallel(
        id=f"{exp_id}_university",
        desc=("University affiliation for the first expedition"
              if index == 1 else "University affiliation for the second expedition"),
        parent=exp_node,
        critical=True
    )
    univ_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.university and exp.university.strip()),
        id=f"{exp_id}_university_gate",
        desc="University: source URL(s) provided and university name present in answer",
        parent=univ_node,
        critical=True
    )

    univ_name_leaf = evaluator.add_leaf(
        id=f"{exp_id}_university_name",
        desc="The organizing or affiliated university/college is identified by name and supported by a reference URL",
        parent=univ_node,
        critical=True
    )
    claim = f"The expedition is organized by or officially affiliated with the university/college '{exp.university}'."
    await evaluator.verify(
        claim=claim,
        node=univ_name_leaf,
        sources=urls,
        additional_instruction=(
            "Look for explicit statements on the provided webpages that the expedition is organized by or officially affiliated "
            "with the named institution (e.g., listed as organizer, hosted by the university, or run by a university department)."
        ),
        extra_prerequisites=[univ_gate]
    )

    accred_leaf = evaluator.add_leaf(
        id=f"{exp_id}_accreditation",
        desc="The institution is confirmed to be an accredited university or college, supported by a reference URL or the institution's official website",
        parent=univ_node,
        critical=True
    )
    claim = f"The institution '{exp.university}' is an accredited university or college."
    await evaluator.verify(
        claim=claim,
        node=accred_leaf,
        sources=urls,
        additional_instruction=(
            "From the provided webpages (including the institution's official site if present), determine if the institution is a bona fide, accredited "
            "university/college (e.g., recognized by relevant national/regional accreditation frameworks). If this cannot be confirmed from the pages, mark unsupported."
        ),
        extra_prerequisites=[univ_gate]
    )

    # --------------------------- Expert leadership ----------------------- #
    expert_node = evaluator.add_parallel(
        id=f"{exp_id}_expert",
        desc=("Expert leadership for the first expedition"
              if index == 1 else "Expert leadership for the second expedition"),
        parent=exp_node,
        critical=True
    )
    expert_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.expert_name and exp.expert_name.strip()),
        id=f"{exp_id}_expert_gate",
        desc="Expert: source URL(s) provided and expert name present in answer",
        parent=expert_node,
        critical=True
    )

    expert_name_leaf = evaluator.add_leaf(
        id=f"{exp_id}_expert_name",
        desc="At least one astronomy expert (professor, astronomer, or planetarium director) leading or speaking at the expedition is identified by name and supported by a reference URL",
        parent=expert_node,
        critical=True
    )
    claim = f"An astronomy expert named '{exp.expert_name}' is leading or speaking at the expedition."
    await evaluator.verify(
        claim=claim,
        node=expert_name_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the person's role (e.g., leading the expedition, giving a talk) from the provided webpages. "
            "Accept roles like professor, astronomer, astrophysicist, or planetarium director."
        ),
        extra_prerequisites=[expert_gate]
    )

    # Credentials subgroup
    cred_node = evaluator.add_parallel(
        id=f"{exp_id}_credentials",
        desc="The expert's academic credentials or position are specified and supported by a reference URL",
        parent=expert_node,
        critical=True
    )
    cred_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.expert_credentials and exp.expert_credentials.strip()),
        id=f"{exp_id}_credentials_gate",
        desc="Credentials: source URL(s) provided and credentials text present in answer",
        parent=cred_node,
        critical=True
    )

    cred_stated_leaf = evaluator.add_leaf(
        id=f"{exp_id}_credentials_stated",
        desc="The expert's academic credentials or position are specified and supported by a reference URL",
        parent=cred_node,
        critical=True
    )
    claim = f"The expert '{exp.expert_name}' holds the position/credentials: '{exp.expert_credentials}'."
    await evaluator.verify(
        claim=claim,
        node=cred_stated_leaf,
        sources=urls,
        additional_instruction=(
            "Verify on the provided webpages that the expert is described with the stated academic role/credential "
            "(e.g., Professor of Astronomy, Assistant Professor, Research Scientist, Astronomer, Planetarium Director)."
        ),
        extra_prerequisites=[cred_gate]
    )

    cred_verify_leaf = evaluator.add_leaf(
        id=f"{exp_id}_credentials_verification",
        desc="The credentials include a professorship, research position, or equivalent academic role",
        parent=cred_node,
        critical=True
    )
    claim = (
        f"The role '{exp.expert_credentials}' is a professorship, research position, or equivalent academic role "
        f"(e.g., professor, assistant professor, associate professor, lecturer in astronomy, research scientist, astronomer, "
        f"planetarium director)."
    )
    await evaluator.verify(
        claim=claim,
        node=cred_verify_leaf,
        additional_instruction=(
            "Classify based on the provided role text only; allow reasonable equivalents. "
            "Return Correct only if the role clearly fits as a professorship or academic/research-equivalent role."
        ),
        # This check is a logical classification based on the provided text; no URLs required.
    )

    # ----------------------- Educational programming --------------------- #
    # Wrap education in its own group to avoid gating affecting siblings
    edu_node = evaluator.add_parallel(
        id=f"{exp_id}_education_main",
        desc="Educational programming group",
        parent=exp_node,
        critical=True
    )
    edu_gate = evaluator.add_custom_node(
        result=sources_present(urls) and bool(exp.education_programming and exp.education_programming.strip()),
        id=f"{exp_id}_education_gate",
        desc="Education: source URL(s) provided and programming text present in answer",
        parent=edu_node,
        critical=True
    )
    edu_leaf = evaluator.add_leaf(
        id=f"{exp_id}_education",
        desc="The expedition includes educational programming such as lectures, astronomy talks, or stargazing activities, supported by a reference URL",
        parent=edu_node,
        critical=True
    )
    claim = f"The expedition includes educational programming such as '{exp.education_programming}'."
    await evaluator.verify(
        claim=claim,
        node=edu_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the presence of educational programming (e.g., lectures, talks, workshops, guided stargazing) "
            "from the provided webpages."
        ),
        extra_prerequisites=[edu_gate]
    )

    # ------------------------------ Dates -------------------------------- #
    dates_node = evaluator.add_parallel(
        id=f"{exp_id}_dates_main",
        desc="Dates group",
        parent=exp_node,
        critical=True
    )
    dates_gate = evaluator.add_custom_node(
        result=sources_present(urls),
        id=f"{exp_id}_dates_gate",
        desc="Dates: source URL(s) provided",
        parent=dates_node,
        critical=True
    )
    dates_leaf = evaluator.add_leaf(
        id=f"{exp_id}_dates",
        desc="The expedition dates include August 12, 2026, supported by a reference URL",
        parent=dates_node,
        critical=True
    )
    claim = f"The expedition dates include {ECLIPSE_DATE_TEXT}."
    await evaluator.verify(
        claim=claim,
        node=dates_leaf,
        sources=urls,
        additional_instruction=(
            "Check the provided webpages for the expedition schedule or date range. "
            "If the event spans multiple days (e.g., Aug 10–14, 2026), consider it as including Aug 12, 2026. "
            "Also accept Spanish date formats such as '12 de agosto de 2026'."
        ),
        extra_prerequisites=[dates_gate]
    )

    # ------------------------ Distinctness (expedition 2) ---------------- #
    if index == 2 and other_exp is not None:
        distinct_leaf = evaluator.add_custom_node(
            result=(
                (normalize_text(exp.viewing_site_name) and normalize_text(other_exp.viewing_site_name) and
                 normalize_text(exp.viewing_site_name) != normalize_text(other_exp.viewing_site_name))
                or
                (normalize_text(exp.university) and normalize_text(other_exp.university) and
                 normalize_text(exp.university) != normalize_text(other_exp.university))
            ),
            id=f"{exp_id}_distinctness",
            desc="The second expedition has either a different viewing site or a different organizing university from the first expedition",
            parent=exp_node,
            critical=True
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
    Evaluate an answer against the rubric for two university-organized eclipse expeditions to mainland Spain.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: expeditions evaluated in parallel
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify two distinct university-organized expeditions to mainland Spain for viewing the August 12, 2026 total solar eclipse, each meeting all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # IMPORTANT: root must be non-critical to satisfy framework constraint
    # (critical parent cannot have non-critical children). We enforce strictness
    # with critical sub-nodes inside each expedition instead.
    root.critical = False

    # 1) Extract structured expeditions info
    extracted = await evaluator.extract(
        prompt=prompt_extract_expeditions(),
        template_class=ExpeditionsExtraction,
        extraction_name="expeditions_extraction",
    )

    exp1 = extracted.expedition1 or ExpeditionInfo()
    exp2 = extracted.expedition2 or ExpeditionInfo()

    # 2) Build verification tree and verify
    await verify_expedition(evaluator, root, exp1, index=1, other_exp=None)
    await verify_expedition(evaluator, root, exp2, index=2, other_exp=exp1)

    # Optional: record some parsed helper info
    evaluator.add_custom_info(
        info={
            "expedition1_duration_seconds": parse_duration_to_seconds(exp1.totality_duration),
            "expedition2_duration_seconds": parse_duration_to_seconds(exp2.totality_duration),
            "expedition1_url_count": len(exp1.source_urls or []),
            "expedition2_url_count": len(exp2.source_urls or []),
        },
        info_type="derived_metrics",
        info_name="parsed_duration_and_url_counts"
    )

    # 3) Return evaluator summary
    return evaluator.get_summary()