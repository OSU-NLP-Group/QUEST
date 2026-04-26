import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "artemis_ii_crew_profiles"
TASK_DESCRIPTION = """
NASA's Artemis II mission, scheduled to launch in April 2026, will be the first crewed lunar flyby in over 50 years. The mission will last approximately 10 days and carry four astronauts around the Moon.

Compile comprehensive professional profiles for each of the four astronauts selected for this historic mission. For each astronaut, provide the following information:

1. Their full name
2. Their specific role on the Artemis II mission (Commander, Pilot, or Mission Specialist)
3. Their nationality
4. The year they were selected as an astronaut by their respective space agency (NASA or Canadian Space Agency)
5. One previous spaceflight mission they participated in (if any), or indicate if they are a rookie astronaut with no previous spaceflights
6. At least one academic degree they hold, including the name of the institution that granted it

All information must be verifiable through official NASA, Canadian Space Agency, or other reputable sources.
""".strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AstronautExtraction(BaseModel):
    full_name: FieldWithSources = Field(default_factory=FieldWithSources)
    role: FieldWithSources = Field(default_factory=FieldWithSources)
    nationality: FieldWithSources = Field(default_factory=FieldWithSources)
    selection_year: FieldWithSources = Field(default_factory=FieldWithSources)
    previous_spaceflight: FieldWithSources = Field(default_factory=FieldWithSources)
    degree: FieldWithSources = Field(default_factory=FieldWithSources)


class CrewExtraction(BaseModel):
    commander: AstronautExtraction = Field(default_factory=AstronautExtraction)
    pilot: AstronautExtraction = Field(default_factory=AstronautExtraction)
    mission_specialist_1: AstronautExtraction = Field(default_factory=AstronautExtraction)
    mission_specialist_2: AstronautExtraction = Field(default_factory=AstronautExtraction)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_crew_profiles() -> str:
    return """
    Extract structured profiles for the four Artemis II astronauts as presented in the answer text. For each role (Commander, Pilot, Mission Specialist 1, Mission Specialist 2), return the following fields, each with a value and the list of URL sources explicitly cited in the answer text that support that field.

    IMPORTANT:
    - Only use URLs explicitly present in the answer text. If the answer uses markdown-style links (e.g., [text](url)), extract the actual URL.
    - Prefer official NASA (.nasa.gov), Canadian Space Agency (.asc-csa.gc.ca), and other reputable sources (e.g., well-known major news or agency pages). Still, extract all URLs that the answer cites for the field, regardless of domain.
    - If a value is provided but there are no explicit URLs cited in the answer for that field, set sources to an empty list.
    - For "previous_spaceflight", if the answer states the astronaut is a rookie (no flights), set value to a clear phrase like "rookie" or "no previous spaceflight".
    - For "degree", ensure the value includes both the degree and the granting institution in a single string (e.g., "M.S. in Systems Engineering — Johns Hopkins University"). If multiple degrees are listed, pick one representative example that includes an institution.

    Return a JSON object with this structure (nulls allowed when missing):
    {
      "commander": {
        "full_name": {"value": str|null, "sources": [urls...]},
        "role": {"value": str|null, "sources": [urls...]},
        "nationality": {"value": str|null, "sources": [urls...]},
        "selection_year": {"value": str|null, "sources": [urls...]},
        "previous_spaceflight": {"value": str|null, "sources": [urls...]},
        "degree": {"value": str|null, "sources": [urls...]}
      },
      "pilot": {
        "full_name": {"value": str|null, "sources": [urls...]},
        "role": {"value": str|null, "sources": [urls...]},
        "nationality": {"value": str|null, "sources": [urls...]},
        "selection_year": {"value": str|null, "sources": [urls...]},
        "previous_spaceflight": {"value": str|null, "sources": [urls...]},
        "degree": {"value": str|null, "sources": [urls...]}
      },
      "mission_specialist_1": {
        "full_name": {"value": str|null, "sources": [urls...]},
        "role": {"value": str|null, "sources": [urls...]},
        "nationality": {"value": str|null, "sources": [urls...]},
        "selection_year": {"value": str|null, "sources": [urls...]},
        "previous_spaceflight": {"value": str|null, "sources": [urls...]},
        "degree": {"value": str|null, "sources": [urls...]}
      },
      "mission_specialist_2": {
        "full_name": {"value": str|null, "sources": [urls...]},
        "role": {"value": str|null, "sources": [urls...]},
        "nationality": {"value": str|null, "sources": [urls...]},
        "selection_year": {"value": str|null, "sources": [urls...]},
        "previous_spaceflight": {"value": str|null, "sources": [urls...]},
        "degree": {"value": str|null, "sources": [urls...]}
      }
    }

    Disambiguation and ordering:
    - If the answer lists two Mission Specialists without numbering, assign them to mission_specialist_1 and mission_specialist_2 in the order they appear.
    - Normalize obvious variants (e.g., "U.S." vs "United States" vs "American") in values only if the answer makes them clear. Do not invent information.

    Return exactly this JSON structure following the model definition.
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def gather_all_sources(profile: AstronautExtraction) -> List[str]:
    all_urls = []
    all_urls.extend(profile.full_name.sources or [])
    all_urls.extend(profile.role.sources or [])
    all_urls.extend(profile.nationality.sources or [])
    all_urls.extend(profile.selection_year.sources or [])
    all_urls.extend(profile.previous_spaceflight.sources or [])
    all_urls.extend(profile.degree.sources or [])
    return _dedup_urls(all_urls)


def pick_sources(primary: List[str], fallback: List[str]) -> List[str]:
    return _dedup_urls(primary if primary else fallback)


def is_rookie_statement(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    patterns = [
        "rookie",
        "no previous spaceflight",
        "no prior spaceflight",
        "has not flown",
        "no spaceflight",
        "never flown",
        "no prior flights",
        "no previous flights",
    ]
    return any(pat in t for pat in patterns)


def extract_first_year(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    m = re.search(r"(19|20)\d{2}", value)
    return m.group(0) if m else None


def norm(s: Optional[str]) -> str:
    return (s or "").strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_astronaut_profile(
    evaluator: Evaluator,
    parent_root,
    role_key: str,
    profile: AstronautExtraction,
) -> None:
    """
    Build the verification subtree for a single Artemis II astronaut profile.
    role_key in {"commander", "pilot", "mission_specialist_1", "mission_specialist_2"}.
    """
    role_label_map = {
        "commander": "Commander",
        "pilot": "Pilot",
        "mission_specialist_1": "Mission Specialist",
        "mission_specialist_2": "Mission Specialist",
    }
    # Parent profile node IDs and descriptions from rubric
    profile_meta = {
        "commander": ("Commander_Profile",
                      "Complete profile for the astronaut serving as Commander on Artemis II, including full name, role, nationality, selection year, previous spaceflight experience, and academic credentials"),
        "pilot": ("Pilot_Profile",
                  "Complete profile for the astronaut serving as Pilot on Artemis II, including full name, role, nationality, selection year, previous spaceflight experience, and academic credentials"),
        "mission_specialist_1": ("Mission_Specialist_1_Profile",
                                 "Complete profile for the first Mission Specialist on Artemis II, including full name, role, nationality, selection year, previous spaceflight experience, and academic credentials"),
        "mission_specialist_2": ("Mission_Specialist_2_Profile",
                                 "Complete profile for the second Mission Specialist on Artemis II, including full name, role, nationality, selection year, previous spaceflight experience, and academic credentials"),
    }

    leaf_ids = {
        "commander": {
            "full_name": "Commander_Full_Name",
            "role": "Commander_Mission_Role",
            "nationality": "Commander_Nationality",
            "selection_year": "Commander_Selection_Year",
            "previous": "Commander_Previous_Spaceflight",
            "degree": "Commander_Academic_Degree",
        },
        "pilot": {
            "full_name": "Pilot_Full_Name",
            "role": "Pilot_Mission_Role",
            "nationality": "Pilot_Nationality",
            "selection_year": "Pilot_Selection_Year",
            "previous": "Pilot_Previous_Spaceflight",
            "degree": "Pilot_Academic_Degree",
        },
        "mission_specialist_1": {
            "full_name": "Mission_Specialist_1_Full_Name",
            "role": "Mission_Specialist_1_Role",
            "nationality": "Mission_Specialist_1_Nationality",
            "selection_year": "Mission_Specialist_1_Selection_Year",
            "previous": "Mission_Specialist_1_Previous_Spaceflight",
            "degree": "Mission_Specialist_1_Academic_Degree",
        },
        "mission_specialist_2": {
            "full_name": "Mission_Specialist_2_Full_Name",
            "role": "Mission_Specialist_2_Role",
            "nationality": "Mission_Specialist_2_Nationality",
            "selection_year": "Mission_Specialist_2_Selection_Year",
            "previous": "Mission_Specialist_2_Previous_Experience",
            "degree": "Mission_Specialist_2_Academic_Degree",
        },
    }

    role_label = role_label_map[role_key]
    node_id, node_desc = profile_meta[role_key]
    profile_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_root,
        critical=False,
    )

    # Reusable union of sources per astronaut (if field-level sources are missing)
    all_sources = gather_all_sources(profile)

    # ---------- 1) Full Name ----------
    name_value = norm(profile.full_name.value)
    name_sources = pick_sources(profile.full_name.sources, all_sources)
    name_exist_node = evaluator.add_custom_node(
        result=bool(name_value) and bool(name_sources),
        id=f"{leaf_ids[role_key]['full_name']}_exists",
        desc=f"{role_label} full name value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    name_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["full_name"],
        desc=f"Full name of the {role_label} as officially announced by NASA/CSA or other reputable sources",
        parent=profile_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The astronaut's full name is '{name_value}'.",
        node=name_leaf,
        sources=name_sources,
        additional_instruction="Verify that the cited page clearly shows this person's official full name. Prefer NASA or CSA pages; allow minor variations (middle initials, capitalization).",
    )

    # ---------- 2) Mission Role ----------
    role_value = norm(profile.role.value)
    role_sources = pick_sources(profile.role.sources, all_sources)
    role_exist_node = evaluator.add_custom_node(
        result=bool(role_value) and bool(role_sources),
        id=f"{leaf_ids[role_key]['role']}_exists",
        desc=f"{role_label} mission role value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    role_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["role"],
        desc=f"Correct identification of the {role_label} role for this crew member",
        parent=profile_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_value} is assigned as the {role_label} for NASA's Artemis II mission.",
        node=role_leaf,
        sources=role_sources,
        additional_instruction="The source should explicitly tie the person to the Artemis II crew and state their role (Commander, Pilot, or Mission Specialist). Allow minor wording differences like 'will serve as' vs 'is the'."
    )

    # ---------- 3) Nationality ----------
    nat_value = norm(profile.nationality.value)
    nat_sources = pick_sources(profile.nationality.sources, all_sources)
    nat_exist_node = evaluator.add_custom_node(
        result=bool(nat_value) and bool(nat_sources),
        id=f"{leaf_ids[role_key]['nationality']}_exists",
        desc=f"{role_label} nationality value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    nat_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["nationality"],
        desc=f"Nationality ({'United States or Canada'}) of the {role_label}",
        parent=profile_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_value}'s nationality is '{nat_value}'.",
        node=nat_leaf,
        sources=nat_sources,
        additional_instruction="Confirm the astronaut's nationality/citizenship. Treat 'American/USA/United States/U.S.' as equivalent and 'Canadian/Canada' as equivalent."
    )

    # ---------- 4) Selection Year ----------
    sel_value_raw = norm(profile.selection_year.value)
    sel_year = extract_first_year(sel_value_raw) or sel_value_raw
    sel_sources = pick_sources(profile.selection_year.sources, all_sources)
    sel_exist_node = evaluator.add_custom_node(
        result=bool(sel_value_raw) and bool(sel_sources),
        id=f"{leaf_ids[role_key]['selection_year']}_exists",
        desc=f"{role_label} selection year value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    sel_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["selection_year"],
        desc=f"Year the {role_label} was selected as an astronaut by NASA or CSA",
        parent=profile_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_value} was selected as an astronaut in {sel_year}.",
        node=sel_leaf,
        sources=sel_sources,
        additional_instruction="Verify the astronaut selection year by NASA (for U.S. astronauts) or CSA (for Canadian astronauts). Accept phrasings like 'selected in 2013' or '2013 astronaut class'."
    )

    # ---------- 5) Previous Spaceflight / Rookie ----------
    prev_value = norm(profile.previous_spaceflight.value)
    prev_sources = pick_sources(profile.previous_spaceflight.sources, all_sources)
    prev_exist_node = evaluator.add_custom_node(
        result=bool(prev_value) and bool(prev_sources),
        id=f"{leaf_ids[role_key]['previous']}_exists",
        desc=f"{role_label} previous spaceflight/rookie value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    prev_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["previous"],
        desc=f"Previous spaceflight experience (or rookie status) for the {role_label}",
        parent=profile_node,
        critical=True
    )
    if is_rookie_statement(prev_value):
        prev_claim = f"{name_value} has not yet flown in space (rookie astronaut) as of their Artemis II assignment."
        prev_addins = "Confirm the astronaut has no prior spaceflights. Accept clear statements like 'rookie', 'has not flown in space', or equivalent."
    else:
        prev_claim = f"{name_value} previously flew on the mission '{prev_value}'."
        prev_addins = "Verify that the cited mission (e.g., ISS Expedition, Soyuz, or SpaceX Crew mission) is indeed part of this astronaut's flight history. Minor naming variations are acceptable."
    await evaluator.verify(
        claim=prev_claim,
        node=prev_leaf,
        sources=prev_sources,
        additional_instruction=prev_addins
    )

    # ---------- 6) Academic Degree ----------
    deg_value = norm(profile.degree.value)
    deg_sources = pick_sources(profile.degree.sources, all_sources)
    deg_exist_node = evaluator.add_custom_node(
        result=bool(deg_value) and bool(deg_sources),
        id=f"{leaf_ids[role_key]['degree']}_exists",
        desc=f"{role_label} degree value and at least one source are provided",
        parent=profile_node,
        critical=True
    )
    deg_leaf = evaluator.add_leaf(
        id=leaf_ids[role_key]["degree"],
        desc=f"At least one academic degree for the {role_label} including institution name",
        parent=profile_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{name_value} holds the academic credential: {deg_value}.",
        node=deg_leaf,
        sources=deg_sources,
        additional_instruction="Confirm both the degree and the granting institution are supported by the source (e.g., NASA/CSA bio or official profile). Allow minor wording differences."
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
    Evaluate an answer for the Artemis II Crew Profiles task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # profiles evaluated independently
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Comprehensive professional profiles for all four astronauts assigned to NASA's Artemis II mission",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extraction
    crew = await evaluator.extract(
        prompt=prompt_extract_crew_profiles(),
        template_class=CrewExtraction,
        extraction_name="crew_profiles",
    )

    # Build tree according to rubric
    # Root already set; child profiles are created inside verify_astronaut_profile

    # Verify each of the four profiles
    await verify_astronaut_profile(evaluator, root, "commander", crew.commander)
    await verify_astronaut_profile(evaluator, root, "pilot", crew.pilot)
    await verify_astronaut_profile(evaluator, root, "mission_specialist_1", crew.mission_specialist_1)
    await verify_astronaut_profile(evaluator, root, "mission_specialist_2", crew.mission_specialist_2)

    return evaluator.get_summary()