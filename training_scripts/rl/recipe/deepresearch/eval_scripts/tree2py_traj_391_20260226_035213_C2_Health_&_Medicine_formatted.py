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
TASK_ID = "nipah_guidelines_wb_2026"
TASK_DESCRIPTION = (
    "According to the official guidelines for managing the January 2026 Nipah virus outbreak in West Bengal, India, "
    "what is the required duration for monitoring close contacts of confirmed cases, and what are the mandatory "
    "infection prevention and control measures that healthcare workers must implement when caring for suspected or "
    "confirmed Nipah patients?"
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class IPCMeasuresExtraction(BaseModel):
    contact_droplet_precautions_text: Optional[str] = None
    ppe_items: List[str] = Field(default_factory=list)
    airborne_precautions_text: Optional[str] = None
    respirator_mentioned: Optional[str] = None
    isolation_room_mentioned: Optional[str] = None


class NipahGuidelineExtraction(BaseModel):
    attribution_text: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)
    monitoring_duration_text: Optional[str] = None
    ipc: Optional[IPCMeasuresExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nipah_guideline() -> str:
    return """
Extract the following information exactly as it appears in the answer:

1) attribution_text: A short quote or summary phrase (if present) where the answer attributes its information to official guidelines or authorities
   such as India NCDC (National Centre for Disease Control), India's Ministry of Health and Family Welfare (MoHFW),
   Government of West Bengal Health Department, WHO, or ICMR. If not present, return null.

2) source_urls: A list of all URLs explicitly included in the answer text (including markdown-style links).
   Only include valid URLs; do not invent URLs.

3) monitoring_duration_text: The exact phrase indicating the duration for following up or monitoring close contacts
   (e.g., "21 days", "21-day monitoring"). If not present, return null.

4) ipc:
   - contact_droplet_precautions_text: A snippet that describes contact and droplet precautions for routine care of suspected/confirmed Nipah patients. If not present, return null.
   - ppe_items: A list of PPE items explicitly listed for routine care under contact/droplet precautions (e.g., "medical mask", "surgical mask", "eye protection", "goggles", "face shield", "fluid-resistant gown", "impermeable gown", "examination gloves", "non-sterile gloves").
   - airborne_precautions_text: A snippet describing airborne precautions during aerosol-generating procedures (AGPs). If not present, return null.
   - respirator_mentioned: The exact words for the respirator used for AGPs (e.g., "fit-tested N95", "FFP2", "FFP3", "filtering facepiece respirator"). If not present, return null.
   - isolation_room_mentioned: The exact term for the isolation room for AGPs (e.g., "airborne infection isolation room", "AIIR", "negative pressure room"). If not present, return null.

Important:
- Do not add information not present in the answer.
- Return null for any field that is not explicitly present.
"""


# --------------------------------------------------------------------------- #
# Helper: official domains (recorded for debugging; not used for gating)      #
# --------------------------------------------------------------------------- #
def _is_official_domain(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    url_l = url.lower()
    official_fragments = [
        "who.int",
        "ncdc.gov.in",
        "mohfw.gov.in",
        "nhm.gov.in",
        "icmr.gov.in",
        "wbhealth.gov.in",
        "health.wb.gov.in",
        "wbsdma.gov.in",
        ".gov.in",
    ]
    return any(frag in url_l for frag in official_fragments)


# --------------------------------------------------------------------------- #
# Build verification tree and run checks                                      #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    extraction: NipahGuidelineExtraction,
) -> None:
    """
    Build the rubric verification tree and perform checks following the provided JSON rubric.
    """

    # Top-level evaluation node (critical, parallel aggregation)
    top_node = evaluator.add_parallel(
        id="NipahGuidelineAnswer",
        desc="Evaluate whether the answer provides the required close-contact monitoring duration and the mandatory IPC measures per official guidelines for the January 2026 Nipah outbreak in West Bengal, India.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) OfficialGuidelineAttribution (critical leaf)
    node_attr = evaluator.add_leaf(
        id="OfficialGuidelineAttribution",
        desc="Attributes the information to official guidelines (e.g., India NCDC and/or WHO) or otherwise clearly indicates it is drawn from official guidance.",
        parent=top_node,
        critical=True,
    )
    # Simple verification: judge from the answer text itself.
    # Allow credit if it explicitly cites official bodies/guidelines or includes official URLs.
    attribution_hint = extraction.attribution_text or ""
    urls_hint = ", ".join(extraction.source_urls[:5]) if extraction.source_urls else ""
    await evaluator.verify(
        claim=(
            "The answer explicitly attributes its information to official guidelines or authorities "
            "(e.g., India's NCDC, India's MoHFW, Government of West Bengal Health Department, WHO, or ICMR) "
            "by naming them or by clearly stating it is based on official guidance."
        ),
        node=node_attr,
        additional_instruction=(
            "Judge based on the answer text. Consider the claim satisfied if the answer names official bodies "
            "like 'NCDC', 'MoHFW', 'WHO', 'Government of West Bengal', 'ICMR', or explicitly says "
            "'according to official guidelines'. Also treat the presence of official-looking URLs as supportive context "
            f"even though you primarily judge from the answer text. Hints from extraction: '{attribution_hint}'. "
            f"Possible URLs present: {urls_hint if urls_hint else 'None'}."
        ),
    )

    # 2) CloseContactMonitoringDuration (critical leaf)
    node_duration = evaluator.add_leaf(
        id="CloseContactMonitoringDuration",
        desc="States the required duration for monitoring/following up close contacts of confirmed cases (21 days).",
        parent=top_node,
        critical=True,
    )
    # Simple verification from answer text: must state 21-day monitoring
    await evaluator.verify(
        claim=(
            "The answer states that close contacts of confirmed Nipah cases should be monitored or followed up for 21 days."
        ),
        node=node_duration,
        additional_instruction=(
            "Focus on the answer text. Accept reasonable phrasings such as 'monitoring for 21 days', '21-day active "
            "surveillance', 'twenty-one days', or 'daily monitoring for 21 days'. Reject if the answer gives a different "
            "duration or omits a duration."
        ),
    )

    # 3) InfectionPreventionAndControlMeasures (critical, parallel aggregation)
    node_ipc = evaluator.add_parallel(
        id="InfectionPreventionAndControlMeasures",
        desc="States the mandatory IPC measures for healthcare workers caring for suspected/confirmed Nipah patients.",
        parent=top_node,
        critical=True,
    )

    # 3.a) ContactAndDropletPrecautions (critical leaf)
    node_contact_droplet = evaluator.add_leaf(
        id="ContactAndDropletPrecautions",
        desc="Includes contact + droplet precautions with required PPE elements (well-fitting medical mask, eye protection, fluid-resistant gown, and examination gloves).",
        parent=node_ipc,
        critical=True,
    )
    # Simple verification from answer text for presence of contact+droplet plus the four PPE elements (allow synonyms).
    ppe_hint = ", ".join(extraction.ipc.ppe_items) if extraction and extraction.ipc and extraction.ipc.ppe_items else "None"
    contact_droplet_hint = extraction.ipc.contact_droplet_precautions_text if extraction and extraction.ipc else ""
    await evaluator.verify(
        claim=(
            "The answer includes contact and droplet precautions for routine care and explicitly lists all of the following PPE: "
            "a well-fitting medical (surgical) mask, eye protection (goggles or face shield), a fluid-resistant (or impermeable) gown, "
            "and examination (non-sterile) gloves."
        ),
        node=node_contact_droplet,
        additional_instruction=(
            "Judge from the answer text. Accept synonymous wording: 'medical mask' or 'surgical mask'; "
            "'eye protection' can be 'goggles' or 'face shield'; 'fluid-resistant' can be 'impermeable' gown; "
            "'examination gloves' can be 'non-sterile gloves'. All four PPE categories must be present, "
            "and contact + droplet precautions must be stated for routine care. "
            f"Hints from extraction -> PPE items: {ppe_hint}; Precautions text: '{contact_droplet_hint}'."
        ),
    )

    # 3.b) AirbornePrecautionsForAGPs (critical leaf)
    node_airborne_agp = evaluator.add_leaf(
        id="AirbornePrecautionsForAGPs",
        desc="States airborne precautions during aerosol-generating procedures, including use of a fit-tested filtering facepiece respirator (instead of a medical mask) and use of an airborne-infection isolation room (or equivalent).",
        parent=node_ipc,
        critical=True,
    )
    respirator_hint = extraction.ipc.respirator_mentioned if extraction and extraction.ipc else ""
    iso_room_hint = extraction.ipc.isolation_room_mentioned if extraction and extraction.ipc else ""
    airborne_hint = extraction.ipc.airborne_precautions_text if extraction and extraction.ipc else ""
    await evaluator.verify(
        claim=(
            "The answer states that airborne precautions are required during aerosol-generating procedures (AGPs), "
            "including use of a fit-tested filtering facepiece respirator (e.g., N95/FFP2/FFP3) instead of a medical mask "
            "and use of an airborne-infection isolation room (AIIR) or equivalent (e.g., a negative-pressure isolation room) when feasible."
        ),
        node=node_airborne_agp,
        additional_instruction=(
            "Judge from the answer text. Accept synonymous wording: 'fit-tested N95', 'FFP2', 'FFP3', 'P2 respirator' as respirator; "
            "'AIIR', 'airborne infection isolation room', or 'negative pressure room' as isolation room equivalents. "
            "Both elements (respirator replacement for AGPs AND AIIR/equivalent) must be present. "
            f"Hints from extraction -> Respirator: '{respirator_hint}', Isolation room: '{iso_room_hint}', Text: '{airborne_hint}'."
        ),
    )

    # Record some non-scoring diagnostics about sources to the summary
    urls = extraction.source_urls if extraction and extraction.source_urls else []
    official_count = sum(1 for u in urls if _is_official_domain(u))
    evaluator.add_custom_info(
        info={
            "total_urls": len(urls),
            "official_like_urls": official_count,
            "sample_urls": urls[:5],
        },
        info_type="diagnostics",
        info_name="source_url_diagnostics"
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
    Evaluate an answer for the Nipah guideline question using the obj_task_eval framework.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The real root; we will add our task node under it
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

    # Extract structured info from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_nipah_guideline(),
        template_class=NipahGuidelineExtraction,
        extraction_name="nipah_guideline_extraction",
    )

    # Optionally include GT/meta info (not strict GT here, but helpful context)
    evaluator.add_ground_truth({
        "required_contact_monitoring_duration_days": 21,
        "required_ppe_elements_routine_care": [
            "medical/surgical mask",
            "eye protection (goggles/face shield)",
            "fluid-resistant/impermeable gown",
            "examination/non-sterile gloves",
        ],
        "airborne_precautions_for_agps": {
            "respirator": "fit-tested N95/FFP2/FFP3",
            "isolation_room": "AIIR or negative pressure room (when feasible)"
        }
    }, gt_type="expected_requirements")

    # Build verification tree and run checks
    await build_and_verify(evaluator, extraction)

    # Return standardized summary
    return evaluator.get_summary()