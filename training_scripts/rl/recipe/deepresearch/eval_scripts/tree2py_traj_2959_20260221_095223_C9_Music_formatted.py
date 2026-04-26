import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "festival_planning_ca_2026"
TASK_DESCRIPTION = """You are planning a comprehensive 3-day outdoor music festival in California for summer 2026. Your festival planning must meet all professional industry standards and regulatory requirements.

Venue Requirements:
Identify and provide details for a suitable outdoor amphitheater or festival grounds venue in California that meets the following specifications:
- Capacity between 10,000 and 25,000 attendees
- At least 1% wheelchair-accessible seating with adjacent companion seats (ADA compliance)
- Minimum required emergency exits for the capacity (3 exits for 501-1,000 occupancy; 4 exits for 1,000+ occupancy)
- Fire sprinkler system installed (required for occupancy over 300)
- Allows minimum 6-8 hours for concert stage setup
- Has designated weather shelter areas
- Provides adequate parking

Artist Lineup Requirements:
Assemble a 4-artist lineup consisting of:

1. Headliner: An artist who has received a Grammy nomination or win in the 2025 or 2026 Grammy cycle AND achieved Billboard Hot 100 charting in 2025. Follow the 9-18 month advance booking standard for major acts.

2. Supporting Act #1: A mid-level artist who has achieved RIAA Gold or Platinum certification (500,000+ units) AND has achieved minimum 75 million U.S. streams for singles or equivalent album streams. Follow the 6-12 month advance booking standard.

3. Supporting Act #2: A second mid-level artist meeting the same criteria as Supporting Act #1.

4. Emerging Artist: An artist who has released a minimum of 5 singles/tracks or 1 complete album AND meets Grammy Best New Artist eligibility criteria (has not exceeded 30 singles/tracks before breakthrough). Follow the 3-6 month advance booking standard.

Operational Requirements:
Your festival operations must include:
- Minimum $1,000,000 general liability insurance coverage
- Medical staffing providing minimum 1 EMT per 250 attendees based on venue capacity
- Documented severe weather contingency plan with evacuation protocols and shelter designation
- Production schedule allocating 6-8 hours for stage setup before each performance day
- All required permits (entertainment, noise, special event)
- Professional sound system with line arrays, subwoofers, and monitoring
- Waste management and restroom facilities plan

Marketing and Ticketing:
- Multi-tier ticket pricing strategy (general admission, VIP, early bird)
- Accessible seating tickets must be priced at the same levels as comparable non-accessible seats (ADA compliance)
- Marketing campaign beginning at least 3-4 weeks before the event

For each component of your festival plan, provide specific details with supporting reference URLs from your research.
"""


# -----------------------------
# Extraction Models
# -----------------------------

class VenueBasic(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    type: Optional[str] = None  # e.g., outdoor amphitheater, festival grounds
    capacity: Optional[str] = None  # allow ranges or textual notes
    urls: List[str] = Field(default_factory=list)


class VenueSafety(BaseModel):
    emergency_exits: Optional[str] = None  # number or compliance statement
    sprinkler: Optional[str] = None  # yes/no or statement
    weather_shelter: Optional[str] = None  # yes/no or statement
    urls: List[str] = Field(default_factory=list)


class VenueADA(BaseModel):
    wheelchair_access_pct: Optional[str] = None  # e.g., ">=1%" or "1%+"
    companion_seats: Optional[str] = None  # yes/no or statement
    urls: List[str] = Field(default_factory=list)


class VenueOps(BaseModel):
    stage_setup_hours: Optional[str] = None  # e.g., "6-8 hours"
    parking: Optional[str] = None  # adequacy statement
    load_in_access: Optional[str] = None  # loading dock/roll-in
    urls: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    basic: Optional[VenueBasic] = None
    safety: Optional[VenueSafety] = None
    ada: Optional[VenueADA] = None
    ops: Optional[VenueOps] = None


class HeadlinerAwards(BaseModel):
    grammy_2025_2026: Optional[str] = None  # nomination or win description
    hot100_2025: Optional[str] = None  # charting description
    urls: List[str] = Field(default_factory=list)


class HeadlinerBooking(BaseModel):
    lead_time_months: Optional[str] = None  # textual months window


class Headliner(BaseModel):
    name: Optional[str] = None
    awards: Optional[HeadlinerAwards] = None
    booking: Optional[HeadlinerBooking] = None


class SupportingCreds(BaseModel):
    riaa_cert: Optional[str] = None  # e.g., Gold/Platinum with work
    streams_75m_us: Optional[str] = None  # description/count
    urls: List[str] = Field(default_factory=list)


class SupportingBooking(BaseModel):
    lead_time_months: Optional[str] = None


class SupportingArtist(BaseModel):
    name: Optional[str] = None
    creds: Optional[SupportingCreds] = None
    booking: Optional[SupportingBooking] = None


class EmergingEligibility(BaseModel):
    releases_count_desc: Optional[str] = None  # e.g., "1 album" or "7 singles"
    grammy_best_new_eligibility: Optional[str] = None  # statement on eligibility
    urls: List[str] = Field(default_factory=list)


class EmergingBooking(BaseModel):
    lead_time_months: Optional[str] = None


class EmergingArtist(BaseModel):
    name: Optional[str] = None
    eligibility: Optional[EmergingEligibility] = None
    booking: Optional[EmergingBooking] = None


class LineupExtraction(BaseModel):
    headliner: Optional[Headliner] = None
    supporting1: Optional[SupportingArtist] = None
    supporting2: Optional[SupportingArtist] = None
    emerging: Optional[EmergingArtist] = None


class LegalInsurance(BaseModel):
    liability_coverage: Optional[str] = None  # e.g., "$1,000,000 general liability"
    permits: List[str] = Field(default_factory=list)  # list of permits like "entertainment", "noise", "special event"
    urls: List[str] = Field(default_factory=list)


class SafetyEmergency(BaseModel):
    medical_staffing: Optional[str] = None  # e.g., "1 EMT per 250 attendees"
    weather_plan: Optional[str] = None  # evacuation and shelter protocols
    security_staffing: Optional[str] = None  # statement
    urls: List[str] = Field(default_factory=list)


class TechnicalProduction(BaseModel):
    daily_stage_setup_hours: Optional[str] = None  # 6-8 hours per day
    sound_system_spec: Optional[str] = None  # line arrays, subs, monitoring
    lighting_plan: Optional[str] = None  # stage and safety lighting
    urls: List[str] = Field(default_factory=list)


class FacilityServices(BaseModel):
    waste_management: Optional[str] = None  # waste and restroom plan
    food_beverage: Optional[str] = None  # vendor arrangements
    urls: List[str] = Field(default_factory=list)


class OpsExtraction(BaseModel):
    legal: Optional[LegalInsurance] = None
    safety: Optional[SafetyEmergency] = None
    technical: Optional[TechnicalProduction] = None
    facility: Optional[FacilityServices] = None


class Ticketing(BaseModel):
    pricing_strategy: Optional[str] = None  # multi-tier: GA, VIP, early bird
    ada_pricing_parity: Optional[str] = None  # statement on parity
    platform: Optional[str] = None  # platform name/link
    urls: List[str] = Field(default_factory=list)


class MarketingCampaign(BaseModel):
    advance_promo_timeline: Optional[str] = None  # at least 3-4 weeks
    social_media_plan: Optional[str] = None
    email_marketing_plan: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Promotions(BaseModel):
    lineup_announcement: Optional[str] = None
    local_media_outreach: Optional[str] = None
    website: Optional[str] = None
    venue_collab: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class MarketingExtraction(BaseModel):
    ticketing: Optional[Ticketing] = None
    campaign: Optional[MarketingCampaign] = None
    promos: Optional[Promotions] = None


class FestivalExtraction(BaseModel):
    venue: Optional[VenueExtraction] = None
    lineup: Optional[LineupExtraction] = None
    ops: Optional[OpsExtraction] = None
    marketing: Optional[MarketingExtraction] = None


# -----------------------------
# Extraction Prompt
# -----------------------------

def prompt_extract_festival() -> str:
    return """
Extract a structured festival plan from the answer. Return JSON conforming to the provided schema. Follow these rules:
- Extract only what is explicitly present in the answer.
- For any field that is missing, set it to null (or empty list for arrays).
- Collect reference URLs for each sub‑section if provided. Include all relevant URLs, whether plain or markdown links.

Schema to fill:
{
  "venue": {
    "basic": {
      "name": string|null,
      "city": string|null,
      "state": string|null,
      "type": string|null,  // e.g., "outdoor amphitheater", "festival grounds"
      "capacity": string|null,  // allow ranges or descriptive text
      "urls": string[]  // venue identification/support pages
    },
    "safety": {
      "emergency_exits": string|null,  // count or compliance statement
      "sprinkler": string|null,  // yes/no or statement
      "weather_shelter": string|null,  // yes/no or statement
      "urls": string[]  // safety compliance/supporting pages
    },
    "ada": {
      "wheelchair_access_pct": string|null,  // e.g., ">=1%" or "1%"
      "companion_seats": string|null,  // yes/no or statement
      "urls": string[]  // ADA/Accessibility pages
    },
    "ops": {
      "stage_setup_hours": string|null,  // e.g., "6-8 hours"
      "parking": string|null,  // adequacy statement
      "load_in_access": string|null,  // loading dock/roll-in
      "urls": string[]  // operations/technical pages
    }
  },
  "lineup": {
    "headliner": {
      "name": string|null,
      "awards": {
        "grammy_2025_2026": string|null,  // nomination/win statement
        "hot100_2025": string|null,  // Billboard Hot 100 in 2025
        "urls": string[]  // awards/press/chart links
      }|null,
      "booking": {
        "lead_time_months": string|null  // e.g., "9-18 months"
      }|null
    },
    "supporting1": {
      "name": string|null,
      "creds": {
        "riaa_cert": string|null,  // Gold/Platinum
        "streams_75m_us": string|null,  // >= 75M U.S. streams
        "urls": string[]  // RIAA/streaming references
      }|null,
      "booking": {
        "lead_time_months": string|null  // e.g., "6-12 months"
      }|null
    },
    "supporting2": {
      "name": string|null,
      "creds": {
        "riaa_cert": string|null,
        "streams_75m_us": string|null,
        "urls": string[]
      }|null,
      "booking": {
        "lead_time_months": string|null  // e.g., "6-12 months"
      }|null
    },
    "emerging": {
      "name": string|null,
      "eligibility": {
        "releases_count_desc": string|null,  // ">=5 singles" or "1 album"
        "grammy_best_new_eligibility": string|null,  // not exceeded 30 singles/tracks before breakthrough
        "urls": string[]  // artist discography/press/Grammy rules
      }|null,
      "booking": {
        "lead_time_months": string|null  // e.g., "3-6 months"
      }|null
    }
  },
  "ops": {
    "legal": {
      "liability_coverage": string|null,  // "$1,000,000 general liability"
      "permits": string[],  // e.g., ["entertainment", "noise", "special event"]
      "urls": string[]  // requirement references
    },
    "safety": {
      "medical_staffing": string|null,  // "1 EMT per 250 attendees"
      "weather_plan": string|null,  // evacuation and shelter designation
      "security_staffing": string|null,  // statement
      "urls": string[]  // EMS/weather/security references
    },
    "technical": {
      "daily_stage_setup_hours": string|null,  // "6-8 hours"
      "sound_system_spec": string|null,  // "line arrays, subwoofers, monitoring"
      "lighting_plan": string|null,  // "stage and safety lighting"
      "urls": string[]  // technical references/specs
    },
    "facility": {
      "waste_management": string|null,  // waste & restroom plan
      "food_beverage": string|null,  // vendor arrangements
      "urls": string[]  // facility service references
    }
  },
  "marketing": {
    "ticketing": {
      "pricing_strategy": string|null,  // multi-tier: GA, VIP, early bird
      "ada_pricing_parity": string|null,  // parity statement
      "platform": string|null,  // ticketing platform
      "urls": string[]  // ticketing references
    },
    "campaign": {
      "advance_promo_timeline": string|null,  // >= 3-4 weeks
      "social_media_plan": string|null,
      "email_marketing_plan": string|null,
      "urls": string[]  // marketing references
    },
    "promos": {
      "lineup_announcement": string|null,
      "local_media_outreach": string|null,
      "website": string|null,
      "venue_collab": string|null,
      "urls": string[]  // promo references
    }
  }
}
"""


# -----------------------------
# Helper Utilities
# -----------------------------

def _urls(lst: Optional[List[str]]) -> List[str]:
    if not lst:
        return []
    # Filter obvious invalids conservatively
    return [u for u in lst if isinstance(u, str) and len(u.strip()) > 0]


async def add_verified_leaf(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    claim: str,
    sources: Optional[List[str] | str],
    critical: bool,
    add_ins: str = "None",
):
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=add_ins
    )


def add_urls_existence_node(
    evaluator: Evaluator,
    parent,
    node_id: str,
    desc: str,
    urls: Optional[List[str]],
    critical: bool = True
):
    exists = bool(urls) and len(_urls(urls)) > 0
    return evaluator.add_custom_node(
        result=exists,
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )


# -----------------------------
# Verification Builders
# -----------------------------

async def verify_venue(evaluator: Evaluator, parent, ext: FestivalExtraction):
    venue = ext.venue or VenueExtraction()

    venue_node = evaluator.add_parallel(
        id="Venue_Specifications",
        desc="Venue selection and capacity requirements for 3-day outdoor music festival",
        parent=parent,
        critical=False
    )

    # Venue Basic Identity
    basic_node = evaluator.add_parallel(
        id="Venue_Basic_Identity",
        desc="Core venue identification and location requirements",
        parent=venue_node,
        critical=False
    )
    basic = venue.basic or VenueBasic()
    # Existence of venue reference URLs (critical gating)
    add_urls_existence_node(
        evaluator,
        basic_node,
        "Venue_Basic_Reference_URLs",
        "Provide reference URLs supporting venue identification",
        basic.urls,
        critical=True
    )
    # Location in California
    await add_verified_leaf(
        evaluator,
        basic_node,
        "Venue_Location_California",
        "Selected venue must be located in California",
        claim=f"The selected venue '{basic.name or ''}' is located in California (CA).",
        sources=_urls(basic.urls),
        critical=True,
        add_ins="Allow minor naming variations. Accept if the page clearly indicates the venue is in California."
    )
    # Venue type Outdoor Amphitheater or Festival Grounds
    await add_verified_leaf(
        evaluator,
        basic_node,
        "Venue_Type_Outdoor",
        "Selected venue must be an outdoor amphitheater or festival grounds",
        claim=f"The venue '{basic.name or ''}' is an outdoor amphitheater or festival grounds.",
        sources=_urls(basic.urls),
        critical=True,
        add_ins="Pass if the venue is clearly described as outdoor amphitheater, outdoor pavilion, or festival grounds."
    )
    # Capacity range 10k-25k
    await add_verified_leaf(
        evaluator,
        basic_node,
        "Venue_Capacity_Range",
        "Venue capacity must be between 10,000 and 25,000 to accommodate target audience",
        claim=f"The venue '{basic.name or ''}' has a capacity between 10,000 and 25,000 attendees.",
        sources=_urls(basic.urls),
        critical=True,
        add_ins="Accept if the stated capacity falls within 10,000 to 25,000, including ranges that lie entirely within."
    )

    # Venue Safety Compliance
    safety_node = evaluator.add_parallel(
        id="Venue_Safety_Compliance",
        desc="Safety systems and emergency preparedness requirements",
        parent=venue_node,
        critical=False
    )
    safety = venue.safety or VenueSafety()
    add_urls_existence_node(
        evaluator,
        safety_node,
        "Venue_Safety_Reference_URLs",
        "Provide reference URLs supporting safety compliance",
        safety.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Emergency_Exit_Compliance",
        "Venue has minimum required emergency exits for capacity (3 exits for 501-1,000; 4 exits for 1,000+)",
        claim=f"The venue '{basic.name or ''}' has at least 4 emergency exits, appropriate for occupancy over 1,000.",
        sources=_urls(safety.urls),
        critical=True,
        add_ins="Look for venue evacuation maps, safety specs, or official documents confirming number of exits or explicit compliance with egress requirements."
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Fire_Sprinkler_System",
        "Venue has fire sprinkler system for occupancy over 300",
        claim=f"The venue '{basic.name or ''}' has a fire sprinkler system installed.",
        sources=_urls(safety.urls),
        critical=True,
        add_ins="Support can be explicit mention of sprinklers or an official facility feature list."
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Weather_Shelter_Areas",
        "Venue has designated weather shelter areas for outdoor attendees",
        claim=f"The venue '{basic.name or ''}' provides designated weather shelter areas.",
        sources=_urls(safety.urls),
        critical=True,
        add_ins="Accept venue maps or policies that designate indoor areas, concourses, or shelters for severe weather."
    )

    # Venue Accessibility Compliance (ADA)
    ada_node = evaluator.add_parallel(
        id="Venue_Accessibility_Compliance",
        desc="ADA accessibility requirements for venue",
        parent=venue_node,
        critical=False
    )
    ada = venue.ada or VenueADA()
    add_urls_existence_node(
        evaluator,
        ada_node,
        "Venue_ADA_Reference_URLs",
        "Provide reference URLs supporting ADA compliance",
        ada.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        ada_node,
        "ADA_Wheelchair_Seating",
        "Venue provides at least 1% wheelchair-accessible seating spaces",
        claim=f"The venue '{basic.name or ''}' provides at least 1% of seating as wheelchair-accessible.",
        sources=_urls(ada.urls),
        critical=True,
        add_ins="Accept explicit statements of 1% or greater accessible seating or documents indicating compliance with code requiring at least 1%."
    )
    await add_verified_leaf(
        evaluator,
        ada_node,
        "ADA_Companion_Seats",
        "Each wheelchair space has adjacent companion seat",
        claim=f"For each wheelchair space at '{basic.name or ''}', an adjacent companion seat is provided.",
        sources=_urls(ada.urls),
        critical=True,
        add_ins="Look for ADA seating policy pages mentioning companion seats adjacent to wheelchair locations."
    )

    # Venue Operational Capabilities
    ops_node = evaluator.add_parallel(
        id="Venue_Operational_Capabilities",
        desc="Operational features supporting festival production",
        parent=venue_node,
        critical=False
    )
    vops = venue.ops or VenueOps()
    add_urls_existence_node(
        evaluator,
        ops_node,
        "Venue_Operational_Reference_URLs",
        "Provide reference URLs supporting operational capabilities",
        vops.urls,
        critical=False
    )
    await add_verified_leaf(
        evaluator,
        ops_node,
        "Stage_Setup_Time",
        "Venue allows minimum 6-8 hours for concert stage setup",
        claim=f"The venue '{basic.name or ''}' allows a minimum load-in/stage setup window of 6–8 hours.",
        sources=_urls(vops.urls),
        critical=True,
        add_ins="Pass if policies, tech riders, or event production guides mention a 6-8 hour setup window or longer."
    )
    await add_verified_leaf(
        evaluator,
        ops_node,
        "Parking_Capacity",
        "Venue provides adequate parking for festival attendees",
        claim=f"The venue '{basic.name or ''}' provides adequate parking for the expected audience.",
        sources=_urls(vops.urls),
        critical=False,
        add_ins="Accept statements indicating ample/large parking or specified capacities that plausibly support 10k–25k attendees."
    )
    await add_verified_leaf(
        evaluator,
        ops_node,
        "Load_In_Access",
        "Venue provides accessible load-in areas for equipment",
        claim=f"The venue '{basic.name or ''}' provides suitable load-in areas for production equipment.",
        sources=_urls(vops.urls),
        critical=False,
        add_ins="Look for mention of loading docks, backstage access, truck bays, ramp access, etc."
    )


async def verify_headliner(evaluator: Evaluator, parent, lineup: LineupExtraction):
    node = evaluator.add_parallel(
        id="Headliner_Artist",
        desc="Primary headliner meeting specific criteria",
        parent=parent,
        critical=False
    )
    head = (lineup.headliner or Headliner())
    awards_node = evaluator.add_parallel(
        id="Headliner_Awards_Recognition",
        desc="Headliner has achieved industry recognition through awards",
        parent=node,
        critical=False
    )
    awards = head.awards or HeadlinerAwards()
    add_urls_existence_node(
        evaluator,
        awards_node,
        "Headliner_Recognition_URLs",
        "Provide reference URLs supporting headliner awards and chart performance",
        awards.urls,
        critical=True
    )
    # Grammy 2025/2026
    await add_verified_leaf(
        evaluator,
        awards_node,
        "Headliner_Grammy_Nomination",
        "Headliner has Grammy nomination or win in 2025 or 2026 cycle",
        claim=f"The headliner '{head.name or ''}' has a Grammy nomination or win in the 2025 or 2026 cycle.",
        sources=_urls(awards.urls),
        critical=True,
        add_ins="Accept official Grammy pages or reputable press confirming 2025/2026 nomination/win."
    )
    # Billboard Hot 100 in 2025
    await add_verified_leaf(
        evaluator,
        awards_node,
        "Headliner_Chart_Performance",
        "Headliner has achieved Billboard Hot 100 charting in 2025",
        claim=f"The headliner '{head.name or ''}' achieved Billboard Hot 100 charting in 2025.",
        sources=_urls(awards.urls),
        critical=True,
        add_ins="Accept Billboard pages, charts, or reputable publications confirming Hot 100 charting in 2025."
    )
    # Booking standards
    booking_node = evaluator.add_parallel(
        id="Headliner_Booking_Standards",
        desc="Headliner booking follows professional industry standards",
        parent=node,
        critical=False
    )
    lead = (head.booking or HeadlinerBooking()).lead_time_months or ""
    await add_verified_leaf(
        evaluator,
        booking_node,
        "Headliner_Booking_Lead_Time",
        "Headliner booking follows 9-18 month advance booking standard for major acts",
        claim=f"The plan specifies a headliner booking lead time of '{lead}', which falls within 9–18 months.",
        sources=None,
        critical=False,
        add_ins="Judge based on the extracted lead-time text; pass if it indicates a window within 9–18 months for the headliner."
    )


async def verify_supporting(
    evaluator: Evaluator,
    parent,
    sup: Optional[SupportingArtist],
    idx: int
):
    id_prefix = f"Supporting_Act_{idx}"
    node = evaluator.add_parallel(
        id=id_prefix,
        desc=f"{'First' if idx == 1 else 'Second'} supporting mid-level artist meeting criteria",
        parent=parent,
        critical=False
    )
    creds_node = evaluator.add_parallel(
        id=f"{id_prefix}_Credentials",
        desc=f"{'First' if idx == 1 else 'Second'} supporting artist commercial success metrics",
        parent=node,
        critical=False
    )
    sup = sup or SupportingArtist()
    creds = sup.creds or SupportingCreds()
    add_urls_existence_node(
        evaluator,
        creds_node,
        f"{id_prefix}_Reference_URLs",
        "Provide reference URLs supporting artist credentials",
        creds.urls,
        critical=True
    )
    # RIAA certification
    await add_verified_leaf(
        evaluator,
        creds_node,
        f"{id_prefix}_RIAA_Certification",
        "Artist has achieved RIAA Gold or Platinum certification (500,000+ units)",
        claim=f"The artist '{sup.name or ''}' has an RIAA Gold or Platinum certification (≥500,000 units).",
        sources=_urls(creds.urls),
        critical=True,
        add_ins="Accept RIAA database pages or reputable sources confirming Gold/Platinum for any release by the artist."
    )
    # Streaming threshold
    await add_verified_leaf(
        evaluator,
        creds_node,
        f"{id_prefix}_Streaming_Threshold",
        "Artist has achieved minimum 75 million U.S. streams for singles or equivalent album streams",
        claim=f"The artist '{sup.name or ''}' has achieved at least 75 million U.S. streams for a single (or equivalent album streams).",
        sources=_urls(creds.urls),
        critical=True,
        add_ins="Accept Spotify, Apple Music, or reputable industry analytics indicating ≥75M U.S. streams or equivalent."
    )
    # Booking standards
    booking_node = evaluator.add_parallel(
        id=f"{id_prefix}_Booking_Standards",
        desc=f"{'First' if idx == 1 else 'Second'} supporting artist booking follows professional standards",
        parent=node,
        critical=False
    )
    lead = (sup.booking or SupportingBooking()).lead_time_months or ""
    await add_verified_leaf(
        evaluator,
        booking_node,
        f"{id_prefix}_Booking_Timeline",
        "Artist booking follows 6-12 month advance booking standard for mid-level acts",
        claim=f"The plan specifies a booking lead time of '{lead}' for this supporting act, which falls within 6–12 months.",
        sources=None,
        critical=False,
        add_ins="Judge against the extracted text; pass if it indicates a 6–12 month window for supporting acts."
    )


async def verify_emerging(evaluator: Evaluator, parent, emerging: Optional[EmergingArtist]):
    node = evaluator.add_parallel(
        id="Emerging_Artist",
        desc="Emerging or Best New Artist candidate meeting criteria",
        parent=parent,
        critical=False
    )
    emerging = emerging or EmergingArtist()
    elig_node = evaluator.add_parallel(
        id="Emerging_Eligibility",
        desc="Emerging artist meets industry eligibility requirements",
        parent=node,
        critical=False
    )
    elig = emerging.eligibility or EmergingEligibility()
    add_urls_existence_node(
        evaluator,
        elig_node,
        "Emerging_Reference_URLs",
        "Provide reference URLs supporting artist eligibility",
        elig.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        elig_node,
        "Emerging_Release_Requirement",
        "Artist has released minimum 5 singles/tracks or 1 complete album",
        claim=f"The emerging artist '{emerging.name or ''}' has released at least 5 singles/tracks or 1 complete album.",
        sources=_urls(elig.urls),
        critical=True,
        add_ins="Discography pages or reputable databases supporting release counts qualify."
    )
    await add_verified_leaf(
        evaluator,
        elig_node,
        "Emerging_Grammy_Eligibility",
        "Artist meets Grammy Best New Artist eligibility criteria (not exceeded 30 singles/tracks before breakthrough)",
        claim=f"The emerging artist '{emerging.name or ''}' meets Grammy Best New Artist eligibility, not exceeding 30 singles/tracks before their breakthrough.",
        sources=_urls(elig.urls),
        critical=True,
        add_ins="Accept references to official Grammy rules and credible evidence about the artist's release count/history."
    )
    # Booking standards
    booking_node = evaluator.add_parallel(
        id="Emerging_Booking_Standards",
        desc="Emerging artist booking follows professional standards",
        parent=node,
        critical=False
    )
    lead = (emerging.booking or EmergingBooking()).lead_time_months or ""
    await add_verified_leaf(
        evaluator,
        booking_node,
        "Emerging_Booking_Timeline",
        "Artist booking follows 3-6 month advance booking standard for rising stars",
        claim=f"The plan specifies an emerging artist booking lead time of '{lead}', which falls within 3–6 months.",
        sources=None,
        critical=False,
        add_ins="Judge based on the extracted lead-time text for emerging artists."
    )


async def verify_lineup(evaluator: Evaluator, parent, ext: FestivalExtraction):
    lineup_node = evaluator.add_parallel(
        id="Artist_Lineup_Requirements",
        desc="Artist booking meeting professional standards and timeline requirements",
        parent=parent,
        critical=False
    )
    lineup = ext.lineup or LineupExtraction()

    await verify_headliner(evaluator, lineup_node, lineup)

    await verify_supporting(evaluator, lineup_node, lineup.supporting1, 1)
    await verify_supporting(evaluator, lineup_node, lineup.supporting2, 2)

    await verify_emerging(evaluator, lineup_node, lineup.emerging)


async def verify_ops(evaluator: Evaluator, parent, ext: FestivalExtraction):
    ops_root = evaluator.add_parallel(
        id="Operational_Requirements",
        desc="Essential operational planning and logistics for 3-day festival",
        parent=parent,
        critical=False
    )
    ops = ext.ops or OpsExtraction()

    # Legal & Insurance
    legal_node = evaluator.add_parallel(
        id="Legal_Insurance_Requirements",
        desc="Legal and insurance compliance requirements",
        parent=ops_root,
        critical=False
    )
    legal = ops.legal or LegalInsurance()
    add_urls_existence_node(
        evaluator,
        legal_node,
        "Legal_Reference_URLs",
        "Provide reference URLs supporting insurance and permit requirements",
        legal.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        legal_node,
        "Insurance_Coverage",
        "Minimum $1,000,000 general liability insurance coverage",
        claim="The festival plan secures a minimum $1,000,000 general liability insurance coverage.",
        sources=_urls(legal.urls),
        critical=True,
        add_ins="Accept insurance requirement documents or policy references that clearly state $1,000,000 general liability coverage."
    )
    await add_verified_leaf(
        evaluator,
        legal_node,
        "Festival_Permits",
        "All required permits obtained including entertainment, noise, and special event permits",
        claim="The festival plan covers all required permits including entertainment, noise, and special event permits.",
        sources=_urls(legal.urls),
        critical=True,
        add_ins="Accept city/county guidance or official references enumerating these permit requirements for events."
    )

    # Safety & Emergency Services
    safety_node = evaluator.add_parallel(
        id="Safety_Emergency_Services",
        desc="Safety services and emergency preparedness",
        parent=ops_root,
        critical=False
    )
    safety = ops.safety or SafetyEmergency()
    add_urls_existence_node(
        evaluator,
        safety_node,
        "Ops_Safety_Reference_URLs",
        "Provide reference URLs supporting safety and emergency services",
        safety.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Medical_Services_Staffing",
        "Medical staffing provides minimum 1 EMT per 250 attendees based on venue capacity",
        claim="Medical staffing is planned at a minimum ratio of 1 EMT per 250 attendees.",
        sources=_urls(safety.urls),
        critical=True,
        add_ins="Accept EMS planning guides or municipal requirements explicitly stating the 1 EMT per 250 attendees standard."
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Weather_Contingency_Plan",
        "Documented severe weather plan including evacuation protocols and shelter designation",
        claim="The festival has a documented severe weather contingency plan with evacuation protocols and designated shelter areas.",
        sources=_urls(safety.urls),
        critical=True,
        add_ins="Accept emergency action plans, venue safety pages, or official documentation describing evacuation and shelter designations."
    )
    await add_verified_leaf(
        evaluator,
        safety_node,
        "Security_Staffing",
        "Security staffing plan appropriate for venue capacity and event type",
        claim="The festival includes a security staffing plan appropriate for the venue capacity and event type.",
        sources=_urls(safety.urls),
        critical=False,
        add_ins="Look for security staffing policies, ratios, or coordination with venue security and local law enforcement."
    )

    # Technical Production
    tech_node = evaluator.add_parallel(
        id="Technical_Production",
        desc="Technical production and equipment specifications",
        parent=ops_root,
        critical=False
    )
    tech = ops.technical or TechnicalProduction()
    add_urls_existence_node(
        evaluator,
        tech_node,
        "Technical_Reference_URLs",
        "Provide reference URLs supporting technical production specifications",
        tech.urls,
        critical=True
    )
    await add_verified_leaf(
        evaluator,
        tech_node,
        "Technical_Production_Setup",
        "Production schedule allocates 6-8 hours for stage setup before each performance day",
        claim="The production schedule allocates 6–8 hours for stage setup before each performance day.",
        sources=_urls(tech.urls),
        critical=True,
        add_ins="Accept production schedules or technical plans that clearly allocate 6–8 hours for setup each day."
    )
    await add_verified_leaf(
        evaluator,
        tech_node,
        "Sound_System_Specifications",
        "Sound system meets professional concert specifications including line arrays, subwoofers, and monitoring",
        claim="The sound system includes professional components: line arrays, subwoofers, and on-stage monitoring.",
        sources=_urls(tech.urls),
        critical=True,
        add_ins="Look for system spec sheets or rider notes referencing line arrays, subs, and monitors."
    )
    await add_verified_leaf(
        evaluator,
        tech_node,
        "Lighting_Production",
        "Lighting production plan includes stage lighting and safety lighting for pathways",
        claim="The lighting plan includes stage lighting and safety lighting for pathways/egress.",
        sources=_urls(tech.urls),
        critical=False,
        add_ins="Accept production plans mentioning stage fixtures plus safety/egress lighting."
    )

    # Facility & Attendee Services
    facility_node = evaluator.add_parallel(
        id="Facility_Services",
        desc="Facility and attendee services planning",
        parent=ops_root,
        critical=False
    )
    facility = ops.facility or FacilityServices()
    add_urls_existence_node(
        evaluator,
        facility_node,
        "Facility_Reference_URLs",
        "Provide reference URLs supporting facility services",
        facility.urls,
        critical=False
    )
    await add_verified_leaf(
        evaluator,
        facility_node,
        "Waste_Management_Plan",
        "Waste management and restroom facilities plan for multi-day event",
        claim="The plan includes waste management and adequate restroom facilities for the multi-day event.",
        sources=_urls(facility.urls),
        critical=False,
        add_ins="Accept vendor plans or municipal guidelines indicating waste services, portable restrooms, servicing schedules, etc."
    )
    await add_verified_leaf(
        evaluator,
        facility_node,
        "Food_Beverage_Services",
        "Food and beverage vendor arrangements",
        claim="The plan includes food and beverage vendor arrangements.",
        sources=_urls(facility.urls),
        critical=False,
        add_ins="Look for vendor agreements, RFPs, or festival planning notes about F&B."
    )


async def verify_marketing(evaluator: Evaluator, parent, ext: FestivalExtraction):
    marketing_root = evaluator.add_parallel(
        id="Marketing_Ticket_Sales",
        desc="Marketing strategy and ticketing approach for festival",
        parent=parent,
        critical=False
    )
    mk = ext.marketing or MarketingExtraction()

    # Ticketing Strategy
    ticket_node = evaluator.add_parallel(
        id="Ticketing_Strategy",
        desc="Comprehensive ticketing approach and pricing",
        parent=marketing_root,
        critical=False
    )
    tix = mk.ticketing or Ticketing()
    add_urls_existence_node(
        evaluator,
        ticket_node,
        "Ticketing_Reference_URLs",
        "Provide reference URLs supporting ticketing approach",
        tix.urls,
        critical=False
    )
    await add_verified_leaf(
        evaluator,
        ticket_node,
        "Ticket_Pricing_Strategy",
        "Multi-tier ticket pricing strategy (general admission, VIP, early bird)",
        claim="The ticketing strategy uses multiple tiers such as General Admission, VIP, and Early Bird.",
        sources=_urls(tix.urls),
        critical=False,
        add_ins="Accept ticketing pages or plan documents that show tiered options."
    )
    await add_verified_leaf(
        evaluator,
        ticket_node,
        "ADA_Ticket_Pricing_Parity",
        "Accessible seating tickets priced at same levels as comparable non-accessible seats",
        claim="Accessible seating tickets are priced at the same levels as comparable non-accessible seats.",
        sources=_urls(tix.urls),
        critical=True,
        add_ins="Accept ADA ticketing policy statements or pricing pages indicating parity for accessible seats."
    )
    await add_verified_leaf(
        evaluator,
        ticket_node,
        "Online_Ticketing_Platform",
        "Online ticketing platform for sales and distribution",
        claim="An online ticketing platform is used for sales and distribution.",
        sources=_urls(tix.urls),
        critical=False,
        add_ins="Look for platform references (e.g., Ticketmaster, Eventbrite) or embedded purchase links."
    )

    # Marketing Campaign
    campaign_node = evaluator.add_parallel(
        id="Marketing_Campaign",
        desc="Marketing campaign execution and timeline",
        parent=marketing_root,
        critical=False
    )
    camp = mk.campaign or MarketingCampaign()
    add_urls_existence_node(
        evaluator,
        campaign_node,
        "Marketing_Reference_URLs",
        "Provide reference URLs supporting marketing campaign",
        camp.urls,
        critical=False
    )
    # Advance promotion timeline (simple check)
    await add_verified_leaf(
        evaluator,
        campaign_node,
        "Advance_Promotion_Timeline",
        "Marketing begins 3-4 weeks minimum before event",
        claim=f"The plan states the marketing campaign begins '{camp.advance_promo_timeline or ''}', which is at least 3–4 weeks before the event.",
        sources=None,
        critical=True,
        add_ins="Judge based on the extracted timeline text; pass if it indicates ≥3–4 weeks lead time before the event."
    )
    await add_verified_leaf(
        evaluator,
        campaign_node,
        "Social_Media_Marketing",
        "Social media marketing plan across multiple platforms",
        claim="The marketing plan includes social media marketing across multiple platforms.",
        sources=_urls(camp.urls),
        critical=False,
        add_ins="Look for references to platforms like Instagram, TikTok, X/Twitter, Facebook, etc."
    )
    await add_verified_leaf(
        evaluator,
        campaign_node,
        "Email_Marketing_Campaign",
        "Email marketing campaign with segmentation strategy",
        claim="The marketing plan includes email marketing, ideally with audience segmentation.",
        sources=_urls(camp.urls),
        critical=False,
        add_ins="Accept marketing plans or articles describing email strategy and segmentation."
    )

    # Promotional Activities
    promo_node = evaluator.add_parallel(
        id="Promotional_Activities",
        desc="Additional promotional strategies and partnerships",
        parent=marketing_root,
        critical=False
    )
    promos = mk.promos or Promotions()
    await add_verified_leaf(
        evaluator,
        promo_node,
        "Lineup_Announcement_Strategy",
        "Phased lineup announcement strategy to maintain engagement",
        claim="The plan includes a phased lineup announcement strategy to maintain engagement.",
        sources=_urls(promos.urls),
        critical=False,
        add_ins="Accept plans that mention staggered announcements, teasers, or wave-based reveals."
    )
    await add_verified_leaf(
        evaluator,
        promo_node,
        "Local_Media_Outreach",
        "Outreach to local California media outlets",
        claim="The plan includes outreach to local California media outlets.",
        sources=_urls(promos.urls),
        critical=False,
        add_ins="Look for references to local press, radio, or regional media partnerships."
    )
    await add_verified_leaf(
        evaluator,
        promo_node,
        "Festival_Website",
        "Dedicated festival website with event information and ticket sales",
        claim="A dedicated festival website (or landing page) is provided with event information and ticket sales.",
        sources=_urls(promos.urls),
        critical=False,
        add_ins="Accept references to the official festival site or a dedicated landing page."
    )
    await add_verified_leaf(
        evaluator,
        promo_node,
        "Venue_Collaboration",
        "Marketing collaboration with venue for cross-promotion",
        claim="The plan includes collaboration with the venue for cross-promotion.",
        sources=_urls(promos.urls),
        critical=False,
        add_ins="Accept statements or examples of co-promotion with the venue."
    )


# -----------------------------
# Main Evaluation Function
# -----------------------------

async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Extraction
    ext: FestivalExtraction = await evaluator.extract(
        prompt=prompt_extract_festival(),
        template_class=FestivalExtraction,
        extraction_name="festival_plan_extraction"
    )

    # Build top-level root node (non-critical to allow mixed critical children below)
    root_node = evaluator.add_parallel(
        id="Festival_Planning_Requirements",
        desc="Root evaluation of comprehensive music festival planning meeting all professional requirements",
        parent=root,
        critical=False
    )

    # Venue
    await verify_venue(evaluator, root_node, ext)

    # Lineup
    await verify_lineup(evaluator, root_node, ext)

    # Operations
    await verify_ops(evaluator, root_node, ext)

    # Marketing & Ticketing
    await verify_marketing(evaluator, root_node, ext)

    return evaluator.get_summary()