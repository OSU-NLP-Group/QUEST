import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

TASK_ID = "ohio_school_emergency_compliance"
TASK_DESCRIPTION = """What are the key regulatory requirements for an Ohio public school district managing emergency weather closures and athletic event postponements? Specifically, provide the following information with supporting references:

1. The minimum instructional hours required by Ohio state law for elementary students (grades K-6) and secondary students (grades 7-12), including an explanation of Ohio's current hour-based tracking system

2. The OHSAA lightning and inclement weather policy requirements for postponing athletic practices and competitions, including the minimum waiting period after lightning or thunder and the conditions that reset this waiting period

3. Who has the authority to declare emergency school closures and the typical decision-making timeline for weather-related closures

4. The communication channels and notification systems that school districts use to alert parents and families about emergency closures

5. Reference to current Ohio state emergency guidelines for schools

Each answer component must include specific factual details (such as hour requirements, time durations, and responsible positions) and provide valid reference URLs from official sources (such as the Ohio Department of Education, OHSAA, or school district policies) to support the information.
"""


# ----------------------------- Data Models ----------------------------------

class HoursExtraction(BaseModel):
    k6_hours_value: Optional[str] = None
    k6_urls: List[str] = Field(default_factory=list)

    secondary_hours_value: Optional[str] = None  # Grades 7–12
    secondary_urls: List[str] = Field(default_factory=list)

    community_hours_value: Optional[str] = None  # Community schools
    community_urls: List[str] = Field(default_factory=list)

    no_calamity_days_hour_based: Optional[bool] = None
    make_up_hours_required_if_below_minimum: Optional[bool] = None
    hour_system_urls: List[str] = Field(default_factory=list)


class OHSAAWeatherExtraction(BaseModel):
    waiting_period_minutes: Optional[str] = None
    reset_timer_on_subsequent_lightning_or_thunder: Optional[bool] = None
    applies_to_practices_and_competitions: Optional[bool] = None
    ohsaa_urls: List[str] = Field(default_factory=list)


class ClosureAuthorityExtraction(BaseModel):
    authority_role: Optional[str] = None
    authority_urls: List[str] = Field(default_factory=list)

    typical_decision_timeline_window: Optional[str] = None  # e.g., "5:00–6:00 AM"
    timeline_urls: List[str] = Field(default_factory=list)


class ParentNotificationExtraction(BaseModel):
    mass_notification_systems_used: Optional[bool] = None
    channels_listed: List[str] = Field(default_factory=list)  # Expect entries like "phone", "email", "text/SMS"
    channels_urls: List[str] = Field(default_factory=list)

    platforms_mentioned: List[str] = Field(default_factory=list)  # e.g., "SchoolMessenger", "Finalsite", "Blackboard Connect", "K12 Alerts"
    platforms_urls: List[str] = Field(default_factory=list)


class StateGuidelinesExtraction(BaseModel):
    guidelines_name_context: Optional[str] = None
    july_2025_revision_mentioned: Optional[bool] = None
    guidelines_urls: List[str] = Field(default_factory=list)


# --------------------------- Extraction Prompts ------------------------------

def prompt_extract_hours() -> str:
    return """
    Extract the instructional-hours details and hour-based tracking system explanations stated in the answer.

    Return:
    - k6_hours_value: The minimum instructional hours for grades K–6 if explicitly stated (e.g., "910"); otherwise null.
    - k6_urls: All official reference URLs the answer cites specifically to support the K–6 minimum hours. Only include URLs explicitly present in the answer.

    - secondary_hours_value: The minimum instructional hours for grades 7–12 if explicitly stated (e.g., "1001"); otherwise null.
    - secondary_urls: All official reference URLs the answer cites specifically to support the 7–12 minimum hours.

    - community_hours_value: The minimum instructional hours for community schools if stated (e.g., "920"); otherwise null.
    - community_urls: All official reference URLs the answer cites specifically to support the community school minimum hours.

    - no_calamity_days_hour_based: true if the answer explicitly says Ohio no longer uses "calamity days" and uses an hour-based instructional time system; false if it explicitly says otherwise; null if not stated.
    - make_up_hours_required_if_below_minimum: true if the answer explicitly says districts must make up time if closures cause them to fall below minimum hours; false if explicitly says otherwise; null if not stated.
    - hour_system_urls: All official reference URLs the answer cites to support the hour-based system/explanation (including calamity days change and make-up requirement).

    Only include URLs explicitly present in the answer. If a field is missing in the answer, set it to null (or empty list for URLs).
    """


def prompt_extract_ohsaa() -> str:
    return """
    Extract OHSAA lightning/inclement weather policy details cited in the answer.

    Return:
    - waiting_period_minutes: The minimum waiting period minutes after lightning/thunder before resuming (e.g., "30") if explicitly stated; otherwise null.
    - reset_timer_on_subsequent_lightning_or_thunder: true if the answer explicitly says any subsequent lightning/thunder resets the 30-minute timer; false if explicitly says otherwise; null if not stated.
    - applies_to_practices_and_competitions: true if the answer explicitly says the policy applies to both practices and competitions; false if explicitly says otherwise; null if not stated.
    - ohsaa_urls: All official OHSAA reference URLs (prefer ohsaa.org pages/PDFs) that the answer cites to support these rules.

    Only include URLs explicitly present in the answer. If a field is missing in the answer, set it to null (or empty list for URLs).
    """


def prompt_extract_closure() -> str:
    return """
    Extract authority and typical timeline for emergency school closures as stated in the answer.

    Return:
    - authority_role: The role stated to have authority to declare emergency closures (e.g., "Superintendent") if explicitly stated; otherwise null.
    - authority_urls: All official reference URLs (e.g., district policy/FAQ pages) cited in the answer supporting the authority role.

    - typical_decision_timeline_window: The typical decision-making window for weather-related closures (e.g., "5:00–6:00 AM") if explicitly stated; otherwise null.
    - timeline_urls: All official reference URLs (e.g., district policy/FAQ pages) cited in the answer supporting the timeline.

    Only include URLs explicitly present in the answer. If a field is missing in the answer, set it to null (or empty list for URLs).
    """


def prompt_extract_notification() -> str:
    return """
    Extract district notification system information for emergency closures as stated in the answer.

    Return:
    - mass_notification_systems_used: true if the answer explicitly says districts use mass notification systems for closures; false if explicitly says otherwise; null if not stated.
    - channels_listed: List the channels explicitly named in the answer (e.g., "phone", "email", "text", "SMS").
    - channels_urls: All official district communications/policy page URLs cited in the answer supporting the use of these channels and mass notification systems.

    - platforms_mentioned: List of platform brand names explicitly mentioned (e.g., "SchoolMessenger", "Finalsite", "Blackboard Connect", "K12 Alerts").
    - platforms_urls: All official district pages/policies cited in the answer that show the use of any of these platforms.

    Only include URLs explicitly present in the answer. If a field is missing in the answer, set it to null (or empty list for URLs).
    """


def prompt_extract_guidelines() -> str:
    return """
    Extract the Ohio state emergency guidelines information as stated in the answer.

    Return:
    - guidelines_name_context: The name/description of the current Ohio state emergency guidelines for schools (document/page name and relevant context) if explicitly stated; otherwise null.
    - july_2025_revision_mentioned: true if the answer explicitly says revised guidelines were released in July 2025; false if explicitly says otherwise; null if not stated.
    - guidelines_urls: All official Ohio Department of Education (or equivalent Ohio education agency) URLs cited in the answer for the referenced guidelines/announcement.

    Only include URLs explicitly present in the answer. If a field is missing in the answer, set it to null (or empty list for URLs).
    """


# ----------------------------- Helper Methods --------------------------------

async def add_reference_support_leaf(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> None:
    """
    Add a leaf that must be supported by URL evidence. If no URLs were provided in the answer, mark failed directly.
    """
    if urls and len(urls) > 0:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=True
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=urls,
            additional_instruction=add_ins
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=True
        )


# --------------------------- Verification Builders ---------------------------

async def build_instructional_hours(
    evaluator: Evaluator,
    parent_node,
    data: HoursExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Instructional_Hours_Requirements",
        desc="Checks minimum instructional-hour requirements and the hour-based tracking system, per constraints, with official references.",
        parent=parent_node,
        critical=True
    )

    # Elementary K–6
    k6_node = evaluator.add_parallel(
        id="Elementary_Hours_K6",
        desc="Grades K–6 minimum instructional hours (910) with official reference.",
        parent=group,
        critical=True
    )
    # Value presence in answer
    k6_value_leaf = evaluator.add_leaf(
        id="K6_Hour_Value_910",
        desc="States that Ohio grades K–6 must provide a minimum of 910 instructional hours per school year.",
        parent=k6_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that Ohio grades K–6 must provide a minimum of 910 instructional hours per school year.",
        node=k6_value_leaf,
        additional_instruction="Read the answer text carefully. If it does not include the numeric value 910 for K–6 minimum hours, mark Incorrect."
    )
    # Official reference support
    await add_reference_support_leaf(
        evaluator,
        k6_node,
        "K6_Official_Reference_URL",
        "Provides a valid official reference URL (e.g., Ohio Department of Education) supporting the 910-hour K–6 requirement.",
        "Ohio grades K–6 must provide a minimum of 910 instructional hours per school year.",
        data.k6_urls,
        add_ins="Confirm the page explicitly states the K–6 minimum hours as 910. Prefer official pages such as education.ohio.gov or codes.ohio.gov. If the URL is not official or does not support the numeric requirement, mark Unsupported."
    )

    # Secondary 7–12
    sec_node = evaluator.add_parallel(
        id="Secondary_Hours_7_12",
        desc="Grades 7–12 minimum instructional hours (1,001) with official reference.",
        parent=group,
        critical=True
    )
    sec_value_leaf = evaluator.add_leaf(
        id="7_12_Hour_Value_1001",
        desc="States that Ohio grades 7–12 must provide a minimum of 1,001 instructional hours per school year.",
        parent=sec_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that Ohio grades 7–12 must provide a minimum of 1,001 instructional hours per school year.",
        node=sec_value_leaf,
        additional_instruction="Read the answer text carefully. If it does not include the numeric value 1,001 for 7–12 minimum hours, mark Incorrect."
    )
    await add_reference_support_leaf(
        evaluator,
        sec_node,
        "7_12_Official_Reference_URL",
        "Provides a valid official reference URL (e.g., Ohio Department of Education) supporting the 1,001-hour 7–12 requirement.",
        "Ohio grades 7–12 must provide a minimum of 1,001 instructional hours per school year.",
        data.secondary_urls,
        add_ins="Confirm the page explicitly states the 7–12 minimum hours as 1,001. Prefer official pages such as education.ohio.gov or codes.ohio.gov. If the URL is not official or does not support the numeric requirement, mark Unsupported."
    )

    # Community schools
    comm_node = evaluator.add_parallel(
        id="Community_School_Hours",
        desc="Community school minimum instructional hours (920) with official reference (included because it is explicitly listed in constraints).",
        parent=group,
        critical=True
    )
    comm_value_leaf = evaluator.add_leaf(
        id="Community_Hour_Value_920",
        desc="States that Ohio community schools must provide a minimum of 920 instructional hours per school year.",
        parent=comm_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that Ohio community schools must provide a minimum of 920 instructional hours per school year.",
        node=comm_value_leaf,
        additional_instruction="Read the answer text carefully. If it does not include the numeric value 920 for community school minimum hours, mark Incorrect."
    )
    await add_reference_support_leaf(
        evaluator,
        comm_node,
        "Community_Official_Reference_URL",
        "Provides a valid official reference URL (e.g., Ohio Department of Education) supporting the 920-hour community school requirement.",
        "Ohio community schools must provide a minimum of 920 instructional hours per school year.",
        data.community_urls,
        add_ins="Confirm the page explicitly states the community school minimum hours as 920. Prefer official pages such as education.ohio.gov or codes.ohio.gov. If the URL is not official or does not support the numeric requirement, mark Unsupported."
    )

    # Hour-based tracking system
    hour_node = evaluator.add_parallel(
        id="Hour_Based_Tracking_System",
        desc="Explains Ohio’s hour-based tracking system (including elimination of calamity days and make-up requirement) with official reference.",
        parent=group,
        critical=True
    )
    # No calamity days
    await add_reference_support_leaf(
        evaluator,
        hour_node,
        "No_Calamity_Days_Hour_Based",
        "Explains that Ohio no longer uses 'calamity days' and instead operates on an hour-based instructional time schedule/tracking system.",
        "Ohio no longer uses 'calamity days' and instead operates on an hour-based instructional time tracking system for instructional hours.",
        data.hour_system_urls,
        add_ins="Verify the source explicitly describes the elimination of 'calamity days' and the hour-based system. Prefer official Ohio education agency pages."
    )
    # Make-up hours requirement
    await add_reference_support_leaf(
        evaluator,
        hour_node,
        "Make_Up_Hours_If_Below_Minimum",
        "Explains that districts must make up instructional time if closures cause them to fall below the minimum required instructional hours.",
        "Districts must make up instructional time if closures cause them to fall below the minimum required instructional hours.",
        data.hour_system_urls,
        add_ins="Verify the source clearly states that districts must make up instructional time when below minimum hours. Prefer official Ohio education agency pages."
    )
    # Official reference for system
    await add_reference_support_leaf(
        evaluator,
        hour_node,
        "Hour_System_Official_Reference_URL",
        "Provides a valid official reference URL supporting the hour-based system explanation (including the above points).",
        "Official guidance describes Ohio’s hour-based instructional time system, including eliminating 'calamity days' and making up time if below minimum.",
        data.hour_system_urls,
        add_ins="Confirm the URL is an official Ohio education agency page (e.g., education.ohio.gov) and supports the hour-based system explanation."
    )


async def build_ohsaa_policy(
    evaluator: Evaluator,
    parent_node,
    data: OHSAAWeatherExtraction
) -> None:
    group = evaluator.add_parallel(
        id="OHSAA_Weather_Policy_Compliance",
        desc="Checks OHSAA lightning/inclement weather policy requirements (30-minute wait, reset rule, applies to practices and competitions) with official OHSAA reference(s).",
        parent=parent_node,
        critical=True
    )

    # 30-minute waiting period
    wait_node = evaluator.add_parallel(
        id="Lightning_Waiting_Period",
        desc="30-minute minimum waiting period with official OHSAA reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        wait_node,
        "Wait_At_Least_30_Minutes",
        "States that practices/competitions must wait at least 30 minutes after the last lightning flash or thunder is heard before resuming.",
        "Practices and competitions must wait at least 30 minutes after the last observed lightning or thunder before resuming.",
        data.ohsaa_urls,
        add_ins="Confirm the OHSAA policy explicitly states a minimum 30-minute waiting period after lightning/thunder. Prefer official ohsaa.org pages or OHSAA PDFs."
    )
    await add_reference_support_leaf(
        evaluator,
        wait_node,
        "OHSAA_Official_Reference_URL_Wait",
        "Provides a valid official OHSAA reference URL supporting the 30-minute waiting period requirement.",
        "The provided source is an official OHSAA policy page/publication that states the 30-minute lightning/thunder waiting period.",
        data.ohsaa_urls,
        add_ins="Confirm the URL belongs to an official OHSAA domain (ohsaa.org) or an OHSAA-published document. If not official, mark Unsupported."
    )

    # Reset timer rule
    reset_node = evaluator.add_parallel(
        id="Timer_Reset_Rule",
        desc="Reset condition for the 30-minute timer with official OHSAA reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        reset_node,
        "Reset_On_Subsequent_Lightning_Thunder",
        "States that any subsequent lightning or thunder after the countdown begins resets the timer back to 30 minutes.",
        "Any subsequent lightning or thunder after the countdown begins resets the timer back to 30 minutes.",
        data.ohsaa_urls,
        add_ins="Verify the OHSAA source states that the 30-minute countdown resets upon subsequent lightning/thunder."
    )
    await add_reference_support_leaf(
        evaluator,
        reset_node,
        "OHSAA_Official_Reference_URL_Reset",
        "Provides a valid official OHSAA reference URL supporting the reset condition.",
        "The OHSAA source describes the reset condition for the 30-minute timer when lightning/thunder occurs again.",
        data.ohsaa_urls,
        add_ins="Confirm the URL is official (ohsaa.org or OHSAA-published document). If not official, mark Unsupported."
    )

    # Scope applies to practices and competitions
    scope_node = evaluator.add_parallel(
        id="Applies_To_Practices_And_Competitions",
        desc="Scope: applies to both practices and competitions, with official OHSAA reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        scope_node,
        "Scope_Practices_And_Competitions",
        "States that OHSAA weather policies apply to both practices and competitions.",
        "OHSAA weather policies apply to both practices and competitions.",
        data.ohsaa_urls,
        add_ins="Confirm the OHSAA source indicates the policy applies to both practices and competitions."
    )
    await add_reference_support_leaf(
        evaluator,
        scope_node,
        "OHSAA_Official_Reference_URL_Scope",
        "Provides a valid official OHSAA reference URL supporting the practices-and-competitions scope.",
        "The OHSAA source clearly shows that the lightning/inclement weather policy applies to practices and competitions.",
        data.ohsaa_urls,
        add_ins="Confirm the URL is official (ohsaa.org or OHSAA-published document). If not official, mark Unsupported."
    )


async def build_closure_authority_timeline(
    evaluator: Evaluator,
    parent_node,
    data: ClosureAuthorityExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Emergency_Closure_Authority_and_Timeline",
        desc="Checks who can declare emergency closures and the typical weather-closure decision timeline, with official reference(s).",
        parent=parent_node,
        critical=True
    )

    # Authority: Superintendent
    auth_node = evaluator.add_parallel(
        id="Authority_Superintendent",
        desc="Authority to declare emergency closures is the superintendent, with official reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        auth_node,
        "Authority_Is_Superintendent",
        "States that school superintendents have the authority to declare emergency school closures.",
        "School superintendents have the authority to declare emergency school closures.",
        data.authority_urls,
        add_ins="Confirm the source (preferably a district policy/FAQ page) indicates the superintendent is the authority for closures."
    )
    await add_reference_support_leaf(
        evaluator,
        auth_node,
        "Authority_Official_Reference_URL",
        "Provides a valid official reference URL (e.g., Ohio school district policy/FAQ page) supporting superintendent authority.",
        "The referenced page is an official school district or Ohio education site confirming superintendent authority for closures.",
        data.authority_urls,
        add_ins="Confirm the URL is an official district website (e.g., k12.oh.us or district domain) or Ohio education agency page. If not official, mark Unsupported."
    )

    # Typical timeline 5:00–6:00 AM
    time_node = evaluator.add_parallel(
        id="Timeline_5_to_6_AM",
        desc="Typical timeline 5:00–6:00 AM with official reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        time_node,
        "Decision_By_5_to_6_AM",
        "States that emergency weather-closure decisions are typically made by 5:00–6:00 AM on the day of closure.",
        "Emergency weather-closure decisions are typically made by 5:00–6:00 AM on the day of closure.",
        data.timeline_urls,
        add_ins="Allow minor variations such as 'by 6 AM', 'between 5 and 6 AM', or specific times within that window. Confirm the page describes this early-morning decision timeline."
    )
    await add_reference_support_leaf(
        evaluator,
        time_node,
        "Timeline_Official_Reference_URL",
        "Provides a valid official reference URL supporting the stated 5:00–6:00 AM timeline (e.g., school district policy/FAQ).",
        "The referenced page is an official district communications/policy page describing the typical early-morning decision timeline around 5–6 AM.",
        data.timeline_urls,
        add_ins="Confirm the URL is an official district website. If not official or if the page does not describe the 5–6 AM timeline, mark Unsupported."
    )


async def build_parent_notification_system(
    evaluator: Evaluator,
    parent_node,
    data: ParentNotificationExtraction
) -> None:
    group = evaluator.add_parallel(
        id="Parent_Notification_System",
        desc="Checks that districts use mass notification systems and specified channels (phone/email/text) and includes the platform examples listed in constraints, with official reference(s).",
        parent=parent_node,
        critical=True
    )

    # Mass notification systems used
    mass_node = evaluator.add_parallel(
        id="Mass_Notification_System_Used",
        desc="States districts use mass notification systems, with official reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        mass_node,
        "States_Mass_Notification_Systems",
        "States that school districts use mass notification systems to communicate emergency closures to parents/families.",
        "School districts use mass notification systems to communicate emergency closures to parents and families.",
        data.channels_urls or data.platforms_urls,
        add_ins="Confirm the official district communications/policy page references the use of mass notification or alert systems (e.g., SchoolMessenger, alert calls, SMS notifications)."
    )
    await add_reference_support_leaf(
        evaluator,
        mass_node,
        "Notification_System_Official_Reference_URL",
        "Provides a valid official reference URL (e.g., a school district communications page/policy) supporting use of a mass notification system.",
        "The referenced page is an official district communications/policy page showing the use of a mass notification system.",
        data.channels_urls or data.platforms_urls,
        add_ins="Confirm the URL is an official district site and the page indicates the use of a mass notification platform/system."
    )

    # Channels include phone, email, text
    chan_node = evaluator.add_parallel(
        id="Notification_Channels_Phone_Email_Text",
        desc="Channels include phone, email, and text, with official reference.",
        parent=group,
        critical=True
    )
    await add_reference_support_leaf(
        evaluator,
        chan_node,
        "Includes_Phone_Email_Text",
        "Lists phone, email, and text as channels used to alert parents/families about emergency closures.",
        "School districts use phone calls, email, and text/SMS messages to alert parents/families about emergency closures.",
        data.channels_urls,
        add_ins="Confirm the official page lists phone/voice calls, email, and text/SMS as notification channels. Synonyms like 'SMS' or 'text alert' are acceptable."
    )
    await add_reference_support_leaf(
        evaluator,
        chan_node,
        "Channels_Official_Reference_URL",
        "Provides a valid official reference URL supporting the listed channels (phone/email/text).",
        "The referenced page is an official district communications/policy page that lists phone, email, and text/SMS as notification channels.",
        data.channels_urls,
        add_ins="Confirm the URL is an official district site and the page explicitly lists the channels."
    )

    # Platform examples listed
    plat_node = evaluator.add_parallel(
        id="Platform_Examples_Listed",
        desc="Includes the platform examples listed in constraints, with reference.",
        parent=group,
        critical=True
    )
    # Mentions in answer (simple verification using answer text)
    mentions_leaf = evaluator.add_leaf(
        id="Mentions_SchoolMessenger_Finalsite_Blackboard_K12Alerts",
        desc="Mentions SchoolMessenger, Finalsite/Blackboard Connect, and K12 Alerts as examples of mass notification platforms used by districts.",
        parent=plat_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer mentions SchoolMessenger, Finalsite or Blackboard Connect, and K12 Alerts as examples of district mass notification platforms.",
        node=mentions_leaf,
        additional_instruction="Check the answer text itself. Accept 'Finalsite' as equivalent to 'Blackboard Connect'. All three examples must be mentioned somewhere in the answer; if any are missing, mark Incorrect."
    )
    await add_reference_support_leaf(
        evaluator,
        plat_node,
        "Platforms_Reference_URL",
        "Provides a valid reference URL supporting the cited platform examples (preferably a school district page/policy showing use of these systems).",
        "At least one referenced official district page/policy demonstrates use of one or more of these platforms: SchoolMessenger, Finalsite/Blackboard Connect, or K12 Alerts.",
        data.platforms_urls,
        add_ins="Confirm the URL is an official district site showing use of the named platform(s). If not official or not showing platform usage, mark Unsupported."
    )


async def build_state_guidelines_reference(
    evaluator: Evaluator,
    parent_node,
    data: StateGuidelinesExtraction
) -> None:
    group = evaluator.add_parallel(
        id="State_Guidelines_Reference",
        desc="Checks reference to current Ohio state emergency guidelines for schools, including the July 2025 revision, with an official ODE reference URL.",
        parent=parent_node,
        critical=True
    )

    await add_reference_support_leaf(
        evaluator,
        group,
        "Guidelines_Name_And_Context",
        "Identifies the current Ohio state emergency guidelines for schools (document/page name and relevant context).",
        "The referenced document/page represents the current Ohio state emergency guidelines for schools, with an identifiable name and context.",
        data.guidelines_urls,
        add_ins="Confirm the page presents official state school emergency guidelines (document/page name and context) for Ohio. Prefer education.ohio.gov."
    )
    await add_reference_support_leaf(
        evaluator,
        group,
        "July_2025_Revised_Guidelines",
        "States that Ohio released revised Emergency and Health Guidelines for Schools in July 2025.",
        "Ohio released revised Emergency and Health Guidelines for Schools in July 2025.",
        data.guidelines_urls,
        add_ins="Confirm the official page or announcement indicates a revision/update in July 2025. If date does not match, mark Unsupported."
    )
    await add_reference_support_leaf(
        evaluator,
        group,
        "Guidelines_Official_Reference_URL",
        "Provides a valid official Ohio Department of Education (or equivalent official Ohio education agency) URL to the referenced guidelines/announcement.",
        "The provided source is an official Ohio Department of Education (education.ohio.gov) or equivalent Ohio education agency page hosting the guidelines or announcement.",
        data.guidelines_urls,
        add_ins="Confirm the URL is official (education.ohio.gov or equivalent state site). If the domain is not official, mark Unsupported."
    )


# ----------------------------- Main Evaluation -------------------------------

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

    # Extract all sections (in parallel)
    hours_task = evaluator.extract(
        prompt=prompt_extract_hours(),
        template_class=HoursExtraction,
        extraction_name="instructional_hours"
    )
    ohsaa_task = evaluator.extract(
        prompt=prompt_extract_ohsaa(),
        template_class=OHSAAWeatherExtraction,
        extraction_name="ohsaa_weather_policy"
    )
    closure_task = evaluator.extract(
        prompt=prompt_extract_closure(),
        template_class=ClosureAuthorityExtraction,
        extraction_name="closure_authority_timeline"
    )
    notify_task = evaluator.extract(
        prompt=prompt_extract_notification(),
        template_class=ParentNotificationExtraction,
        extraction_name="parent_notification_system"
    )
    guide_task = evaluator.extract(
        prompt=prompt_extract_guidelines(),
        template_class=StateGuidelinesExtraction,
        extraction_name="state_guidelines_reference"
    )

    hours_extraction, ohsaa_extraction, closure_extraction, notification_extraction, guidelines_extraction = await asyncio.gather(
        hours_task, ohsaa_task, closure_task, notify_task, guide_task
    )

    # Add ground truth info (for transparency; not used directly in scoring)
    evaluator.add_ground_truth({
        "expected_min_hours": {"K-6": 910, "7-12": 1001, "community_school": 920},
        "ohsaa_wait_minutes": 30,
        "typical_decision_window": "5:00–6:00 AM",
        "guidelines_revision": "July 2025"
    }, gt_type="reference_expectations")

    # Build verification tree under a critical top-level compliance node
    compliance_node = evaluator.add_parallel(
        id="Ohio_School_Emergency_Management_Compliance",
        desc="Evaluates whether the response covers all required Ohio instructional-hour rules, OHSAA inclement weather postponement rules, closure authority/timeline, family notification methods, and current Ohio emergency guidelines, each supported by official references as required.",
        parent=root,
        critical=True
    )

    await build_instructional_hours(evaluator, compliance_node, hours_extraction)
    await build_ohsaa_policy(evaluator, compliance_node, ohsaa_extraction)
    await build_closure_authority_timeline(evaluator, compliance_node, closure_extraction)
    await build_parent_notification_system(evaluator, compliance_node, notification_extraction)
    await build_state_guidelines_reference(evaluator, compliance_node, guidelines_extraction)

    return evaluator.get_summary()