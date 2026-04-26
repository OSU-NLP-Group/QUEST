import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "news_events_2026_early"
TASK_DESCRIPTION = """In early 2026, several major news events occurred in the United States and internationally. Please provide verified information about the following four events:

1. President Trump's 2026 State of the Union address: What was the exact date, time (EST), and location of the address? What major Venezuelan-related topic did he mention?

2. The U.S. capture of Nicolás Maduro: What was the date of the capture operation and its official operation name? When and where was Maduro arraigned, and what was his plea?

3. Kevin Warsh's Federal Reserve nomination: When was Kevin Warsh nominated, for what position, and whom would he replace? When does that person's term end?

4. The Israel-Iran 12-Day War: What were the start and end dates of this conflict? What was Israel's operation name for this campaign, and what type of facilities did they target in Iran?

For each event, provide specific dates, names, and key details along with a URL source that verifies the information.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Event1SOTU(BaseModel):
    date: Optional[str] = None
    time_est: Optional[str] = None
    location: Optional[str] = None
    venezuela_topic: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event2Maduro(BaseModel):
    operation_date: Optional[str] = None
    operation_name: Optional[str] = None
    arraignment_date: Optional[str] = None
    court_location: Optional[str] = None
    plea: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event3Warsh(BaseModel):
    nomination_date: Optional[str] = None
    position: Optional[str] = None
    predecessor: Optional[str] = None
    predecessor_term_end: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class Event4IranWar(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration: Optional[str] = None
    operation_name: Optional[str] = None
    target_type: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    event1: Optional[Event1SOTU] = None
    event2: Optional[Event2Maduro] = None
    event3: Optional[Event3Warsh] = None
    event4: Optional[Event4IranWar] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_events() -> str:
    return """
    Extract structured details for the four events described below exactly as presented in the answer. Do not invent or infer information. For each event, also extract the URL sources that the answer cites for that event (explicit URLs only).

    Event 1: President Trump's 2026 State of the Union
    - date: Exact date of the address (e.g., "February 24, 2026")
    - time_est: Exact start time in Eastern Time (e.g., "9:12 p.m. EST" or "9:00 PM ET")
    - location: Location where the address was delivered (e.g., "chamber of the United States Capitol" or "House chamber, U.S. Capitol")
    - venezuela_topic: The major Venezuelan-related topic mentioned (e.g., "capture of Nicolás Maduro")
    - urls: List of URL(s) provided in the answer that verify these details for Event 1

    Event 2: U.S. capture of Nicolás Maduro
    - operation_date: Date of the capture operation (e.g., "January 3, 2026")
    - operation_name: Official codename of the operation (e.g., "Operation Absolute Resolve")
    - arraignment_date: Date of arraignment (e.g., "January 5, 2026")
    - court_location: Location of arraignment (e.g., "Manhattan federal court")
    - plea: Plea entered by Maduro (and spouse if mentioned) (e.g., "not guilty")
    - urls: List of URL(s) provided in the answer that verify these details for Event 2

    Event 3: Kevin Warsh's Federal Reserve nomination
    - nomination_date: Date when Kevin Warsh was nominated (e.g., "January 30, 2026")
    - position: Position he was nominated for (e.g., "Chair of the Federal Reserve")
    - predecessor: Person he would replace (e.g., "Jerome Powell")
    - predecessor_term_end: When that person's term ends (e.g., "May 2026")
    - urls: List of URL(s) provided in the answer that verify these details for Event 3

    Event 4: Israel-Iran 12-Day War
    - start_date: Start date of the conflict (e.g., "June 13, 2025")
    - end_date: End date or ceasefire date (e.g., "June 24, 2025")
    - duration: Duration of the conflict in days (e.g., "12 days")
    - operation_name: Israel's operation name for this campaign (e.g., "Rising Lion")
    - target_type: Type(s) of facilities targeted in Iran (e.g., "nuclear and military facilities")
    - urls: List of URL(s) provided in the answer that verify these details for Event 4

    Output as a JSON object with keys: event1, event2, event3, event4, each containing the specified fields. If a field or URL is missing in the answer, set the field to null (or empty list for urls).
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _urls_present(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0 and all(isinstance(u, str) and len(u.strip()) > 0 for u in urls)


# --------------------------------------------------------------------------- #
# Event verification subtrees                                                 #
# --------------------------------------------------------------------------- #
async def build_and_verify_event1(
    evaluator: Evaluator,
    parent_node,
    e1: Optional[Event1SOTU],
) -> None:
    node = evaluator.add_parallel(
        id="Event1_Trump_State_of_Union",
        desc="Verify details of Trump's 2026 State of the Union address",
        parent=parent_node,
        critical=False,
    )

    urls_ok = _urls_present(e1.urls if e1 else [])
    evaluator.add_custom_node(
        result=urls_ok,
        id="Event1_URL",
        desc="A verifiable URL source is provided for the State of the Union details",
        parent=node,
        critical=True,
    )

    # Date
    date_leaf = evaluator.add_leaf(
        id="Event1_Date",
        desc="The State of the Union address was delivered on February 24, 2026",
        parent=node,
        critical=True,
    )
    date_val = e1.date if e1 else None
    date_claim = f"President Trump's 2026 State of the Union address was delivered on {date_val}."
    # Verify against provided URLs
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=(e1.urls if e1 else []),
        additional_instruction="Verify the exact date of the address from the cited source(s). Allow minor formatting variants like 'Feb' vs 'February'.",
    )

    # Time
    time_leaf = evaluator.add_leaf(
        id="Event1_Time",
        desc="The address began at 9:12 p.m. EST",
        parent=node,
        critical=True,
    )
    time_val = e1.time_est if e1 else None
    time_claim = f"The address began at {time_val} Eastern Time (ET)."
    await evaluator.verify(
        claim=time_claim,
        node=time_leaf,
        sources=(e1.urls if e1 else []),
        additional_instruction="Verify the official start time of the address in Eastern Time from the source(s). Accept 'ET', 'EST', or 'EDT' as context-appropriate labeling; focus on the clock time.",
    )

    # Location
    loc_leaf = evaluator.add_leaf(
        id="Event1_Location",
        desc="The address was delivered in the chamber of the United States Capitol",
        parent=node,
        critical=True,
    )
    loc_val = e1.location if e1 else None
    loc_claim = f"The address was delivered at {loc_val}."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=(e1.urls if e1 else []),
        additional_instruction="Verify that the address location matches the U.S. Capitol chamber (e.g., House chamber). Minor naming variations are acceptable if equivalent.",
    )

    # Venezuela topic
    ven_leaf = evaluator.add_leaf(
        id="Event1_Venezuela_Topic",
        desc="The address mentioned the capture of Venezuelan leader Nicolás Maduro",
        parent=node,
        critical=True,
    )
    ven_val = e1.venezuela_topic if e1 else None
    ven_claim = f"In the address, President Trump mentioned {ven_val}."
    await evaluator.verify(
        claim=ven_claim,
        node=ven_leaf,
        sources=(e1.urls if e1 else []),
        additional_instruction="Confirm from the transcript or reporting that the address mentions a Venezuelan-related topic. Specifically check for reference to the capture of Nicolás Maduro. Allow equivalent phrasing.",
    )


async def build_and_verify_event2(
    evaluator: Evaluator,
    parent_node,
    e2: Optional[Event2Maduro],
) -> None:
    node = evaluator.add_parallel(
        id="Event2_Maduro_Capture",
        desc="Verify details of the U.S. operation to capture Nicolás Maduro",
        parent=parent_node,
        critical=False,
    )

    urls_ok = _urls_present(e2.urls if e2 else [])
    evaluator.add_custom_node(
        result=urls_ok,
        id="Event2_URL",
        desc="A verifiable URL source is provided for the Maduro capture details",
        parent=node,
        critical=True,
    )

    # Operation Date
    op_date_leaf = evaluator.add_leaf(
        id="Event2_Operation_Date",
        desc="The capture operation occurred on January 3, 2026",
        parent=node,
        critical=True,
    )
    op_date_val = e2.operation_date if e2 else None
    op_date_claim = f"The U.S. capture operation of Nicolás Maduro occurred on {op_date_val}."
    await evaluator.verify(
        claim=op_date_claim,
        node=op_date_leaf,
        sources=(e2.urls if e2 else []),
        additional_instruction="Verify the date of the capture operation from cited sources.",
    )

    # Operation Name
    op_name_leaf = evaluator.add_leaf(
        id="Event2_Operation_Name",
        desc="The operation was codenamed Operation Absolute Resolve",
        parent=node,
        critical=True,
    )
    op_name_val = e2.operation_name if e2 else None
    op_name_claim = f"The operation was codenamed {op_name_val}."
    await evaluator.verify(
        claim=op_name_claim,
        node=op_name_leaf,
        sources=(e2.urls if e2 else []),
        additional_instruction="Verify the official operation codename from cited sources. Allow minor styling variants (e.g., with/without quotes).",
    )

    # Arraignment Date
    arr_date_leaf = evaluator.add_leaf(
        id="Event2_Arraignment_Date",
        desc="Maduro and his wife were arraigned on January 5, 2026",
        parent=node,
        critical=True,
    )
    arr_date_val = e2.arraignment_date if e2 else None
    arr_date_claim = f"Nicolás Maduro (and his wife, if applicable) were arraigned on {arr_date_val}."
    await evaluator.verify(
        claim=arr_date_claim,
        node=arr_date_leaf,
        sources=(e2.urls if e2 else []),
        additional_instruction="Verify the arraignment date from the cited sources. If the spouse is mentioned, ensure the statement aligns with the page.",
    )

    # Court Location
    court_leaf = evaluator.add_leaf(
        id="Event2_Court_Location",
        desc="The arraignment took place in Manhattan federal court",
        parent=node,
        critical=True,
    )
    court_val = e2.court_location if e2 else None
    court_claim = f"The arraignment took place in {court_val}."
    await evaluator.verify(
        claim=court_claim,
        node=court_leaf,
        sources=(e2.urls if e2 else []),
        additional_instruction="Verify the court location (e.g., Manhattan federal court) from the cited sources. Accept equivalent official naming variants.",
    )

    # Plea
    plea_leaf = evaluator.add_leaf(
        id="Event2_Plea",
        desc="Both Maduro and his wife pleaded not guilty",
        parent=node,
        critical=True,
    )
    plea_val = e2.plea if e2 else None
    plea_claim = f"Both Nicolás Maduro and his wife pleaded {plea_val}."
    await evaluator.verify(
        claim=plea_claim,
        node=plea_leaf,
        sources=(e2.urls if e2 else []),
        additional_instruction="Verify the plea entered (e.g., 'not guilty') from the cited sources. If the spouse is not mentioned in sources, ensure the claim's scope matches the source.",
    )


async def build_and_verify_event3(
    evaluator: Evaluator,
    parent_node,
    e3: Optional[Event3Warsh],
) -> None:
    node = evaluator.add_parallel(
        id="Event3_Warsh_Nomination",
        desc="Verify details of Kevin Warsh's Federal Reserve nomination",
        parent=parent_node,
        critical=False,
    )

    urls_ok = _urls_present(e3.urls if e3 else [])
    evaluator.add_custom_node(
        result=urls_ok,
        id="Event3_URL",
        desc="A verifiable URL source is provided for the Warsh nomination details",
        parent=node,
        critical=True,
    )

    # Nomination Date
    nom_date_leaf = evaluator.add_leaf(
        id="Event3_Nomination_Date",
        desc="Kevin Warsh was nominated on January 30, 2026",
        parent=node,
        critical=True,
    )
    nom_date_val = e3.nomination_date if e3 else None
    nom_date_claim = f"Kevin Warsh was nominated on {nom_date_val}."
    await evaluator.verify(
        claim=nom_date_claim,
        node=nom_date_leaf,
        sources=(e3.urls if e3 else []),
        additional_instruction="Verify the nomination date from the cited source(s).",
    )

    # Position
    pos_leaf = evaluator.add_leaf(
        id="Event3_Position",
        desc="He was nominated to serve as Chair of the Federal Reserve",
        parent=node,
        critical=True,
    )
    pos_val = e3.position if e3 else None
    pos_claim = f"Kevin Warsh was nominated to serve as {pos_val} of the Federal Reserve."
    await evaluator.verify(
        claim=pos_claim,
        node=pos_leaf,
        sources=(e3.urls if e3 else []),
        additional_instruction="Verify the position for which Kevin Warsh was nominated (e.g., Chair of the Federal Reserve).",
    )

    # Predecessor
    pred_leaf = evaluator.add_leaf(
        id="Event3_Predecessor",
        desc="He would replace Jerome Powell",
        parent=node,
        critical=True,
    )
    pred_val = e3.predecessor if e3 else None
    pred_claim = f"Kevin Warsh would replace {pred_val}."
    await evaluator.verify(
        claim=pred_claim,
        node=pred_leaf,
        sources=(e3.urls if e3 else []),
        additional_instruction="Verify whom Kevin Warsh would replace. Accept minor name variants (e.g., with/without middle initial).",
    )

    # Predecessor Term End
    term_leaf = evaluator.add_leaf(
        id="Event3_Predecessor_Term_End",
        desc="Powell's term as Chair ends in May 2026",
        parent=node,
        critical=True,
    )
    term_val = e3.predecessor_term_end if e3 else None
    # If the predecessor is Powell, phrase accordingly; else generic phrasing
    if e3 and e3.predecessor and ("Powell" in e3.predecessor or "Jerome Powell" in e3.predecessor):
        term_claim = f"Jerome Powell's term as Chair ends in {term_val}."
    else:
        term_claim = f"The predecessor's term as Chair ends in {term_val}."
    await evaluator.verify(
        claim=term_claim,
        node=term_leaf,
        sources=(e3.urls if e3 else []),
        additional_instruction="Verify the end of term month/year for the predecessor (typically Jerome Powell). Accept minor phrasing variants as long as the date is explicit.",
    )


async def build_and_verify_event4(
    evaluator: Evaluator,
    parent_node,
    e4: Optional[Event4IranWar],
) -> None:
    node = evaluator.add_parallel(
        id="Event4_Iran_War",
        desc="Verify details of the Israel-Iran 12-Day War",
        parent=parent_node,
        critical=False,
    )

    urls_ok = _urls_present(e4.urls if e4 else [])
    evaluator.add_custom_node(
        result=urls_ok,
        id="Event4_URL",
        desc="A verifiable URL source is provided for the Iran War details",
        parent=node,
        critical=True,
    )

    # Start Date
    start_leaf = evaluator.add_leaf(
        id="Event4_Start_Date",
        desc="The conflict began on June 13, 2025",
        parent=node,
        critical=True,
    )
    start_val = e4.start_date if e4 else None
    start_claim = f"The Israel-Iran conflict began on {start_val}."
    await evaluator.verify(
        claim=start_claim,
        node=start_leaf,
        sources=(e4.urls if e4 else []),
        additional_instruction="Verify the start date of the conflict. Accept minor date formatting variations.",
    )

    # End Date
    end_leaf = evaluator.add_leaf(
        id="Event4_End_Date",
        desc="The conflict ended with a ceasefire on June 24, 2025",
        parent=node,
        critical=True,
    )
    end_val = e4.end_date if e4 else None
    end_claim = f"The conflict ended (ceasefire) on {end_val}."
    await evaluator.verify(
        claim=end_claim,
        node=end_leaf,
        sources=(e4.urls if e4 else []),
        additional_instruction="Verify the end or ceasefire date of the conflict.",
    )

    # Duration
    dur_leaf = evaluator.add_leaf(
        id="Event4_Duration",
        desc="The conflict lasted 12 days",
        parent=node,
        critical=True,
    )
    dur_val = e4.duration if e4 else None
    dur_claim = f"The conflict lasted {dur_val}."
    await evaluator.verify(
        claim=dur_claim,
        node=dur_leaf,
        sources=(e4.urls if e4 else []),
        additional_instruction="Verify the stated duration in days. Allow numeric and textual variants (e.g., '12' vs 'twelve').",
    )

    # Operation Name
    op_leaf = evaluator.add_leaf(
        id="Event4_Operation_Name",
        desc="Israel's operation was named Rising Lion",
        parent=node,
        critical=True,
    )
    op_val = e4.operation_name if e4 else None
    op_claim = f"Israel's operation for this campaign was named {op_val}."
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=(e4.urls if e4 else []),
        additional_instruction="Verify Israel's operation name for this campaign. Accept minor styling variants.",
    )

    # Target Type
    tgt_leaf = evaluator.add_leaf(
        id="Event4_Target",
        desc="Israeli strikes targeted Iranian nuclear and military facilities",
        parent=node,
        critical=True,
    )
    tgt_val = e4.target_type if e4 else None
    tgt_claim = f"Israeli strikes targeted {tgt_val} in Iran."
    await evaluator.verify(
        claim=tgt_claim,
        node=tgt_leaf,
        sources=(e4.urls if e4 else []),
        additional_instruction="Verify the types of Iranian facilities targeted (e.g., nuclear and military facilities).",
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
    Evaluate an answer for the multi-event verification task (early 2026 and late 2025 events).
    """
    # Initialize evaluator with parallel aggregation at root
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

    # Extract event details from the answer
    events = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Build verification tree for each event
    await build_and_verify_event1(evaluator, root, events.event1)
    await build_and_verify_event2(evaluator, root, events.event2)
    await build_and_verify_event3(evaluator, root, events.event3)
    await build_and_verify_event4(evaluator, root, events.event4)

    # Return structured evaluation summary
    return evaluator.get_summary()