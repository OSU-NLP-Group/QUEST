import asyncio
import logging
from typing import Optional, List, Dict, Any

import openai
from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "protein_vis_software"
TASK_DESCRIPTION = """
Identify four widely-used protein visualization software tools commonly employed in structural biology. For each tool, provide the developer's name, supported operating systems (Windows, macOS, Linux/Unix), a direct link to the software's official website, and indicate whether it is free, paid, or has both free and paid versions available.
"""

JUDGE_MODEL = "o4-mini"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SoftwareTool(BaseModel):
    """Represents a single protein visualization software tool."""
    name: Optional[str] = None
    developer: Optional[str] = None
    operating_systems: List[str] = Field(default_factory=list)
    website_url: Optional[str] = None
    pricing_model: Optional[str] = None

class SoftwareToolsList(BaseModel):
    """List of protein visualization software tools."""
    tools: List[SoftwareTool] = Field(default_factory=list)

class ToolURLs(BaseModel):
    """Simple list of URLs related to a specific software tool."""
    urls: List[str] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_software_tools() -> str:
    return """
    Please extract all protein visualization software tools mentioned in the answer.
    
    For each tool, extract:
    1. The name of the software tool
    2. The developer's name
    3. The supported operating systems (Windows, macOS, Linux/Unix)
    4. The direct link to the software's official website
    5. The pricing model (free, paid, or both free and paid versions)
    
    Return the information in a structured format where each tool is a separate object with the fields:
    - name: The name of the software
    - developer: The name of the developer or organization
    - operating_systems: List of supported operating systems
    - website_url: The official website URL
    - pricing_model: The pricing model ('free', 'paid', or 'both')
    
    If any information is missing for a specific tool, set the corresponding field to null.
    """

def prompt_extract_urls_for_tool(tool_name: str) -> str:
    return f"""
    Extract all the URLs mentioned in the answer that are related to the software tool named "{tool_name}".
    This should include the official website URL and any other relevant URLs mentioned in connection with this tool.
    
    Simply return a list of URL strings.
    """

# --------------------------------------------------------------------------- #
# Tool verification function                                                  #
# --------------------------------------------------------------------------- #
async def verify_single_tool(
    evaluator: Evaluator,
    parent_node,
    tool_index: int,
    tool: SoftwareTool
) -> None:
    """
    Verify all aspects of a single protein visualization software tool.
    """
    # Create tool node with parallel strategy
    tool_node = evaluator.add_parallel(
        id=f"tool_{tool_index}_verification",
        desc=f"Verify all information for tool #{tool_index+1}: '{tool.name or 'Missing'}'",
        parent=parent_node,
        critical=False  # Non-critical to allow partial credit across tools
    )
    
    # Single completeness check for ALL required fields
    tool_completeness = evaluator.add_custom_node(
        result=(bool(tool.name) and bool(tool.website_url) and 
                bool(tool.developer) and bool(tool.operating_systems) and 
                bool(tool.pricing_model)),
        id=f"tool_{tool_index}_completeness",
        desc=f"Check if all required information is provided for tool #{tool_index+1}",
        parent=tool_node,
        critical=True  # Critical gate for all verifications
    )
    
    # Extract all URLs related to this tool (only if tool has a name)
    tool_urls = []
    if tool.name:
        urls_info = await evaluator.extract(
            prompt=prompt_extract_urls_for_tool(tool.name),
            template_class=ToolURLs,
            extraction_name=f"tool_{tool_index}_urls"
        )
        tool_urls = urls_info.urls
        
    # Add the website_url if it's not already in the list
    if tool.website_url and tool.website_url not in tool_urls:
        tool_urls.append(tool.website_url)
    
    # Website verification
    website_verification = evaluator.add_leaf(
        id=f"tool_{tool_index}_website_verification",
        desc=f"Verify that the website is official for '{tool.name}'",
        parent=tool_node,
        critical=True  
    )
    
    claim = f"The website is the official website for the protein visualization software tool '{tool.name}'."
    await evaluator.verify(
        claim=claim,
        node=website_verification,
        sources=tool.website_url,
        additional_instruction="Verify that this is the official website for the software tool."
    )
    
    # Developer verification
    developer_verification = evaluator.add_leaf(
        id=f"tool_{tool_index}_developer_verification",
        desc=f"Verify developer information",
        parent=tool_node,
        critical=True  
    )
    
    claim = f"The developer of the protein visualization software tool '{tool.name}' is '{tool.developer}'."
    await evaluator.verify(
        claim=claim,
        node=developer_verification,
        sources=tool_urls if tool_urls else None
    )
    
    # OS verification
    os_verification = evaluator.add_leaf(
        id=f"tool_{tool_index}_os_verification",
        desc=f"Verify operating system compatibility",
        parent=tool_node,
        critical=True  
    )
    
    os_list = ", ".join(tool.operating_systems) if tool.operating_systems else ""
    claim = f"The protein visualization software tool '{tool.name}' supports the following operating systems: {os_list}."
    await evaluator.verify(
        claim=claim,
        node=os_verification,
        sources=tool_urls if tool_urls else None
    )
    
    # Pricing verification
    pricing_verification = evaluator.add_leaf(
        id=f"tool_{tool_index}_pricing_verification",
        desc=f"Verify pricing model",
        parent=tool_node,
        critical=True  
    )
    
    if tool.pricing_model == "both":
        claim = f"The protein visualization software tool '{tool.name}' has a free version and a paid version."
    else:
        claim = f"The protein visualization software tool '{tool.name}' has a {tool.pricing_model} version."
    
    await evaluator.verify(
        claim=claim,
        node=pricing_verification,
        sources=tool_urls if tool_urls else None
    )

# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: openai.AsyncAzureOpenAI,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer and return a structured result dictionary.
    """
    # -------- 1. Set up evaluator ---------------------------------------- #
    evaluator = Evaluator()
    
    # Initialize evaluator
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
        default_model=model if model else JUDGE_MODEL
    )
    
    # -------- 2. Extract software tools information --------------------- #
    tools_list = await evaluator.extract(
        prompt=prompt_extract_software_tools(),
        template_class=SoftwareToolsList,
        extraction_name="software_tools"
    )
    
    # -------- 3. Build verification tree -------------------------------- #
    # Ensure we have exactly 4 tools (pad with empty tools if needed)
    required_tool_count = 4
    while len(tools_list.tools) < required_tool_count:
        tools_list.tools.append(SoftwareTool())
    
    # Verify each tool (only first 4 if more were provided)
    tools_to_verify = tools_list.tools[:required_tool_count]
    
    for i, tool in enumerate(tools_to_verify):
        await verify_single_tool(evaluator, root, i, tool)
    
    # -------- 4. Return structured result ------------------------------- #
    return evaluator.get_summary()