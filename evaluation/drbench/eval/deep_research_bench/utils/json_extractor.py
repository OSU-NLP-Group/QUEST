import json
import re

def extract_json_from_markdown(text):
    """Extract JSON from a markdown text that may contain ```json ... ``` blocks
    
    Args:
        text (str): The input text which might contain JSON blocks
        
    Returns:
        str or None: The extracted JSON string or None if not found/not valid
    """
    if not isinstance(text, str):
        return None
    
    # Method 0: Try to parse the complete text directly, if it's valid JSON return it
    # This handles cases where LLM returns a direct JSON object
    if text.strip().startswith('{') and text.strip().endswith('}'):
        try:
            # Only validate, no need to assign to variable
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            # If not valid JSON, continue with other methods
            pass
    
    # Method 1: Use string operations to extract JSON directly from code blocks
    if "```json" in text and "```" in text[text.find("```json")+7:]:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            json_str = text[start:end].strip()
            try:
                # Validate JSON
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # If validation fails, continue with other methods
                pass
    
    # Method 2: As backup, retain original regex matching
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        json_str = match.group(1).strip()
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # If validation fails, continue with other methods
            pass
    
    # Method 3: Check if the entire text is directly JSON
    try:
        json_obj = json.loads(text.strip())
        return text.strip()
    except json.JSONDecodeError:
        # Not valid JSON, continue trying
        pass
    
    # Method 4: Try to extract content within outermost curly braces
    # Find the first left curly brace
    start = text.find('{')
    if start != -1:
        # Calculate nesting level to find the matching closing brace
        level = 0
        for i, char in enumerate(text[start:]):
            if char == '{':
                level += 1
            elif char == '}':
                level -= 1
                if level == 0:
                    # Found matching closing brace
                    end = start + i + 1
                    potential_json = text[start:end]
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        # If validation fails, continue with other methods
                        pass
                    break
    
    # Method 5: Final fallback method, try simple start and end brace matching
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = text[start:end+1]
        try:
            json.loads(potential_json)
            return potential_json
        except json.JSONDecodeError:
            # All methods failed, try final fallback method
            pass
    
    # Method 6: Special fallback method - build simplified JSON object by keyword pattern matching
    if "comprehensiveness" in text and "article_1_score" in text and "article_2_score" in text:
        try:
            dimensions = ["comprehensiveness", "insight", "instruction_following", "readability"]
            result = {}
            
            for dim in dimensions:
                if dim in text:
                    result[dim] = []
                    
                    # Search for all scoring entries under this dimension
                    # First find the dimension starting position
                    dim_start = text.find(f'"{dim}"') 
                    if dim_start == -1:
                        dim_start = text.find(f"'{dim}'")
                    if dim_start == -1:
                        dim_start = text.find(dim)
                    
                    if dim_start != -1:
                        # Determine dimension end position (next dimension start or end of text)
                        next_dim_start = len(text)
                        for next_dim in dimensions:
                            if next_dim != dim:
                                pos = text.find(f'"{next_dim}"', dim_start)
                                if pos == -1:
                                    pos = text.find(f"'{next_dim}'", dim_start)
                                if pos == -1:
                                    pos = text.find(next_dim, dim_start + len(dim))
                                if pos != -1 and pos < next_dim_start:
                                    next_dim_start = pos
                        
                        # Extract content for this dimension
                        dim_content = text[dim_start:next_dim_start]
                        
                        # Use regex to find all "criterion", "article_1_score" and "article_2_score"
                        criterion_matches = re.finditer(r'"criterion"\s*:\s*"([^"]+)"', dim_content)
                        score1_matches = re.finditer(r'"article_1_score"\s*:\s*(\d+\.?\d*)', dim_content)
                        score2_matches = re.finditer(r'"article_2_score"\s*:\s*(\d+\.?\d*)', dim_content)
                        
                        # Convert to lists for multiple access
                        criteria = [m.group(1) for m in criterion_matches]
                        scores1 = [float(m.group(1)) for m in score1_matches]
                        scores2 = [float(m.group(1)) for m in score2_matches]
                        
                        # Combine into scoring entries
                        for i in range(min(len(criteria), len(scores1), len(scores2))):
                            result[dim].append({
                                "criterion": criteria[i],
                                "article_1_score": scores1[i],
                                "article_2_score": scores2[i]
                            })
            
            # Validate if we successfully extracted scoring data
            if any(len(scores) > 0 for scores in result.values()):
                return json.dumps(result)
        except Exception as e:
            # Fallback extraction method failed, log error but continue trying
            pass
    
    # All methods failed, return None
    return None 