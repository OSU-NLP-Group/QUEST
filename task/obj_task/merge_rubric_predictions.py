#!/usr/bin/env python3
"""Documentation omitted."""

import json
import re
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

def extract_rubric_tree_from_second_last(content: str) -> Optional[tuple]:
    """Documentation omitted."""
    patterns = [
        r'(?i)##\s*STEP\s+4[^\n]*Rubric\s+Tree',
        r'(?i)##\s*Rubric\s+Tree',
        r'(?i)#\s*Rubric\s+Tree',
        r'(?i)Rubric\s+Tree:',
    ]
    
    start_pos = None
    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            start_pos = match.start()
            break
    
    if start_pos is None:
        return None
    end_tag_match = re.search(r'</rubric_tree>', content[start_pos:], re.IGNORECASE)
    if end_tag_match:
        end_pos = start_pos + end_tag_match.end()
        return (start_pos, end_pos, content[start_pos:end_pos])
    next_section = re.search(r'(?i)\n##\s*STEP\s+5', content[start_pos:])
    if next_section:
        end_pos = start_pos + next_section.start()
        return (start_pos, end_pos, content[start_pos:end_pos])
    next_header = re.search(r'\n##\s+', content[start_pos + 100:])
    if next_header:
        end_pos = start_pos + 100 + next_header.start()
        return (start_pos, end_pos, content[start_pos:end_pos])
    return (start_pos, len(content), content[start_pos:])

def extract_rubric_tree_from_last(content: str) -> Optional[str]:
    """Documentation omitted."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and 'rubric_tree' in data:
            rubric_tree = data['rubric_tree']
            if isinstance(rubric_tree, str):
                return rubric_tree
            elif isinstance(rubric_tree, dict):
                return json.dumps(rubric_tree, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    rubric_start = re.search(r'(?i)##\s*STEP\s+4[^\n]*Rubric\s+Tree', content)
    if rubric_start:
        start_pos = rubric_start.start()
        end_tag_match = re.search(r'</rubric_tree>', content[start_pos:], re.IGNORECASE)
        if end_tag_match:
            end_pos = start_pos + end_tag_match.end()
            return content[start_pos:end_pos]
    
    return None

def extract_answer_from_prediction(prediction: str) -> Optional[str]:
    """Documentation omitted."""
    answer_match = re.search(r'<answer[^>]*>(.*?)</answer>', prediction, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_content = answer_match.group(1)
        return answer_content.strip()
    
    return None

def replace_rubric_tree_in_answer(answer_content: str, new_rubric_tree_dict: Optional[Dict]) -> str:
    """Documentation omitted."""
    if new_rubric_tree_dict is None:
        return answer_content
    
    try:
        answer_json = json.loads(answer_content)
        if isinstance(answer_json, dict) and 'rubric_tree' in answer_json:
            answer_json['rubric_tree'] = new_rubric_tree_dict
            return json.dumps(answer_json, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        pass
    
    return answer_content

def process_file(file_path: Path) -> bool:
    """Documentation omitted."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'messages' not in data:
            print(f"Warning: {file_path.name} has no messages field, skipping")
            return False
        assistants = [m for m in data['messages'] if m.get('role') == 'assistant']
        
        if len(assistants) < 2:
            print(f"Warning: {file_path.name} only has {len(assistants)} assistant messages; need at least 2, skipping")
            return False
        second_last_content = assistants[-2].get('content', '')
        last_content = assistants[-1].get('content', '')
        
        if not second_last_content:
            print(f"Warning: second-last assistant content is empty in {file_path.name}, skipping")
            return False
        new_rubric_tree = extract_rubric_tree_from_last(last_content)
        new_rubric_tree_dict = None
        try:
            last_json = json.loads(last_content)
            if isinstance(last_json, dict) and 'rubric_tree' in last_json:
                new_rubric_tree_dict = last_json['rubric_tree']
        except (json.JSONDecodeError, TypeError):
            pass
        
        if new_rubric_tree is None:
            print(f"Warning: could not extract rubric tree from the last assistant message in {file_path.name}; using original content")
            data['prediction'] = second_last_content
        else:
            rubric_info = extract_rubric_tree_from_second_last(second_last_content)
            
            if rubric_info is None:
                print(f"Warning: could not find rubric tree in the second-last assistant message in {file_path.name}; using original content")
                data['prediction'] = second_last_content
            else:
                start_pos, end_pos, old_rubric_tree = rubric_info
                title_match = re.match(r'(##\s*STEP\s+4[^\n]*Rubric\s+Tree[^\n]*)', old_rubric_tree, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1)
                    if not re.match(r'(?i)##\s*STEP\s+4[^\n]*Rubric\s+Tree', new_rubric_tree):
                        new_rubric_tree = title + '\n\n' + new_rubric_tree
                new_content = (
                    second_last_content[:start_pos] + 
                    new_rubric_tree + 
                    second_last_content[end_pos:]
                )
                
                data['prediction'] = new_content
        answer_content = extract_answer_from_prediction(data.get('prediction', ''))
        if answer_content:
            answer_content = replace_rubric_tree_in_answer(answer_content, new_rubric_tree_dict)
            data['answer'] = answer_content
        else:
            data['answer'] = ''
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed: {file_path.name}")
        return True
        
    except Exception as e:
        print(f"Error: failed to process {file_path.name}: {str(e)}")
        return False

def main():
    """Documentation omitted."""
    parser = argparse.ArgumentParser(description="Merge refined rubric predictions back into trajectory JSON files.")
    parser.add_argument(
        "--input-dir",
        default="./outputs/objective_trajectories",
        help="Directory containing raw trajectory JSON files.",
    )
    args = parser.parse_args()

    target_dir = Path(args.input_dir)
    if not target_dir.exists():
        print(f"Error: directory does not exist: {target_dir}")
        return
    
    if not target_dir.is_dir():
        print(f"Error: not a directory: {target_dir}")
        return
    json_files = list(target_dir.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {target_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files in {target_dir}")
    print("Starting processing...\n")
    
    success_count = 0
    fail_count = 0
    
    for json_file in json_files:
        if process_file(json_file):
            success_count += 1
        else:
            fail_count += 1
    
    print("\nDone!")
    print(f"Succeeded: {success_count} files")
    print(f"Failed: {fail_count} files")

if __name__ == '__main__':
    main()
