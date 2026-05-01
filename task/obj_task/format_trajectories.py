import json
import os
import glob
import argparse
from typing import Dict, Any, Tuple, Optional

def calculate_rubric_tree_depth(node: Dict[str, Any], current_depth: int = 1) -> int:
    """Documentation omitted."""
    if not isinstance(node, dict):
        return current_depth
    
    max_depth = current_depth
    if "children" in node and isinstance(node["children"], dict):
        for child in node["children"].values():
            child_depth = calculate_rubric_tree_depth(child, current_depth + 1)
            max_depth = max(max_depth, child_depth)
    
    return max_depth

def calculate_rubric_tree_width(node: Dict[str, Any]) -> Tuple[int, Dict[int, int]]:
    """Documentation omitted."""
    def count_nodes_at_level(node: Dict[str, Any], level: int, level_counts: Dict[int, int]):
        if not isinstance(node, dict):
            return
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
        if "children" in node and isinstance(node["children"], dict):
            for child in node["children"].values():
                count_nodes_at_level(child, level + 1, level_counts)
    
    level_counts = {}
    count_nodes_at_level(node, 1, level_counts)
    
    max_width = max(level_counts.values()) if level_counts else 0
    return max_width, level_counts

def analyze_rubric_tree(rubric_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Documentation omitted."""
    if not isinstance(rubric_tree, dict):
        return {"max_depth": 0, "max_width": 0, "width_by_level": {}}
    all_level_counts = {}
    max_depth = 0
    max_width = 0
    
    for root_name, root_node in rubric_tree.items():
        if isinstance(root_node, dict):
            root_depth = calculate_rubric_tree_depth(root_node, 1)
            root_width, root_level_counts = calculate_rubric_tree_width(root_node)
            
            max_depth = max(max_depth, root_depth)
            max_width = max(max_width, root_width)
            for level, count in root_level_counts.items():
                if level not in all_level_counts:
                    all_level_counts[level] = 0
                all_level_counts[level] += count
    
    return {
        "max_depth": max_depth,
        "max_width": max_width,
        "width_by_level": all_level_counts
    }

def format_rubric_tree_node(node: Dict[str, Any], node_name: str = "", level: int = 1, prefix: str = "", is_last: bool = True) -> str:
    """Documentation omitted."""
    lines = []
    
    if isinstance(node, dict):
        description = node.get("description", "")
        critical = node.get("critical", False)
        strategy = node.get("aggregation_strategy", "")
        critical_mark = "🔴 CRITICAL" if critical else "⚪ NON-CRITICAL"
        if level > 1:
            if is_last:
                connector = "└── "
                continuation = "    "
            else:
                connector = "├── "
                continuation = "│   "
            current_prefix = prefix + connector
            next_prefix = prefix + continuation
        else:
            current_prefix = ""
            next_prefix = ""
        if node_name:
            lines.append(f"{current_prefix}📌 {node_name}")
            lines.append(f"{next_prefix}   Type: {critical_mark}")
            lines.append(f"{next_prefix}   Aggregation Strategy: {strategy}")
            lines.append(f"{next_prefix}   Description: {description}")
        else:
            lines.append(f"{current_prefix}Type: {critical_mark}")
            lines.append(f"{current_prefix}Aggregation Strategy: {strategy}")
            lines.append(f"{current_prefix}Description: {description}")
        if "children" in node and isinstance(node["children"], dict):
            children = list(node["children"].items())
            for idx, (child_name, child_node) in enumerate(children):
                child_is_last = (idx == len(children) - 1)
                child_lines = format_rubric_tree_node(child_node, child_name, level + 1, next_prefix, child_is_last)
                if child_lines:
                    lines.append(child_lines)
    
    return "\n".join(lines)

def format_rubric_tree_to_json(rubric_tree: Dict[str, Any]) -> Dict[str, Any]:
    """Documentation omitted."""
    def node_to_dict(node: Dict[str, Any], node_name: str = "") -> Dict[str, Any]:
        """Documentation omitted."""
        result = {}
        if not isinstance(node, dict):
            return result
        if node_name:
            result["node_name"] = node_name
        result["description"] = node.get("description", "")
        result["critical"] = bool(node.get("critical", False))
        result["critical_type"] = "CRITICAL" if result["critical"] else "NON-CRITICAL"
        aggregation_strategy = node.get("aggregation_strategy", "")
        result["aggregation_strategy"] = aggregation_strategy if aggregation_strategy else ""
        result["children"] = []
        if "children" in node:
            children = node["children"]
            if isinstance(children, dict) and len(children) > 0:
                for child_name, child_node in children.items():
                    if isinstance(child_node, dict):
                        child_dict = node_to_dict(child_node, child_name)
                        result["children"].append(child_dict)
            elif isinstance(children, list) and len(children) > 0:
                for idx, child_node in enumerate(children):
                    if isinstance(child_node, dict):
                        child_name = child_node.get("node_name", f"child_{idx}")
                        child_dict = node_to_dict(child_node, child_name)
                        result["children"].append(child_dict)
        
        return result
    
    formatted = {}
    if not isinstance(rubric_tree, dict):
        return formatted
    
    for root_name, root_node in rubric_tree.items():
        if isinstance(root_node, dict):
            formatted[root_name] = node_to_dict(root_node, root_name)
    
    return formatted

def format_rubric_tree(rubric_tree: Dict[str, Any]) -> str:
    """Documentation omitted."""
    lines = []
    if not isinstance(rubric_tree, dict):
        return ""
    
    for root_idx, (root_name, root_node) in enumerate(rubric_tree.items(), 1):
        if isinstance(root_node, dict):
            lines.append("")
            lines.append("═" * 80)
            lines.append(f"ROOT NODE {root_idx}: {root_name}")
            lines.append("═" * 80)
            lines.append("")
            root_description = root_node.get("description", "")
            root_critical = root_node.get("critical", False)
            root_strategy = root_node.get("aggregation_strategy", "")
            root_critical_mark = "🔴 CRITICAL" if root_critical else "⚪ NON-CRITICAL"
            
            lines.append(f"Type: {root_critical_mark}")
            lines.append(f"Aggregation Strategy: {root_strategy}")
            lines.append(f"Description: {root_description}")
            lines.append("")
            if "children" in root_node and isinstance(root_node["children"], dict):
                children = list(root_node["children"].items())
                lines.append("Children:")
                lines.append("")
                for idx, (child_name, child_node) in enumerate(children):
                    is_last = (idx == len(children) - 1)
                    child_lines = format_rubric_tree_node(child_node, child_name, level=2, prefix="", is_last=is_last)
                    if child_lines:
                        lines.append(child_lines)
                    if idx < len(children) - 1:
                        lines.append("")
            
            lines.append("")
    
    return "\n".join(lines)

def parse_trajectory_file(filepath: str) -> Dict[str, Any]:
    """Documentation omitted."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get("messages", [])
    result = {
        "metadata": {
            "question": data.get("question", ""),
            "answer": data.get("answer", ""),
            "complexity_class": data.get("complexity_class", "Unknown"),
            "termination": data.get("termination", ""),
        },
        "rubric_tree_analysis": {},
        "answer_data": {},
        "messages": messages,
    }
    answer_str = data.get("answer", "")
    answer_data = {}
    if answer_str:
        try:
            if isinstance(answer_str, str):
                answer_data = json.loads(answer_str)
                result["answer_data"] = answer_data
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse answer field as JSON: {e}")
            result["answer_data"] = {}
    rubric_tree = None
    if "refined_rubric_tree" in data and data["refined_rubric_tree"]:
        rubric_tree = data["refined_rubric_tree"]
    elif data.get("decision", "").lower() == "valid":
        if answer_data and "rubric_tree" in answer_data:
            rubric_tree = answer_data["rubric_tree"]
            print(f"[Info] Using rubric_tree from answer field (decision=valid, no refined_rubric_tree)")
    elif answer_data and "rubric_tree" in answer_data:
        rubric_tree = answer_data["rubric_tree"]
        print(f"[Info] Using rubric_tree from answer field (fallback)")
    if rubric_tree and isinstance(rubric_tree, dict):
        analysis = analyze_rubric_tree(rubric_tree)
        result["rubric_tree_analysis"] = {
            "max_depth": analysis["max_depth"],
            "max_width": analysis["max_width"],
            "width_by_level": analysis["width_by_level"],
            "formatted_tree": format_rubric_tree_to_json(rubric_tree)
        }
    
    return result

def save_formatted_output(parsed_data: Dict[str, Any], output_filepath: str):
    """Documentation omitted."""
    refined_analysis = parsed_data.get("rubric_tree_analysis", {})
    answer_data = parsed_data.get("answer_data") or {}
    meta_question = parsed_data.get("metadata", {}).get("question", "")
    proposed_question = answer_data.get("proposed_question") or meta_question
    constraints = answer_data.get("constraints", [])
    solution = answer_data.get("solution", {})
    
    output = {
        "metadata": parsed_data["metadata"],
        "proposed_question": proposed_question,
        "constraints": constraints,
        "rubric_tree_analysis_refined": refined_analysis,
        "solution": solution
    }
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"Formatted output saved to: {output_filepath}")

def process_trajectory_file(input_file: str, output_file: Optional[str] = None, verbose: bool = True):
    """Documentation omitted."""
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return None
    if verbose:
        print(f"Parsing trajectory file: {input_file}")
    parsed_data = parse_trajectory_file(input_file)
    if verbose:
        print("\n" + "="*80)
        print("METADATA")
        print("="*80)
        metadata = parsed_data["metadata"]
        print(f"Question: {metadata['question']}")
        print(f"Complexity Class: {metadata['complexity_class']}")
        print(f"Termination: {metadata['termination']}")

        def _print_rubric_analysis_block(analysis: Dict[str, Any], title: str):
            print("\n" + "="*80)
            print(title)
            print("="*80)
            if not analysis:
                print("No rubric tree found")
                return

            max_depth = analysis.get('max_depth', 0)
            max_width = analysis.get('max_width', 0)
            print(f"Maximum Depth: {max_depth}")
            print(f"Maximum Width: {max_width}")
            complexity_class = metadata.get('complexity_class', '')
            if complexity_class:
                complexity_requirements = {
                    'C1': {'breadth': (1, 8), 'depth': (2, 2)},
                    'C2': {'breadth': (1, 8), 'depth': (3, 4)},
                    'C3': {'breadth': (1, 8), 'depth': (5, 6)},
                    'C4': {'breadth': (8, 20), 'depth': (2, 2)},
                    'C5': {'breadth': (8, 20), 'depth': (3, 4)},
                    'C6': {'breadth': (8, 20), 'depth': (5, 6)},
                    'C7': {'breadth': (21, float('inf')), 'depth': (2, 2)},
                    'C8': {'breadth': (21, float('inf')), 'depth': (3, 4)},
                    'C9': {'breadth': (21, float('inf')), 'depth': (5, 6)},
                }
                
                if complexity_class in complexity_requirements:
                    req = complexity_requirements[complexity_class]
                    breadth_min, breadth_max = req['breadth']
                    depth_min, depth_max = req['depth']
                    
                    breadth_ok = breadth_min <= max_width <= breadth_max if breadth_max != float('inf') else max_width >= breadth_min
                    depth_ok = depth_min <= max_depth <= depth_max
                    
                    print(f"\nCompliance Check for {complexity_class}:")
                    print(f"  Breadth requirement: {breadth_min}-{breadth_max if breadth_max != float('inf') else '∞'} nodes per layer")
                    print(f"  Actual Breadth: {max_width} nodes {'✓' if breadth_ok else '✗ (NOT COMPLIANT)'}")
                    print(f"  Depth requirement: {depth_min}-{depth_max} layers")
                    print(f"  Actual Depth: {max_depth} layers {'✓' if depth_ok else '✗ (NOT COMPLIANT)'}")

                    if not breadth_ok or not depth_ok:
                        suggested_class = None
                        if max_width < 8:
                            if max_depth == 2:
                                suggested_class = 'C1'
                            elif 3 <= max_depth <= 4:
                                suggested_class = 'C2'
                            elif max_depth >= 5:
                                suggested_class = 'C3'
                        elif 8 <= max_width <= 20:
                            if max_depth == 2:
                                suggested_class = 'C4'
                            elif 3 <= max_depth <= 4:
                                suggested_class = 'C5'
                            elif max_depth >= 5:
                                suggested_class = 'C6'
                        elif max_width >= 21:
                            if max_depth == 2:
                                suggested_class = 'C7'
                            elif 3 <= max_depth <= 4:
                                suggested_class = 'C8'
                            elif max_depth >= 5:
                                suggested_class = 'C9'

                        if suggested_class and suggested_class != complexity_class:
                            print(f"\n  ⚠️  Suggested Complexity Class: {suggested_class} (based on actual Breadth={max_width}, Depth={max_depth})")
                        elif not suggested_class:
                            if max_width > 20:
                                print(f"\n  ⚠️  Note: Breadth={max_width} exceeds C6 requirement (8-20). Based on Breadth alone, this should be C7/C8/C9 (Breadth ≥21)")
                            if max_depth > 6:
                                print(f"  ⚠️  Note: Depth={max_depth} exceeds C6 requirement (5-6). Maximum defined depth is 6 layers.")

            print("\nWidth by Level:")
            width_by_level = analysis.get('width_by_level', {})
            for level in sorted(width_by_level.keys()):
                print(f"  Level {level}: {width_by_level[level]} nodes")

            print("\n" + "-"*80)
            print("FORMATTED RUBRIC TREE (JSON):")
            print("-"*80)
            formatted_tree = analysis.get('formatted_tree', {})
            if formatted_tree:
                print(json.dumps(formatted_tree, ensure_ascii=False, indent=2))
            else:
                print('N/A')
        refined_analysis = parsed_data.get("rubric_tree_analysis", {})
        if refined_analysis:
            _print_rubric_analysis_block(
                refined_analysis,
                "RUBRIC TREE ANALYSIS (REFINED)",
            )

        print("\n" + "="*80)
        print("PROPOSED QUESTION")
        print("="*80)
        answer_data = parsed_data.get("answer_data") or {}
        proposed_question = answer_data.get("proposed_question", "") or parsed_data["metadata"].get("question", "")
        print(proposed_question)
        
        print("\n" + "="*80)
        print("CONSTRAINTS")
        print("="*80)
        constraints = answer_data.get("constraints", [])
        for i, constraint in enumerate(constraints, 1):
            print(f"{i}. {constraint}")
    if output_file is None:
        input_dir = os.path.dirname(os.path.abspath(input_file))
        formatted_dir = os.path.join(input_dir, "formatted")
        os.makedirs(formatted_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(formatted_dir, f"{base_name}_formatted.json")
    else:
        if not os.path.isabs(output_file):
            input_dir = os.path.dirname(os.path.abspath(input_file))
            output_file = os.path.join(input_dir, output_file)
    
    try:
        save_formatted_output(parsed_data, output_file)
        if verbose:
            print(f"\n✓ Formatted output saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving formatted output: {e}")
        import traceback
        traceback.print_exc()
    
    return parsed_data

def main():
    """Documentation omitted."""
    parser = argparse.ArgumentParser(description="Convert trajectory JSON files into formatted verifier inputs.")
    parser.add_argument(
        "--input-dir",
        default="./outputs/objective_trajectories",
        help="Directory containing trajectory JSON files.",
    )
    args = parser.parse_args()

    current_dir = args.input_dir
    pattern = os.path.join(current_dir, "*.json")
    files = glob.glob(pattern)
    files = [f for f in files if not f.endswith("_formatted.json")]
    
    if not files:
        print(f"No JSON files found in {current_dir}")
        return
    
    print(f"Found {len(files)} JSON file(s). Processing...")
    for file in sorted(files):
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(file)}")
        print(f"{'='*80}")
        process_trajectory_file(file, None, verbose=True)

if __name__ == "__main__":
    main()
