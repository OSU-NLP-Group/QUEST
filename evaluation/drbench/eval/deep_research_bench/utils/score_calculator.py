import logging

def calculate_weighted_scores(llm_output_json, criteria_data, language="en"):
    """
    Calculates weighted scores based on LLM output and criteria weights.
    
    Args:
        llm_output_json: JSON output from LLM with scoring data
        criteria_data: Criteria configuration with dimension weights
        language: Language of the evaluation (default: "en")
        
    Returns:
        Dictionary with weighted scores for target and reference models
    """
    results = {
        "target": {"dims": {}, "total": 0.0},
        "reference": {"dims": {}, "total": 0.0}
    }
    total_target_score = 0.0
    total_reference_score = 0.0
    
    # Get dimension weights
    dimension_weights = criteria_data.get("dimension_weight", {})
    task_id = criteria_data.get("id", "Unknown") 
    
    # Check if criterions exist
    if "criterions" not in criteria_data or not criteria_data["criterions"]:
        error_msg = f"ID: {task_id} - Missing required criterions data, cannot calculate weighted scores"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Create a mapping from criterion text to weight for easier lookup
    criterion_weights = {}
    for dim, criterions in criteria_data.get("criterions", {}).items():
        criterion_weights[dim] = {crit['criterion']: crit['weight'] for crit in criterions}

    # Record all unmatched criteria for warnings
    unmatched_criteria = set()
    
    for dim, scores_list in llm_output_json.items():
        if not isinstance(scores_list, list):
            logging.warning(f"ID: {task_id} - Dimension '{dim}' in LLM output is not a list. Skipping.")
            continue

        if dim not in dimension_weights:
            logging.warning(f"ID: {task_id} - Dimension '{dim}' from LLM output not found in criteria dimension weights. Skipping dimension.")
            continue
            
        if dim not in criterion_weights:
            logging.warning(f"ID: {task_id} - Dimension '{dim}' from LLM output not found in criteria details. Skipping dimension.")
            continue

        dim_target_weighted_sum = 0.0
        dim_reference_weighted_sum = 0.0
        dim_total_weight = 0.0

        dim_criteria_map = criterion_weights.get(dim, {})
        
        # Skip dimension if no criteria mapping exists
        if not dim_criteria_map:
            logging.warning(f"ID: {task_id} - No criteria mapping found for dimension '{dim}'. Skipping dimension.")
            continue

        for score_item in scores_list:
            if not isinstance(score_item, dict):
                logging.warning(f"ID: {task_id} - Item in scores_list for dimension '{dim}' is not a dictionary. Skipping item: {score_item}")
                continue

            criterion_text_raw = score_item.get("criterion")
            criterion_text = criterion_text_raw.strip() if isinstance(criterion_text_raw, str) else None

            # Check different score field formats
            # 1. Comparative scoring mode: article_1_score and article_2_score
            # 2. Single scoring mode: target_score
            article_1_score_raw = score_item.get("article_1_score")
            article_2_score_raw = score_item.get("article_2_score")
            target_score_raw = score_item.get("target_score")  # Single scoring mode

            # If target_score exists but article_1_score doesn't, assign target_score to article_1_score
            if target_score_raw is not None and article_1_score_raw is None:
                article_1_score_raw = target_score_raw
                # In single scoring mode, article_2_score is not needed

            try:
                article_1_score = float(article_1_score_raw) if article_1_score_raw is not None else None
                article_2_score = float(article_2_score_raw) if article_2_score_raw is not None else None
            except (ValueError, TypeError):
                logging.warning(f"ID: {task_id} - Invalid score format for criterion '{criterion_text}' in dimension '{dim}'. Scores: art1='{article_1_score_raw}', art2='{article_2_score_raw}'. Skipping criterion.")
                continue

            # If criterion_text exists and article_1_score is not None
            if criterion_text and article_1_score is not None:
                # Check for exact match
                weight = dim_criteria_map.get(criterion_text)
                
                # If exact match not found, try fuzzy matching
                if weight is None:
                    # First try case-insensitive matching
                    criterion_lower = criterion_text.lower()
                    for key, val in dim_criteria_map.items():
                        if key.lower() == criterion_lower:
                            weight = val
                            break
                    
                    # If still not found, try substring matching
                    if weight is None:
                        for key, val in dim_criteria_map.items():
                            # Check if criterion text contains criteria or criteria contains criterion text
                            if criterion_lower in key.lower() or key.lower() in criterion_lower:
                                weight = val
                                break
                
                # If still no match found, record and use average weight
                if weight is None:
                    unmatched_criteria.add(f"{dim}:{criterion_text}")
                    # Calculate average weight for this dimension's criteria
                    weight = sum(dim_criteria_map.values()) / len(dim_criteria_map)
                    
                dim_target_weighted_sum += article_1_score * weight
                dim_total_weight += weight
                
                # Only calculate reference part if article_2_score exists
                if article_2_score is not None:
                    dim_reference_weighted_sum += article_2_score * weight
            else:
                if criterion_text:
                    if dim not in getattr(calculate_weighted_scores, '_warned_dims', set()):
                        logging.warning(f"ID: {task_id} - Criterion text mismatch for dimension '{dim}': '{criterion_text}'. Available criteria keys start with: {list(dim_criteria_map.keys())[:1] if dim_criteria_map else []}... Check prompt/output/criteria file consistency. Further mismatches in this dim won't be logged fully.")
                        if not hasattr(calculate_weighted_scores, '_warned_dims'): calculate_weighted_scores._warned_dims = set()
                        calculate_weighted_scores._warned_dims.add(dim)
                    else:
                        logging.debug(f"ID: {task_id} - Another criterion mismatch for dimension '{dim}': '{criterion_text}'")
                elif not criterion_text:
                    logging.warning(f"ID: {task_id} - Missing 'criterion' key in score item for dimension '{dim}': {score_item}. Skipping item.")

        if dim_total_weight > 0:
            dim_target_avg = dim_target_weighted_sum / dim_total_weight
            # Only calculate reference average if reference scores exist
            dim_reference_avg = dim_reference_weighted_sum / dim_total_weight if article_2_score is not None else 0
        else:
            dim_target_avg = 0
            dim_reference_avg = 0
            if len(scores_list) > 0:
                logging.warning(f"ID: {task_id} - No valid criteria scored for dimension '{dim}' despite {len(scores_list)} items in LLM output. Check for systematic mismatches. Dimension average set to 0.")

        results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
        results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_reference_avg

        dim_weight = dimension_weights.get(dim, 0)
        total_target_score += dim_target_avg * dim_weight
        total_reference_score += dim_reference_avg * dim_weight

    # Log warning if there are unmatched criteria
    if unmatched_criteria:
        logging.warning(f"ID: {task_id} - {len(unmatched_criteria)} criteria without exact matches: {unmatched_criteria}")
    
    results["target"]["total"] = total_target_score
    results["reference"]["total"] = total_reference_score

    return results