API_COST = {
    # model_name: input, output
    "gpt-4.1": [2.00, 8.00],
    "o4-mini": [1.10, 4.40],
    "gpt-4o": [2.50, 10.00],
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": [3.00, 15.00]
}

UNIT = 1000000


def calculate_api_cost(input_tokens, output_tokens, model_name):
    if model_name not in API_COST:
        raise ValueError(f"Cannot get the price of calling {model_name}")
    return API_COST[model_name][0] * input_tokens / UNIT + API_COST[model_name][1] * output_tokens / UNIT
