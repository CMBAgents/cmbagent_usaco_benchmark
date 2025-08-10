# --- Output normalization utility ---
def normalize_llm_output(output):
    """
    Ensures the output is a list of lists, splits single string elements by whitespace,
    and converts numeric strings to int or float as appropriate.
    Example:
        [['hello world']] -> [['hello', 'world']]
        [['5.4']] -> [[5.4]]
        [['5']] -> [[5]]
        [['hello', '5']] -> [['hello', 5]]
    """
    def convert_token(token):
        # Try int, then float, else keep as string
        try:
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token

    normalized = []
    for row in output:
        if len(row) == 1 and isinstance(row[0], str):
            # Split by whitespace
            tokens = row[0].strip().split()
            normalized.append([convert_token(tok) for tok in tokens])
        else:
            normalized.append([convert_token(tok) for tok in row])
    return normalized

# this file handles everything related to the LLMs, like creating the prompt, calling the API, and processing the response

import re
import time
from dataclasses import dataclass

# Only import cmbagent once if needed
cmbagent_module = None

# data classes

@dataclass
class LLM_response:
    generated_code: str
    generation_time: float
    generation_cost: float
    prompt_tokens: int
    completion_tokens: int

# functions

def prompt_wrapper(problem_description: str):
    return f"""
You are an expert computer science olympiad problem solver.

Task:
- Write a Python function called `main_function(input_data: list[list[int | str]])` that solves the problem you will be provided with. You may use multiple functions, but the entry point must be main_function.

Key Requirements for generating the solution:
- Wrap the full solution in a <code> ... </code> block.
- The function receives input_data as a list where each element represents a line of input which was read from a file. Example: If the data from the file was originally "3 5\n1 2 3\n\"hello\" 4\n3\n\"hi\"", then input_data will be [[3,5], [1,2,3], ["hello", 4], [3], ["hi"]].
- Return the final answer in the exact format expected, as the input_data, even for single-line or single-value outputs. For example, if the answer is 9, return [[9]]; if the answer is [1, 5], return [[1, 5]]; if the answer is "hello hello hello", return [["hello", "hello", "hello"]]. Do not return a flat list or a single string.
- Ensure efficiency (must be executed in under 10 seconds) and handle all constraints/edge cases.
- Do not print anything, just return the answer
- The code must be executable and free of syntax errors, unnecessary comments and explanations.

Example format:
<code>
def optional_helper_function(...):
    # code here
    return something

def main_function(input_data: list[list[int | str]]):
    # Your solution here
    return final_answer
</code>

Problem you need to solve:
{problem_description}
"""

def find_llm_type(agent: str, llm_token_prices: dict) -> str:
    for category, models in llm_token_prices.items():
        # Direct match for custom agent categories
        if agent == category:
            return category
        # Standard dict-style models
        if isinstance(models, dict) and agent in models:
            return category
    raise ValueError(f"Agent '{agent}' not found in any category of model_config.yaml")

def calculate_generation_cost(prompt_tokens: int, completion_tokens: int, agent: str, llm_token_prices: dict) -> float:
    for category, models in llm_token_prices.items():
        if agent in models:
            model_info = models[agent]
            input_price = model_info.get('input_price_per_1m', 0)
            output_price = model_info.get('output_price_per_1m', 0)
            if category == 'openai_gpt' or category == 'anthropic_claude':
                input_cost = (prompt_tokens / 1_000_000) * input_price
                output_cost = (completion_tokens / 1_000_000) * output_price
                return input_cost + output_cost
            else:
                return 0.0
    raise ValueError(f"Agent '{agent}' not found in model_config.yaml for cost calculation.")

def extract_code_block(code_str: str) -> str:
    """Extract code from the first <code>...</code> block in a string."""
    # Try <code>...</code> block first
    match = re.search(r'<code>(.*?)</code>', code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try ```python ... ``` block
    match = re.search(r'```python\s+(.*?)```', code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic triple backtick block
    match = re.search(r'```\s*(.*?)```', code_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If nothing found, return empty string
    return ''


def get_llm_response(prompt: str, agent: str, llm_clients: dict, llm_token_prices: dict, work_dir: str = None, engineer_model: str = None) -> LLM_response:
    # Handle oneshot/planning_and_control agent naming
    if agent.startswith('oneshot-'):
        llm_type = 'oneshot'
        engineer_model = agent[len('oneshot-'):]
    elif agent == 'planning_and_control':
        llm_type = 'planning_and_control'
        engineer_model = None
    else:
        llm_type = find_llm_type(agent, llm_token_prices)

    # Get max_output_tokens from config
    max_tokens = None
    for category, models in llm_token_prices.items():
        if agent in models:
            max_tokens = models[agent].get('max_output_tokens', 6000)
            break
    if max_tokens is None:
        max_tokens = 6000

    import importlib
    global cmbagent_module
    if llm_type in ['oneshot', 'planning_and_control'] and cmbagent_module is None:
        cmbagent_module = importlib.import_module('cmbagent')

    extraction_map = {
        'openai_gpt': {
            'client': llm_clients.get('openai_gpt'),
            'call': lambda client: client.chat.completions.create(
                model=agent,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            ),
            'extract_code': lambda response: extract_code_block(response.choices[0].message.content.strip()),
            'extract_tokens': lambda response: (
                getattr(response.usage, 'prompt_tokens', 0),
                getattr(response.usage, 'completion_tokens', 0)
            )
        },
        #gem
        'anthropic_claude': {
            'client': llm_clients.get('anthropic_claude'),
            'call': lambda client: client.messages.create(
                model=agent,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            ),
            'extract_code': lambda response: extract_code_block(response.content[0].text.strip()),
            'extract_tokens': lambda response: (
                getattr(response.usage, 'input_tokens', 0),
                getattr(response.usage, 'output_tokens', 0)
            )
        },
        'oneshot': {
            'client': None,
            'call': lambda client: (
                print("[INFO] Using cached cmbagent module..."),
                cmbagent_module.one_shot(
                    task=prompt,
                    max_rounds=5,
                    agent='engineer',
                    engineer_model=engineer_model if engineer_model else 'gpt-4.1-2025-04-14',
                    work_dir=work_dir
                )
            )[1],
            'extract_code': lambda response: next((extract_code_block(msg['content']) for msg in reversed(response['chat_history']) if extract_code_block(msg['content'])), ''),
            'extract_tokens': lambda response: (
                0, 0
            )
        },
        'planning_and_control': {
            'client': None,
            'call': lambda client: (
                print("[INFO] Using cached cmbagent module..."),
                cmbagent_module.planning_and_control_context_carryover(
                    task=prompt,
                    max_rounds_control=100, # how many messages exchanged between all agents
                    n_plan_reviews=1,
                    max_n_attempts=3, # attemps of writing code after executing it
                    max_plan_steps=5,
                    engineer_model="gemini-2.5-pro",
                    researcher_model="gpt-4.1-2025-04-14",
                    plan_reviewer_model="claude-sonnet-4-20250514",
                    plan_instructions=r"""
Use engineer for whole analysis, and return final code with researcher at the very end. Plan must have between 3 and 5 steps.
""",
                    work_dir=work_dir,
                    clear_work_dir=False
                )
            )[1],
            'extract_code': lambda response: '',  # code will be read from file in run_benchmark.py
            'extract_tokens': lambda response: (
                0, 0
            )
        },
    }

    client = extraction_map[llm_type]['client']
    start_time = time.time()
    response = extraction_map[llm_type]['call'](client)
    end_time = time.time()
    generation_time = end_time - start_time
    generated_code = extraction_map[llm_type]['extract_code'](response)
    prompt_tokens, completion_tokens = extraction_map[llm_type]['extract_tokens'](response)
    if llm_type in ['oneshot', 'planning_and_control']:
        cost_df = response['final_context'].get('cost_dataframe', '')
        cost_df_str = str(cost_df)
        lines = cost_df_str.splitlines()
        if lines:
            last_row = lines[-1]
            columns = last_row.split()
            float_candidates = [col for col in columns if col.replace('.', '', 1).isdigit()]
            extracted_cost = float(float_candidates[0]) if float_candidates else 0.0
        else:
            extracted_cost = 0.0
        generation_cost = extracted_cost
    else:
        generation_cost = calculate_generation_cost(prompt_tokens, completion_tokens, agent, llm_token_prices)
    return LLM_response(
        generated_code=generated_code,
        generation_time=generation_time,
        generation_cost=generation_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )
        
        