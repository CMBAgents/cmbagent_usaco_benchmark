import argparse
import json
import os
import yaml
import pathlib
from llm import prompt_wrapper, get_llm_response, find_llm_type
from executor import run_test_cases
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import datetime
# load argument from command line
parser = argparse.ArgumentParser()
parser.add_argument('--benchmark_file', type=str, required=True, help='Path to the run json benchmark file')
args = parser.parse_args()
benchmark_file = args.benchmark_file


# Determine run folder structure based on the run file location
run_file_path = pathlib.Path(benchmark_file).resolve()
run_base_dir = run_file_path.parent
run_base_dir.mkdir(parents=True, exist_ok=True)


# Load .env from config directory
load_dotenv(pathlib.Path(__file__).parent.parent / "config" / ".env")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/mnt/p/stage/cmbagent_benchmark/cmbagent/camels-453517-7c2faf50eda2.json"

# Start the benchmark
print("\n\033[1;36m====================[ BENCHMARK RUNNER ]====================\033[0m")
print(f"\033[1;33mConfiguration file:\033[0m {benchmark_file}\n")

with open(benchmark_file, 'r') as f:
    benchmark_dict = json.load(f)

config_path = benchmark_dict['config_path']
problem_ids = benchmark_dict['problem_ids']

with open(config_path, 'r') as f:
    config_data = yaml.safe_load(f)

json_file_path = config_data['json_file_path']
test_cases_folder_path = config_data['test_cases_folder_path']
agents = config_data['agents']

# Load LLM token prices/config
llm_token_prices_path = pathlib.Path(__file__).parent.parent / "config" / "model_config.yaml"
with open(llm_token_prices_path, 'r') as f:
    llm_token_prices = yaml.safe_load(f)

# get problem info
# Load the whole JSON from json_file_path
with open(json_file_path, 'r') as f:
    full_json_file = json.load(f)

# Create a new empty dict called 'problems'
problems = {}

# Loop through ids in problem_ids
for problem_id in problem_ids:
    # For each id, find the key inside full_json_file and the value of it
    value = full_json_file[problem_id]
    # The value is also a json containing 4 keys, but we want only 3
    problem_level = value['problem_level']
    description = value['description']
    num_tests = value['num_tests']
    # Append to the problems dict
    problems[problem_id] = {
        "level": problem_level,
        "description": description,
        "num_tests": num_tests,
        "test_cases_path": str(pathlib.Path(test_cases_folder_path) / problem_id)
    }

# now we have everything we need: 
# - problems (dict with problem level, description, num_tests and test cases folder path)
# - agents (list of agent names we'll use in the benchmark)
# what's left to do is run every problem on each agent and append it in benchmark_dict (create 'results' key and as the value we have another dict with keys as agent names and values as yet another dict for every problem)

# initialise agents
load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / "config" / ".env")


llm_types = []
for agent in agents:
    if agent.startswith('oneshot-'):
        llm_type = 'oneshot'
    else:
        llm_type = find_llm_type(agent, llm_token_prices)
    if llm_type not in llm_types:
        llm_types.append(llm_type)

llm_clients = {}
for llm_type in llm_types:
    if llm_type == 'openai_gpt':
        try:
            import openai
            llm_clients['openai_gpt'] = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with 'pip install openai'.")
    elif llm_type == 'anthropic_claude':
        try:
            import anthropic
            llm_clients['anthropic_claude'] = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except ImportError:
            raise ImportError("Anthropic library is not installed. Please install it with 'pip install anthropic'.")
    elif llm_type == 'google-gemini':
        try:
            from google import genai
            llm_clients['google-gemini'] = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
        except ImportError:
            raise ImportError("Google Gemini library is not installed. Please install it with 'pip install google-generativeai'.")
    elif llm_type in ['oneshot', 'planning_and_control']:
        # No client object needed, but add a placeholder to show it's a valid type
        llm_clients[llm_type] = None
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")

benchmark_dict['results'] = {agent: {} for agent in agents}

def process_single_task(task_data):
    """Process a single (agent, problem_id) task"""
    agent, problem_id, problems, llm_token_prices, test_cases_folder_path, run_base_dir = task_data
    
    # Load environment variables in each thread
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent.parent / "config" / ".env")
    
    # Recreate LLM clients inside each thread (can't share them across threads)
    if agent.startswith('oneshot-'):
        llm_type = 'oneshot'
        engineer_model = agent[len('oneshot-'):]
    elif agent == 'planning_and_control':
        llm_type = 'planning_and_control'
        engineer_model = None
    else:
        llm_type = find_llm_type(agent, llm_token_prices)
        engineer_model = None
    
    # Create LLM clients for this thread
    thread_llm_clients = {}
    if llm_type == 'openai_gpt':
        try:
            import openai
            thread_llm_clients['openai_gpt'] = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            raise ImportError("OpenAI library is not installed. Please install it with 'pip install openai'.")
    elif llm_type == 'anthropic_claude':
        try:
            import anthropic
            thread_llm_clients['anthropic_claude'] = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        except ImportError:
            raise ImportError("Anthropic library is not installed. Please install it with 'pip install anthropic'.")
    elif llm_type == 'google-gemini':
        try:
            from google import genai
            thread_llm_clients['google-gemini'] = genai.Client(api_key=os.getenv('GOOGLE_GEMINI_API_KEY'))
        except ImportError:
            raise ImportError("Google Gemini library is not installed. Please install it with 'pip install google-generativeai'.")
    elif llm_type in ['oneshot', 'planning_and_control']:
        thread_llm_clients[llm_type] = None
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    # Create agent directories if needed
    if llm_type in ['oneshot', 'planning_and_control']:
        agent_dir = run_base_dir / llm_type
        agent_dir.mkdir(parents=True, exist_ok=True)
        problem_dir = agent_dir / problem_id
        problem_dir.mkdir(parents=True, exist_ok=True)
        work_dir = str(problem_dir)
    else:
        work_dir = None
    
    # Process the problem
    prompt = prompt_wrapper(problems[problem_id]['description'])
    
    if llm_type in ['oneshot', 'planning_and_control']:
        llm_response = get_llm_response(prompt, agent, thread_llm_clients, llm_token_prices, work_dir=work_dir, engineer_model=engineer_model)
        # For planning_and_control, override generated_code by reading from work_dir/control/codebase/*.py
        if llm_type == 'planning_and_control':
            control_codebase_dir = pathlib.Path(work_dir) / 'control' / 'codebase'
            py_files = list(control_codebase_dir.glob('*.py'))
            if py_files:
                with open(py_files[0], 'r') as f:
                    llm_response.generated_code = f.read()
    else:
        llm_response = get_llm_response(prompt, agent, thread_llm_clients, llm_token_prices)
    
    test_case_result = run_test_cases(llm_response.generated_code, pathlib.Path(test_cases_folder_path) / problem_id)
    generation_info = llm_response.__dict__ if hasattr(llm_response, '__dict__') else llm_response
    
    result_data = {
        'generation_info': generation_info,
        'execution_info': test_case_result
    }
    
    return agent, problem_id, result_data


# Create tasks list: [(agent, problem_id), ...]
tasks = []
for agent in agents:
    for problem_id in problems.keys():
        task_data = (agent, problem_id, problems, llm_token_prices, test_cases_folder_path, run_base_dir)
        tasks.append(task_data)

# Thread-safe progress tracking
completed_tasks = 0
total_tasks = len(tasks)
progress_lock = threading.Lock()

print(f"\n\033[1;33mStarting {total_tasks} tasks across {len(agents)} agents and {len(problems)} problems\033[0m")

# Process tasks in parallel
max_workers = min(8, os.cpu_count())  # Don't exceed CPU count
start_time = time.time()

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_task = {executor.submit(process_single_task, task_data): task_data for task_data in tasks}
    
    # Collect results as they complete
    for future in as_completed(future_to_task):
        try:
            agent, problem_id, result_data = future.result()
            
            # Thread-safe result storage
            benchmark_dict['results'][agent][problem_id] = result_data
            
            # Thread-safe progress update
            with progress_lock:
                completed_tasks += 1
                status = result_data['execution_info'].get("status", "system_error")
                status_color = "\033[1;32m" if status == "success" else "\033[1;31m"
                print(f"{status_color}Completed {completed_tasks}/{total_tasks}: {agent} - {problem_id} ({status})\033[0m")
                
        except Exception as exc:
            task_data = future_to_task[future]
            agent, problem_id = task_data[0], task_data[1]
            print(f'\033[1;31mTask {agent}-{problem_id} generated an exception: {exc}\033[0m')

end_time = time.time()
real_benchmark_time = end_time - start_time
print(f"\n\033[1;36mAll tasks completed in {real_benchmark_time:.2f} seconds\033[0m")

# Calculate agent summaries from collected results
for agent in agents:
    agent_summary = {
        "total_generation_time": 0.0,
        "total_cost": 0.0,
        "accuracy": 0.0,
        "number_per_failure_type": {
            "timeout": 0,
            "runtime_error": 0,
            "compilation_error": 0,
            "wrong_answer": 0,
            "system_error": 0
        }
    }
    correct_count = 0
    total_count = 0
    
    for problem_id in problems.keys():
        if problem_id in benchmark_dict['results'][agent]:
            result = benchmark_dict['results'][agent][problem_id]
            generation_info = result['generation_info']
            execution_info = result['execution_info']
            
            agent_summary["total_generation_time"] += generation_info.get("generation_time", 0.0)
            agent_summary["total_cost"] += generation_info.get("generation_cost", 0.0)
            status = execution_info.get("status", "system_error")
            total_count += 1
            
            if status == "success":
                correct_count += 1
            else:
                if status in agent_summary["number_per_failure_type"]:
                    agent_summary["number_per_failure_type"][status] += 1
                else:
                    agent_summary["number_per_failure_type"][status] = 1
    
    agent_summary["accuracy"] = round(100.0 * correct_count / total_count, 2) if total_count > 0 else 0.0
    benchmark_dict['results'][agent]['agent_summary'] = agent_summary


# Calculate benchmark_summary
total_generation_cost = 0.0
total_generation_time = 0.0
agent_accuracies = []
agent_times = []
for agent in agents:
    agent_summary = benchmark_dict['results'][agent]['agent_summary']
    total_generation_cost += agent_summary['total_cost']
    total_generation_time += agent_summary['total_generation_time']
    agent_accuracies.append((agent, agent_summary['accuracy']))
    agent_times.append((agent, agent_summary['total_generation_time']))

if len(agents) > 1:
    agent_comparison = {
        "by_accuracy": sorted(agent_accuracies, key=lambda x: -x[1]),
        "by_time": sorted(agent_times, key=lambda x: x[1])
    }
else:
    agent_comparison = None


benchmark_summary = {
    "total_generation_cost": round(total_generation_cost, 6),
    "total_generation_time": round(total_generation_time, 3),
    "agent_comparison": agent_comparison,
    "real_benchmark_time_with_multiprocessing": round(real_benchmark_time, 3)
}
benchmark_dict['results']['benchmark_summary'] = benchmark_summary

# overwrite the benchmark file with the results
with open(benchmark_file, 'w') as f:
    json.dump(benchmark_dict, f, indent=4)

# Print a nice summary
def print_benchmark_summary(benchmark_dict):
    results = benchmark_dict['results']
    print("\n\033[1;35m================[ BENCHMARK SUMMARY ]================\033[0m")
    print(f"\033[1;33mTotal generation cost:\033[0m {results['benchmark_summary']['total_generation_cost']}")
    print(f"\033[1;33mTotal generation time:\033[0m {results['benchmark_summary']['total_generation_time']} seconds")
    if results['benchmark_summary']['agent_comparison']:
        print("\n\033[1;36mAgent comparison by accuracy:\033[0m")
        for agent, acc in results['benchmark_summary']['agent_comparison']['by_accuracy']:
            print(f"  \033[1;34m{agent}:\033[0m {acc}% correct")
        print("\n\033[1;36mAgent comparison by total generation time:\033[0m")
        for agent, t in results['benchmark_summary']['agent_comparison']['by_time']:
            print(f"  \033[1;34m{agent}:\033[0m {round(t, 3)} seconds")
    else:
        print("\n\033[1;36mOnly one agent, no comparison available.\033[0m")
    print("\n\033[1;35m----------------[ AGENT DETAILS ]----------------\033[0m")
    for agent in agents:
        agent_summary = results[agent]['agent_summary']
        print(f"\n\033[1;34mAgent: {agent}\033[0m")
        print(f"  Total generation time: \033[1;33m{round(agent_summary['total_generation_time'], 3)} seconds\033[0m")
        print(f"  Total cost: \033[1;33m{round(agent_summary['total_cost'], 6)}\033[0m")
        print(f"  Accuracy: \033[1;32m{agent_summary['accuracy']}%\033[0m")
        print(f"  Failure types:")
        for k, v in agent_summary['number_per_failure_type'].items():
            print(f"    {k}: {v}")
    print("\n\033[1;35m================================================\033[0m\n")
    print(f"\033[1;32mBenchmark completed successfully! Run file located at: {benchmark_file} \033[0m")

print_benchmark_summary(benchmark_dict)