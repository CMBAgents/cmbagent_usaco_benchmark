

# cmbagent_usaco_benchmark

## Overview
This project provides a framework for benchmarking large language models (LLMs) on USACO-style competitive programming problems. It supports flexible configuration, custom test sets, and multiple agent (model) types.

## Project Structure

- `create_benchmark.sh`: **Entry point script.** Run this to start a new benchmark or repeat an existing one. It guides you through selecting problems, agents, and settings.

- `config/`
	- `.env.example`: Template for environment variables (API keys, credentials). **Copy to `.env` and fill in your keys.**
	- `.env`: Your actual API keys and credentials (not tracked by git).
	- `model_config.yaml`: Lists supported LLM models and their configuration (name, token limits, pricing, etc.).

`data/`
	- JSON files (e.g., `easy_custom_samples.json`): Map problem IDs to metadata (level, description, number of tests). **Each problem ID in the JSON must match a folder name in the relevant test cases subfolder.** This is how the system links metadata to the actual test cases. For example, `easy_custom_samples.json` contains the metadata for the problems in the `easy_tests/` subfolder. 
	- Subfolders (e.g., `easy_tests/`, `diverse_tests/`, `usaco_tests/`): Each contains directories for individual problems, named by their problem ID, each with input/output files named like `I.1`, `O.1`, etc.
		- **Example:**
			- `data/easy_tests/0001_easy_addition/I.1` (input for test 1)
			- `data/easy_tests/0001_easy_addition/O.1` (output for test 1)

- `python_scripts/`: All Python scripts for running and evaluating benchmarks (no need to run directly).

- `benchmark_output/`: Created automatically when you run a benchmark. Stores configs, run logs, and per-problem results. If you use "oneshot" or "planning_and_control" modes, subfolders are created for each problem.

## Setup & Usage

1. **Configure API Keys:**
	 - Copy `config/.env.example` to `config/.env` and fill in your API keys and credentials.

2. **Edit Model Config (Optional):**
	 - Edit `config/model_config.yaml` to add or modify supported LLM models.

3. **Prepare Test Cases:**
	 - Add or edit JSON files in `data/` to define which problems to use.
	 - Place problem folders and their test cases in the appropriate subfolder (see above for structure).

4. **Run the Benchmark:**
	 - In your terminal:
		 ```bash
		 ./create_benchmark.sh
		 ```
	 - Follow the prompts to create a new benchmark or repeat an existing one.
	 - You can select problems by level, randomly, or by specific IDs.
	 - Choose which agents (models) to run. (type "<end>" to finish adding agents)

5. **View Results:**
	 - Results and logs are saved in `benchmark_output/`.
	 - For advanced modes, each problem gets its own subfolder with detailed outputs.


## Benchmark Output Structure

When you run a benchmark, a new folder is created under `benchmark_output/runs/` (e.g., `run_YYYYMMDD_HHMMSS/`). This folder contains all results and logs for that run, organized as follows:

- `run_*.json`: The main run file, recording the configuration, list of problem IDs, and a summary of results for each agent and problem.
- `oneshot/`: Contains results for each problem solved using the "oneshot" agent/mode. Each problem gets its own subfolder, and all files created by cmbagent for that problem are saved there.
- `planning_and_control/`: Contains results for each problem solved using the "planning_and_control" agent/mode. Each problem gets its own subfolder, and all files created by cmbagent for that problem are saved there.

---

- Each JSON in `data/` (e.g., `easy_custom_samples.json`) looks like:
	```json
	{
		"0001_easy_addition": {
			"problem_id": "0001_easy_addition",
			"problem_level": "bronze",
			"description": "Given two integers A and B, output their sum.",
			"num_tests": 3
		},
		...
	}
	```
		- Each run folder is linked to a YAML config file in `benchmark_output/configs/` (e.g., `config_YYYYMMDD_HHMMSS.yaml`). This YAML stores the settings and choices you made when running `create_benchmark.sh` (such as how to select problems and which agents to use).
		- You can re-run a benchmark in two ways:
			- By providing the YAML config file: this will repeat the benchmark with the same settings (problem dataset, problem-choice method and agents), but will most likely select a new random set of problems (if random or by_level selection was used).
			- By providing the JSON run file: this will repeat the benchmark with the exact same problems as the original run.
- Each problem folder (e.g., `data/easy_tests/0001_easy_addition/`) contains input/output files for each test case:
	- `I.1`, `O.1`, `I.2`, `O.2`, ...

