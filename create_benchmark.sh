#!/bin/bash

# --- HELPER FUNCTIONS ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to get a valid path from the user
get_valid_path() {
    local prompt="$1"
    local type="$2" # "file" or "dir"
    local extension="$3"
    local path

    while true; do
        read -p "$prompt" path
        if [[ "$type" == "file" ]]; then
            if [[ ! -f "$path" ]]; then
                echo "File does not exist. Please enter a valid JSON file path:" >&2
            elif [[ -n "$extension" && "$path" != *"$extension" ]]; then
                echo "Invalid file type (not ending in $extension). Please enter a valid JSON file path:" >&2
            else
                echo "$path"
                return
            fi
        elif [[ "$type" == "dir" ]]; then
            if [[ ! -d "$path" ]]; then
                echo "Folder does not exist. Please enter a valid test cases folder path:" >&2
            else
                echo "$path"
                return
            fi
        else
            echo "Invalid type specified for validation." >&2
            exit 1
        fi
    done
}

# Function to get a valid number from the user
get_valid_number() {
    local prompt="$1"
    local number
    read -p "$prompt" number
    while ! [[ "$number" =~ ^[0-9]+$ ]]; do
        echo "Invalid input. Please enter a valid number for ${prompt%:*}:"
        read -p "$prompt" number
    done
    echo "$number"
}

# Function to get valid level count with validation
get_valid_level_count() {
    local level="$1"
    local available="$2"
    local count
    while true; do
        count=$(get_valid_number "$level: ")
        if [[ "$count" -gt "$available" ]]; then
            echo "You cannot select more $level problems than available ($available). Please enter a valid number:"
        else
            echo "$count"
            return
        fi
    done
}

# Function to select problems by level
select_problems_by_level() {
    local json_file="$1"
    local level="$2"
    local count="$3"
    if (( count > 0 )); then
        mapfile -t level_ids < <(jq -r "to_entries[] | select(.value.problem_level == \"$level\") | .key" "$json_file")
        printf "%s\n" "${level_ids[@]}" | shuf | head -n "$count"
    fi
}

# --- PROBLEM SELECTION FUNCTIONS ---

# Function to handle problem selection by level
select_by_level() {
    local benchmark_file="$1"
    local real_bronze_count="$2"
    local real_silver_count="$3"
    local real_gold_count="$4"
    local real_platinum_count="$5"
    echo "Choose number of problems by level:"
    local bronze_count
    while true; do
        bronze_count=$(get_valid_number "Bronze: ")
        if [[ "$bronze_count" -gt "$real_bronze_count" ]]; then
            echo "You cannot select more bronze problems than available ($real_bronze_count). Please enter a valid number:"
        else
            break
        fi
    done
    local silver_count
    while true; do
        silver_count=$(get_valid_number "Silver: ")
        if [[ "$silver_count" -gt "$real_silver_count" ]]; then
            echo "You cannot select more silver problems than available ($real_silver_count). Please enter a valid number:"
        else
            break
        fi
    done
    local gold_count
    while true; do
        gold_count=$(get_valid_number "Gold: ")
        if [[ "$gold_count" -gt "$real_gold_count" ]]; then
            echo "You cannot select more gold problems than available ($real_gold_count). Please enter a valid number:"
        else
            break
        fi
    done
    local platinum_count
    while true; do
        platinum_count=$(get_valid_number "Platinum: ")
        if [[ "$platinum_count" -gt "$real_platinum_count" ]]; then
            echo "You cannot select more platinum problems than available ($real_platinum_count). Please enter a valid number:"
        else
            break
        fi
    done

    printf "problem:\n    selection_type: \"by_level\"\n    bronze: %s\n    silver: %s\n    gold: %s\n    platinum: %s\n" \
        "$bronze_count" "$silver_count" "$gold_count" "$platinum_count" >> "$benchmark_file"
}

# Function to handle random problem selection
# Function to handle random problem selection
select_randomly() {
    local benchmark_file="$1"
    local total_problem_count="$2"
    local bronze_problem_count="$3"
    local silver_problem_count="$4"
    local gold_problem_count="$5"
    local platinum_problem_count="$6"
    local random_count
    local avoid_levels
    local available_count

    while true; do
        random_count=$(get_valid_number "How many problems do you want to select randomly? ")
        read -p "Want to avoid certain types of problem levels? (y/n) " avoid_levels
        while [[ "$avoid_levels" != "y" && "$avoid_levels" != "n" ]]; do
            echo "Invalid input. Please enter 'y' or 'n':"
            read -p "Want to avoid certain types of problem levels? (y/n) " avoid_levels
        done

        if [[ "$avoid_levels" == "y" ]]; then
            local avoid_levels_list
            local valid_input=false
            while [ "$valid_input" = false ]; do
                read -p "Enter the levels to avoid (comma-separated, no spaces, e.g., bronze,silver): " avoid_levels_list
                local all_valid=true
                IFS=',' read -ra levels_to_avoid <<< "$avoid_levels_list"
                for level in "${levels_to_avoid[@]}"; do
                    if [[ "$level" != "bronze" && "$level" != "silver" && "$level" != "gold" && "$level" != "platinum" ]]; then
                        all_valid=false
                        echo "Invalid level found: $level. Please only use 'bronze', 'silver', 'gold', 'platinum'."
                        break
                    fi
                done
                if [ "$all_valid" = true ]; then
                    valid_input=true
                fi
            done
            # Calculate available_count by subtracting avoided levels
            available_count=$total_problem_count
            for level in "${levels_to_avoid[@]}"; do
                case "$level" in
                    bronze)
                        available_count=$((available_count - bronze_problem_count))
                        ;;
                    silver)
                        available_count=$((available_count - silver_problem_count))
                        ;;
                    gold)
                        available_count=$((available_count - gold_problem_count))
                        ;;
                    platinum)
                        available_count=$((available_count - platinum_problem_count))
                        ;;
                esac
            done
            if [[ "$available_count" -eq 0 ]]; then
                echo "No problems are available after avoidance. Please choose different levels to avoid."
                continue
            fi
            if [[ "$random_count" -gt "$available_count" ]]; then
                echo "You cannot select more problems ($random_count) than available after avoidance ($available_count). Please try again."
                continue
            fi
            printf "problem:\n    selection_type: \"random\"\n    count: %s\n    avoid_levels: [%s]\n" "$random_count" "$avoid_levels_list" >> "$benchmark_file"
            break
        else
            available_count=$total_problem_count
            if [[ "$random_count" -gt "$available_count" ]]; then
                echo "You cannot select more problems ($random_count) than available ($available_count). Please try again."
                continue
            fi
            printf "problem:\n    selection_type: \"random\"\n    count: %s\n" "$random_count" >> "$benchmark_file"
            break
        fi
    done
}

# Function to handle problem selection by IDs
select_by_ids() {
    local benchmark_file="$1"
    local json_keys="$2"  # newline-separated list of valid IDs
    local problem_ids=()
    local id
    local valid=false

    echo "Enter problem IDs one by one. When you're done, type <end>."
    while true; do
        read -p "ID: " id
        if [[ "$id" == "<end>" ]]; then
            if [ "${#problem_ids[@]}" -eq 0 ]; then
                echo "Please provide at least one problem."
                continue
            else
                break
            fi
        fi
        # Check if id exists in json_keys
        if echo "$json_keys" | grep -qx "$id"; then
            problem_ids+=("$id")
        else
            echo "ID doesn't exist."
        fi
    done
    # Join IDs with commas for YAML output
    local id_list
    id_list=$(IFS=','; echo "${problem_ids[*]}")
    printf "problem:\n    selection_type: \"by_id\"\n    list: [%s]\n" "$id_list" >> "$benchmark_file"
}

# --- MAIN WORKFLOW FUNCTIONS ---

# Function to print the welcome message
print_welcome_message() {
    echo -e "\n\033[1;36m====================[ BENCHMARK SETUP ]====================\033[0m"
    echo -e "\033[1;33mWhat do you want to do?\033[0m Available options:"
    echo -e "\033[1;34mMenu:\033[0m"
    echo ""
    echo -e "  (1) Create new benchmark"
    echo -e "  (2) Run existing benchmark preset (exact repeat)"
    echo -e "\033[1;36m==========================================================\033[0m"
}

# Function to read the user's main choice
read_user_option_for_benchmark_type() {
    local option
    read -p $'\033[1;32mSelect an option (1-2): \033[0m' option
    while [[ "$option" != "1" && "$option" != "2" ]]; do
        read -p $'\033[1;31mInvalid input. Please enter 1 or 2: \033[0m' option
    done
    echo "$option"
}

get_problem_counts() {
    local json_file="$1"
    local total bronze silver gold platinum
    total=$(jq 'keys | length' "$json_file")
    bronze=$(jq '[to_entries[] | select(.value.problem_level == "bronze")] | length' "$json_file")
    silver=$(jq '[to_entries[] | select(.value.problem_level == "silver")] | length' "$json_file")
    gold=$(jq '[to_entries[] | select(.value.problem_level == "gold")] | length' "$json_file")
    platinum=$(jq '[to_entries[] | select(.value.problem_level == "platinum")] | length' "$json_file")
    echo "$total $bronze $silver $gold $platinum"
}

# Function to guide the user through creating a new benchmark
create_new_benchmark() {
    echo -e "\n\033[1;36mStarting configuration for new benchmark...\033[0m"
    mkdir -p "$SCRIPT_DIR/benchmark_output/configs"
    mkdir -p "$SCRIPT_DIR/benchmark_output/runs"

    local run_timestamp=$(date +%Y%m%d_%H%M%S)
    local config_file="$SCRIPT_DIR/benchmark_output/configs/config_${run_timestamp}.yaml"
    local run_dir="$SCRIPT_DIR/benchmark_output/runs/run_${run_timestamp}"
    mkdir -p "$run_dir"
    local run_file="$run_dir/run_${run_timestamp}.json"
    touch "$config_file"
    echo -e "\033[1;32mConfig file created:\033[0m $config_file"
    echo -e "\033[1;32mRun directory created:\033[0m $run_dir"
    echo -e "\033[1;32mRun file created:\033[0m $run_file"

    # Initialize run file with config path
    echo "{\"config_path\": \"$config_file\"}" > "$run_file"

    local json_file_path=$(get_valid_path "Please enter the path to the JSON file: " "file" ".json")
    echo "json_file_path: $json_file_path" >> "$config_file"

    local test_cases_folder_path=$(get_valid_path "Please enter the path to the test cases folder: " "dir")
    echo "test_cases_folder_path: $test_cases_folder_path" >> "$config_file"

    # verify if all keys from the json exist in the list made out of the dir names from the test cases folder
    json_keys=$(jq -r 'keys[]' "$json_file_path")
    dir_names=$(ls -1 "$test_cases_folder_path")
    for key in $json_keys; do
        if ! echo "$dir_names" | grep -qx "$key"; then
            echo "Missing test cases folder for problem: $key (is present in json but not in test cases folder)"
            exit 1
        fi
    done

    echo -e "\n\033[1;33mHow do you want the problems to be chosen?\033[0m\n"
    echo -e "  (1) By level"
    echo -e "  (2) Randomly"
    echo -e "  (3) By problem ids"
    local problem_choice
    read -p "Select an option (1-3): " problem_choice
    while [[ "$problem_choice" != "1" && "$problem_choice" != "2" && "$problem_choice" != "3" ]]; do
        echo "Invalid input. Please enter 1, 2, or 3:"
        read -p "Select an option (1-3): " problem_choice
    done

    read total bronze silver gold platinum < <(get_problem_counts "$json_file_path")
    echo -e "\033[1;34mTotal problems:\033[0m $total"
    echo -e "\033[1;33mBronze problems:\033[0m $bronze"
    echo -e "\033[1;37mSilver problems:\033[0m $silver"
    echo -e "\033[1;33mGold problems:\033[0m $gold"
    echo -e "\033[1;35mPlatinum problems:\033[0m $platinum"

    local selected_problem_ids=()
    if [[ "$problem_choice" == "1" ]]; then
        # By level
        local bronze_count=$(get_valid_level_count "Bronze" "$bronze")
        local silver_count=$(get_valid_level_count "Silver" "$silver")
        local gold_count=$(get_valid_level_count "Gold" "$gold")
        local platinum_count=$(get_valid_level_count "Platinum" "$platinum")
        
        # Select problems by level
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "bronze" "$bronze_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "silver" "$silver_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "gold" "$gold_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "platinum" "$platinum_count") )
        
        printf "problem:\n    selection_type: \"by_level\"\n    bronze: %s\n    silver: %s\n    gold: %s\n    platinum: %s\n" \
            "$bronze_count" "$silver_count" "$gold_count" "$platinum_count" >> "$config_file"
    elif [[ "$problem_choice" == "2" ]]; then
        # Random
        local random_count=$(get_valid_number "How many problems do you want to select randomly? ")
        read -p "Want to avoid certain types of problem levels? (y/n) " avoid_levels
        while [[ "$avoid_levels" != "y" && "$avoid_levels" != "n" ]]; do
            echo "Invalid input. Please enter 'y' or 'n':"
            read -p "Want to avoid certain types of problem levels? (y/n) " avoid_levels
        done
        local avoid_str=""
        local jq_query='to_entries[]'
        if [[ "$avoid_levels" == "y" ]]; then
            read -p "Enter the levels to avoid (comma-separated, no spaces, e.g., bronze,silver): " avoid_str
            IFS=',' read -ra avoid <<< "$avoid_str"
            for level in "${avoid[@]}"; do
                jq_query+=" | select(.value.problem_level != \"$level\")"
            done
            mapfile -t all_ids < <(jq -r "$jq_query | .key" "$json_file_path")
            selected_problem_ids=( $(printf "%s\n" "${all_ids[@]}" | shuf | head -n "$random_count") )
            printf "problem:\n    selection_type: \"random\"\n    count: %s\n    avoid_levels: [%s]\n" "$random_count" "$avoid_str" >> "$config_file"
        else
            mapfile -t all_ids < <(jq -r "$jq_query | .key" "$json_file_path")
            selected_problem_ids=( $(printf "%s\n" "${all_ids[@]}" | shuf | head -n "$random_count") )
            printf "problem:\n    selection_type: \"random\"\n    count: %s\n" "$random_count" >> "$config_file"
        fi
    elif [[ "$problem_choice" == "3" ]]; then
        # By ID
        echo "Enter problem IDs one by one. When you're done, type <end>."
        local ids_list=()
        while true; do
            read -p "ID: " id
            if [[ "$id" == "<end>" ]]; then
                if [ "${#ids_list[@]}" -eq 0 ]; then
                    echo "Please provide at least one problem."
                    continue
                else
                    break
                fi
            fi
            # Check if id exists in json_keys
            if echo "$json_keys" | grep -qx "$id"; then
                ids_list+=("$id")
            else
                echo "ID doesn't exist."
            fi
        done
        selected_problem_ids=( "${ids_list[@]}" )
        local id_list
        id_list=$(IFS=','; echo "${ids_list[*]}")
        printf "problem:\n    selection_type: \"by_id\"\n    list: [%s]\n" "$id_list" >> "$config_file"
    fi

    # Prompt for agent names
    local model_config_path="$SCRIPT_DIR/config/model_config.yaml"
    local valid_agents=( $(grep -E '^[[:space:]]*name:' "$model_config_path" | sed -E 's/^[[:space:]]*name:[[:space:]]*"?([^\"]*)"?.*/\1/') )
    local agent_names_array=()
    local agent_name

    echo -e "\033[1;33mEnter agent names one by one. When you're done, type <end>.\033[0m"
    while true; do
        read -p "Agent: " agent_name
        if [[ "$agent_name" == "<end>" ]]; then
            if [ "${#agent_names_array[@]}" -eq 0 ]; then
                echo "Please provide at least one agent."
                continue
            else
                break
            fi
        fi
        # Trim whitespace from both input and valid agent names
        agent_name_trimmed=$(echo "$agent_name" | xargs)
        found=false
        for valid_agent in "${valid_agents[@]}"; do
            valid_agent_trimmed=$(echo "$valid_agent" | xargs)
            if [[ "$agent_name_trimmed" == "$valid_agent_trimmed" ]]; then
                found=true
                break
            fi
        done
        # Accept oneshot-* and planning_and_control as valid agent names
        if [[ "$found" == true && -n "$agent_name_trimmed" ]] || [[ "$agent_name_trimmed" =~ ^oneshot-.* ]] || [[ "$agent_name_trimmed" == "planning_and_control" ]]; then
            if [[ "$agent_name_trimmed" == "oneshot" ]]; then
                agent_names_array+=("oneshot-$engineer_model")
            else
                agent_names_array+=("$agent_name_trimmed")
            fi
        else
            echo "Invalid agent name: '$agent_name_trimmed'. Please enter valid agent names from model_config.yaml, use oneshot-<model>, or planning_and_control."
            echo -e "\033[1;31mInvalid agent name: '$agent_name_trimmed'. Please enter valid agent names from model_config.yaml, use oneshot-<model>, or planning_and_control.\033[0m"
            echo -e "\033[1;31mPlease provide at least one agent.\033[0m"
        fi
    done
    # Join agent names with commas for YAML output
    local agent_names
    agent_names=$(IFS=','; echo "${agent_names_array[*]}")
    printf "agents: [%s]\n" "$agent_names" >> "$config_file"
    
    # Write run file with config_path and problem_ids
    jq -n --arg config_path "$config_file" --argjson problem_ids "$(printf '%s\n' "${selected_problem_ids[@]}" | jq -R . | jq -s .)" '{config_path: $config_path, problem_ids: $problem_ids}' > "$run_file"
    
    python "$SCRIPT_DIR/python_scripts/run_benchmark.py" --benchmark_file "$run_file"
}

# Function to generate problems based on config file settings
generate_problems_from_config() {
    local config_file="$1"
    local selected_problem_ids=()
    
    # Extract settings from config file
    local json_file_path=$(grep "^json_file_path:" "$config_file" | cut -d' ' -f2)
    local selection_type=$(grep -A1 "^problem:" "$config_file" | grep "selection_type:" | sed 's/.*selection_type: *"\([^"]*\)".*/\1/')
    
    if [[ "$selection_type" == "by_level" ]]; then
        local bronze_count=$(grep -A5 "^problem:" "$config_file" | grep "bronze:" | sed 's/.*bronze: *\([0-9]*\).*/\1/')
        local silver_count=$(grep -A5 "^problem:" "$config_file" | grep "silver:" | sed 's/.*silver: *\([0-9]*\).*/\1/')
        local gold_count=$(grep -A5 "^problem:" "$config_file" | grep "gold:" | sed 's/.*gold: *\([0-9]*\).*/\1/')
        local platinum_count=$(grep -A5 "^problem:" "$config_file" | grep "platinum:" | sed 's/.*platinum: *\([0-9]*\).*/\1/')
        
        # Select problems by level using our helper function
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "bronze" "$bronze_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "silver" "$silver_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "gold" "$gold_count") )
        selected_problem_ids+=( $(select_problems_by_level "$json_file_path" "platinum" "$platinum_count") )
        
    elif [[ "$selection_type" == "random" ]]; then
        local count=$(grep -A3 "^problem:" "$config_file" | grep "count:" | sed 's/.*count: *\([0-9]*\).*/\1/')
        local avoid_levels_line=$(grep -A4 "^problem:" "$config_file" | grep "avoid_levels:")
        
        if [[ -n "$avoid_levels_line" ]]; then
            # Extract avoid levels (remove brackets and split by comma)
            local avoid_str=$(echo "$avoid_levels_line" | sed 's/.*avoid_levels: *\[\([^\]]*\)\].*/\1/' | tr -d ' ')
            local jq_query='to_entries[]'
            IFS=',' read -ra avoid <<< "$avoid_str"
            for level in "${avoid[@]}"; do
                jq_query+=" | select(.value.problem_level != \"$level\")"
            done
            mapfile -t all_ids < <(jq -r "$jq_query | .key" "$json_file_path")
        else
            mapfile -t all_ids < <(jq -r 'to_entries[] | .key' "$json_file_path")
        fi
        selected_problem_ids=( $(printf "%s\n" "${all_ids[@]}" | shuf | head -n "$count") )
        
    elif [[ "$selection_type" == "by_id" ]]; then
        # Extract the list of IDs from config
        local id_list=$(grep -A2 "^problem:" "$config_file" | grep "list" | sed 's/.*\[\([^]]*\)\].*/\1/')
        IFS=',' read -ra selected_problem_ids <<< "$id_list"
    fi
    
    # Return the selected problem IDs
    printf "%s\n" "${selected_problem_ids[@]}"
}

# Function for running an existing benchmark
run_existing_benchmark() {
    echo "How do you want to run the existing benchmark?"
    echo "(1) Use config YAML file (will generate new random problems based on settings)"
    echo "(2) Use run JSON file (will use exact same problems as previous run)"
    local run_choice
    read -p "Select an option (1-2): " run_choice
    while [[ "$run_choice" != "1" && "$run_choice" != "2" ]]; do
        echo "Invalid input. Please enter 1 or 2:"
        read -p "Select an option (1-2): " run_choice
    done

    if [[ "$run_choice" == "1" ]]; then
        # Use config YAML file
        local config_file
        config_file=$(get_valid_path "Please enter the path to the config YAML file: " "file" ".yaml")
        
        # Generate new problems based on config settings
        echo "Generating new problems based on config settings..."
        mapfile -t selected_problem_ids < <(generate_problems_from_config "$config_file")
        
        local run_timestamp=$(date +%Y%m%d_%H%M%S)
        local run_dir="$SCRIPT_DIR/benchmark_output/runs/run_${run_timestamp}"
        mkdir -p "$run_dir"
        local run_file="$run_dir/run_${run_timestamp}.json"
        jq -n --arg config_path "$config_file" --argjson problem_ids "$(printf '%s\n' "${selected_problem_ids[@]}" | jq -R . | jq -s .)" '{config_path: $config_path, problem_ids: $problem_ids}' > "$run_file"
        echo "Run file created: $run_file"
        echo "Selected $(echo "${selected_problem_ids[@]}" | wc -w) problems based on config settings"
        echo -e "\033[1;32mRun directory created:\033[0m $run_dir"
        echo -e "\033[1;32mRun file created:\033[0m $run_file"
        echo -e "\033[1;34mSelected $(echo "${selected_problem_ids[@]}" | wc -w) problems based on config settings\033[0m"
        python "$SCRIPT_DIR/python_scripts/run_benchmark.py" --benchmark_file "$run_file"
    elif [[ "$run_choice" == "2" ]]; then
        # Use run JSON file
        local run_file
        run_file=$(get_valid_path "Please enter the path to the run JSON file: " "file" ".json")
        
        # Extract config_path and problem_ids from the existing run file
        local config_path
        local problem_ids
        config_path=$(jq -r '.config_path' "$run_file")
        problem_ids=$(jq -r '.problem_ids' "$run_file")
        
        # Verify the config file exists
        if [[ ! -f "$config_path" ]]; then
            echo "Error: Config file referenced in run file does not exist: $config_path"
            exit 1
        fi
        
        # Create new run file with same settings
        local run_timestamp=$(date +%Y%m%d_%H%M%S)
        local run_dir="$SCRIPT_DIR/benchmark_output/runs/run_${run_timestamp}"
        mkdir -p "$run_dir"
        local run_file="$run_dir/run_${run_timestamp}.json"
        jq -n --arg config_path "$config_path" --argjson problem_ids "$problem_ids" '{config_path: $config_path, problem_ids: $problem_ids}' > "$run_file"
        echo "New run file created: $run_file"
        echo "Using same problems as previous run: $(echo "$problem_ids" | jq -r '. | length') problems"
        echo -e "\033[1;32mRun directory created:\033[0m $run_dir"
        echo -e "\033[1;32mNew run file created:\033[0m $run_file"
        echo -e "\033[1;34mUsing same problems as previous run: $(echo "$problem_ids" | jq -r '. | length') problems\033[0m"
        python "$SCRIPT_DIR/python_scripts/run_benchmark.py" --benchmark_file "$run_file"
    fi
}

# --- MAIN FUNCTION ---

main() {
    print_welcome_message
    local option=$(read_user_option_for_benchmark_type)

    if [[ "$option" == "1" ]]; then
        create_new_benchmark
    elif [[ "$option" == "2" ]]; then
        run_existing_benchmark
    fi
}

# Run the main function
main
