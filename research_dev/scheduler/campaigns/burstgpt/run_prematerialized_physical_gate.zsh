#!/usr/bin/env zsh
set -eu

if (( $# != 9 )); then
    print -u2 -- "usage: $0 <base-command> <output> <selection-mode> <catalog> <request-indices> <observation-store> <adaptive-observation-store> <phone-session-root> <energy-attribution-kind>"
    exit 2
fi

base_command=$1
output=$2
selection_mode=$3
catalog=$4
request_indices=$5
observation_store=$6
adaptive_observation_store=$7
phone_session_root=$8
energy_attribution_kind=$9

for input_path in "$base_command" "$catalog" "$observation_store" \
        "$adaptive_observation_store"; do
    if [[ ! -f $input_path ]]; then
        print -u2 -- "physical gate input is absent: $input_path"
        exit 2
    fi
done
if [[ $output != /* || -e $output ]]; then
    print -u2 -- "output must be a new absolute path: $output"
    exit 2
fi
if [[ ! $request_indices =~ '^[0-9]+(,[0-9]+)*$' ]]; then
    print -u2 -- "request indices are invalid"
    exit 2
fi

command_line=$(tail -n 1 "$base_command")
command=(${(z)command_line})

set_option() {
    local option=$1
    local value=$2
    local found=0
    local index
    for ((index = 1; index <= ${#command}; index++)); do
        if [[ ${command[$index]} == $option ]]; then
            if (( index == ${#command} )); then
                print -u2 -- "physical gate option lacks a value: $option"
                exit 2
            fi
            command[$((index + 1))]=$value
            found=1
        fi
    done
    if (( ! found )); then
        command+=("$option" "$value")
    fi
}

set_option --capability-catalog "$catalog"
set_option --selection-mode "$selection_mode"
set_option --energy-attribution-kind "$energy_attribution_kind"
set_option --output "$output/run"
set_option --request-indices "$request_indices"
set_option --observation-store-input "$observation_store"
set_option --adaptive-observation-store-input "$adaptive_observation_store"
set_option --phone-session-root "$phone_session_root"

mkdir -p -- "$output"
printf '%q ' "${command[@]}" > "$output/RUN_COMMAND.txt"
printf '\n' >> "$output/RUN_COMMAND.txt"

here=${0:A:h}
repo_root=${S42_UNIFIED_REPO_ROOT:-${here:h:h:h:h}}
cd "$repo_root"
"${command[@]}" 2>&1 | tee "$output/runner.log"

sha256sum "$catalog" "$output/RUN_COMMAND.txt" \
    "$output/run/RESULT.json" \
    "$output/run/SCHEDULER_DECISION_LOG.json" \
    "$output/run/AUTOMATED_OBSERVATIONS.json" \
    > "$output/SHA256SUMS.txt"
