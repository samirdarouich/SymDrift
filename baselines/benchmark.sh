#!/bin/bash
# Run 50-sample timing benchmark for all baselines and report results.
# Each sample script prints a line: [METHOD] Total: Xs | Avg/sample: Yms

BASELINES_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$BASELINES_DIR/scripts"
RESULTS_FILE="$BASELINES_DIR/benchmark_results.txt"

echo "=== Conformer Generation Benchmark (50 QM9 samples each) ==="
echo "Results will be saved to: $RESULTS_FILE"
echo ""

> "$RESULTS_FILE"  # clear previous results

METHODS=(geodiff geomol mcf tordiff etflow)
METHOD_NAMES=("GeoDiff" "GeoMol" "MCF" "TorsionalDiff" "ETFlow")

declare -A TIMING

for i in "${!METHODS[@]}"; do
    method="${METHODS[$i]}"
    name="${METHOD_NAMES[$i]}"
    script="$SCRIPTS_DIR/${method}_sample.sh"

    if [ ! -f "$script" ]; then
        echo "[$name] SKIP — script not found: $script"
        continue
    fi

    echo "--- Running: $name ---"
    output=$(bash "$script" 2>&1)
    exit_code=$?
    echo "$output"

    if [ $exit_code -ne 0 ]; then
        TIMING[$name]="FAILED"
        echo "[$name] FAILED (exit code $exit_code)" >> "$RESULTS_FILE"
    else
        # Extract the timing line printed by each sample script
        timing_line=$(echo "$output" | grep -E "Avg inference speed" | tail -1)
        TIMING[$name]="$timing_line"
        echo "$timing_line" >> "$RESULTS_FILE"
    fi
    echo ""
done

echo "=============================="
echo "  Benchmark Summary"
echo "=============================="
printf "%-20s | %s\n" "Method" "Result"
printf "%-20s-+-%s\n" "--------------------" "-------------------------------"
for name in "${METHOD_NAMES[@]}"; do
    printf "%-20s | %s\n" "$name" "${TIMING[$name]:-NOT RUN}"
done
echo ""
echo "Full results saved to: $RESULTS_FILE"
