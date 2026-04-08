#!/bin/bash
# Run all tasks sequentially, cleaning up sandboxes between runs.
# Usage: pixi run bash run_all.sh [jobs_dir]
set -euo pipefail

JOBS_DIR="${1:-jobs/latest}"
rm -rf "$JOBS_DIR"
mkdir -p "$JOBS_DIR"

TASKS_DIR="tasks"
RESULTS_FILE="$JOBS_DIR/all_results.jsonl"
> "$RESULTS_FILE"

cleanup_sandboxes() {
    python3 -c "
import daytona
d = daytona.Daytona()
for sb in d.list().items:
    try: d.delete(sb)
    except: pass
" 2>/dev/null || true
}

TOTAL=0
PASSED=0
TOTAL_SCORE=0
TASK_SCORES=""

for task_dir in $(ls -d "$TASKS_DIR"/*/); do
    task_name=$(basename "$task_dir")
    echo "=== Running $task_name ==="

    cleanup_sandboxes

    TASK_JOBS="$JOBS_DIR/$task_name"
    mkdir -p "$TASK_JOBS"

    if pixi run harbor run -p "$task_dir" -n 1 --agent-import-path agent:AutoAgent -e daytona -o "$TASK_JOBS" --job-name run > "$TASK_JOBS/run.log" 2>&1; then
        echo "  Completed"
    else
        echo "  Failed (exit code $?)"
    fi

    # Extract score from result.json
    SCORE=$(python3 -c "
import json, sys
try:
    with open('$TASK_JOBS/run/result.json') as f:
        d = json.load(f)
    stats = d.get('stats', {}).get('evals', {})
    for eval_name, eval_data in stats.items():
        metrics = eval_data.get('metrics', [{}])
        if metrics:
            print(f'{metrics[0].get(\"mean\", 0.0):.6f}')
            sys.exit(0)
    print('0.000000')
except Exception as e:
    print('0.000000', file=sys.stderr)
    print(f'Error: {e}', file=sys.stderr)
    print('0.000000')
" 2>/dev/null)

    echo "  Score: $SCORE"
    echo "{\"task\": \"$task_name\", \"score\": $SCORE}" >> "$RESULTS_FILE"

    TOTAL=$((TOTAL + 1))
    if (( $(echo "$SCORE > 0" | bc -l) )); then
        PASSED=$((PASSED + 1))
    fi
    TOTAL_SCORE=$(echo "$TOTAL_SCORE + $SCORE" | bc -l)

    if [ -n "$TASK_SCORES" ]; then
        TASK_SCORES="$TASK_SCORES,$task_name=$SCORE"
    else
        TASK_SCORES="$task_name=$SCORE"
    fi
done

cleanup_sandboxes

AVG_SCORE=$(echo "scale=6; $TOTAL_SCORE / $TOTAL" | bc -l)
echo ""
echo "=== SUMMARY ==="
echo "Total: $TOTAL"
echo "Passed: $PASSED/$TOTAL"
echo "Average score: $AVG_SCORE"
echo ""
echo "Per-task scores:"
cat "$RESULTS_FILE"
