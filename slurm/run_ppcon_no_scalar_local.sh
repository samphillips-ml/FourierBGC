#!/bin/bash
# Local replacement for the ppcon_no_scalar tasks (array indices 60-74) of
# slurm/train_eval_final_ablation.slurm, which never produced output on the
# cluster. Same recipe, same epoch budget, same seeds, same eval path -- the
# only difference is that these runs happen on this machine, so PPCon-NoCoord
# is the one model in the spine whose provenance is local rather than the
# single cluster job. Note that in decisions.md and the paper's reproducibility
# statement.
#
#   5 seeds x 3 targets = 15 runs, PPCon's own per-variable epoch budget.
#   ~17 s/epoch, 1875 epochs total; 3 concurrent workers -> roughly 3-4 h.
#
# Results land in local-output/ppcon_no_scalar/seed{N}/{VAR}/, mirroring the
# cluster-output/ layout so the aggregation scripts need no special-casing.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

OUT=$REPO/local-output/ppcon_no_scalar
mkdir -p "$OUT"

# PPCon's published per-variable epoch budget, not the common recipe's 100.
# A case statement rather than an associative array: macOS ships bash 3.2,
# which has no `declare -A`.
epochs_for() {
    case $1 in
        NITRATE) echo 100 ;;
        CHLA)    echo 150 ;;
        BBP700)  echo 125 ;;
    esac
}

run_one() {
    local SEED=$1 VAR=$2 EP=$3
    local ROOT=$OUT/seed${SEED}/${VAR}
    if [ -s "$ROOT/eval_best.txt" ]; then
        echo "[skip] seed$SEED $VAR already done"
        return 0
    fi
    mkdir -p "$ROOT"
    echo "[start] seed$SEED $VAR ($EP epochs) $(date +%H:%M:%S)"
    OMP_NUM_THREADS=3 "$REPO/.venv-ppcon/bin/python" -u ppcon_no_scalar/run.py \
        --variable "$VAR" --epochs "$EP" --seed "$SEED" --save_dir "$ROOT" \
        > "$ROOT/train.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "[FAIL train] seed$SEED $VAR -- see $ROOT/train.log"
        return 1
    fi
    OMP_NUM_THREADS=3 "$REPO/.venv/bin/python" -u evaluate.py \
        --model ppcon_no_scalar --target_var "$VAR" \
        --checkpoint_dir "$ROOT/model" --epoch "$EP" \
        > "$ROOT/eval_best.txt" 2> "$ROOT/eval.log"
    if [ $? -ne 0 ]; then
        echo "[FAIL eval] seed$SEED $VAR -- see $ROOT/eval.log"
        rm -f "$ROOT/eval_best.txt"
        return 1
    fi
    echo "[done]  seed$SEED $VAR $(date +%H:%M:%S)  $(head -1 "$ROOT/eval_best.txt")"
}
export -f run_one
export OUT REPO

# Longest jobs first so the tail of the run isn't a single straggler.
JOBS=""
for VAR in CHLA BBP700 NITRATE; do
    for SEED in 0 1 2 3 4; do
        JOBS+="$SEED $VAR $(epochs_for $VAR)"$'\n'
    done
done

echo "$JOBS" | grep -v '^$' | xargs -P 3 -L 1 bash -c 'run_one $0 $1 $2'

echo "======================================================"
echo "all runs finished $(date)"
find "$OUT" -name eval_best.txt | wc -l | xargs echo "eval files produced:"
