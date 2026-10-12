#!/usr/bin/env bash
set -euo pipefail

ENGINE_TYPE="$1"
ENGINE_BIN="$2"
shift 2

SLOT_DIR="/tmp/talbot_slots"
mkdir -p "$SLOT_DIR/talbot" "$SLOT_DIR/opponent"

declare -A TALBOT_MCTS=(   [1]="2"       [2]="5"       [3]="10"       [4]="13" )
declare -A TALBOT_BATCH=(  [1]="18,3,19" [2]="21,6,22" [3]="26,11,27" [4]="29,14,30" )
declare -A OPPONENT=(      [1]="4,20"    [2]="7,23"    [3]="12,28"    [4]="15,31" )

POOL="$SLOT_DIR/$ENGINE_TYPE"

SLOT=""
for attempt in $(seq 1 100); do
  for i in 1 2 3 4; do
    exec 9>"$POOL/$i.lock"
    if flock -n 9; then SLOT="$i"; break 2; fi
    exec 9>&-
  done
  sleep 0.1
done

[ -z "$SLOT" ] && { echo "no slot free ($ENGINE_TYPE)" >&2; exit 1; }

if [ "$ENGINE_TYPE" = "talbot" ]; then
  MCTS="${TALBOT_MCTS[$SLOT]}"
  BATCH="${TALBOT_BATCH[$SLOT]}"
  exec taskset -c "$MCTS,$BATCH" "$ENGINE_BIN" \
    --worker-cores "$MCTS" --batcher-cores "$BATCH" "$@"
else
  exec taskset -c "${OPPONENT[$SLOT]}" "$ENGINE_BIN" "$@"
fi