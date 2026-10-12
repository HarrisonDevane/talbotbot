#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=4

RUN_DIR="/app/logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_DIR"

rm -rf /tmp/talbot_slots

FASTCHESS=/app/data/fastchess/fastchess
WRAP=/app/scripts/pin_wrapper.sh

throttle() {  # block until fewer than MAX_JOBS background jobs are running
  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
    wait -n || true   # wait for any one job to finish; ignore its exit code (set -e)
  done
}

run() {  # $1 = opponent name, rest = opponent -engine options
  local opp=$1; shift
  throttle
  mkdir -p "$RUN_DIR/$opp"
  (
    cd "$RUN_DIR/$opp"
    "$FASTCHESS" \
      -engine cmd=$WRAP args="talbot /app/build/talbot_uci --opponent $opp --log-dir $RUN_DIR/$opp" name=talbot \
        option.SyzygyPath=/app/data/tablebases/3-4-5 \
      -engine cmd=$WRAP "$@" name=$opp \
      -each tc=65.3+0.54 restart=on \
      -openings file=/app/data/opening_pgn/8moves_v3.pgn format=pgn order=random plies=8 \
      -srand 12345 \
      -rounds 50 -repeat -concurrency 1 \
      -resign movecount=3 score=600 \
      -draw movenumber=40 movecount=8 score=10 \
      -tb /app/data/tablebases/3-4-5 \
      -pgnout file=match.pgn \
      > fastchess.log 2>&1
  ) &
}

run olithink   args="opponent /app/data/engines/olithink/olithink" option.SyzygyPath=/app/data/tablebases/3-4-5
run tomitank   args="opponent /usr/bin/node /app/data/engines/tomitank/tomitankChess.js" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run sentinel   args="opponent /app/data/engines/sentinel/sentinel" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run casanchess args="opponent /app/data/engines/casanchess/build/casanchess" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5

wait
echo "Done. Results in $RUN_DIR/<opponent>/"