#!/usr/bin/env bash
set -euo pipefail

MAX_JOBS=4

RUN_DIR="/app/logs/external_blitz/$(date +%Y-%m-%d_%H-%M-%S)"
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
      -engine cmd=$WRAP args="talbot /app/build/talbot_uci --opponent $opp" name=talbot \
        option.SyzygyPath=/app/data/tablebases/3-4-5 \
      -engine cmd=$WRAP "$@" name=$opp \
      -each tc=65.3+0.54 restart=on \
      -openings file=/app/data/opening_pgn/8moves_v3_tune.pgn format=pgn order=random plies=8 \
      -srand 12345 \
      -rounds 10 -repeat -concurrency 1 \
      -resign movecount=3 score=600 \
      -draw movenumber=40 movecount=8 score=10 \
      -tb /app/data/tablebases/3-4-5 \
      -pgnout file=match.pgn nodes=true nps=true \
      > fastchess.log 2>&1
  ) &
}

run zurichess-neuchatel   args="opponent /app/data/engines/zurichess/zurichess" option.Hash=128 option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run tomitank-7.0          args="opponent /usr/bin/node /app/data/engines/tomitank/tomitankChess.js" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run sentinel-2.0          args="opponent /app/data/engines/sentinel/sentinel" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run vault-0.4.1           args="opponent /app/data/engines/vault/vault" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5 option.EvalFile=/app/data/engines/vault/default.nn
run erinn-2               args="opponent /usr/bin/java -jar /app/data/engines/erinn/Erinn-2.jar" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run pingu-5.0.0           args="opponent /app/data/engines/pingu/Pingu" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run casanchess-2.0        args="opponent /app/data/engines/casanchess/build/casanchess" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run svart-6               args="opponent /app/data/engines/svart/svart6-x86_64-linux-v3" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run dual-0.4.3            args="opponent /app/data/engines/dual/target/release/Dual" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
run scarlet-2.0           args="opponent /app/data/engines/scarlet/target/release/scarlet" option.Hash=128 option.SyzygyPath=/app/data/tablebases/3-4-5
wait
echo "Done. Results in $RUN_DIR/<opponent>/"