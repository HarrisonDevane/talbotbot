#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/app/logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

rm -rf /tmp/talbot_slots

/app/data/fastchess/fastchess \
  -engine cmd=/app/scripts/pin_wrapper.sh args="talbot /app/build/talbot_uci" name=talbot \
    option.SyzygyPath=/app/data/tablebases/3-4-5 \
  -engine cmd=/app/scripts/pin_wrapper.sh args="opponent /app/data/engines/apotheosis/ApotheosisV4.0.1_Linux" name=apotheosis \
    option.Hash=256 \
  -each tc=120+1 \
  -openings file=/app/data/opening_pgn/8moves_v3.pgn format=pgn order=random plies=16 \
  -rounds 50 -repeat -concurrency 4 \
  -resign movecount=3 score=600 \
  -draw movenumber=40 movecount=8 score=10 \
  -pgnout file=match.pgn