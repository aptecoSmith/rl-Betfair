#!/bin/bash
# Simple durable monitor: sleep, then report cohort progress + GPU. Re-arm by
# relaunching from the parent on wake. No nested PowerShell (that broke the
# inline version with exit 127).
cd "C:/Users/jsmit/source/repos/rl-betfair"
sleep "${1:-14400}"
echo "=== MONITOR WAKE ==="
date
for L in registry/scalping_ga_c1_*.log registry/scalping_ga_c2_*.log registry/scalping_ga_c3_*.log; do
  [ -f "$L" ] || continue
  echo "--- $L ---"
  grep -E "Generation [0-9]/|Cohort complete|Top-3 by|Gen [0-9].*composite" "$L" 2>/dev/null | tail -4
done
echo "--- scoreboards (completed cohorts) ---"
for S in registry/scalping_ga_c*/scoreboard.jsonl; do
  [ -f "$S" ] || continue
  echo "$S rows=$(wc -l < "$S" 2>/dev/null)"
done
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
echo "=== END MONITOR WAKE ==="
