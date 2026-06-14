#!/usr/bin/env bash
# Full reproduction: run baselines, then regenerate both derived figures.
#
# Text-only needs no imagery and downloads CLIP weights on first run.
# Vision approaches additionally need the source log PDF (see DATASHEET.md)
# passed via LOGS_PDF.
#
# Usage:
#   bash scripts/reproduce.sh                       # text-only + figures
#   LOGS_PDF=path/to/vilkyciai22_logs500.pdf \
#     bash scripts/reproduce.sh full                # all approaches + figures
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p images results

MODE="${1:-text}"
if [ "$MODE" = "full" ]; then
  if [ -z "${LOGS_PDF:-}" ]; then
    echo "full mode needs LOGS_PDF=path/to/vilkyciai22_logs500.pdf"; exit 2
  fi
  APPROACHES="text_only vision_clip fusion grounding_dino blip2"
  EXTRA="--logs-pdf $LOGS_PDF"
else
  APPROACHES="text_only"
  EXTRA=""
fi

echo ">> Running baselines: $APPROACHES"
python scripts/run_evaluation.py --approaches $APPROACHES $EXTRA \
  --out results/reproduced_results.json

echo ">> Generating results figure"
python scripts/make_results_figure.py \
  --results results/reproduced_results.json \
  --out images/Figure2_Results_Comparison.png

echo ">> Generating architecture figure"
python scripts/make_architecture_figure.py --out images/architecture.png

echo ">> Done. Figures in images/, results in results/reproduced_results.json"
