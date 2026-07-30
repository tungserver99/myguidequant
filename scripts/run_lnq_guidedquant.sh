#!/usr/bin/env bash
set -euo pipefail
set -x

# One-shot runner for LNQ + GuidedQuant.
# Defaults: 3-bit, RedPajama calibration with 1024 samples / 4096 tokens.
# Override with positional args if needed:
#   bash scripts/run_lnq_guidedquant.sh [MODEL_NAME] [BITS] [NUM_GROUPS] [MODE]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${1:-meta-llama/Llama-2-7b-hf}"
BITS="${2:-3}"
NUM_GROUPS="${3:-4}"
MODE="${4:-pack}"
DATASET="redpajama"
SEQ_LEN="4096"
NUM_EXAMPLES="1024"

# Colab often renders carriage-return progress updates as new lines.
# Keep tqdm enabled, but refresh less often so logs stay readable.
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TQDM_MAXINTERVAL="${TQDM_MAXINTERVAL:-30}"
export TQDM_POSITION="${TQDM_POSITION:--1}"

MODEL_BASENAME="${MODEL_NAME##*/}"
TOKEN_CACHE="cache/tokens/${MODEL_BASENAME}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}.pt"

if [[ ! -f "${TOKEN_CACHE}" ]]; then
  bash scripts/download_calibration.sh
fi

if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing calibration token cache: ${TOKEN_CACHE}" >&2
  echo "Check scripts/download_calibration.sh or place the matching .pt file under cache/tokens/." >&2
  exit 1
fi

python quantize.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --parent_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --mode "${MODE}"

python layerwise_nuq.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --mode "${MODE}"