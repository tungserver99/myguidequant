#!/usr/bin/env bash
set -euo pipefail
set -x

# Re-pack an existing LNQ + GuidedQuant layerwise-quantized cache, then run PPL eval.
# This intentionally does not call quantize.py.
# Defaults match scripts/run_lnq_guidedquant_llama32_1b_c4_eval_ppl.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-1B}"
BITS="${BITS:-3}"
NUM_GROUPS="${NUM_GROUPS:-4}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CD_CYCLES="${CD_CYCLES:-4}"
RANDOM_STATE="${RANDOM_STATE:-42}"
CACHE_DIR="${CACHE_DIR:-cache}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-dataset_cache}"
EVAL_METHOD="${EVAL_METHOD:-block}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_DTYPE="${EVAL_DTYPE:-fp16}"

export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TQDM_MAXINTERVAL="${TQDM_MAXINTERVAL:-30}"
export TQDM_POSITION="${TQDM_POSITION:--1}"

MODEL_BASENAME="${MODEL_NAME##*/}"
QUANTIZED_DIR="${CACHE_DIR}/layerwise_quantized/${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}"
PACKED_MODEL_DIR="${CACHE_DIR}/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}"
PPL_JSON="${PACKED_MODEL_DIR}/ppl_${EVAL_METHOD}.json"
PPL_TAG="llama32_1b_guidedquant_${BITS}bit_${DATASET}_${NUM_EXAMPLES}_${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}_${EVAL_METHOD}"

if [[ ! -d "${QUANTIZED_DIR}" ]]; then
  echo "Missing layerwise quantized cache, refusing to quantize again: ${QUANTIZED_DIR}" >&2
  exit 1
fi

python layerwise_nuq.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --cd_cycles "${CD_CYCLES}" \
  --mode pack \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  --overwrite_pack

python eval_ppl.py \
  --model-path "${PACKED_MODEL_DIR}" \
  --datasets wikitext2 c4 \
  --seqlen "${SEQ_LEN}" \
  --method "${EVAL_METHOD}" \
  --dtype "${EVAL_DTYPE}" \
  --cache-dir "${EVAL_CACHE_DIR}" \
  --out-json "${PPL_JSON}" \
  --tag "${PPL_TAG}"

echo "Done."
echo "Packed model: ${PACKED_MODEL_DIR}"
echo "PPL results: ${PPL_JSON}"