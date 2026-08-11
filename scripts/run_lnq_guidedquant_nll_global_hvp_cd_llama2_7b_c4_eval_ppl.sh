#!/usr/bin/env bash
set -euo pipefail
set -x

# LNQ + full-NLL global HVP GuideQuant saliency on Llama-2-7B, then PPL eval.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
BITS="${BITS:-3}"
NUM_GROUPS="${NUM_GROUPS:-4}"
MODE="${MODE:-pack}"
DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
NUM_EXAMPLES="${NUM_EXAMPLES:-128}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"
CD_CYCLES="${CD_CYCLES:-4}"
RANDOM_STATE="${RANDOM_STATE:-42}"
OVERWRITE="${OVERWRITE:-0}"
CACHE_DIR="${CACHE_DIR:-cache_nll_global_hvp_cd_llama2_7b}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-dataset_cache}"
EVAL_METHOD="${EVAL_METHOD:-block}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_DTYPE="${EVAL_DTYPE:-fp16}"
NLL_HVP_PROBES="${NLL_HVP_PROBES:-1}"
NLL_HVP_SEED="${NLL_HVP_SEED:-0}"
NLL_HVP_LAYER_CHUNK_SIZE="${NLL_HVP_LAYER_CHUNK_SIZE:-1}"
NLL_HVP_SDPA_BACKEND="${NLL_HVP_SDPA_BACKEND:-math}"
NLL_HVP_SAVED_TENSORS_DEVICE="${NLL_HVP_SAVED_TENSORS_DEVICE:-cpu}"
RESULT_SUFFIX="nll_global_hvp_cd_llama2_7b"
HESSIAN_SUFFIX="nll_global_hvp_p${NLL_HVP_PROBES}_seed${NLL_HVP_SEED}_lc${NLL_HVP_LAYER_CHUNK_SIZE}_sum"

has_packed_model() {
  local dir="$1"
  [[ -f "${dir}/model.safetensors" || -f "${dir}/pytorch_model.bin" ]] && return 0
  compgen -G "${dir}/*.index.json" >/dev/null && return 0
  compgen -G "${dir}/model-*.safetensors" >/dev/null && return 0
  compgen -G "${dir}/pytorch_model-*.bin" >/dev/null && return 0
  return 1
}

export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export TQDM_MAXINTERVAL="${TQDM_MAXINTERVAL:-30}"
export TQDM_POSITION="${TQDM_POSITION:--1}"

MODEL_BASENAME="${MODEL_NAME##*/}"
MODEL_TAG="${MODEL_TAG:-${MODEL_BASENAME//[^[:alnum:]]/_}}"
PACKED_MODEL_DIR="${CACHE_DIR}/layerwise_packed/layerwise-${MODEL_BASENAME}-w${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}_${HESSIAN_SUFFIX}"
PPL_JSON="${PACKED_MODEL_DIR}/ppl_${EVAL_METHOD}_${RESULT_SUFFIX}.json"
PPL_TAG="${MODEL_TAG}_guidedquant_${RESULT_SUFFIX}_${HESSIAN_SUFFIX}_${BITS}bit_${DATASET}_${NUM_EXAMPLES}_${SEQ_LEN}_g${NUM_GROUPS}_iter${NUM_ITERATIONS}_cd${CD_CYCLES}_${EVAL_METHOD}"
QUANT_LOG_DIR="logs_layer"

quantize_overwrite_args=()
layerwise_overwrite_args=()
hessian_overwrite_args=()
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  quantize_overwrite_args=(--overwrite_tokens --overwrite_gradients --overwrite_quantize --overwrite_pack)
  layerwise_overwrite_args=(--overwrite_quantize --overwrite_pack)
  hessian_overwrite_args=(--overwrite_saliency --overwrite_hessians)
fi

if [[ -d "${PACKED_MODEL_DIR}" ]] && ! has_packed_model "${PACKED_MODEL_DIR}"; then
  echo "Detected incomplete packed NLL global HVP model directory, will re-pack: ${PACKED_MODEL_DIR}" >&2
  layerwise_overwrite_args+=(--overwrite_pack)
fi

python quantize.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --parent_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --mode "${MODE}" \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  "${quantize_overwrite_args[@]}"

python layerwise_nuq.py "${MODEL_NAME}" \
  --seed_precision "${BITS}" \
  --dataset "${DATASET}" \
  --seq_len "${SEQ_LEN}" \
  --num_examples "${NUM_EXAMPLES}" \
  --num_groups "${NUM_GROUPS}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --cd_cycles "${CD_CYCLES}" \
  --hessian_source nll_global_hvp \
  --nll_hvp_probes "${NLL_HVP_PROBES}" \
  --nll_hvp_seed "${NLL_HVP_SEED}" \
  --nll_hvp_layer_chunk_size "${NLL_HVP_LAYER_CHUNK_SIZE}" \
  --nll_hvp_sdpa_backend "${NLL_HVP_SDPA_BACKEND}" \
  --nll_hvp_saved_tensors_device "${NLL_HVP_SAVED_TENSORS_DEVICE}" \
  --mode "${MODE}" \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  "${hessian_overwrite_args[@]}" \
  "${layerwise_overwrite_args[@]}"

latest_quant_log="$(ls -t "${QUANT_LOG_DIR}"/*.txt 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_quant_log}" ]]; then
  echo "Latest quantization log: ${latest_quant_log}"
else
  echo "WARNING: No quantization log found under ${QUANT_LOG_DIR}." >&2
fi
if [[ ! -d "${PACKED_MODEL_DIR}" ]] || ! has_packed_model "${PACKED_MODEL_DIR}"; then
  echo "Packed NLL global HVP model directory is missing model weights: ${PACKED_MODEL_DIR}" >&2
  exit 1
fi

eval_args=(
  --model-path "${PACKED_MODEL_DIR}"
  --datasets wikitext2 c4
  --seqlen "${SEQ_LEN}"
  --method "${EVAL_METHOD}"
  --dtype "${EVAL_DTYPE}"
  --cache-dir "${EVAL_CACHE_DIR}"
  --out-json "${PPL_JSON}"
  --tag "${PPL_TAG}"
)

if [[ "${EVAL_METHOD}" == "sliding" ]]; then
  eval_args+=(--stride "${EVAL_STRIDE}")
fi

python eval_ppl.py "${eval_args[@]}"

echo "Done."
echo "Packed NLL global HVP model: ${PACKED_MODEL_DIR}"
echo "PPL results: ${PPL_JSON}"
echo "PPL tag: ${PPL_TAG}"


