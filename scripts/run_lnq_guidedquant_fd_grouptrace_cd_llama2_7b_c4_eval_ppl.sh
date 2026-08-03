#!/usr/bin/env bash
set -euo pipefail
set -x

# LNQ + finite-difference GroupTrace HNLL on Llama-2-7B with scalar CD, then PPL eval.

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
CACHE_DIR="${CACHE_DIR:-cache_hnll_fd_grouptrace_cd_llama2_7b}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-dataset_cache}"
EVAL_METHOD="${EVAL_METHOD:-block}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_DTYPE="${EVAL_DTYPE:-fp16}"
ASSIGNMENT_SOLVER="cd"
FD_NUM_PROBES="${FD_NUM_PROBES:-1}"
FD_LAYER_CHUNK_SIZE="${FD_LAYER_CHUNK_SIZE:-0}"
FD_EXECUTION_MODE="${FD_EXECUTION_MODE:-paired_batch}"
FD_SCALE_MODE="${FD_SCALE_MODE:-activation_rms}"
FD_SCALE_MULTIPLIER="${FD_SCALE_MULTIPLIER:-0.01}"
FD_BUILD_FLUSH_INTERVAL="${FD_BUILD_FLUSH_INTERVAL:-8}"
NLL_FD_PROFILE="${NLL_FD_PROFILE:-0}"
NLL_FD_DTYPE="${NLL_FD_DTYPE:-auto}"
NLL_HESSIAN_BUILDER="batched_shared_x"
NLL_HESSIAN_GROUP_CHUNK_SIZE="${NLL_HESSIAN_GROUP_CHUNK_SIZE:-4}"
NLL_HESSIAN_VALIDATE_SHARED_X="${NLL_HESSIAN_VALIDATE_SHARED_X:-0}"
RESULT_SUFFIX="hnll_fd_grouptrace_cd_llama2_7b"
HESSIAN_SUFFIX="hnll_fd_grouptrace_p${FD_NUM_PROBES}_mode${FD_EXECUTION_MODE}_scale${FD_SCALE_MODE}_mul${FD_SCALE_MULTIPLIER}_flush${FD_BUILD_FLUSH_INTERVAL}_dt${NLL_FD_DTYPE}_hb${NLL_HESSIAN_BUILDER}_gcs${NLL_HESSIAN_GROUP_CHUNK_SIZE}"

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
fd_profile_args=()
fd_validate_shared_x_args=()
if [[ "${NLL_FD_PROFILE}" == "1" || "${NLL_FD_PROFILE}" == "true" ]]; then
  fd_profile_args=(--nll_hvp_profile)
fi
if [[ "${NLL_HESSIAN_VALIDATE_SHARED_X}" == "1" || "${NLL_HESSIAN_VALIDATE_SHARED_X}" == "true" ]]; then
  fd_validate_shared_x_args=(--nll_hessian_validate_shared_x)
fi
if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" ]]; then
  quantize_overwrite_args=(--overwrite_tokens --overwrite_gradients --overwrite_quantize --overwrite_pack)
  layerwise_overwrite_args=(--overwrite_quantize --overwrite_pack)
  hessian_overwrite_args=(--overwrite_hessians)
fi

if [[ -d "${PACKED_MODEL_DIR}" ]] && ! has_packed_model "${PACKED_MODEL_DIR}"; then
  echo "Detected incomplete packed FD-GroupTrace HNLL model directory, will re-pack: ${PACKED_MODEL_DIR}" >&2
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
  --assignment_solver "${ASSIGNMENT_SOLVER}" \
  --hessian_source nll_fd_grouptrace \
  --nll_hvp_probes "${FD_NUM_PROBES}" \
  --nll_hvp_layer_chunk_size "${FD_LAYER_CHUNK_SIZE}" \
  --nll_hvp_dtype "${NLL_FD_DTYPE}" \
  --nll_hessian_builder "${NLL_HESSIAN_BUILDER}" \
  --nll_hessian_group_chunk_size "${NLL_HESSIAN_GROUP_CHUNK_SIZE}" \
  --fd_execution_mode "${FD_EXECUTION_MODE}" \
  --fd_scale_mode "${FD_SCALE_MODE}" \
  --fd_scale_multiplier "${FD_SCALE_MULTIPLIER}" \
  --fd_build_flush_interval "${FD_BUILD_FLUSH_INTERVAL}" \
  --mode "${MODE}" \
  --cache_dir "${CACHE_DIR}" \
  --random_state "${RANDOM_STATE}" \
  "${fd_profile_args[@]}" \
  "${fd_validate_shared_x_args[@]}" \
  "${hessian_overwrite_args[@]}" \
  "${layerwise_overwrite_args[@]}"

latest_quant_log="$(ls -t "${QUANT_LOG_DIR}"/*.txt 2>/dev/null | head -n 1 || true)"
if [[ -n "${latest_quant_log}" ]]; then
  echo "Latest quantization log: ${latest_quant_log}"
else
  echo "WARNING: No quantization log found under ${QUANT_LOG_DIR}." >&2
fi
if [[ ! -d "${PACKED_MODEL_DIR}" ]] || ! has_packed_model "${PACKED_MODEL_DIR}"; then
  echo "Packed FD-GroupTrace HNLL model directory is missing model weights: ${PACKED_MODEL_DIR}" >&2
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
echo "Packed FD-GroupTrace HNLL model: ${PACKED_MODEL_DIR}"
echo "PPL results: ${PPL_JSON}"
echo "PPL tag: ${PPL_TAG}"
