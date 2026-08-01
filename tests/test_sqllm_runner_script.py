from pathlib import Path
import unittest


class SqllmRunnerScriptTest(unittest.TestCase):
    def test_sqllm_runner_uses_c4_and_anyprec_packed_output(self):
        script = Path("scripts/run_sqllm_llama32_1b_c4_eval_ppl.sh")
        text = script.read_text()

        self.assertIn('DATASET="${DATASET:-c4}"', text)
        self.assertIn('python quantize.py "${MODEL_NAME}"', text)
        self.assertNotIn("layerwise_nuq.py", text)
        self.assertIn('${CACHE_DIR}/packed/anyprec-${MODEL_BASENAME}-w${BITS}_orig${BITS}-${DATASET}_s${NUM_EXAMPLES}_blk${SEQ_LEN}', text)
        self.assertIn("--method \"${EVAL_METHOD}\"", text)

    def test_guidedquant_runner_repackages_incomplete_output(self):
        script = Path("scripts/run_lnq_guidedquant_llama32_1b_c4_eval_ppl.sh")
        text = script.read_text()

        self.assertIn("has_packed_model()", text)
        self.assertIn("Detected incomplete packed GuidedQuant model directory", text)
        self.assertIn("missing model weights", text)
        self.assertIn('layerwise_overwrite_args+=(--overwrite_pack)', text)

    def test_guidedquant_pack_eval_only_runner_does_not_call_quantize(self):
        script = Path("scripts/run_lnq_guidedquant_pack_eval_only_llama32_1b_c4.sh")
        text = script.read_text()

        self.assertNotIn("python quantize.py", text)
        self.assertIn('QUANTIZED_DIR="${CACHE_DIR}/layerwise_quantized/', text)
        self.assertIn("refusing to quantize again", text)
        self.assertIn("python layerwise_nuq.py", text)
        self.assertIn("--mode pack", text)
        self.assertIn("--overwrite_pack", text)
        self.assertIn("python eval_ppl.py", text)
    def test_hnll_cd_runner_uses_nll_hvp_hessian_and_distinct_outputs(self):
        script = Path("scripts/run_lnq_guidedquant_hnll_cd_c4_eval_ppl.sh")
        text = script.read_text()

        self.assertIn('CACHE_DIR="${CACHE_DIR:-cache_hnll_cd}"', text)
        self.assertIn('RESULT_SUFFIX="hnll_cd"', text)
        self.assertIn('HESSIAN_SUFFIX="hnll_global_hvp_p${NLL_HVP_PROBES}_lc${NLL_HVP_LAYER_CHUNK_SIZE}_dt${NLL_HVP_DTYPE}"', text)
        self.assertIn('ASSIGNMENT_SOLVER="cd"', text)
        self.assertIn('NLL_HVP_PROBES="${NLL_HVP_PROBES:-1}"', text)
        self.assertIn('NLL_HVP_LAYER_CHUNK_SIZE="${NLL_HVP_LAYER_CHUNK_SIZE:-0}"', text)
        self.assertIn('NLL_HVP_DTYPE="${NLL_HVP_DTYPE:-auto}"', text)
        self.assertIn("--hessian_source nll_hvp", text)
        self.assertIn('--nll_hvp_probes "${NLL_HVP_PROBES}"', text)
        self.assertIn('--nll_hvp_layer_chunk_size "${NLL_HVP_LAYER_CHUNK_SIZE}"', text)
        self.assertIn('--nll_hvp_dtype "${NLL_HVP_DTYPE}"', text)
        self.assertIn('"${hnll_profile_args[@]}" \\', text)
        self.assertIn("has_packed_model()", text)
        self.assertIn("missing model weights", text)
        self.assertNotIn("pair", text.lower())
        self.assertNotIn("triton", text.lower())


    def test_hnll_fast_sharedx_runner_uses_combined_builder(self):
        script = Path("scripts/run_lnq_guidedquant_hnll_fast_sharedx_cd_c4_eval_ppl.sh")
        text = script.read_text()

        self.assertIn('CACHE_DIR="${CACHE_DIR:-cache_hnll_fast_sharedx_cd}"', text)
        self.assertIn('RESULT_SUFFIX="hnll_fast_sharedx_cd"', text)
        self.assertIn('NLL_HESSIAN_BUILDER="batched_shared_x"', text)
        self.assertIn('NLL_HVP_SDPA_BACKEND="${NLL_HVP_SDPA_BACKEND:-math}"', text)
        self.assertIn('NLL_HESSIAN_GROUP_CHUNK_SIZE="${NLL_HESSIAN_GROUP_CHUNK_SIZE:-4}"', text)
        self.assertIn('_hb${NLL_HESSIAN_BUILDER}_gcs${NLL_HESSIAN_GROUP_CHUNK_SIZE}', text)
        self.assertIn('_sdpa${NLL_HVP_SDPA_BACKEND}', text)
        self.assertIn('--nll_hessian_builder "${NLL_HESSIAN_BUILDER}"', text)
        self.assertIn('--nll_hvp_sdpa_backend "${NLL_HVP_SDPA_BACKEND}"', text)
        self.assertIn('--nll_hessian_group_chunk_size "${NLL_HESSIAN_GROUP_CHUNK_SIZE}"', text)
        self.assertIn('"${hnll_validate_shared_x_args[@]}" \\', text)
        self.assertIn('ASSIGNMENT_SOLVER="cd"', text)
        self.assertNotIn("pair", text.lower())
        self.assertNotIn("triton", text.lower())
if __name__ == "__main__":
    unittest.main()





