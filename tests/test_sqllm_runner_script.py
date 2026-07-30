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


if __name__ == "__main__":
    unittest.main()
