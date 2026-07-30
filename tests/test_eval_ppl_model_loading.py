import types
import unittest
import sys
from unittest import mock

import torch

import eval_ppl


class EvalPplModelLoadingTest(unittest.TestCase):
    def test_loads_anyprecision_model_when_config_has_anyprec(self):
        tokenizer = types.SimpleNamespace(pad_token=None, eos_token="<eos>")
        config = types.SimpleNamespace(anyprec={"seed_precision": 3})
        quantized_model = mock.Mock()
        anyprec_loader = mock.Mock(return_value=quantized_model)
        fake_anyprec_module = types.ModuleType("any_precision.modules.AnyPrecisionForCausalLM")
        fake_anyprec_module.AnyPrecisionForCausalLM = types.SimpleNamespace(
            from_quantized=anyprec_loader)
        modules = {
            "any_precision": types.ModuleType("any_precision"),
            "any_precision.modules": types.ModuleType("any_precision.modules"),
            "any_precision.modules.AnyPrecisionForCausalLM": fake_anyprec_module,
        }

        with mock.patch.dict(sys.modules, modules), \
             mock.patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained") as auto_model:
            model, loaded_tokenizer = eval_ppl._load_model_and_tokenizer(
                "cache/layerwise_packed/layerwise-Llama-3.2-1B-w3-c4_s128_blk2048_g4_iter3_cd4",
                dtype=torch.float16,
                device_map="auto",
            )

        self.assertIs(model, quantized_model)
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertEqual(tokenizer.pad_token, "<eos>")
        auto_model.assert_not_called()
        anyprec_loader.assert_called_once_with(
            "cache/layerwise_packed/layerwise-Llama-3.2-1B-w3-c4_s128_blk2048_g4_iter3_cd4",
            torch_dtype=torch.float16,
        )


if __name__ == "__main__":
    unittest.main()
