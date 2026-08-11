import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
from .activations import accumulate_saliency_weighted_hessians
from .nll_hvp_saliency import collect_full_nll_hvp_saliencies, complete_saliency_cache_exists, log_cached_hessian_diagnostics
from .layerwise_quantize import seed
from .pack import pack
from .datautils import get_tokens
import datetime
import sys

import warnings

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_hessian_source(hessian_source: str, curvature_mode: str = None) -> str:
    if curvature_mode is None:
        return hessian_source
    if hessian_source != "saliency" and hessian_source != curvature_mode:
        raise ValueError(
            f"Conflicting curvature selectors: hessian_source={hessian_source!r}, "
            f"curvature_mode={curvature_mode!r}"
        )
    return curvature_mode


def _nll_global_hvp_tag(num_probes: int, seed_value: int, layer_chunk_size: int) -> str:
    return f"nll_global_hvp_p{num_probes}_seed{seed_value}_lc{layer_chunk_size}_sum"


def layerwise_nuq(
        model,
        seed_precision=DEFAULT_SEED_PRECISION,
        mode='pack',
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,
        dataset=DEFAULT_DATASET, seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,
        cpu_count=None,
        overwrite_tokens=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        overwrite_saliency=False,
        overwrite_hessians=False,
        random_state=None,
        num_groups=None,
        num_iterations=3,
        cd_cycles=4,
        hessian_source="saliency",
        curvature_mode=None,
        nll_hvp_probes=1,
        nll_hvp_seed=0,
        nll_hvp_layer_chunk_size=1,
        nll_hvp_sdpa_backend="math",
        nll_hvp_saved_tensors_device="cpu",
        sub_hessian=None,
        sub_qlayer=None,
        is_nosal=False,
):

    # ------------------- Set cache paths -------------------

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]
    hessian_source = _resolve_hessian_source(hessian_source, curvature_mode)
    if hessian_source not in ("saliency", "nll_global_hvp"):
        raise ValueError(
            f"Unsupported hessian_source={hessian_source!r}; expected 'saliency' or 'nll_global_hvp'"
        )

    initialization_cache_path = (f"{cache_dir}/quantized/"
                          f"{model_name}-w{seed_precision}_orig{seed_precision}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}")

    tokens_cache_path = (f"{cache_dir}/tokens/"
                         f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt")

    saliency_cache_path = (f"{cache_dir}/saliency/"
                          f"{model_name}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}")

    hessian_suffix = "_nosal" if is_nosal else ""
    if hessian_source == "nll_global_hvp":
        nll_hvp_cache_tag = _nll_global_hvp_tag(
            nll_hvp_probes,
            nll_hvp_seed,
            nll_hvp_layer_chunk_size,
        )
        saliency_cache_path = (f"{cache_dir}/saliency_nll_global_hvp/"
                              f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}"
                              f"_g{num_groups}_{nll_hvp_cache_tag}")
        hessians_cache_path = (f"{cache_dir}/hessians_nll_global_hvp/"
                              f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}"
                              f"_g{num_groups}_{nll_hvp_cache_tag}")
        hessian_suffix = f"_{nll_hvp_cache_tag}"
    else:
        hessians_cache_path = (f"{cache_dir}/hessians/"
                              f"{model_name}"
                              f"-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}{hessian_suffix}")

    quantized_cache_path = (f"{cache_dir}/layerwise_quantized/"
                          f"{model_name}-w{seed_precision}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}_iter{num_iterations}_cd{cd_cycles}{hessian_suffix}")

    model_output_path = (f"{cache_dir}/layerwise_packed/"
                         f"layerwise-{model_name}-w{seed_precision}"
                         f"-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}_iter{num_iterations}_cd{cd_cycles}{hessian_suffix}")


    # Logging with time sans date, level name, and message
    log_dir = "logs_layer"
    log_file_name = os.path.basename(quantized_cache_path)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_file_name = f"{log_file_name}_{timestamp}"

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s | %(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/{log_file_name}.txt"),
        ]
    )

    logging.info(f"Initialization cache path: {initialization_cache_path}")
    logging.info(f"Tokens cache path: {tokens_cache_path}")
    logging.info(f"Saliency cache path: {saliency_cache_path}")
    logging.info(f"Hessians cache path: {hessians_cache_path}")
    logging.info(f"Quantized cache path: {quantized_cache_path}")
    logging.info(f"Model output path: {model_output_path}")
    logging.info(f"Hessian source: {hessian_source}")
    logging.info(f"Overwrite saliency: {overwrite_saliency}")
    logging.info(f"Overwrite hessians: {overwrite_hessians}")
    if hessian_source == "nll_global_hvp":
        logging.info(f"NLL global HVP probes: {nll_hvp_probes}")
        logging.info(f"NLL global HVP seed: {nll_hvp_seed}")
        logging.info(f"NLL global HVP layer chunk size: {nll_hvp_layer_chunk_size}")
        logging.info(f"NLL global HVP SDPA backend: {nll_hvp_sdpa_backend}")
        logging.info(f"NLL global HVP saved tensors device: {nll_hvp_saved_tensors_device}")

    # ------------------- Log mode and other options -------------------

    assert mode in ['tokens', 'hessians', 'quantize', 'pack'], \
        "mode must be one of 'tokens', 'hessians', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."

    if overwrite_tokens:
        if not overwrite_quantize:
            logging.warning("Quantized model needs to be recalculated if tokens are recalculated. "
                            "Setting overwrite_quantize to True.")
            overwrite_quantize = True

    if overwrite_quantize:
        if not overwrite_pack:
            logging.warning("Packed model needs to be recalculated if parent model is recalculated. "
                            "Setting overwrite_pack to True.")
            overwrite_pack = True

    if overwrite_hessians:
        if not overwrite_quantize:
            logging.warning("Quantized model needs to be recalculated if Hessians are recalculated. "
                            "Setting overwrite_quantize to True.")
            overwrite_quantize = True
        if not overwrite_pack:
            logging.warning("Packed model needs to be recalculated if Hessians are recalculated. "
                            "Setting overwrite_pack to True.")
            overwrite_pack = True

    if overwrite_saliency:
        overwrite_hessians = True
        overwrite_quantize = True
        overwrite_pack = True

    if mode == 'tokens':
        logging.info("Running: [Tokens]")
    elif mode == 'hessians':
        logging.info("Running: [Tokens -> Hessians]")
    elif mode == 'quantize':
        logging.info("Running: [Tokens -> Hessians -> Quantize]")
    else:
        logging.info("Running: [Tokens -> Hessians -> Quantize -> Pack]")

    logging.info(f"Running Non-Uniform Layerwise Quantization on {model_name} with precision {seed_precision} "
                 f"using {dataset} for calibration")

    # ------------------- Load model -------------------

    analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)
    module_names = analyzer.module_names
    if sub_hessian is not None:
        logging.warning("sub_hessian is accepted for CLI compatibility but is not used by layerwise_nuq.")

    # ------------------- Get tokens -------------------

    logging.info("------------------- Get tokens -------------------")
    logging.info(f"Getting tokens for {dataset} with sequence length {seq_len} and {num_examples} examples")
    tokens = get_tokens(dataset, "train", analyzer.tokenizer, seq_len, num_examples, tokens_cache_path, random_state)
    logging.info("Tokens loading complete.")

    if mode == 'tokens':
        return

    # ------------------- Get Hessians -------------------
    logging.info("------------------- Get Hessians -------------------")
    if overwrite_hessians and os.path.exists(hessians_cache_path):
        logging.info(f"Detected cached Hessians at {hessians_cache_path}. Will delete and recalculate.")
        shutil.rmtree(hessians_cache_path)

    if hessian_source == "nll_global_hvp":
        if overwrite_saliency and os.path.exists(saliency_cache_path):
            logging.info(f"Detected cached NLL global HVP saliency at {saliency_cache_path}. Will delete and recalculate.")
            shutil.rmtree(saliency_cache_path)
        if overwrite_saliency or not complete_saliency_cache_exists(
            analyzer,
            saliency_cache_path,
            num_examples,
            seq_len,
            num_groups,
            num_probes=nll_hvp_probes,
            base_seed=nll_hvp_seed,
            layer_chunk_size=nll_hvp_layer_chunk_size,
        ):
            collect_full_nll_hvp_saliencies(
                analyzer=analyzer,
                input_tokens=tokens,
                output_folder=saliency_cache_path,
                num_groups=num_groups,
                num_probes=nll_hvp_probes,
                base_seed=nll_hvp_seed,
                layer_chunk_size=nll_hvp_layer_chunk_size,
                overwrite=overwrite_saliency,
                sdpa_backend=nll_hvp_sdpa_backend,
                saved_tensors_device=nll_hvp_saved_tensors_device,
            )

    logging.info(f"Getting Hessians for {dataset} with sequence length {seq_len} and {num_examples} examples")
    from_cache = accumulate_saliency_weighted_hessians(analyzer, tokens, saliency_cache_path, hessians_cache_path, num_groups)
    if hessian_source == "nll_global_hvp":
        log_cached_hessian_diagnostics(analyzer, hessians_cache_path)
    logging.info("Hessians loading complete.")

    if mode == 'hessians':
        return
    if not from_cache:
        # We dropped the layers while calculating the Hessians, so we need to reload the analyzer
        analyzer = get_analyzer(model, yaml_path=yaml_path, include_tokenizer=True)

    # ------------------- Check initialization cache -------------------

    if not os.path.exists(initialization_cache_path):
        logging.info(f"Initialization cache path {initialization_cache_path} does not exist. Need to provide it.")
        return

    # ------------------- Quantize: Seed -------------------

    logging.info("------------------- Quantize -------------------")

    # Calculate or load parent
    logging.info(f"Beginning {seed_precision}-bit Non-Uniform Layerwise Quantization...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)

    # this skips over existing layers in the cache, and doesn't overwrite them
    seed(
        analyzer=analyzer,
        module_names=module_names,
        initialization_path=initialization_cache_path,
        hessians_path=hessians_cache_path,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        cpu_count=cpu_count,
        num_iterations=num_iterations,
        cd_cycles=cd_cycles,
        sub_qlayer=sub_qlayer,
    )

    if mode == 'quantize':
        return

    analyzer.drop_original_weights()  # drop the original weights to save memory

    logging.info("Quantization(Seed) complete.")

    # ------------------- Pack -------------------
    logging.info("------------------- Pack -------------------")

    # check for non-empty directory
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and "
                         f"re-pack.")
            shutil.rmtree(model_output_path)
        else:
            # if the user doesn't want to overwrite the pack, but the directory is not empty, skip packing
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return

    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=seed_precision,
        cpu_count=cpu_count,
    )

    logging.info("Packing complete.")








