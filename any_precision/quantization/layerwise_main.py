
import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
from .activations import accumulate_fast_hnll_base_residual_hvp_hessians, accumulate_fd_grouptrace_hnll_hessians, accumulate_nll_ggn_hessians, accumulate_nll_hvp_hessians, accumulate_saliency_weighted_hessians
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
        overwrite_hessians=False,
        random_state=None,
        num_groups=None,
        num_iterations=3,
        cd_cycles=4,
        assignment_solver="pair",
        hessian_source="saliency",
        nll_hvp_probes=1,
        nll_hvp_layer_chunk_size=0,
        nll_hvp_profile=False,
        nll_hvp_dtype="auto",
        nll_hessian_builder="legacy",
        nll_hessian_group_chunk_size=4,
        nll_hessian_validate_shared_x=False,
        nll_hvp_sdpa_backend="math",
        nll_hvp_engine="autograd",
        nll_hvp_fd_epsilon=1e-3,
        nll_hvp_fd_batched_signs=False,
        nll_base_mode="teacher_real_fisher",
        nll_residual_hvp_probes=1,
        nll_residual_hvp_layer_chunk_size=0,
        nll_residual_hvp_sdpa_backend="math",
        fd_execution_mode="paired_batch",
        fd_scale_mode="activation_rms",
        fd_scale_multiplier=1e-2,
        fd_build_flush_interval=8,
        sub_qlayer=None,
        is_nosal=False,
):

    # ------------------- Set cache paths -------------------

    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]

    initialization_cache_path = (f"{cache_dir}/quantized/"
                          f"{model_name}-w{seed_precision}_orig{seed_precision}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}")

    tokens_cache_path = (f"{cache_dir}/tokens/"
                         f"{model_name}-{dataset}_s{num_examples}_blk{seq_len}.pt")

    saliency_cache_path = (f"{cache_dir}/saliency/"
                          f"{model_name}"
                          f"-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}")

    if hessian_source not in ("saliency", "nll_hvp", "nll_ggn", "nll_base_residual_hvp", "nll_fd_grouptrace"):
        raise ValueError(f"Unsupported hessian_source={hessian_source!r}; expected saliency, nll_hvp, nll_ggn, nll_base_residual_hvp, or nll_fd_grouptrace")

    hessian_suffix = "_nosal" if is_nosal else ""
    if hessian_source == "nll_hvp":
        hessian_suffix = f"_hnll_global_hvp_p{nll_hvp_probes}_lc{nll_hvp_layer_chunk_size}_dt{nll_hvp_dtype}"
        if nll_hessian_builder != "legacy":
            hessian_suffix += f"_hb{nll_hessian_builder}_gcs{nll_hessian_group_chunk_size}"
        if nll_hvp_sdpa_backend != "math":
            hessian_suffix += f"_sdpa{nll_hvp_sdpa_backend}"
        if nll_hvp_engine != "autograd":
            hessian_suffix += f"_eng{nll_hvp_engine}_eps{nll_hvp_fd_epsilon:g}"
            if nll_hvp_fd_batched_signs:
                hessian_suffix += "_fdbs1"
    elif hessian_source == "nll_ggn":
        hessian_suffix = f"_nll_ggn_p{nll_hvp_probes}_dt{nll_hvp_dtype}"
        if nll_hessian_builder != "legacy":
            hessian_suffix += f"_hb{nll_hessian_builder}_gcs{nll_hessian_group_chunk_size}"
    elif hessian_source == "nll_base_residual_hvp":
        hessian_suffix = f"_fast_hnll_base{nll_base_mode}_residual_p{nll_residual_hvp_probes}_lc{nll_residual_hvp_layer_chunk_size}_dt{nll_hvp_dtype}"
        if nll_hessian_builder != "legacy":
            hessian_suffix += f"_hb{nll_hessian_builder}_gcs{nll_hessian_group_chunk_size}"
        if nll_residual_hvp_sdpa_backend != "math":
            hessian_suffix += f"_sdpa{nll_residual_hvp_sdpa_backend}"

    elif hessian_source == "nll_fd_grouptrace":
        hessian_suffix = f"_hnll_fd_grouptrace_p{nll_hvp_probes}_mode{fd_execution_mode}_scale{fd_scale_mode}_mul{fd_scale_multiplier:g}_flush{fd_build_flush_interval}_dt{nll_hvp_dtype}"
        if nll_hessian_builder != "legacy":
            hessian_suffix += f"_hb{nll_hessian_builder}_gcs{nll_hessian_group_chunk_size}"
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
    logging.info(f"Hessians cache path: {hessians_cache_path}")
    logging.info(f"Quantized cache path: {quantized_cache_path}")
    logging.info(f"Model output path: {model_output_path}")
    logging.info(f"Overwrite hessians: {overwrite_hessians}")
    logging.info(f"Hessian source: {hessian_source}")
    logging.info(f"NLL HVP probes: {nll_hvp_probes}")
    logging.info(f"NLL HVP layer chunk size: {nll_hvp_layer_chunk_size}")
    logging.info(f"NLL HVP profile: {nll_hvp_profile}")
    logging.info(f"NLL HVP dtype: {nll_hvp_dtype}")
    logging.info(f"NLL Hessian builder: {nll_hessian_builder}")
    logging.info(f"NLL Hessian group chunk size: {nll_hessian_group_chunk_size}")
    logging.info(f"NLL Hessian validate shared-X: {nll_hessian_validate_shared_x}")
    logging.info(f"NLL HVP SDPA backend: {nll_hvp_sdpa_backend}")
    logging.info(f"NLL HVP engine: {nll_hvp_engine}")
    logging.info(f"NLL HVP finite-difference epsilon: {nll_hvp_fd_epsilon}")
    logging.info(f"NLL HVP finite-difference batched +/- signs: {nll_hvp_fd_batched_signs}")
    logging.info(f"NLL base mode: {nll_base_mode}")
    logging.info(f"NLL residual HVP probes: {nll_residual_hvp_probes}")
    logging.info(f"NLL residual HVP layer chunk size: {nll_residual_hvp_layer_chunk_size}")
    logging.info(f"NLL residual HVP SDPA backend: {nll_residual_hvp_sdpa_backend}")
    logging.info(f"FD GroupTrace execution mode: {fd_execution_mode}")
    logging.info(f"FD GroupTrace scale mode: {fd_scale_mode}")
    logging.info(f"FD GroupTrace scale multiplier: {fd_scale_multiplier}")
    logging.info(f"FD GroupTrace build flush interval: {fd_build_flush_interval}")

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
    logging.info(f"Getting Hessians for {dataset} with sequence length {seq_len} and {num_examples} examples")
    if hessian_source == "nll_hvp":
        from_cache = accumulate_nll_hvp_hessians(
            analyzer, tokens, hessians_cache_path, num_groups,
            num_probes=nll_hvp_probes, random_state=random_state,
            layer_chunk_size=nll_hvp_layer_chunk_size,
            profile=nll_hvp_profile,
            curvature_dtype=nll_hvp_dtype,
            hessian_builder=nll_hessian_builder,
            hessian_group_chunk_size=nll_hessian_group_chunk_size,
            validate_shared_x=nll_hessian_validate_shared_x,
            sdpa_backend=nll_hvp_sdpa_backend,
            hvp_engine=nll_hvp_engine,
            fd_epsilon=nll_hvp_fd_epsilon,
            fd_batched_signs=nll_hvp_fd_batched_signs,
        )
    elif hessian_source == "nll_ggn":
        from_cache = accumulate_nll_ggn_hessians(
            analyzer, tokens, hessians_cache_path, num_groups,
            num_probes=nll_hvp_probes, random_state=random_state,
            layer_chunk_size=nll_hvp_layer_chunk_size,
            profile=nll_hvp_profile,
            curvature_dtype=nll_hvp_dtype,
            hessian_builder=nll_hessian_builder,
            hessian_group_chunk_size=nll_hessian_group_chunk_size,
            validate_shared_x=nll_hessian_validate_shared_x,
        )
    elif hessian_source == "nll_base_residual_hvp":
        from_cache = accumulate_fast_hnll_base_residual_hvp_hessians(
            analyzer, tokens, hessians_cache_path, num_groups,
            num_probes=nll_residual_hvp_probes, random_state=random_state,
            layer_chunk_size=nll_residual_hvp_layer_chunk_size,
            profile=nll_hvp_profile,
            curvature_dtype=nll_hvp_dtype,
            hessian_builder=nll_hessian_builder,
            hessian_group_chunk_size=nll_hessian_group_chunk_size,
            validate_shared_x=nll_hessian_validate_shared_x,
            base_mode=nll_base_mode,
            sdpa_backend=nll_residual_hvp_sdpa_backend,
        )
    elif hessian_source == "nll_fd_grouptrace":
        from_cache = accumulate_fd_grouptrace_hnll_hessians(
            analyzer, tokens, hessians_cache_path, num_groups,
            num_probes=nll_hvp_probes, random_state=random_state,
            layer_chunk_size=nll_hvp_layer_chunk_size,
            profile=nll_hvp_profile,
            curvature_dtype=nll_hvp_dtype,
            hessian_builder=nll_hessian_builder,
            hessian_group_chunk_size=nll_hessian_group_chunk_size,
            validate_shared_x=nll_hessian_validate_shared_x,
            fd_execution_mode=fd_execution_mode,
            fd_scale_mode=fd_scale_mode,
            fd_scale_multiplier=fd_scale_multiplier,
            fd_build_flush_interval=fd_build_flush_interval,
        )
    else:
        from_cache = accumulate_saliency_weighted_hessians(analyzer, tokens, saliency_cache_path, hessians_cache_path, num_groups)
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
        assignment_solver=assignment_solver,
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












