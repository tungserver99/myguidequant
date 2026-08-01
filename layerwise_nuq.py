import argparse
from any_precision.quantization import layerwise_nuq

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("model", type=str, help="The model to quantize")
    parser.add_argument("--seed_precision", type=int, help="The precision to quantize the seed to")
    parser.add_argument("--mode", type=str, default="pack", help="The mode to run in")
    parser.add_argument("--yaml_path", type=str, help="The path to the architecture config yaml file")
    parser.add_argument("--cache_dir", type=str, help="The directory to cache results in")
    parser.add_argument("--dataset", type=str, help="The dataset to use")
    parser.add_argument("--seq_len", type=int, help="The sequence length to use")
    parser.add_argument("--num_examples", type=int, help="The number of examples to use")
    parser.add_argument("--cpu_count", type=int, help="The number of CPUs to use for parallelization")
    parser.add_argument("--overwrite_quantize", action="store_true",
                        help="Whether to overwrite the quantized model stored to disk")
    parser.add_argument("--overwrite_pack", action="store_true",
                        help="Whether to overwrite the packed model stored to disk")
    parser.add_argument("--random_state", type=int,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    parser.add_argument("--sub_hessian", nargs='+', type=int, default=None,
                         help="(start, end) of layers to use for hessian saving")
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of groups $g$ to use for GuidedQuant Hessian")
    parser.add_argument("--num_iterations", type=int, default=3,
                        help="Number of iterations to run")
    parser.add_argument('--cd_cycles', type=int, default=4,
                        help='Number of CD cycles to run')
    parser.add_argument('--assignment_solver', choices=['cd', 'pair', 'pair_k2'], default='pair',
                        help='Assignment solver to use; pair is the default on this branch')
    parser.add_argument('--hessian_source', choices=['saliency', 'nll_hvp'], default='saliency',
                        help='Hessian source for layerwise LNQ; nll_hvp uses Group-Trace Positive NLL HVP Hessian')
    parser.add_argument('--nll_hvp_probes', type=int, default=1,
                        help='Number of Hutchinson probes for --hessian_source nll_hvp')
    parser.add_argument('--nll_hvp_layer_chunk_size', type=int, default=0,
                        help='Target-layer chunk size for global HNLL HVP; 0 means all target layers')
    parser.add_argument('--nll_hvp_layer_batch_size', dest='nll_hvp_layer_chunk_size', type=int,
                        help='Deprecated alias for --nll_hvp_layer_chunk_size')
    parser.add_argument('--nll_hvp_profile', action='store_true',
                        help='Log per-calibration timing for HNLL forward/backward/HVP/build-H')
    parser.add_argument('--nll_hvp_dtype', choices=['auto', 'current', 'bf16', 'fp16', 'fp32'], default='auto',
                        help='Dtype for HNLL curvature pass; auto uses BF16 on supported CUDA GPUs')
    parser.add_argument('--nll_hessian_builder', choices=['legacy', 'batched', 'batched_shared_x'], default='legacy',
                        help='Hessian builder for --hessian_source nll_hvp')
    parser.add_argument('--nll_hessian_group_chunk_size', type=int, default=4,
                        help='Group chunk size for batched HNLL Hessian builders')
    parser.add_argument('--nll_hessian_validate_shared_x', action='store_true',
                        help='Validate fused shared-X activations and fall back if they differ')
    parser.add_argument('--nll_hvp_sdpa_backend', choices=['math', 'auto', 'flash', 'efficient', 'cudnn'], default='math',
                        help='SDPA backend for HNLL curvature pass; math is safest for double backward')
    parser.add_argument("--sub_qlayer", nargs='+', type=int, default=None,
                        help="(start, end) of layers to use for quantization")
    parser.add_argument("--is_nosal", type=str2bool, default=False,
                        help="Do not use GuidedQuant Hessian")

    args = parser.parse_args()
    args.sub_hessian = tuple(args.sub_hessian) if args.sub_hessian else None
    args.sub_qlayer = tuple(args.sub_qlayer) if args.sub_qlayer else None

    # only pass options that are not None
    layerwise_nuq(**{k: v for k, v in args.__dict__.items() if v is not None})



